import json
import uuid
from dataclasses import asdict, dataclass, field
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from .agent import Agent, HarnessKnobs


@dataclass
class ProgramRecord:
    """One program an agent wrote for one task, scored by the evaluator pool on that task alone.

    `cell` is its MAP-Elites cell in the task's grid (see PopulationArchive.add_program); `feedback` is the
    benchmark's description of where it fails, which the next repair prompt shows the model.
    """
    task_id: str
    solution: str
    agent_id: str
    generation: int
    scores: Dict[str, float]
    cell: str
    solved: bool = False
    feedback: str = ""
    parent_program_id: Optional[str] = None
    usage: Dict[str, Any] = field(default_factory=dict)  # the call's tokens and finish_reason, when the client reports them
    id: str = field(default_factory=lambda: f"prog_{uuid.uuid4().hex[:10]}")


def weighted_score(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    """sum_j w_j mu_j over the evaluators that scored this program, renormalised over their weights so a
    program scored on a pruned subset is not ranked below one scored on the full pool."""
    total_w = sum(weights.get(k, 0.0) for k in scores)
    if total_w <= 0:
        return sum(scores.values()) / len(scores) if scores else 0.0
    return sum(weights.get(k, 0.0) * v for k, v in scores.items()) / total_w


class PopulationArchive:
    """Maintains the population history, Pareto frontier, lineage tracking, and Gap 5 Per-Evaluator Dimension Archive."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.best_agent_id: Optional[str] = None
        self.generation_records: List[Dict[str, Any]] = []
        # Gap 5: Per-Evaluator Dimension Elite Archive
        # Maps metric_name -> {"agent_id": str, "score": float, "generation": int, "prompt": str, "mutation": str}
        self.dimension_elites: Dict[str, Dict[str, Any]] = {}

        # Program view (program-archive mode, one archive for the whole run): every program written for a
        # task, a MAP-Elites grid per task keeping the best program per cell, and the islands agents live on.
        self.programs: Dict[str, ProgramRecord] = {}
        self.task_cells: Dict[str, Dict[str, str]] = {}  # task_id -> {cell: program_id}
        self.solved_tasks: Dict[str, str] = {}  # task_id -> id of the first program passing every demonstration
        self.islands: List[List[str]] = []  # island -> agent ids (an agent can sit on several after migration)
        # (parent program id or FRESH, Delta_F) per task: the observations the per-task TPE splits
        self.task_trials: Dict[str, List[Tuple[str, float]]] = {}

    FRESH = "fresh"  # program-parent choice meaning "write the task from scratch"

    def add_agent(self, agent: Agent, generation_metadata: Optional[Dict[str, Any]] = None):
        """Adds an evaluated agent to the population archive and updates per-dimension elites."""
        self.agents[agent.id] = agent
        
        # Update overall best agent tracking
        if self.best_agent_id is None:
            self.best_agent_id = agent.id
        else:
            current_best = self.agents[self.best_agent_id]
            if agent.fitness > current_best.fitness:
                self.best_agent_id = agent.id

        # Gap 5: Update Per-Evaluator Dimension Elite Archive
        if agent.metrics:
            for metric_name, score in agent.metrics.items():
                current_elite = self.dimension_elites.get(metric_name)
                if current_elite is None or score > current_elite["score"]:
                    self.dimension_elites[metric_name] = {
                        "agent_id": agent.id,
                        "score": float(score),
                        "generation": agent.generation,
                        "prompt": agent.system_prompt,
                        "mutation": agent.mutation_type,
                        "fitness": float(agent.fitness),
                    }

        # Record generation summary
        record = {
            "generation": agent.generation,
            "agent_id": agent.id,
            "parent_id": agent.parent_id,
            "fitness": agent.fitness,
            "cost_spent": agent.cost_spent,
            "mutation_type": agent.mutation_type,
            "temperature": agent.harness_knobs.temperature,
            "max_retries": agent.harness_knobs.max_retries,
            "active_evaluators_count": len(agent.active_evaluators),
        }
        if agent.metrics:
            for k, v in agent.metrics.items():
                record[k] = v
        if generation_metadata:
            record.update(generation_metadata)

        self.generation_records.append(record)

    # ------------------------------------------------------------------ program view

    def add_program(self, program: ProgramRecord, weights: Dict[str, float]) -> bool:
        """Adds a program and lets it compete for its MAP-Elites cell; returns True if it took the cell.

        A cell is (which demonstration pairs the program passes, how many outputs have the right shape), so a
        program right on pair 1 and one right on pair 3 are both kept even when one scores lower overall
        (OpenEvolve's per-island MAP-Elites grid, here per task). Within a cell the higher weighted score wins.
        """
        self.programs[program.id] = program
        if program.solved and program.task_id not in self.solved_tasks:
            self.solved_tasks[program.task_id] = program.id
        cells = self.task_cells.setdefault(program.task_id, {})
        incumbent = self.programs.get(cells.get(program.cell, ""))
        if incumbent is None or weighted_score(program.scores, weights) > weighted_score(incumbent.scores, weights):
            cells[program.cell] = program.id
            return True
        return False

    def task_elites(self, task_id: str) -> List[ProgramRecord]:
        """The programs currently holding a cell of the task's grid."""
        return [self.programs[pid] for pid in self.task_cells.get(task_id, {}).values()]

    def task_best(self, task_id: str, weights: Dict[str, float]) -> Optional[ProgramRecord]:
        elites = self.task_elites(task_id)
        return max(elites, key=lambda p: (p.solved, weighted_score(p.scores, weights)), default=None)

    def task_best_score(self, task_id: str, weights: Dict[str, float]) -> float:
        best = self.task_best(task_id, weights)
        return weighted_score(best.scores, weights) if best else 0.0

    def is_solved(self, task_id: str) -> bool:
        return task_id in self.solved_tasks

    def record_task_trial(self, task_id: str, parent_choice: str, delta_f: float) -> None:
        self.task_trials.setdefault(task_id, []).append((parent_choice, float(delta_f)))

    def task_pareto_agents(self, weights: Dict[str, float]) -> List[str]:
        """Agents holding the best program on at least one task: kept as parents even with a lower mean."""
        owners = set()
        for task_id in self.task_cells:
            best = self.task_best(task_id, weights)
            if best is not None:
                owners.add(best.agent_id)
        return sorted(owners)

    # ------------------------------------------------------------------ islands

    def init_islands(self, n: int, seed_agent_ids: List[str]) -> None:
        self.islands = [list(seed_agent_ids) for _ in range(max(1, n))]

    def add_to_island(self, island: int, agent_id: str) -> None:
        if agent_id not in self.islands[island]:
            self.islands[island].append(agent_id)

    def island_agents(self, island: int) -> List[Agent]:
        return [self.agents[a] for a in self.islands[island] if a in self.agents]

    def migrate_ring(self) -> List[Tuple[int, int, str]]:
        """Copies each island's best agent to its two ring neighbours (OpenEvolve's default topology).
        Returns the (from, to, agent_id) moves made."""
        n = len(self.islands)
        if n < 2:
            return []
        bests = []
        for i in range(n):
            members = self.island_agents(i)
            bests.append(max(members, key=lambda a: a.fitness).id if members else None)
        moves = []
        for i, agent_id in enumerate(bests):
            if agent_id is None:
                continue
            for j in {(i + 1) % n, (i - 1) % n} - {i}:
                if agent_id not in self.islands[j]:
                    self.islands[j].append(agent_id)
                    moves.append((i, j, agent_id))
        return moves

    # ------------------------------------------------------------------ JSON checkpoint

    def to_state(self) -> Dict[str, Any]:
        agents = {}
        for a in self.agents.values():
            d = a.to_dict()
            d.pop("task_solutions", None)  # programs live in self.programs
            agents[a.id] = d
        return {
            "agents": agents,
            "best_agent_id": self.best_agent_id,
            "generation_records": self.generation_records,
            "dimension_elites": self.dimension_elites,
            "programs": {pid: asdict(p) for pid, p in self.programs.items()},
            "task_cells": self.task_cells,
            "solved_tasks": self.solved_tasks,
            "islands": self.islands,
            "task_trials": self.task_trials,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "PopulationArchive":
        archive = cls()
        for agent_id, d in state["agents"].items():
            d = dict(d)
            d["harness_knobs"] = HarnessKnobs.from_dict(d["harness_knobs"])
            archive.agents[agent_id] = Agent(**d)
        archive.best_agent_id = state.get("best_agent_id")
        archive.generation_records = list(state.get("generation_records", []))
        archive.dimension_elites = dict(state.get("dimension_elites", {}))
        archive.programs = {pid: ProgramRecord(**p) for pid, p in state.get("programs", {}).items()}
        archive.task_cells = {t: dict(c) for t, c in state.get("task_cells", {}).items()}
        archive.solved_tasks = dict(state.get("solved_tasks", {}))
        archive.islands = [list(i) for i in state.get("islands", [])]
        archive.task_trials = {t: [tuple(x) for x in v] for t, v in state.get("task_trials", {}).items()}
        return archive

    def save_json(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_state(), f)
        return path

    def get_best_agent(self) -> Optional[Agent]:
        """Returns the current highest-fitness agent in the archive."""
        if self.best_agent_id and self.best_agent_id in self.agents:
            return self.agents[self.best_agent_id]
        return None

    def get_dimension_elites(self) -> Dict[str, Dict[str, Any]]:
        """Gap 5: Returns the historical peak agent information for every individual evaluator dimension."""
        return dict(self.dimension_elites)

    def get_dimension_archive_df(self) -> pd.DataFrame:
        """Returns a formatted DataFrame of per-evaluator elites for UI display."""
        rows = []
        for metric, data in self.dimension_elites.items():
            rows.append({
                "Metric Dimension": metric,
                "Peak Score": round(data["score"], 4),
                "Generation Found": data["generation"],
                "Elite Agent ID": data["agent_id"],
                "Mutation Type": data["mutation"],
                "Fitness": round(data["fitness"], 4),
                "Prompt Preview": (data["prompt"][:60] + "...") if len(data["prompt"]) > 60 else data["prompt"],
            })
        return pd.DataFrame(rows)

    def detect_silent_regression(
        self,
        current_agent: Agent,
        drop_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """Gap 5: Checks if current_agent has silently regressed on any dimension compared to historical elites."""
        regressions = []
        if not current_agent.metrics or not self.dimension_elites:
            return {"has_regression": False, "regressions": []}

        for metric, elite in self.dimension_elites.items():
            if metric in current_agent.metrics:
                curr_score = current_agent.metrics[metric]
                peak_score = elite["score"]
                delta = peak_score - curr_score
                if peak_score > 0.3 and delta > drop_threshold:
                    regressions.append({
                        "metric": metric,
                        "current_score": round(curr_score, 3),
                        "peak_score": round(peak_score, 3),
                        "regression_magnitude": round(delta, 3),
                        "elite_agent_id": elite["agent_id"],
                    })

        return {
            "has_regression": len(regressions) > 0,
            "regression_count": len(regressions),
            "regressions": regressions,
        }

    def get_all_agents(self) -> List[Agent]:
        """Returns all agents sorted by generation."""
        return sorted(self.agents.values(), key=lambda a: (a.generation, a.created_at))

    def get_pareto_front(self, metric_names: Optional[List[str]] = None) -> List[Agent]:
        """Calculates non-dominated Pareto front agents across specified metrics."""
        agents = list(self.agents.values())
        if not agents:
            return []

        if not metric_names:
            # Default to all metrics present in the first agent with metrics
            all_keys = set()
            for a in agents:
                all_keys.update(a.metrics.keys())
            metric_names = sorted(list(all_keys))

        if not metric_names:
            return agents

        pareto_front: List[Agent] = []
        for i, a1 in enumerate(agents):
            is_dominated = False
            for j, a2 in enumerate(agents):
                if i == j:
                    continue
                # Check if a2 dominates a1 (a2 >= a1 on all metrics, and a2 > a1 on at least one)
                better_or_equal = True
                strictly_better = False
                for m in metric_names:
                    v1 = a1.metrics.get(m, 0.0)
                    v2 = a2.metrics.get(m, 0.0)
                    if v2 < v1:
                        better_or_equal = False
                        break
                    if v2 > v1:
                        strictly_better = True
                
                if better_or_equal and strictly_better:
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_front.append(a1)

        return pareto_front

    def to_dataframe(self) -> pd.DataFrame:
        """Exports a summary of the archive to a pandas DataFrame for UI display."""
        rows = []
        for a in self.get_all_agents():
            row = {
                "Generation": a.generation,
                "Agent ID": a.id,
                "Parent ID": a.parent_id or "Root",
                "Fitness": round(a.fitness, 4),
                "Cost ($)": round(a.cost_spent, 5),
                "Mutation": a.mutation_type,
                "Temp": round(a.harness_knobs.temperature, 2),
                "Retries": a.harness_knobs.max_retries,
                "Prompt Preview": (a.system_prompt[:60] + "...") if len(a.system_prompt) > 60 else a.system_prompt,
            }
            for k, v in a.metrics.items():
                row[k] = round(v, 3)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_detailed_dataframe(self) -> pd.DataFrame:
        """Exports complete parameters, evaluations, weights, and costs into a comprehensive DataFrame matching exact schema."""
        if not self.generation_records:
            return self.to_dataframe()

        exact_columns = [
            "generation", "agent_id", "parent_id", "fitness", "cost_spent", "mutation_type",
            "temperature", "max_retries", "active_evaluators_count", "mu_1_correctness",
            "mu_2_latency", "mu_3_conciseness", "mu_4_reasoning", "mu_5_edge_cases", "mu_6_safety",
            "strategy", "naive_cost_spent", "cumulative_adaptive_cost", "cumulative_naive_cost",
            "task_id", "system_prompt", "top_p", "max_tokens", "delta_f", "cost_saved_this_gen",
            "solution_preview", "weight_mu_1_correctness", "weight_mu_2_latency",
            "weight_mu_3_conciseness", "weight_mu_4_reasoning", "weight_mu_5_edge_cases",
            "weight_mu_6_safety",
        ]

        rows = []
        for rec in self.generation_records:
            row = dict(rec)
            # Flatten weights dictionary if present
            if "weights" in row and isinstance(row["weights"], dict):
                weights_dict = row.pop("weights")
                for w_name, w_val in weights_dict.items():
                    row[f"weight_{w_name}"] = w_val
            
            # Format list of active evaluators as comma-separated string
            if "active_evaluators" in row and isinstance(row["active_evaluators"], list):
                row["active_evaluators"] = ", ".join(row["active_evaluators"])

            # Include full system prompt and knobs if agent exists
            agent_id = row.get("agent_id")
            if agent_id and agent_id in self.agents:
                agent = self.agents[agent_id]
                row["system_prompt"] = agent.system_prompt
                row["top_p"] = agent.harness_knobs.top_p
                row["max_tokens"] = agent.harness_knobs.max_tokens

            # Ensure all exact columns exist in row
            ordered_row = {col: row.get(col, None) for col in exact_columns}
            # Append any additional non-standard metadata keys at end if any
            for k, v in row.items():
                if k not in ordered_row:
                    ordered_row[k] = v

            rows.append(ordered_row)

        df = pd.DataFrame(rows)
        # Reorder columns to place exact_columns first
        ordered_cols = [c for c in exact_columns if c in df.columns] + [c for c in df.columns if c not in exact_columns]
        return df[ordered_cols]

    def export_to_csv(self, filepath: str = "optimization_results.csv") -> str:
        """Saves all optimization metrics, evaluation results, and parameters to a CSV file."""
        df = self.to_detailed_dataframe()
        df.to_csv(filepath, index=False)
        return filepath

import pandas as pd
from typing import List, Dict, Optional, Any
from .agent import Agent


class PopulationArchive:
    """Maintains the population history, Pareto frontier, and lineage tracking."""

    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.best_agent_id: Optional[str] = None
        self.generation_records: List[Dict[str, Any]] = []

    def add_agent(self, agent: Agent, generation_metadata: Optional[Dict[str, Any]] = None):
        """Adds an evaluated agent to the population archive."""
        self.agents[agent.id] = agent
        
        # Update best agent tracking
        if self.best_agent_id is None:
            self.best_agent_id = agent.id
        else:
            current_best = self.agents[self.best_agent_id]
            if agent.fitness > current_best.fitness:
                self.best_agent_id = agent.id

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

    def get_best_agent(self) -> Optional[Agent]:
        """Returns the current highest-fitness agent in the archive."""
        if self.best_agent_id and self.best_agent_id in self.agents:
            return self.agents[self.best_agent_id]
        return None

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

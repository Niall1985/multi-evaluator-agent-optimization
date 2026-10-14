import json
import os
import random
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from core.agent import Agent, HarnessKnobs, AGENT_ARCHETYPES
from core.archive import PopulationArchive, ProgramRecord, weighted_score
from core.llm import LLMClient, make_llm_client
from core.mutator import PromptMutator
from evaluation.pool import EvaluatorPool
from optimization.bayesian_weights import BayesianWeightOptimizer
from optimization.scoring import calculate_cost_penalized_fitness, calculate_delta_f
from optimization.joint_sampler import JointSearchSampler
from optimization.bandit import UCB1MutationBandit, UCB1TaskSelector
from optimization.fanova import FunctionalANOVA
# `tasks/` was renamed to `benchmarks/` on this fork when the ARC-AGI suite was added.
from benchmarks.benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask, get_random_task, get_benchmark_tasks
from benchmarks.arc_challenge import ARC_BENCHMARK_ID, is_arc_benchmark

logger = logging.getLogger(__name__)


class EvolutionController:
    """Evolutionary optimization loop orchestrator implementing Gaps 1, 2, 3, 4, 5, and 6."""

    # Headline metric per suite: never pruned, and kept over its correlate by the redundancy rule
    PRIMARY_METRICS = ("mu_1_correctness", "arc_pass_at_2_test", "arc_pass_at_2_train")

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        lambda_penalty: float = 0.5,
        default_strategy: str = "full_adaptive",
        initial_archetype: str = "General Balanced Assistant",
        selected_task_id: Optional[str] = None,
        inter_call_delay: float = 1.5,
        is_mock: Optional[bool] = None,
        evaluator_pool_factory: Optional[Callable[[LLMClient], EvaluatorPool]] = None,
        reasoning_effort: Optional[str] = None,
        min_max_tokens: int = 0,
        acquisition_type: str = "gp_ucb",
        fanova_window: int = 3,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        rpm: int = 0,
        llm_client: Optional[LLMClient] = None,
        enable_thinking: Optional[bool] = None,
        thinking_budget: Optional[int] = None,
        max_concurrency: int = 1,
        metric_smoothing: Optional[Dict[str, float]] = None,
        evaluate_seed: bool = True,
        evolution_mode: str = "agent",
        n_islands: int = 4,
        tasks_per_child: int = 30,
        migration_interval: int = 5,
        knob_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.lambda_penalty = lambda_penalty
        self.default_strategy = default_strategy
        self.initial_archetype = initial_archetype
        self.selected_task_id = selected_task_id
        self.inter_call_delay = inter_call_delay
        self.is_mock = is_mock
        self.acquisition_type = acquisition_type
        # "agent": one child per generation, scored on every task (the original loop).
        # "program_archive": the archive also keeps every program per task; each generation n_islands children
        # each work on `tasks_per_child` tasks picked by UCB1, repairing an archived program or starting fresh
        # (see _run_archive_generation). Needs a benchmark with program hooks (ARC: benchmarks.arc_challenge.repair).
        if evolution_mode not in ("agent", "program_archive"):
            raise ValueError(f"unknown evolution_mode {evolution_mode!r}")
        self.evolution_mode = evolution_mode
        self.n_islands = max(1, n_islands)
        self.tasks_per_child = max(1, tasks_per_child)
        self.migration_interval = max(1, migration_interval)
        self.task_selector = UCB1TaskSelector()
        # Spec §3.3's W: fANOVA is measured every W generations and prunes after W consecutive low windows
        self.fanova_window = max(1, fanova_window)
        self.csv_filepath = "optimization_results.csv"
        # Tasks of one run (agent calls, then evaluators) in flight at once. 1 = sequential; raise it only for a
        # server that batches requests (vLLM, llama.cpp with --parallel), since every task is one LLM call.
        self.max_concurrency = max(1, max_concurrency)
        # Additive (Laplace) smoothing of a pass-rate metric's mean across tasks: (sum + a) / (n + 2a), {name: a}
        self.metric_smoothing: Dict[str, float] = dict(metric_smoothing or {})

        # Core components. The provider (Groq, or an OpenAI-compatible server such as llama.cpp) is chosen by
        # core.llm.make_llm_client from `provider` / LLM_PROVIDER / base_url / OPENAI_BASE_URL.
        self.llm_client = llm_client or make_llm_client(
            provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
            is_mock=is_mock,
            inter_call_delay=inter_call_delay,
            min_max_tokens=min_max_tokens,
            reasoning_effort=reasoning_effort,
            rpm=rpm,
            enable_thinking=enable_thinking,
            thinking_budget=thinking_budget,
        )
        # Default: the 6 canonical mu evaluators. A factory swaps in a benchmark-specific suite
        # (e.g. benchmarks.arc_challenge.build_arc_evaluator_pool).
        self.evaluator_pool = (
            evaluator_pool_factory(self.llm_client) if evaluator_pool_factory else EvaluatorPool(llm_client=self.llm_client)
        )
        self.metric_names = self.evaluator_pool.get_evaluator_names()
        self.knob_bounds = knob_bounds
        self.bayesian_optimizer = BayesianWeightOptimizer(
            metric_names=self.metric_names,
            acquisition_type=acquisition_type,
            knob_bounds=knob_bounds,
        )
        self.fanova = FunctionalANOVA(evaluator_names=self.metric_names)
        self.mutator = PromptMutator(llm_client=self.llm_client)
        self.bandit = UCB1MutationBandit(arms=self.mutator.get_available_strategies())
        self.sampler = JointSearchSampler(
            evaluator_pool=self.evaluator_pool,
            bayesian_optimizer=self.bayesian_optimizer,
            fanova=self.fanova,
            prune_patience=self.fanova_window,
        )
        self.archive = PopulationArchive()

        # Cumulative tracking telemetry
        self.current_generation = 0
        self.cumulative_adaptive_cost = 0.0
        self.cumulative_naive_cost = 0.0
        self.history_records: List[Dict[str, Any]] = []

        # Initialize seed population. evaluate_seed=False leaves the archive empty for restore_from_csv.
        if evaluate_seed:
            if self.evolution_mode == "program_archive":
                self._initialize_archive_seed()
            else:
                self.initialize_seed(archetype_name=self.initial_archetype, task_id=self.selected_task_id)

    def reset(self, initial_archetype: Optional[str] = None, selected_task_id: Optional[str] = None):
        """Resets the controller state, archive, and optimizers."""
        if initial_archetype:
            self.initial_archetype = initial_archetype
        if selected_task_id is not None:
            self.selected_task_id = selected_task_id
        self.archive = PopulationArchive()
        self.bayesian_optimizer = BayesianWeightOptimizer(
            metric_names=self.metric_names,
            acquisition_type=self.acquisition_type,
            knob_bounds=self.knob_bounds,
        )
        self.task_selector = UCB1TaskSelector()
        self.fanova = FunctionalANOVA(evaluator_names=self.metric_names)
        self.bandit = UCB1MutationBandit(arms=self.mutator.get_available_strategies())
        self.sampler = JointSearchSampler(
            evaluator_pool=self.evaluator_pool,
            bayesian_optimizer=self.bayesian_optimizer,
            fanova=self.fanova,
            prune_patience=self.fanova_window,
        )
        self.current_generation = 0
        self.cumulative_adaptive_cost = 0.0
        self.cumulative_naive_cost = 0.0
        self.history_records = []
        if self.evolution_mode == "program_archive":
            self._initialize_archive_seed()
        else:
            self.initialize_seed(archetype_name=self.initial_archetype, task_id=self.selected_task_id)

    def initialize_seed(
        self,
        archetype_name: Optional[str] = None,
        task_id: Optional[str] = None,
        seed_prompt: Optional[str] = None,
    ):
        """Creates and evaluates the generation 0 seed baseline agent using selected archetype."""
        arch_name = archetype_name or self.initial_archetype
        arch_info = AGENT_ARCHETYPES.get(arch_name, AGENT_ARCHETYPES["General Balanced Assistant"])

        prompt = seed_prompt or arch_info["system_prompt"]
        knobs = arch_info["harness_knobs"]
        id_suffix = arch_info.get("id_suffix", "seed")

        seed_agent = Agent(
            id=f"agent_seed_{id_suffix}_g0",
            parent_id=None,
            generation=0,
            system_prompt=prompt,
            harness_knobs=HarnessKnobs(
                temperature=knobs.temperature,
                max_retries=knobs.max_retries,
                top_p=knobs.top_p,
                max_tokens=knobs.max_tokens,
            ),
            mutation_type=f"seed_{id_suffix}",
            mutation_description=f"Initial seed archetype: {arch_name}",
        )

        # Baseline uniform weights for seed
        uniform_weights = {name: 1.0 / len(self.metric_names) for name in self.metric_names}
        full_subset = self.evaluator_pool.get_evaluator_names()

        # Evaluate seed agent across benchmark (one task, or every task of a whole benchmark)
        selector, tasks = self._resolve_tasks(task_id)
        run = self._run_agent_on_tasks(seed_agent, tasks)
        scores, costs, per_task = self._evaluate_run(seed_agent, tasks, run, full_subset)

        fitness, spent_cost = calculate_cost_penalized_fitness(
            evaluator_scores=scores,
            weights=uniform_weights,
            active_evaluator_costs=costs,
            lambda_penalty=self.lambda_penalty,
        )

        self._record_run_on_agent(seed_agent, selector, tasks, run, scores, fitness, spent_cost, full_subset, per_task)
        agent_solution = seed_agent.last_solution
        naive_cost = self.evaluator_pool.get_full_eval_cost() * len(tasks)

        self.cumulative_adaptive_cost += spent_cost
        self.cumulative_naive_cost += naive_cost

        self.archive.add_agent(
            seed_agent,
            generation_metadata={
                "strategy": "seed",
                "naive_cost_spent": naive_cost,
                "cost_saved_this_gen": 0.0,
                "cumulative_adaptive_cost": self.cumulative_adaptive_cost,
                "cumulative_naive_cost": self.cumulative_naive_cost,
                "task_id": selector,
                "delta_f": 0.0,
                "weights": {k: round(v, 3) for k, v in uniform_weights.items()},
                "solution_preview": agent_solution[:100] + "..." if len(agent_solution) > 100 else agent_solution,
            },
        )
        self.archive.export_to_csv(self.csv_filepath)

        # Extract test case execution details for UI test suite viewer
        test_suite_data = self._build_test_suite_data(selector, tasks, run)

        seed_record = {
            "generation": 0,
            "child_id": seed_agent.id,
            "parent_id": "Root",
            "parent_prompt": "None (Initial Archetype Seed)",
            "child_prompt": seed_agent.system_prompt,
            "solution": agent_solution,
            "strategy": "seed",
            "fitness": fitness,
            "delta_f": 0.0,
            "cost_spent": spent_cost,
            "naive_cost": naive_cost,
            "active_evaluators": full_subset,
            "weights": uniform_weights,
            "metrics": scores,
            "task_id": selector,
            "task_metrics": dict(seed_agent.task_metrics),
            "mutation_type": seed_agent.mutation_type,
            "mutation_goal": seed_agent.mutation_description,
            "silent_regression": {"has_regression": False},
            "test_suite": test_suite_data,
        }
        self.history_records.append(seed_record)

    def restore_from_csv(self, results_csv: str, task_csv: Optional[str] = None) -> int:
        """Rebuilds the archive, history and search state from a stopped run's CSVs so run_generation continues
        after its last completed generation. Returns that generation.

        Restored: agents (prompt, knobs, metrics, fitness), per-task metrics (from `task_csv`), the parent/idea
        trials, the idea study and the GP observations. Not restored: the programs themselves (the CSVs keep
        only a preview) and the RNG state."""
        import pandas as pd

        df = pd.read_csv(results_csv).sort_values("generation")
        tasks_df = pd.read_csv(task_csv) if task_csv else None
        menu = [name for name, _ in self.mutator.get_available_strategies()]

        def num(row, key, default=None):
            value = row.get(key)
            return default if value is None or pd.isna(value) else value

        for _, row in df.iterrows():
            row = row.to_dict()
            gen = int(row["generation"])
            parent_id = num(row, "parent_id")
            parent = self.archive.agents.get(parent_id) if parent_id else None
            metrics = {m: float(row[m]) for m in self.metric_names if num(row, m) is not None}
            weights = {m: float(num(row, f"weight_{m}", 0.0)) for m in self.metric_names}
            task_metrics: Dict[str, Dict[str, float]] = {}
            held_out: Dict[str, float] = {}
            if tasks_df is not None:
                for _, t in tasks_df[tasks_df["generation"] == gen].iterrows():
                    t = t.to_dict()
                    task_metrics[t["task_id"]] = {m: float(t[m]) for m in self.metric_names if num(t, m) is not None}
                    if num(t, "arc_pass_at_2_test_reported") is not None:
                        held_out[t["task_id"]] = float(t["arc_pass_at_2_test_reported"])
            agent = Agent(
                id=row["agent_id"],
                parent_id=parent_id,
                generation=gen,
                system_prompt=row["system_prompt"],
                harness_knobs=HarnessKnobs(
                    temperature=float(row["temperature"]),
                    max_retries=int(row["max_retries"]),
                    top_p=float(num(row, "top_p", 0.95)),
                    max_tokens=int(num(row, "max_tokens", 1024)),
                ),
                metrics=metrics,
                fitness=float(row["fitness"]),
                cost_spent=float(num(row, "cost_spent", 0.0)),
                active_evaluators=list(metrics),
                mutation_type=row["mutation_type"],
                mutation_description="restored from " + results_csv,
                lineage_path=(parent.lineage_path if parent else []) + [row["agent_id"]],
                task_id=num(row, "task_id"),
                task_metrics=task_metrics,
            )
            delta_f = float(num(row, "delta_f", 0.0))
            self.archive.add_agent(
                agent,
                generation_metadata={
                    "strategy": row["strategy"],
                    "delta_f": delta_f,
                    "weights": weights,
                    "naive_cost_spent": num(row, "naive_cost_spent", 0.0),
                    "cost_saved_this_gen": num(row, "cost_saved_this_gen", 0.0),
                    "cumulative_adaptive_cost": num(row, "cumulative_adaptive_cost", 0.0),
                    "cumulative_naive_cost": num(row, "cumulative_naive_cost", 0.0),
                    "task_id": agent.task_id,
                    "solution_preview": num(row, "solution_preview", ""),
                    "soft_probe": bool(num(row, "soft_probe", False)),
                    "fanova_pruned_count": int(num(row, "fanova_pruned_count", 0)),
                },
            )
            if gen > 0:
                # The same updates run_generation makes after scoring a child
                self.sampler.restore_trial(parent_id, agent.mutation_type, delta_f, menu)
                if row["strategy"] in ["no_pruning", "full_adaptive"]:
                    self.bayesian_optimizer.add_observation(
                        weights=weights, delta_f=delta_f, knobs=agent.harness_knobs.to_dict()
                    )
                elif row["strategy"] == "ucb1_bandit":
                    self.bandit.update(arm_name=agent.mutation_type, reward=delta_f)
                if gen % self.fanova_window == 0:
                    self._update_fanova()
            self.history_records.append({
                "generation": gen,
                "child_id": agent.id,
                "parent_id": parent_id or "Root",
                "child_prompt": agent.system_prompt,
                "strategy": row["strategy"],
                "fitness": agent.fitness,
                "delta_f": delta_f,
                "cost_spent": agent.cost_spent,
                "active_evaluators": agent.active_evaluators,
                "weights": weights,
                "metrics": metrics,
                "task_id": agent.task_id,
                "task_metrics": task_metrics,
                "held_out_reported": held_out,
                "mutation_type": agent.mutation_type,
            })
            self.current_generation = gen
            self.cumulative_adaptive_cost = float(num(row, "cumulative_adaptive_cost", 0.0))
            self.cumulative_naive_cost = float(num(row, "cumulative_naive_cost", 0.0))
        return self.current_generation

    # ------------------------------------------------------------------ task resolution & multi-task runs

    def _resolve_tasks(self, task_param: Optional[str] = None) -> Tuple[str, List[BenchmarkTask]]:
        """Resolves the selector to (record id, tasks). A whole benchmark (e.g. `arc_benchmark`) yields
        every one of its tasks; a single task yields [task]; anything else a random canonical task."""
        target = task_param or self.selected_task_id
        if target and target != "All Benchmark Tasks (Suite / Random)":
            tasks = get_benchmark_tasks(target)
            if tasks:
                return (ARC_BENCHMARK_ID if is_arc_benchmark(target) else tasks[0].id), tasks
        task = get_random_task()
        return task.id, [task]

    def _run_agent_on_tasks(self, agent: Agent, tasks: List[BenchmarkTask]) -> Dict[str, Dict[str, Any]]:
        """Calls the agent LLM once per task. Returns {task_id: {"solution", "latency_ms"}}."""

        def call(task: BenchmarkTask) -> Dict[str, Any]:
            start = time.time()
            solution = self._execute_agent_on_task(agent, task)
            return {"solution": solution, "latency_ms": (time.time() - start) * 1000}

        return {task.id: r for task, r in zip(tasks, self._map_tasks(call, tasks))}

    def _map_tasks(self, fn: Callable[[BenchmarkTask], Any], tasks: List[BenchmarkTask]) -> List[Any]:
        """fn(task) for every task, results in task order, up to max_concurrency at once.

        With concurrency > 1, latency_ms (mu_2) includes time queued behind other requests, so compare it only
        between runs at the same concurrency."""
        if self.max_concurrency <= 1 or len(tasks) <= 1:
            return [fn(task) for task in tasks]
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(tasks))) as pool:
            return list(pool.map(fn, tasks))

    def _evaluate_run(
        self,
        agent: Agent,
        tasks: List[BenchmarkTask],
        run: Dict[str, Dict[str, Any]],
        active_subset: List[str],
    ) -> Tuple[Dict[str, float], List[float], Dict[str, Dict[str, float]]]:
        """Runs the active evaluator subset on every task of the run.

        Returns (mean scores across tasks, costs summed across tasks, per-task scores).
        For ARC this makes `arc_pass_at_2_test` the fraction of the benchmark solved, i.e. the official score.
        """
        def evaluate(task: BenchmarkTask):
            r = run[task.id]
            return self.evaluator_pool.evaluate_subset(
                agent=agent,
                task=self._task_to_dict(task),
                agent_output=r["solution"],
                active_subset=active_subset,
                execution_context={"generation_latency_ms": r["latency_ms"]},
            )

        per_task: Dict[str, Dict[str, float]] = {}
        cost_totals: Dict[str, float] = {}
        for task, (scores, costs, _) in zip(tasks, self._map_tasks(evaluate, tasks)):
            per_task[task.id] = scores
            for name, cost in zip([n for n in active_subset if n in scores], costs):
                cost_totals[name] = cost_totals.get(name, 0.0) + cost

        n = max(1, len(tasks))
        mean_scores = {}
        for name in active_subset:
            if any(name in per_task[t.id] for t in tasks):
                alpha = self.metric_smoothing.get(name, 0.0)
                total = sum(per_task[t.id].get(name, 0.0) for t in tasks)
                mean_scores[name] = round((total + alpha) / (n + 2 * alpha), 6 if alpha else 4)
        summed_costs = [cost_totals[name] for name in mean_scores]
        return mean_scores, summed_costs, per_task

    @staticmethod
    def _build_test_suite_data(
        selector: str,
        tasks: List[BenchmarkTask],
        run: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Per-case execution detail for the UI's test-suite viewer.

        Upstream read this off the mu_1 evaluator result for a single task; a run here can cover a whole
        benchmark (every ARC task), and `_evaluate_run` does not keep per-evaluator details, so the cases
        are re-run per task. Each case carries its task id so a multi-task run stays readable.
        """
        from evaluation.runner import CodeExecutionRunner

        cases: List[Dict[str, Any]] = []
        for task in tasks:
            rep = CodeExecutionRunner.run_task(
                task=task,
                agent_output=run[task.id]["solution"],
                include_test_cases=True,
                include_edge_cases=True,
            )
            cases.extend(
                {
                    "task_id": task.id,
                    "index": c.case_index,
                    "type": c.case_type,
                    "args": str(c.args),
                    "expected": str(c.expected),
                    "actual": str(c.actual),
                    "passed": c.passed,
                    "error": c.error or c.exception_raised,
                }
                for c in rep.case_results
            )

        primary = tasks[0] if len(tasks) == 1 else None
        return {
            "task_id": selector,
            "task_name": primary.name if primary else f"{len(tasks)} tasks",
            "task_description": primary.description if primary else "",
            "entry_point": primary.entry_point if primary else "solve",
            "cases": cases,
            "total_count": len(cases),
            "passed_count": sum(1 for c in cases if c.get("passed")),
        }

    @staticmethod
    def _record_run_on_agent(
        agent: Agent,
        selector: str,
        tasks: List[BenchmarkTask],
        run: Dict[str, Dict[str, Any]],
        scores: Dict[str, float],
        fitness: float,
        spent_cost: float,
        active_subset: List[str],
        per_task: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        agent.metrics = scores
        agent.fitness = fitness
        agent.cost_spent = spent_cost
        agent.active_evaluators = active_subset
        agent.task_id = selector
        agent.task_solutions = {tid: r["solution"] for tid, r in run.items()}
        agent.task_metrics = dict(per_task or {})
        if len(tasks) == 1:
            agent.last_solution = run[tasks[0].id]["solution"]
        else:
            # Whole-benchmark run: keep every program, separated by task headers, for the inspector views
            agent.last_solution = "\n\n".join(
                f"# ===== {tid} =====\n{r['solution']}" for tid, r in run.items()
            )

    def run_generation(self, strategy: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs a single generation step of the multi-objective optimization loop across 5 conditions."""
        if self.evolution_mode == "program_archive":
            return self._run_archive_generation()
        self.current_generation += 1
        active_strategy = strategy or self.default_strategy
        gen = self.current_generation

        # Normalize strategy aliases
        if active_strategy == "baseline":
            active_strategy = "static_cascade"
        elif active_strategy == "adaptive":
            active_strategy = "full_adaptive"

        # The joint conditions search pi/idea by density ratio and (w, theta_H) by GP; the baselines keep
        # their fixed mechanisms so the ablations stay what spec §5.1 describes.
        joint_search = active_strategy in ["no_pruning", "full_adaptive"]

        # 1. Sample Parent
        parent = self.sampler.sample_parent(self.archive, use_tpe=joint_search)

        # 2. Mutate Agent (prompt; theta_H takes one random-walk step inside the mutator)
        if active_strategy == "ucb1_bandit":
            selected_arm = self.bandit.select_arm()
        else:
            selected_arm = self.sampler.sample_idea(self.mutator.get_available_strategies(), use_tpe=joint_search)
        child, mut_type, mut_goal = self.mutator.mutate(parent, generation=gen, strategy_tuple=selected_arm)

        # 3. Propose Weights (Gap 2 & Gap 6: GP-UCB with sublinear regret) and, for the joint conditions,
        # theta_H from the same GP in place of the random walk
        if active_strategy == "single_metric":
            weights = {name: (1.0 if name == "mu_1_correctness" else 0.0) for name in self.metric_names}
        elif active_strategy in ["static_cascade", "ucb1_bandit"]:
            weights = {name: 1.0 / len(self.metric_names) for name in self.metric_names}
        else:  # no_pruning, full_adaptive
            weights, child.harness_knobs = self.sampler.sample_continuous(child.harness_knobs)

        # 4. Select Benchmark Task(s): a single task, or every task of a whole benchmark (e.g. ARC)
        selector, tasks = self._resolve_tasks(task_id)

        # 5. Execute Agent on each task
        run = self._run_agent_on_tasks(child, tasks)

        # 6-7. Active Evaluators determination (Gap 1, 4, 5: 2-Tier Cascade + fANOVA + Epsilon-Probing) and
        # evaluation (scores averaged, costs summed across tasks)
        subset_metadata: Dict[str, Any] = {}
        if active_strategy in ["static_cascade", "full_adaptive"]:
            # Multi-tier selective search: run Core first, then only the deep evaluators the gate admits.
            # The core results are reused, not re-run. fANOVA pruning is part of the full system only: the
            # static cascade's x_E is fixed by hand (spec §5.1).
            prune = active_strategy == "full_adaptive"
            core_active, subset_metadata = self.sampler.select_core_subset(enable_pruning=prune)
            scores, costs, per_task = self._evaluate_run(child, tasks, run, core_active)
            core_preview = sum(scores.values()) / max(1, len(scores))
            deep_active = self.sampler.select_deep_subset(core_preview, subset_metadata, enable_pruning=prune)
            if deep_active:
                deep_scores, deep_costs, deep_per_task = self._evaluate_run(child, tasks, run, deep_active)
                scores = {**scores, **deep_scores}
                costs = costs + deep_costs
                per_task = {tid: {**s, **deep_per_task.get(tid, {})} for tid, s in per_task.items()}
            active_evaluators = core_active + deep_active
        else:
            if active_strategy == "single_metric":
                active_evaluators = [self.metric_names[0]] if "mu_1_correctness" not in self.metric_names else ["mu_1_correctness"]
            else:  # ucb1_bandit, no_pruning
                active_evaluators = list(self.metric_names)
            scores, costs, per_task = self._evaluate_run(child, tasks, run, active_evaluators)

        # 8. Cost-Penalized Fitness Score (Gap 3)
        fitness, spent_cost = calculate_cost_penalized_fitness(
            evaluator_scores=scores,
            weights=weights,
            active_evaluator_costs=costs,
            lambda_penalty=self.lambda_penalty,
        )

        # 9. Compute Delta_F (spec §3.2: both sides under the current w, no cost term) and update the
        # GP / density-ratio trials / bandit / fANOVA models
        delta_f = calculate_delta_f(child_scores=scores, parent_metrics=parent.metrics, weights=weights)
        self.sampler.record_trial(parent_id=parent.id, idea=mut_type, delta_f=delta_f)
        if joint_search:
            self.bayesian_optimizer.add_observation(
                weights=weights, delta_f=delta_f, knobs=child.harness_knobs.to_dict()
            )
        elif active_strategy == "ucb1_bandit":
            self.bandit.update(arm_name=mut_type, reward=delta_f)

        # Record the run on the agent (metrics, fitness, per-task solutions) before the surrogate models
        # read it back out of the archive.
        self._record_run_on_agent(child, selector, tasks, run, scores, fitness, spent_cost, active_evaluators, per_task)
        agent_solution = child.last_solution

        # fANOVA is measured once per window of W generations (spec §3.3), not every generation
        if gen % self.fanova_window == 0:
            self._update_fanova()

        # Gap 5: Check for silent regression against dimension elites
        regression_info = self.archive.detect_silent_regression(child)

        # 10. Update telemetry and archive. Naive cost is per task: a whole-benchmark run would pay the
        # full pool on every task, so the pruning telemetry has to scale with the run's task count.
        full_naive_cost = self.evaluator_pool.get_full_eval_cost() * len(tasks)
        self.cumulative_adaptive_cost += spent_cost
        self.cumulative_naive_cost += full_naive_cost

        gen_metadata = {
            "strategy": active_strategy,
            "delta_f": round(delta_f, 4),
            "weights": {k: round(v, 3) for k, v in weights.items()},
            "naive_cost_spent": full_naive_cost,
            "cost_saved_this_gen": round(full_naive_cost - spent_cost, 6),
            "cumulative_adaptive_cost": round(self.cumulative_adaptive_cost, 6),
            "cumulative_naive_cost": round(self.cumulative_naive_cost, 6),
            "task_id": selector,
            "solution_preview": agent_solution[:100] + "..." if len(agent_solution) > 100 else agent_solution,
            "soft_probe": subset_metadata.get("soft_probe_triggered", False),
            "fanova_pruned_count": len(subset_metadata.get("fanova_pruned", [])),
        }
        self.archive.add_agent(child, generation_metadata=gen_metadata)
        self.archive.export_to_csv(self.csv_filepath)

        # Extract test case execution details for UI test suite viewer
        test_suite_data = self._build_test_suite_data(selector, tasks, run)

        record = {
            "generation": gen,
            "child_id": child.id,
            "parent_id": parent.id,
            "parent_prompt": parent.system_prompt,
            "child_prompt": child.system_prompt,
            "solution": agent_solution,
            "strategy": active_strategy,
            "fitness": fitness,
            "delta_f": delta_f,
            "cost_spent": spent_cost,
            "naive_cost": full_naive_cost,
            "active_evaluators": active_evaluators,
            "weights": weights,
            "metrics": scores,
            "task_id": selector,
            "task_metrics": dict(per_task),
            "mutation_type": child.mutation_type,
            "mutation_goal": mut_goal,
            "silent_regression": regression_info,
            "fanova_pruned": subset_metadata.get("fanova_pruned", []),
            "soft_probe_triggered": subset_metadata.get("soft_probe_triggered", False),
            "test_suite": test_suite_data,
        }
        self.history_records.append(record)
        return record

    # ------------------------------------------------------------------ program-archive mode

    def _program_hooks(self) -> Tuple[Callable, Callable]:
        """(inspect_program, build_repair_prompt) for the run's benchmark. Only ARC has them so far."""
        if not is_arc_benchmark(self.selected_task_id):
            raise ValueError("program_archive mode needs a benchmark with program hooks (the ARC benchmark)")
        from benchmarks.arc_challenge.repair import build_repair_prompt, inspect_program

        return inspect_program, build_repair_prompt

    def _call_agent(self, agent: Agent, user_prompt: str) -> Dict[str, Any]:
        """One LLM call with the agent's system prompt and knobs; keeps the call's usage when the client has it."""
        knobs = agent.harness_knobs
        start = time.time()
        solution = self.llm_client.generate(
            system_prompt=agent.system_prompt,
            user_prompt=user_prompt,
            temperature=knobs.temperature,
            top_p=knobs.top_p,
            max_tokens=knobs.max_tokens,
        )
        info_fn = getattr(self.llm_client, "last_call_info", None)
        return {"solution": solution, "latency_ms": (time.time() - start) * 1000, "usage": info_fn() if info_fn else {}}

    def _cascade_evaluate(
        self, agent: Agent, tasks: List[BenchmarkTask], run: Dict[str, Dict[str, Any]]
    ) -> Tuple[Dict[str, float], List[float], Dict[str, Dict[str, float]], List[str], Dict[str, Any]]:
        """The full_adaptive evaluation of run_generation: core tier (fANOVA soft pruning + epsilon probes),
        then the deep tier if the core preview passes the gate."""
        core_active, metadata = self.sampler.select_core_subset(enable_pruning=True)
        scores, costs, per_task = self._evaluate_run(agent, tasks, run, core_active)
        preview = sum(scores.values()) / max(1, len(scores))
        deep_active = self.sampler.select_deep_subset(preview, metadata, enable_pruning=True)
        if deep_active:
            deep_scores, deep_costs, deep_per_task = self._evaluate_run(agent, tasks, run, deep_active)
            scores = {**scores, **deep_scores}
            costs = costs + deep_costs
            per_task = {tid: {**s, **deep_per_task.get(tid, {})} for tid, s in per_task.items()}
        return scores, costs, per_task, core_active + deep_active, metadata

    def _archive_program(
        self,
        agent: Agent,
        task: BenchmarkTask,
        result: Dict[str, Any],
        scores: Dict[str, float],
        parent: Optional[ProgramRecord],
        weights: Dict[str, float],
        generation: int,
    ) -> ProgramRecord:
        inspect_program, _ = self._program_hooks()
        inspection = inspect_program(task, result["solution"])
        record = ProgramRecord(
            task_id=task.id,
            solution=result["solution"],
            agent_id=agent.id,
            generation=generation,
            scores=dict(scores),
            cell=inspection.cell,
            solved=inspection.solved,
            feedback=inspection.feedback,
            parent_program_id=parent.id if parent else None,
            usage=dict(result.get("usage") or {}),
        )
        self.archive.add_program(record, weights)
        return record

    def _inspirations(
        self, task_id: str, parent: Optional[ProgramRecord], weights: Dict[str, float], n: int = 2
    ) -> List[ProgramRecord]:
        """Other programs of the task shown next to the one being repaired (OpenEvolve's inspirations): the
        task's best elite, then a random elite from another cell."""
        if parent is None:
            return []
        others = [p for p in self.archive.task_elites(task_id) if p.id != parent.id]
        if not others:
            return []
        best = max(others, key=lambda p: weighted_score(p.scores, weights))
        rest = [p for p in others if p.id != best.id]
        return ([best] + random.sample(rest, min(len(rest), n - 1)))[:n]

    def _initialize_archive_seed(self) -> None:
        """Generation 0 of program-archive mode: the seed agent writes one program per task (UCB1's first pull
        of every task arm), scored on every evaluator under uniform weights. Every island starts from it."""
        arch_info = AGENT_ARCHETYPES.get(self.initial_archetype, AGENT_ARCHETYPES["General Balanced Assistant"])
        knobs = arch_info["harness_knobs"]
        suffix = arch_info.get("id_suffix", "seed")
        seed = Agent(
            id=f"agent_seed_{suffix}_g0",
            generation=0,
            system_prompt=arch_info["system_prompt"],
            harness_knobs=HarnessKnobs(knobs.temperature, knobs.max_retries, knobs.top_p, knobs.max_tokens),
            mutation_type=f"seed_{suffix}",
            mutation_description=f"Initial seed archetype: {self.initial_archetype}",
        )
        weights = {name: 1.0 / len(self.metric_names) for name in self.metric_names}
        _, build_prompt = self._program_hooks()
        selector, tasks = self._resolve_tasks()
        results = self._map_tasks(lambda t: self._call_agent(seed, build_prompt(t, None)), tasks)
        run = {t.id: r for t, r in zip(tasks, results)}
        full = self.evaluator_pool.get_evaluator_names()
        scores, costs, per_task = self._evaluate_run(seed, tasks, run, full)
        for task in tasks:
            record = self._archive_program(seed, task, run[task.id], per_task[task.id], None, weights, 0)
            self.task_selector.update(task.id, weighted_score(record.scores, weights))
            if record.solved:
                self.task_selector.retire(task.id)

        fitness, spent = calculate_cost_penalized_fitness(scores, weights, costs, self.lambda_penalty)
        seed.metrics, seed.fitness, seed.cost_spent = scores, fitness, spent
        seed.active_evaluators, seed.task_id, seed.task_metrics = full, selector, dict(per_task)
        seed.task_solutions = {tid: r["solution"] for tid, r in run.items()}
        naive = self.evaluator_pool.get_full_eval_cost() * len(tasks)
        self.cumulative_adaptive_cost += spent
        self.cumulative_naive_cost += naive
        self.archive.add_agent(seed, generation_metadata={
            "strategy": "program_archive_seed",
            "delta_f": 0.0,
            "weights": {k: round(v, 3) for k, v in weights.items()},
            "naive_cost_spent": naive,
            "cost_saved_this_gen": 0.0,
            "cumulative_adaptive_cost": self.cumulative_adaptive_cost,
            "cumulative_naive_cost": self.cumulative_naive_cost,
            "task_id": selector,
            "tasks_attempted": len(tasks),
            "solved_tasks": len(self.archive.solved_tasks),
        })
        self.archive.init_islands(self.n_islands, [seed.id])
        self.archive.export_to_csv(self.csv_filepath)
        self.history_records.append({
            "generation": 0,
            "child_id": seed.id,
            "parent_id": "Root",
            "child_prompt": seed.system_prompt,
            "strategy": "program_archive_seed",
            "fitness": fitness,
            "delta_f": 0.0,
            "cost_spent": spent,
            "active_evaluators": full,
            "weights": weights,
            "metrics": scores,
            "task_id": selector,
            "task_metrics": dict(per_task),
            "mutation_type": seed.mutation_type,
            "mutation_goal": seed.mutation_description,
            "island": None,
        })

    def _run_archive_generation(self) -> Dict[str, Any]:
        """One generation of program-archive mode. Each decision has one selector:

        - (w, theta_H): the joint GP, one proposal per generation (temperature capped by `knob_bounds`)
        - which agent to mutate, per island: TPE with archived fitness as the prior (sample_weighted_parent)
        - mutation idea: Optuna TPE, one open trial per child
        - which tasks to work on: UCB1 over the unsolved tasks (UCB1TaskSelector), split across the children
        - per task, which program to repair or a fresh start: TPE with archived score as the prior
          (sample_program_parent)

        Every call's Delta_F is paired (child program against the program it started from, or against the
        task's best for a fresh start, both under the current w). It is the per-task TPE trial; the gain in the
        task's best archived score is its UCB1 reward; a child's mean Delta_F is its TPE / Optuna / GP trial.
        """
        selector, tasks = self._resolve_tasks()
        by_id = {t.id: t for t in tasks}
        open_ids = [t.id for t in tasks if not self.archive.is_solved(t.id)]
        if not open_ids:
            return {"generation": self.current_generation, "children": [], "all_solved": True,
                    "solved_tasks": len(self.archive.solved_tasks), "open_tasks": 0}
        self.current_generation += 1
        gen = self.current_generation
        _, build_prompt = self._program_hooks()

        best = self.archive.get_best_agent()
        weights, knobs = self.sampler.sample_continuous(best.harness_knobs)

        # Children: one per island
        menu = self.mutator.get_available_strategies()
        plans = []
        for island in range(len(self.archive.islands)):
            members = self.archive.island_agents(island)
            if not members:
                continue
            key = f"g{gen}_i{island}"
            parent = self.sampler.sample_weighted_parent(members)
            idea = self.sampler.sample_idea(menu, use_tpe=True, key=key)
            plans.append({"island": island, "parent": parent, "idea": idea, "key": key})
        mutated = self._map_tasks(
            lambda p: self.mutator.mutate(p["parent"], generation=gen, strategy_tuple=p["idea"]), plans
        )
        for plan, (child, mut_type, mut_goal) in zip(plans, mutated):
            child.harness_knobs = HarnessKnobs(
                temperature=knobs.temperature,
                max_retries=child.harness_knobs.max_retries,
                top_p=knobs.top_p,
                max_tokens=plan["parent"].harness_knobs.max_tokens,
            )
            plan.update(child=child, mutation_type=mut_type, mutation_goal=mut_goal)

        # Tasks: UCB1 over the open tasks, distinct across children
        picked = self.task_selector.select(open_ids, self.tasks_per_child * len(plans))
        for j, plan in enumerate(plans):
            plan["tasks"] = picked[j::len(plans)]

        # One call per (child, task): repair the program TPE picks, or start fresh
        jobs = []
        for plan in plans:
            for tid in plan["tasks"]:
                parent_prog = self.sampler.sample_program_parent(self.archive, tid, weights)
                inspirations = self._inspirations(tid, parent_prog, weights)
                prompt = build_prompt(
                    by_id[tid],
                    parent_prog.solution if parent_prog else None,
                    parent_prog.feedback if parent_prog else "",
                    [p.solution for p in inspirations],
                )
                jobs.append({"plan": plan, "task_id": tid, "parent_prog": parent_prog, "prompt": prompt})
        results = self._map_tasks(lambda j: self._call_agent(j["plan"]["child"], j["prompt"]), jobs)
        for job, result in zip(jobs, results):
            job["result"] = result

        child_records = []
        full_cost = self.evaluator_pool.get_full_eval_cost()
        for plan in plans:
            child, parent = plan["child"], plan["parent"]
            cjobs = [j for j in jobs if j["plan"] is plan]
            if not cjobs:
                self.sampler.record_trial(parent.id, plan["mutation_type"], 0.0, key=plan["key"])
                continue
            ctasks = [by_id[j["task_id"]] for j in cjobs]
            run = {j["task_id"]: j["result"] for j in cjobs}
            scores, costs, per_task, active, subset_meta = self._cascade_evaluate(child, ctasks, run)

            task_deltas: Dict[str, float] = {}
            new_solves = []
            for job in cjobs:
                tid, parent_prog = job["task_id"], job["parent_prog"]
                before = self.archive.task_best_score(tid, weights)
                reference = parent_prog or self.archive.task_best(tid, weights)
                record = self._archive_program(child, by_id[tid], job["result"], per_task[tid], parent_prog, weights, gen)
                delta = calculate_delta_f(record.scores, reference.scores, weights) if reference else 0.0
                task_deltas[tid] = delta
                self.archive.record_task_trial(tid, parent_prog.id if parent_prog else self.archive.FRESH, delta)
                self.task_selector.update(tid, max(0.0, self.archive.task_best_score(tid, weights) - before))
                if record.solved:
                    self.task_selector.retire(tid)
                    if self.archive.solved_tasks.get(tid) == record.id:
                        new_solves.append(tid)

            fitness, spent = calculate_cost_penalized_fitness(scores, weights, costs, self.lambda_penalty)
            delta_f = sum(task_deltas.values()) / len(task_deltas)
            self.sampler.record_trial(parent.id, plan["mutation_type"], delta_f, key=plan["key"])
            self.bayesian_optimizer.add_observation(
                weights=weights, delta_f=delta_f, knobs=child.harness_knobs.to_dict()
            )

            child.metrics, child.fitness, child.cost_spent = scores, fitness, spent
            child.active_evaluators, child.task_id, child.task_metrics = active, selector, dict(per_task)
            child.task_solutions = {tid: r["solution"] for tid, r in run.items()}
            naive = full_cost * len(ctasks)
            self.cumulative_adaptive_cost += spent
            self.cumulative_naive_cost += naive
            repairs = sum(1 for j in cjobs if j["parent_prog"] is not None)
            self.archive.add_agent(child, generation_metadata={
                "strategy": "program_archive",
                "delta_f": round(delta_f, 4),
                "weights": {k: round(v, 3) for k, v in weights.items()},
                "naive_cost_spent": naive,
                "cost_saved_this_gen": round(naive - spent, 6),
                "cumulative_adaptive_cost": round(self.cumulative_adaptive_cost, 6),
                "cumulative_naive_cost": round(self.cumulative_naive_cost, 6),
                "task_id": selector,
                "island": plan["island"],
                "tasks_attempted": len(cjobs),
                "repairs": repairs,
                "new_solves": len(new_solves),
                "soft_probe": subset_meta.get("soft_probe_triggered", False),
                "fanova_pruned_count": len(subset_meta.get("fanova_pruned", [])),
            })
            self.archive.add_to_island(plan["island"], child.id)
            record = {
                "generation": gen,
                "child_id": child.id,
                "parent_id": parent.id,
                "parent_prompt": parent.system_prompt,
                "child_prompt": child.system_prompt,
                "strategy": "program_archive",
                "fitness": fitness,
                "delta_f": delta_f,
                "cost_spent": spent,
                "active_evaluators": active,
                "weights": weights,
                "metrics": scores,
                "task_id": selector,
                "task_metrics": dict(per_task),
                "task_deltas": task_deltas,
                "mutation_type": plan["mutation_type"],
                "mutation_goal": plan["mutation_goal"],
                "island": plan["island"],
                "repairs": repairs,
                "new_solves": new_solves,
                "fanova_pruned": subset_meta.get("fanova_pruned", []),
            }
            self.history_records.append(record)
            child_records.append(record)

        if gen % self.fanova_window == 0:
            self._update_fanova()
        migrations = self.archive.migrate_ring() if gen % self.migration_interval == 0 else []
        self.archive.export_to_csv(self.csv_filepath)
        return {
            "generation": gen,
            "children": child_records,
            "weights": weights,
            "knobs": knobs.to_dict(),
            "calls": len(jobs) + len(plans),
            "repairs": sum(r["repairs"] for r in child_records),
            "new_solves": [t for r in child_records for t in r["new_solves"]],
            "solved_tasks": len(self.archive.solved_tasks),
            "open_tasks": len(open_ids) - sum(len(r["new_solves"]) for r in child_records),
            "migrations": migrations,
        }

    def archive_held_out(self, weights: Optional[Dict[str, float]] = None) -> Dict[str, Dict[str, Any]]:
        """Official held-out pass@2 per task with both attempts taken from the archive (reporting only: nothing
        here feeds back into selection). A task's elites are ranked on its demonstration pairs: solved first,
        then weighted score under `weights` (default: the latest generation's w)."""
        from benchmarks.arc_challenge.repair import archive_pass_at_2

        if weights is None:
            weights = self.history_records[-1]["weights"] if self.history_records else {}
        _, tasks = self._resolve_tasks()

        def score(task: BenchmarkTask) -> Dict[str, Any]:
            ranked = sorted(self.archive.task_elites(task.id),
                            key=lambda p: (p.solved, weighted_score(p.scores, weights)), reverse=True)
            return archive_pass_at_2(task, [(p.id, p.solution) for p in ranked])

        return {t.id: r for t, r in zip(tasks, self._map_tasks(score, tasks))}

    def save_state(self, path: str) -> str:
        """Writes the program-archive run (archive with every program, UCB1 tasks, TPE / Optuna trials, GP
        observations) to JSON; load_state continues from it."""
        gp = self.bayesian_optimizer
        state = {
            "evolution_mode": self.evolution_mode,
            "current_generation": self.current_generation,
            "cumulative_adaptive_cost": self.cumulative_adaptive_cost,
            "cumulative_naive_cost": self.cumulative_naive_cost,
            "archive": self.archive.to_state(),
            "task_selector": self.task_selector.state(),
            "sampler_trials": [list(t) for t in self.sampler.trials],
            "gp": {"weights": gp.X_history, "knobs": gp.knob_history, "delta_f": gp.y_history},
            "history_records": self.history_records,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, default=str)
        os.replace(tmp, path)  # a crash mid-write leaves the previous checkpoint intact
        return path

    def load_state(self, path: str) -> int:
        """Restores a save_state checkpoint into this (seedless) controller; returns the last generation."""
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
        self.archive = PopulationArchive.from_state(state["archive"])
        self.task_selector = UCB1TaskSelector.from_state(state["task_selector"])
        menu = [name for name, _ in self.mutator.get_available_strategies()]
        for parent_id, idea, delta_f in state["sampler_trials"]:
            self.sampler.restore_trial(parent_id, idea, delta_f, menu)
        gp = self.bayesian_optimizer
        for w, k, y in zip(state["gp"]["weights"], state["gp"]["knobs"], state["gp"]["delta_f"]):
            gp.add_observation(dict(zip(self.metric_names, w)), y, knobs=gp._unscale_knobs(np.array(k)))
        self.history_records = list(state["history_records"])
        self.current_generation = int(state["current_generation"])
        self.cumulative_adaptive_cost = float(state["cumulative_adaptive_cost"])
        self.cumulative_naive_cost = float(state["cumulative_naive_cost"])
        return self.current_generation

    def _update_fanova(self):
        """Measures one pruning window (spec §3.3) over every archived agent's per-evaluator scores, then
        advances each evaluator's below-tau streak.

        Informativeness is leave-one-out (FunctionalANOVA.fit_informativeness), not the spec's literal
        "importance of w_j": regressing an objective on the evaluators that define it credits a pure-noise
        evaluator with the variance it injects, and with w on the simplex E[dF | w_j] has slope
        m_j - mean_{i != j} m_i, so a noise weight looks as important as an informative one.
        scripts/synthetic_pruning_probe.py reproduces both failures and the leave-one-out fix."""
        agents = [a for a in self.archive.get_all_agents() if a.metrics]
        if len(agents) < 4:
            return
        scores = np.array([[a.metrics.get(m, np.nan) for m in self.metric_names] for a in agents], dtype=float)
        costs = [self.evaluator_pool.evaluators[m].cost for m in self.metric_names]
        protected = [m for m in self.PRIMARY_METRICS if m in self.metric_names] or self.metric_names[:1]
        self.fanova.fit_informativeness(scores, costs=costs, protected=protected)
        self.fanova.close_window()

    def run_n_generations(self, n: int, strategy: Optional[str] = None) -> List[Dict[str, Any]]:
        """Runs N consecutive generation cycles."""
        results = []
        for _ in range(n):
            res = self.run_generation(strategy=strategy)
            results.append(res)
        return results

    def _execute_agent_on_task(self, agent: Agent, task: BenchmarkTask) -> str:
        """Executes the agent LLM on the given benchmark task."""
        system_prompt = agent.system_prompt
        user_prompt = agent.user_template.format(task_description=task.description)
        knobs = agent.harness_knobs

        return self.llm_client.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=knobs.temperature,
            top_p=knobs.top_p,
            max_tokens=knobs.max_tokens,
        )

    def _task_to_dict(self, task: BenchmarkTask) -> Dict[str, Any]:
        return {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "entry_point": task.entry_point,
            "test_cases": task.test_cases,
            "edge_cases": task.edge_cases,
            "constraints": task.constraints,
        }

    def get_fanova_summary(self) -> Dict[str, Any]:
        """Returns current fANOVA sensitivity analysis."""
        return self.fanova.get_fanova_summary()

    def get_dimension_elites_summary(self) -> Dict[str, Any]:
        """Returns per-evaluator archive view for Gap 5."""
        return self.archive.get_dimension_elites()

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Returns overall KPI metrics for UI cards."""
        best_agent = self.archive.get_best_agent()
        total_saved = max(0.0, self.cumulative_naive_cost - self.cumulative_adaptive_cost)
        pct_saved = (
            (total_saved / self.cumulative_naive_cost * 100.0)
            if self.cumulative_naive_cost > 0
            else 0.0
        )
        fanova_info = self.get_fanova_summary()

        return {
            "total_generations": self.current_generation,
            "best_fitness": best_agent.fitness if best_agent else 0.0,
            "best_agent_id": best_agent.id if best_agent else "N/A",
            "cumulative_cost_spent": self.cumulative_adaptive_cost,
            "cumulative_naive_cost": self.cumulative_naive_cost,
            "cost_saved_usd": total_saved,
            "cost_saved_pct": pct_saved,
            "archive_size": len(self.archive.agents),
            "pareto_size": len(self.archive.get_pareto_front()),
            "redundant_evaluators_count": len(fanova_info.get("redundant_evaluators", [])),
            "fanova_total_variance": fanova_info.get("total_variance", 0.0),
        }

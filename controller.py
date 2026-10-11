import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Tuple
from core.agent import Agent, HarnessKnobs, AGENT_ARCHETYPES
from core.archive import PopulationArchive
from core.groq_client import GroqLLMClient
from core.mutator import PromptMutator
from evaluation.pool import EvaluatorPool
from optimization.bayesian_weights import BayesianWeightOptimizer
from optimization.scoring import calculate_cost_penalized_fitness, calculate_delta_f
from optimization.joint_sampler import JointSearchSampler
from optimization.bandit import UCB1MutationBandit
from optimization.fanova import FunctionalANOVA
# `tasks/` was renamed to `benchmarks/` on this fork when the ARC-AGI suite was added.
from benchmarks.benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask, get_random_task, get_benchmark_tasks
from benchmarks.arc_challenge import ARC_BENCHMARK_ID, is_arc_benchmark

logger = logging.getLogger(__name__)


class EvolutionController:
    """Evolutionary optimization loop orchestrator implementing Gaps 1, 2, 3, 4, 5, and 6."""

    # Headline metric per suite: never pruned, and kept over its correlate by the redundancy rule
    PRIMARY_METRICS = ("mu_1_correctness", "arc_pass_at_2_test")

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
        evaluator_pool_factory: Optional[Callable[[GroqLLMClient], EvaluatorPool]] = None,
        reasoning_effort: Optional[str] = None,
        min_max_tokens: int = 0,
        acquisition_type: str = "gp_ucb",
        fanova_window: int = 3,
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
        # Spec §3.3's W: fANOVA is measured every W generations and prunes after W consecutive low windows
        self.fanova_window = max(1, fanova_window)
        self.csv_filepath = "optimization_results.csv"

        # Core components
        self.llm_client = GroqLLMClient(
            api_key=api_key,
            model=model,
            inter_call_delay=inter_call_delay,
            is_mock=is_mock,
            reasoning_effort=reasoning_effort,
            min_max_tokens=min_max_tokens,
        )
        # Default: the 6 canonical mu evaluators. A factory swaps in a benchmark-specific suite
        # (e.g. benchmarks.arc_challenge.build_arc_evaluator_pool).
        self.evaluator_pool = (
            evaluator_pool_factory(self.llm_client) if evaluator_pool_factory else EvaluatorPool(llm_client=self.llm_client)
        )
        self.metric_names = self.evaluator_pool.get_evaluator_names()
        self.bayesian_optimizer = BayesianWeightOptimizer(
            metric_names=self.metric_names,
            acquisition_type=acquisition_type,
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

        # Initialize seed population
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
        )
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
        run: Dict[str, Dict[str, Any]] = {}
        for task in tasks:
            start = time.time()
            solution = self._execute_agent_on_task(agent, task)
            run[task.id] = {"solution": solution, "latency_ms": (time.time() - start) * 1000}
        return run

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
        per_task: Dict[str, Dict[str, float]] = {}
        cost_totals: Dict[str, float] = {}
        for task in tasks:
            r = run[task.id]
            scores, costs, _ = self.evaluator_pool.evaluate_subset(
                agent=agent,
                task=self._task_to_dict(task),
                agent_output=r["solution"],
                active_subset=active_subset,
                execution_context={"generation_latency_ms": r["latency_ms"]},
            )
            per_task[task.id] = scores
            for name, cost in zip([n for n in active_subset if n in scores], costs):
                cost_totals[name] = cost_totals.get(name, 0.0) + cost

        n = max(1, len(tasks))
        mean_scores = {
            name: round(sum(per_task[t.id].get(name, 0.0) for t in tasks) / n, 4)
            for name in active_subset
            if any(name in per_task[t.id] for t in tasks)
        }
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

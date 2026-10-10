import time
import logging
import numpy as np
from typing import Dict, List, Any, Optional
from core.agent import Agent, HarnessKnobs, AGENT_ARCHETYPES
from core.archive import PopulationArchive
from core.groq_client import GroqLLMClient
from core.mutator import PromptMutator
from evaluation.pool import EvaluatorPool
from optimization.bayesian_weights import BayesianWeightOptimizer
from optimization.scoring import calculate_cost_penalized_fitness, calculate_relative_improvement
from optimization.joint_sampler import JointSearchSampler
from optimization.bandit import UCB1MutationBandit
from optimization.fanova import FunctionalANOVA
from tasks.benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask, get_random_task, get_benchmark_task

logger = logging.getLogger(__name__)


class EvolutionController:
    """Evolutionary optimization loop orchestrator implementing Gaps 1, 2, 3, 4, 5, and 6."""

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
        acquisition_type: str = "gp_ucb",
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
        self.csv_filepath = "optimization_results.csv"

        # Core components
        self.llm_client = GroqLLMClient(
            api_key=api_key,
            model=model,
            inter_call_delay=inter_call_delay,
            is_mock=is_mock,
        )
        self.evaluator_pool = EvaluatorPool(llm_client=self.llm_client)
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

        # Evaluate seed agent across benchmark
        benchmark = self._resolve_task(task_id)
        exec_start = time.time()
        agent_solution = self._execute_agent_on_task(seed_agent, benchmark)
        exec_time_ms = (time.time() - exec_start) * 1000

        scores, costs, details = self.evaluator_pool.evaluate_subset(
            agent=seed_agent,
            task=self._task_to_dict(benchmark),
            agent_output=agent_solution,
            active_subset=full_subset,
            execution_context={"generation_latency_ms": exec_time_ms},
        )

        fitness, spent_cost = calculate_cost_penalized_fitness(
            evaluator_scores=scores,
            weights=uniform_weights,
            active_evaluator_costs=costs,
            lambda_penalty=self.lambda_penalty,
        )

        seed_agent.metrics = scores
        seed_agent.fitness = fitness
        seed_agent.cost_spent = spent_cost
        seed_agent.active_evaluators = full_subset
        seed_agent.last_solution = agent_solution
        seed_agent.task_id = benchmark.id

        self.cumulative_adaptive_cost += spent_cost
        self.cumulative_naive_cost += self.evaluator_pool.get_full_eval_cost()

        self.archive.add_agent(
            seed_agent,
            generation_metadata={
                "strategy": "seed",
                "naive_cost_spent": self.evaluator_pool.get_full_eval_cost(),
                "cost_saved_this_gen": 0.0,
                "cumulative_adaptive_cost": self.cumulative_adaptive_cost,
                "cumulative_naive_cost": self.cumulative_naive_cost,
                "task_id": benchmark.id,
                "delta_f": 0.0,
                "weights": {k: round(v, 3) for k, v in uniform_weights.items()},
                "solution_preview": agent_solution[:100] + "..." if len(agent_solution) > 100 else agent_solution,
            },
        )
        self.archive.export_to_csv(self.csv_filepath)

        # Extract test case execution details for UI test suite viewer
        test_cases_results = []
        eval_res_1 = details.get("mu_1_correctness")
        if eval_res_1 and hasattr(eval_res_1, "details") and isinstance(eval_res_1.details, dict):
            test_cases_results.extend(eval_res_1.details.get("case_results", []))
        if not test_cases_results:
            from evaluation.runner import CodeExecutionRunner
            rep = CodeExecutionRunner.run_task(task=benchmark, agent_output=agent_solution, include_test_cases=True, include_edge_cases=True)
            test_cases_results = [
                {
                    "index": c.case_index,
                    "type": c.case_type,
                    "args": str(c.args),
                    "expected": str(c.expected),
                    "actual": str(c.actual),
                    "passed": c.passed,
                    "error": c.error or c.exception_raised,
                }
                for c in rep.case_results
            ]

        test_suite_data = {
            "task_id": benchmark.id,
            "task_name": benchmark.name,
            "task_description": benchmark.description,
            "entry_point": benchmark.entry_point,
            "cases": test_cases_results,
            "total_count": len(test_cases_results),
            "passed_count": sum(1 for c in test_cases_results if c.get("passed")),
        }

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
            "naive_cost": self.evaluator_pool.get_full_eval_cost(),
            "active_evaluators": full_subset,
            "weights": uniform_weights,
            "metrics": scores,
            "task_id": benchmark.id,
            "mutation_type": seed_agent.mutation_type,
            "mutation_goal": seed_agent.mutation_description,
            "silent_regression": {"has_regression": False},
            "test_suite": test_suite_data,
        }
        self.history_records.append(seed_record)

    def _resolve_task(self, task_param: Optional[str] = None) -> BenchmarkTask:
        target = task_param or self.selected_task_id
        if target and target != "All Benchmark Tasks (Suite / Random)":
            found = get_benchmark_task(target)
            if found:
                return found
        return get_random_task()

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

        # 1. Sample Parent
        parent = self.sampler.sample_parent(self.archive)

        # 2. Mutate Agent (prompt + continuous theta_H knobs)
        if active_strategy == "ucb1_bandit":
            selected_arm = self.bandit.select_arm()
            child, mut_type, mut_goal = self.mutator.mutate(parent, generation=gen, strategy_tuple=selected_arm)
        else:
            child, mut_type, mut_goal = self.mutator.mutate(parent, generation=gen)

        # Gap 6: Sample continuous harness knobs (temperature, top_p)
        child.harness_knobs = self.sampler.sample_harness_knobs(child.harness_knobs)

        # 3. Propose Weights (Gap 2 & Gap 6: GP-UCB with sublinear regret)
        if active_strategy == "single_metric":
            weights = {name: (1.0 if name == "mu_1_correctness" else 0.0) for name in self.metric_names}
        elif active_strategy in ["static_cascade", "ucb1_bandit"]:
            weights = {name: 1.0 / len(self.metric_names) for name in self.metric_names}
        else:  # no_pruning, full_adaptive
            weights = self.sampler.sample_weights()

        # 4. Select Benchmark Task
        task = self._resolve_task(task_id)
        task_dict = self._task_to_dict(task)

        # 5. Execute Agent on Task
        exec_start = time.time()
        agent_solution = self._execute_agent_on_task(child, task)
        exec_time_ms = (time.time() - exec_start) * 1000

        # 6. Active Evaluators determination (Gap 1, 4, 5: 2-Tier Cascade + fANOVA + Epsilon-Probing)
        subset_metadata: Dict[str, Any] = {}
        if active_strategy == "single_metric":
            active_evaluators = ["mu_1_correctness"]
        elif active_strategy in ["ucb1_bandit", "no_pruning"]:
            active_evaluators = list(self.metric_names)
        else:  # static_cascade, full_adaptive
            core_evals = self.evaluator_pool.get_evaluators_by_tier("core")
            core_scores, core_costs, _ = self.evaluator_pool.evaluate_subset(
                agent=child,
                task=task_dict,
                agent_output=agent_solution,
                active_subset=core_evals,
                execution_context={"generation_latency_ms": exec_time_ms},
            )
            core_preview = sum(core_scores.values()) / max(1, len(core_scores))
            active_evaluators, subset_metadata = self.sampler.determine_active_evaluator_subset(
                strategy="adaptive",
                core_score_preview=core_preview,
                generation=gen,
            )

        # 7. Run evaluation on active evaluators
        scores, costs, details = self.evaluator_pool.evaluate_subset(
            agent=child,
            task=task_dict,
            agent_output=agent_solution,
            active_subset=active_evaluators,
            execution_context={"generation_latency_ms": exec_time_ms},
        )

        # 8. Cost-Penalized Fitness Score (Gap 3)
        fitness, spent_cost = calculate_cost_penalized_fitness(
            evaluator_scores=scores,
            weights=weights,
            active_evaluator_costs=costs,
            lambda_penalty=self.lambda_penalty,
        )

        # 9. Compute Delta_F and update Bayesian GP / Bandit / fANOVA models
        delta_f = calculate_relative_improvement(
            child_fitness=fitness,
            parent_fitness=parent.fitness,
        )
        if active_strategy in ["no_pruning", "full_adaptive"]:
            self.bayesian_optimizer.add_observation(weights=weights, delta_f=delta_f)
        elif active_strategy == "ucb1_bandit":
            self.bandit.update(arm_name=mut_type, reward=delta_f)

        # Update fANOVA surrogate model with observed metrics
        child.metrics = scores
        child.fitness = fitness
        child.cost_spent = spent_cost
        child.active_evaluators = active_evaluators
        child.last_solution = agent_solution
        child.task_id = task.id

        self._update_fanova()

        # Gap 5: Check for silent regression against dimension elites
        regression_info = self.archive.detect_silent_regression(child)

        # 10. Update telemetry and archive
        full_naive_cost = self.evaluator_pool.get_full_eval_cost()
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
            "task_id": task.id,
            "solution_preview": agent_solution[:100] + "..." if len(agent_solution) > 100 else agent_solution,
            "soft_probe": subset_metadata.get("soft_probe_triggered", False),
            "fanova_pruned_count": len(subset_metadata.get("fanova_pruned", [])),
        }
        self.archive.add_agent(child, generation_metadata=gen_metadata)
        self.archive.export_to_csv(self.csv_filepath)

        # Extract test case execution details for UI test suite viewer
        test_cases_results = []
        eval_res_gen = details.get("mu_1_correctness")
        if eval_res_gen and hasattr(eval_res_gen, "details") and isinstance(eval_res_gen.details, dict):
            test_cases_results.extend(eval_res_gen.details.get("case_results", []))
        if not test_cases_results:
            from evaluation.runner import CodeExecutionRunner
            rep = CodeExecutionRunner.run_task(task=task, agent_output=agent_solution, include_test_cases=True, include_edge_cases=True)
            test_cases_results = [
                {
                    "index": c.case_index,
                    "type": c.case_type,
                    "args": str(c.args),
                    "expected": str(c.expected),
                    "actual": str(c.actual),
                    "passed": c.passed,
                    "error": c.error or c.exception_raised,
                }
                for c in rep.case_results
            ]

        test_suite_data = {
            "task_id": task.id,
            "task_name": task.name,
            "task_description": task.description,
            "entry_point": task.entry_point,
            "cases": test_cases_results,
            "total_count": len(test_cases_results),
            "passed_count": sum(1 for c in test_cases_results if c.get("passed")),
        }

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
            "task_id": task.id,
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
        """Updates Functional ANOVA variance decomposition using historical agent evaluations."""
        agents = self.archive.get_all_agents()
        if len(agents) < 4:
            return

        X_rows = []
        y_vals = []
        for a in agents:
            if a.metrics and len(a.metrics) >= 3:
                row = [a.metrics.get(m, 0.0) for m in self.metric_names]
                X_rows.append(row)
                y_vals.append(a.fitness)

        if len(X_rows) >= 4:
            X_mat = np.array(X_rows)
            y_arr = np.array(y_vals)
            self.fanova.fit(X_mat, y_arr)

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

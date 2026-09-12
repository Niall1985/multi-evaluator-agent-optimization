import time
import logging
from typing import Dict, List, Any, Optional
from core.agent import Agent, HarnessKnobs
from core.archive import PopulationArchive
from core.groq_client import GroqLLMClient
from core.mutator import PromptMutator
from evaluation.pool import EvaluatorPool
from optimization.bayesian_weights import BayesianWeightOptimizer
from optimization.scoring import calculate_cost_penalized_fitness, calculate_relative_improvement
from optimization.joint_sampler import JointSearchSampler
from tasks.benchmark_tasks import BENCHMARK_TASKS, BenchmarkTask, get_random_task, get_benchmark_task
from core.agent import Agent, HarnessKnobs, AGENT_ARCHETYPES

logger = logging.getLogger(__name__)


class EvolutionController:
    """Evolutionary optimization loop orchestrator implementing Gaps 1, 2, and 3."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-oss-120b",
        lambda_penalty: float = 0.5,
        default_strategy: str = "adaptive",
        initial_archetype: str = "General Balanced Assistant",
        selected_task_id: Optional[str] = None,
        inter_call_delay: float = 1.5,
    ):
        self.api_key = api_key
        self.model = model
        self.lambda_penalty = lambda_penalty
        self.default_strategy = default_strategy
        self.initial_archetype = initial_archetype
        self.selected_task_id = selected_task_id
        self.inter_call_delay = inter_call_delay
        self.csv_filepath = "optimization_results.csv"

        # Core components
        self.llm_client = GroqLLMClient(api_key=api_key, model=model, inter_call_delay=inter_call_delay)
        self.evaluator_pool = EvaluatorPool(llm_client=self.llm_client)
        self.metric_names = self.evaluator_pool.get_evaluator_names()
        self.bayesian_optimizer = BayesianWeightOptimizer(metric_names=self.metric_names)
        self.mutator = PromptMutator(llm_client=self.llm_client)
        self.sampler = JointSearchSampler(
            evaluator_pool=self.evaluator_pool,
            bayesian_optimizer=self.bayesian_optimizer,
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
        """Resets the controller state, archive, and optimizer."""
        if initial_archetype:
            self.initial_archetype = initial_archetype
        if selected_task_id is not None:
            self.selected_task_id = selected_task_id
        self.archive = PopulationArchive()
        self.bayesian_optimizer = BayesianWeightOptimizer(metric_names=self.metric_names)
        self.sampler = JointSearchSampler(
            evaluator_pool=self.evaluator_pool,
            bayesian_optimizer=self.bayesian_optimizer,
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

        self.cumulative_adaptive_cost += spent_cost
        self.cumulative_naive_cost += self.evaluator_pool.get_full_eval_cost()

        self.archive.add_agent(
            seed_agent,
            generation_metadata={
                "strategy": "seed",
                "naive_cost_spent": self.evaluator_pool.get_full_eval_cost(),
                "cumulative_adaptive_cost": self.cumulative_adaptive_cost,
                "cumulative_naive_cost": self.cumulative_naive_cost,
                "task_id": benchmark.id,
            },
        )
        self.archive.export_to_csv(self.csv_filepath)

    def _resolve_task(self, task_param: Optional[str] = None) -> BenchmarkTask:
        target = task_param or self.selected_task_id
        if target and target != "All Benchmark Tasks (Suite / Random)":
            found = get_benchmark_task(target)
            if found:
                return found
        return get_random_task()

    def run_generation(self, strategy: Optional[str] = None, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Runs a single generation step of the multi-objective optimization loop."""
        self.current_generation += 1
        active_strategy = strategy or self.default_strategy
        gen = self.current_generation

        # 1. Sample Parent
        parent = self.sampler.sample_parent(self.archive)

        # 2. Mutate Agent (prompt + theta_H knobs)
        child, mut_type, mut_goal = self.mutator.mutate(parent, generation=gen)

        # 3. Propose Weights via Bayesian GP + Expected Improvement (Gap 2)
        if active_strategy == "baseline":
            # Static hand-weighted uniform distribution
            weights = {name: 1.0 / len(self.metric_names) for name in self.metric_names}
        else:
            weights = self.sampler.sample_weights()

        # 4. Select Benchmark Task
        task = self._resolve_task(task_id)
        task_dict = self._task_to_dict(task)

        # 5. Execute Agent on Task
        exec_start = time.time()
        agent_solution = self._execute_agent_on_task(child, task)
        exec_time_ms = (time.time() - exec_start) * 1000

        # 6. Gap 1 Selective Evaluator Search
        if active_strategy == "baseline":
            active_evaluators = self.evaluator_pool.get_evaluator_names()
        else:
            # Multi-tier selective search: run Core first
            core_evals = self.evaluator_pool.get_evaluators_by_tier("core")
            core_scores, core_costs, _ = self.evaluator_pool.evaluate_subset(
                agent=child,
                task=task_dict,
                agent_output=agent_solution,
                active_subset=core_evals,
                execution_context={"generation_latency_ms": exec_time_ms},
            )
            # Preview score across core metrics
            core_preview = sum(core_scores.values()) / max(1, len(core_scores))
            active_evaluators = self.sampler.determine_active_evaluator_subset(
                strategy=active_strategy,
                core_score_preview=core_preview,
            )

        # Run final active evaluators
        scores, costs, details = self.evaluator_pool.evaluate_subset(
            agent=child,
            task=task_dict,
            agent_output=agent_solution,
            active_subset=active_evaluators,
            execution_context={"generation_latency_ms": exec_time_ms},
        )

        # 7. Gap 3 Cost-Penalized Fitness Score
        fitness, spent_cost = calculate_cost_penalized_fitness(
            evaluator_scores=scores,
            weights=weights,
            active_evaluator_costs=costs,
            lambda_penalty=self.lambda_penalty,
        )

        # 8. Compute Delta_F and update Bayesian GP model
        delta_f = calculate_relative_improvement(
            child_fitness=fitness,
            parent_fitness=parent.fitness,
        )
        if active_strategy != "baseline":
            self.bayesian_optimizer.add_observation(weights=weights, delta_f=delta_f)

        # 9. Update telemetry and archive
        full_naive_cost = self.evaluator_pool.get_full_eval_cost()
        self.cumulative_adaptive_cost += spent_cost
        self.cumulative_naive_cost += full_naive_cost

        child.metrics = scores
        child.fitness = fitness
        child.cost_spent = spent_cost
        child.active_evaluators = active_evaluators

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
        }
        self.archive.add_agent(child, generation_metadata=gen_metadata)
        self.archive.export_to_csv(self.csv_filepath)

        record = {
            "generation": gen,
            "child_id": child.id,
            "parent_id": parent.id,
            "strategy": active_strategy,
            "fitness": fitness,
            "delta_f": delta_f,
            "cost_spent": spent_cost,
            "naive_cost": full_naive_cost,
            "active_evaluators": active_evaluators,
            "weights": weights,
            "metrics": scores,
        }
        self.history_records.append(record)
        return record

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

    def get_telemetry_summary(self) -> Dict[str, Any]:
        """Returns overall KPI metrics for Streamlit cards."""
        best_agent = self.archive.get_best_agent()
        total_saved = max(0.0, self.cumulative_naive_cost - self.cumulative_adaptive_cost)
        pct_saved = (
            (total_saved / self.cumulative_naive_cost * 100.0)
            if self.cumulative_naive_cost > 0
            else 0.0
        )
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
        }

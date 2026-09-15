import os
import sys
import unittest
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.agent import Agent, HarnessKnobs
from core.groq_client import GroqLLMClient
from core.mutator import PromptMutator
from core.archive import PopulationArchive
from evaluation.pool import EvaluatorPool
from evaluation.metrics import (
    FunctionalCorrectnessEvaluator,
    ExecutionLatencyEvaluator,
    TokenConcisenessEvaluator,
    LLMReasoningQualityEvaluator,
    EdgeCaseStressEvaluator,
    LLMSafetyHallucinationEvaluator,
)
from optimization.bayesian_weights import BayesianWeightOptimizer
from optimization.scoring import calculate_cost_penalized_fitness, calculate_relative_improvement
from optimization.joint_sampler import JointSearchSampler
from benchmarks.benchmarks_tasks import BENCHMARK_TASKS, get_benchmark_task
from controller import EvolutionController


class TestMultiObjectiveAgentOptimization(unittest.TestCase):
    """Unit and Integration tests for the Multi-Objective Agent Optimization system."""

    def setUp(self):
        self.llm_client = GroqLLMClient(is_mock=True, model="openai/gpt-oss-120b")
        self.pool = EvaluatorPool(llm_client=self.llm_client)

    def test_agent_dataclass_and_clone(self):
        agent = Agent(
            id="agent_001",
            system_prompt="Initial prompt",
            harness_knobs=HarnessKnobs(temperature=0.7, max_retries=2),
        )
        self.assertEqual(agent.id, "agent_001")
        self.assertEqual(agent.lineage_path, ["agent_001"])

        child = agent.clone(generation=1, mutation_type="cot", description="Added chain of thought")
        self.assertEqual(child.parent_id, "agent_001")
        self.assertEqual(child.generation, 1)
        self.assertEqual(len(child.lineage_path), 2)
        self.assertEqual(child.lineage_path[0], "agent_001")

    def test_groq_client_generation(self):
        response = self.llm_client.generate("Evaluate this code", "def solve(): return True")
        self.assertTrue(len(response) > 0)
        
        # Test explicit mock mode
        mock_client = GroqLLMClient(is_mock=True)
        mock_resp = mock_client._mock_generate("evaluate code", "def solve(): return True", 0.7)
        self.assertTrue("Score:" in mock_resp or "0." in mock_resp)

    def test_prompt_mutations_produce_distinct_prompts(self):
        parent = Agent(system_prompt="Base system prompt")
        mutator = PromptMutator(llm_client=self.llm_client)
        mutated_prompts = set()
        for g in range(1, 10):
            child, strategy, goal = mutator.mutate(parent, generation=g)
            self.assertNotEqual(child.system_prompt, parent.system_prompt)
            self.assertFalse(child.system_prompt.startswith("def solve"))
            mutated_prompts.add(child.system_prompt)
        # Should have generated multiple distinct prompt formulations
        self.assertTrue(len(mutated_prompts) >= 3)

    def test_all_6_evaluators(self):
        task = get_benchmark_task("he_055_fib")
        sample_solution = (
            "```python\n"
            "def solve(n: int) -> int:\n"
            "    if n < 0:\n"
            "        raise ValueError('n must be non-negative')\n"
            "    if n <= 1:\n"
            "        return n\n"
            "    a, b = 0, 1\n"
            "    for _ in range(2, n + 1):\n"
            "        a, b = b, a + b\n"
            "    return b\n"
            "```"
        )
        agent = Agent()
        task_dict = {
            "id": task.id,
            "description": task.description,
            "entry_point": task.entry_point,
            "test_cases": task.test_cases,
            "edge_cases": task.edge_cases,
            "constraints": task.constraints,
        }

        # Test Subset 1 (Core)
        ev1 = FunctionalCorrectnessEvaluator()
        r1 = ev1.evaluate(agent, task_dict, sample_solution)
        self.assertEqual(r1.score, 1.0)
        self.assertEqual(r1.cost, 0.001)

        ev2 = ExecutionLatencyEvaluator()
        r2 = ev2.evaluate(agent, task_dict, sample_solution, execution_context={"generation_latency_ms": 100})
        self.assertTrue(0.0 <= r2.score <= 1.0)
        self.assertEqual(r2.cost, 0.0)

        ev3 = TokenConcisenessEvaluator()
        r3 = ev3.evaluate(agent, task_dict, sample_solution)
        self.assertTrue(0.0 <= r3.score <= 1.0)
        self.assertEqual(r3.cost, 0.0)

        # Test Subset 2 (Deep)
        ev4 = LLMReasoningQualityEvaluator(self.llm_client)
        r4 = ev4.evaluate(agent, task_dict, sample_solution)
        self.assertTrue(0.0 <= r4.score <= 1.0)
        self.assertEqual(r4.cost, 0.015)

        ev5 = EdgeCaseStressEvaluator()
        r5 = ev5.evaluate(agent, task_dict, sample_solution)
        self.assertEqual(r5.score, 1.0)
        self.assertEqual(r5.cost, 0.005)

        ev6 = LLMSafetyHallucinationEvaluator(self.llm_client)
        r6 = ev6.evaluate(agent, task_dict, sample_solution)
        self.assertTrue(0.0 <= r6.score <= 1.0)
        self.assertEqual(r6.cost, 0.015)

    def test_bayesian_weight_optimizer_gp_and_ei(self):
        metric_names = self.pool.get_evaluator_names()
        opt = BayesianWeightOptimizer(metric_names=metric_names)

        # Propose initial weights
        w0 = opt.propose_next_weights()
        self.assertEqual(len(w0), len(metric_names))
        self.assertAlmostEqual(sum(w0.values()), 1.0, places=4)

        # Add observations
        for i in range(5):
            w = opt.propose_next_weights()
            delta_f = float(np.random.normal(0.1, 0.05))
            opt.add_observation(w, delta_f)

        # After >=3 observations, GP is fitted
        self.assertTrue(len(opt.X_history) >= 3)
        w_next = opt.propose_next_weights()
        self.assertEqual(len(w_next), len(metric_names))
        self.assertAlmostEqual(sum(w_next.values()), 1.0, places=4)

        # Check 1D slice generation for Streamlit
        w_grid, mu, sigma, ei, sampled_x, sampled_y = opt.get_1d_gp_slice(target_idx=0, resolution=50)
        self.assertEqual(len(w_grid), 50)
        self.assertEqual(len(mu), 50)
        self.assertEqual(len(sigma), 50)
        self.assertEqual(len(ei), 50)
        self.assertEqual(len(sampled_x), 5)

    def test_cost_penalized_scoring(self):
        scores = {"mu_1": 0.9, "mu_2": 0.8}
        weights = {"mu_1": 0.6, "mu_2": 0.4}
        costs = [0.001, 0.015]
        fitness, total_cost = calculate_cost_penalized_fitness(
            evaluator_scores=scores,
            weights=weights,
            active_evaluator_costs=costs,
            lambda_penalty=0.5,
        )
        expected_weighted = 0.6 * 0.9 + 0.4 * 0.8  # 0.54 + 0.32 = 0.86
        expected_cost = 0.016
        expected_fitness = 0.86 - (0.5 * 0.016)   # 0.852
        self.assertAlmostEqual(total_cost, expected_cost, places=5)
        self.assertAlmostEqual(fitness, expected_fitness, places=5)

    def test_end_to_end_controller_evolution(self):
        controller = EvolutionController(
            is_mock=True,
            model="openai/gpt-oss-120b",
            lambda_penalty=0.5,
            default_strategy="full_adaptive",
        )
        self.assertEqual(controller.current_generation, 0)
        self.assertEqual(len(controller.archive.agents), 1)

        # Run 3 generations
        results = controller.run_n_generations(3, strategy="full_adaptive")
        self.assertEqual(len(results), 3)
        self.assertEqual(controller.current_generation, 3)
        self.assertEqual(len(controller.archive.agents), 4)

        # Check telemetry
        telemetry = controller.get_telemetry_summary()
        self.assertEqual(telemetry["total_generations"], 3)
        self.assertTrue(telemetry["cumulative_cost_spent"] > 0)
        self.assertTrue(telemetry["cumulative_naive_cost"] >= telemetry["cumulative_cost_spent"])

        # Check Pareto front
        pareto = controller.archive.get_pareto_front()
        self.assertTrue(len(pareto) >= 1)

        # Test archive DataFrame export and CSV creation
        df = controller.archive.to_dataframe()
        self.assertEqual(len(df), 4)

        df_detailed = controller.archive.to_detailed_dataframe()
        self.assertEqual(len(df_detailed), 4)
        self.assertTrue("fitness" in df_detailed.columns)

        # Verify CSV export
        csv_file = controller.archive.export_to_csv("test_export.csv")
        self.assertTrue(os.path.exists(csv_file))
        if os.path.exists(csv_file):
            os.remove(csv_file)

    def test_stock_exchange_and_task_selection(self):
        stock_task = get_benchmark_task("mbpp_274_stock_exchange")
        self.assertIsNotNone(stock_task)
        self.assertEqual(stock_task.name, "Best Time to Buy & Sell Stock (MBPP/274)")

        controller = EvolutionController(
            is_mock=True,
            model="openai/gpt-oss-120b",
            initial_archetype="Algorithmic Specialist Agent",
            selected_task_id="mbpp_274_stock_exchange",
        )
        self.assertEqual(controller.initial_archetype, "Algorithmic Specialist Agent")
        res = controller.run_generation(strategy="full_adaptive")
        self.assertEqual(controller.current_generation, 1)

    def test_all_7_canonical_tasks_with_runner(self):
        from evaluation.runner import CodeExecutionRunner
        runner = CodeExecutionRunner(timeout_sec=2.0)
        self.assertEqual(len(BENCHMARK_TASKS), 7)

        # Ensure mock client produces solutions for each task that execute cleanly
        for task in BENCHMARK_TASKS:
            code = self.llm_client.generate("Write code", f"Task: {task.id}\n{task.description}")
            report = runner.run_task(
                task=task,
                agent_output=code,
                include_test_cases=True,
                include_edge_cases=True,
            )
            self.assertEqual(report.total_test_cases, len(task.test_cases))
            self.assertEqual(report.total_edge_cases, len(task.edge_cases))
            # Mock solutions should pass the standard test cases
            self.assertEqual(report.passed_test_cases, report.total_test_cases, f"Task {task.id} failed test cases: {report.errors}")

    def test_ucb1_mutation_bandit(self):
        from optimization.bandit import UCB1MutationBandit
        from core.mutator import MUTATION_STRATEGIES

        bandit = UCB1MutationBandit(arms=MUTATION_STRATEGIES, c=1.414)
        # Pull all arms once first
        pulled = []
        for _ in range(len(MUTATION_STRATEGIES)):
            arm = bandit.select_arm()
            pulled.append(arm[0])
            bandit.update(arm[0], reward=0.5)

        self.assertEqual(set(pulled), set([a[0] for a in MUTATION_STRATEGIES]))
        self.assertEqual(bandit.total_pulls, len(MUTATION_STRATEGIES))

        # Reward one arm heavily
        favored = MUTATION_STRATEGIES[0][0]
        bandit.update(favored, reward=10.0)
        next_arm = bandit.select_arm()
        self.assertEqual(next_arm[0], favored)

    def test_all_5_conditions_execution(self):
        conditions = ["single_metric", "static_cascade", "ucb1_bandit", "no_pruning", "full_adaptive"]
        controller = EvolutionController(is_mock=True, model="openai/gpt-oss-120b")

        for cond in conditions:
            res = controller.run_generation(strategy=cond, task_id="he_055_fib")
            self.assertEqual(res["strategy"], cond)
            self.assertIn("fitness", res)
            self.assertIn("cost_spent", res)

            if cond == "single_metric":
                self.assertEqual(res["active_evaluators"], ["mu_1_correctness"])
                self.assertEqual(res["weights"]["mu_1_correctness"], 1.0)
                self.assertEqual(res["weights"]["mu_2_latency"], 0.0)
            elif cond in ["static_cascade", "ucb1_bandit"]:
                for w in res["weights"].values():
                    self.assertAlmostEqual(w, 1.0 / 6.0, places=3)

    def test_csv_schema_compatibility_exact_32_columns(self):
        expected_columns = [
            "generation", "agent_id", "parent_id", "fitness", "cost_spent", "mutation_type",
            "temperature", "max_retries", "active_evaluators_count", "mu_1_correctness",
            "mu_2_latency", "mu_3_conciseness", "mu_4_reasoning", "mu_5_edge_cases", "mu_6_safety",
            "strategy", "naive_cost_spent", "cumulative_adaptive_cost", "cumulative_naive_cost",
            "task_id", "system_prompt", "top_p", "max_tokens", "delta_f", "cost_saved_this_gen",
            "solution_preview", "weight_mu_1_correctness", "weight_mu_2_latency",
            "weight_mu_3_conciseness", "weight_mu_4_reasoning", "weight_mu_5_edge_cases",
            "weight_mu_6_safety",
        ]
        controller = EvolutionController(is_mock=True, model="openai/gpt-oss-120b")
        controller.run_generation(strategy="full_adaptive")
        df = controller.archive.to_detailed_dataframe()
        for col in expected_columns:
            self.assertIn(col, df.columns, f"Missing required column: {col}")
        # Verify first 32 columns match exact expected order
        self.assertEqual(list(df.columns[:len(expected_columns)]), expected_columns)

    def test_harness_and_statistical_analysis(self):
        import pandas as pd
        from run_experiments import run_experiment_suite
        from analyze_results import analyze_csv_results

        test_csv = "test_mini_experiments.csv"
        df = run_experiment_suite(
            conditions=["single_metric", "full_adaptive"],
            tasks=["he_055_fib"],
            num_runs=2,
            num_generations=2,
            is_mock=True,
            inter_call_delay=0.0,
            output_csv=test_csv,
        )
        self.assertTrue(os.path.exists(test_csv))
        self.assertTrue(len(df) > 0)

        wilcoxon_df, conv_df = analyze_csv_results(test_csv)
        self.assertFalse(wilcoxon_df.empty)
        self.assertFalse(conv_df.empty)

        # Cleanup
        if os.path.exists(test_csv):
            os.remove(test_csv)
        if os.path.exists("statistical_analysis_summary.csv"):
            os.remove("statistical_analysis_summary.csv")
        if os.path.exists("task_condition_metrics.csv"):
            os.remove("task_condition_metrics.csv")


if __name__ == "__main__":
    unittest.main()

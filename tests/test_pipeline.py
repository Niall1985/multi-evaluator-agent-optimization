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
from tasks.benchmark_tasks import BENCHMARK_TASKS, get_benchmark_task
from controller import EvolutionController


class TestMultiObjectiveAgentOptimization(unittest.TestCase):
    """Unit and Integration tests for the Multi-Objective Agent Optimization system."""

    def setUp(self):
        self.llm_client = GroqLLMClient(api_key="", model="openai/gpt-oss-120b")
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
        mock_client = GroqLLMClient(api_key="")
        mock_client.is_mock = True
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
        task = BENCHMARK_TASKS[0]  # Fibonacci
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
            api_key="",
            model="openai/gpt-oss-120b",
            lambda_penalty=0.5,
            default_strategy="adaptive",
        )
        self.assertEqual(controller.current_generation, 0)
        self.assertEqual(len(controller.archive.agents), 1)

        # Run 3 generations
        results = controller.run_n_generations(3, strategy="adaptive")
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


if __name__ == "__main__":
    unittest.main()

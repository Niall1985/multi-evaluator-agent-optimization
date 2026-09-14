import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.arc_challenge import (
    ARC_TASKS,
    ARC_TASKS_DIR,
    arc_task_to_benchmark,
    get_arc_task,
    load_kaggle_split,
    pass_at_2_single,
    pixel_accuracy,
    score_arc_program,
)
from core.groq_client import GroqLLMClient
from evaluation.runner import CodeExecutionRunner
from tasks.benchmark_tasks import BENCHMARK_TASKS, get_benchmark_task


class TestArcChallengeBenchmark(unittest.TestCase):
    """Tests for the ARC-AGI benchmark adapter in benchmarks/arc_challenge."""

    def setUp(self):
        self.llm_client = GroqLLMClient(is_mock=True, model="openai/gpt-oss-120b")

    def test_bundled_tasks_load_with_correct_mapping(self):
        self.assertEqual(len(ARC_TASKS), len(list(ARC_TASKS_DIR.glob("*.json"))))
        self.assertGreaterEqual(len(ARC_TASKS), 4)
        for task in ARC_TASKS:
            with open(ARC_TASKS_DIR / f"{task.id.removeprefix('arc_')}.json") as f:
                raw = json.load(f)
            self.assertTrue(task.id.startswith("arc_"))
            self.assertEqual(task.entry_point, "solve")
            # train pairs -> test_cases, test pairs -> edge_cases
            self.assertEqual(len(task.test_cases), len(raw["train"]))
            self.assertEqual(len(task.edge_cases), len(raw["test"]))
            self.assertEqual(task.test_cases[0]["args"][0], raw["train"][0]["input"])
            self.assertEqual(task.test_cases[0]["expected"], raw["train"][0]["output"])
            self.assertIn("ARC-AGI task", task.description)
            # Prompt must not trip the mock client's coding-task keyword branches
            for kw in ("target", "differ", "stock", "fib", "bracket", "parenthes"):
                self.assertNotIn(kw, task.description.lower(), f"{task.id} description contains '{kw}'")

    def test_canonical_suite_unchanged_and_lookup_falls_back_to_arc(self):
        self.assertEqual(len(BENCHMARK_TASKS), 7)
        self.assertIsNone(get_arc_task("nope"))
        self.assertIs(get_benchmark_task("arc_3c9b0459"), get_arc_task("3c9b0459"))
        self.assertIs(get_benchmark_task(ARC_TASKS[0].name), ARC_TASKS[0])
        self.assertIsNotNone(get_benchmark_task("he_055_fib"))

    def test_mock_solutions_pass_train_and_held_out(self):
        runner = CodeExecutionRunner()
        for task in ARC_TASKS:
            code = self.llm_client.generate("You are a coding agent.", task.description)
            report = runner.run_task(task=task, agent_output=code)
            self.assertIsNone(report.syntax_error, task.id)
            self.assertEqual(report.passed_count, report.total_count, f"{task.id}: {report.errors}")

    def test_runner_accepts_numpy_grid_outputs(self):
        task = get_arc_task("3c9b0459")
        code = "import numpy as np\ndef solve(grid):\n    return np.rot90(np.array(grid), 2)\n"
        report = CodeExecutionRunner.run_task(task=task, agent_output=code)
        self.assertEqual(report.passed_count, report.total_count, report.errors)

    def test_wrong_program_fails_all_cases(self):
        task = get_arc_task("c8f0f002")
        report = CodeExecutionRunner.run_task(task=task, agent_output="def solve(grid):\n    return grid\n")
        self.assertEqual(report.passed_count, 0)

    def test_pass_at_2_and_pixel_accuracy(self):
        truth = [[1, 2], [3, 4]]
        self.assertEqual(pass_at_2_single([[[0, 0], [0, 0]], np.array(truth)], truth), 1)
        self.assertEqual(pass_at_2_single([[[0, 0], [0, 0]], [[1, 2], [3, 5]]], truth), 0)
        # Only the first two attempts count
        self.assertEqual(pass_at_2_single([[[0]], [[0]], truth], truth), 0)
        self.assertAlmostEqual(pixel_accuracy([[1, 2], [3, 5]], truth), 0.75)
        self.assertEqual(pixel_accuracy([[1, 2, 3]], truth), 0.0)  # shape mismatch

    def test_score_arc_program_openevolve_style_metrics(self):
        task = get_arc_task("25ff71a9")
        program = (
            "def transform_grid_attempt_1(grid):\n"
            "    return grid  # wrong\n"
            "def transform_grid_attempt_2(grid):\n"
            "    return [[0] * len(grid[0])] + [list(r) for r in grid[:-1]]\n"
        )
        train = score_arc_program(task, program)
        self.assertEqual(train["runs_successfully"], 1.0)
        self.assertEqual(train["combined_score"], 1.0)
        self.assertFalse(train["train_example_0_attempt_0"])
        self.assertTrue(train["train_example_0_attempt_1"])
        self.assertEqual(train["train_example_0_pass_at_2"], 1)

        held_out = score_arc_program(task, program, use_held_out=True)
        self.assertEqual(held_out["combined_score"], 1.0)
        self.assertIn("test_example_1_pass_at_2", held_out)  # this task has 2 test inputs

        broken = score_arc_program(task, "def solve(grid):\n    raise ValueError('boom')\n")
        self.assertEqual(broken["combined_score"], 0.0)
        self.assertIn("errors", broken)
        self.assertEqual(score_arc_program(task, "x = (")["runs_successfully"], 0.0)

    def test_kaggle_split_loader_merges_solutions(self):
        raw = json.load(open(ARC_TASKS_DIR / "c8f0f002.json"))
        challenges = {"c8f0f002": {"train": raw["train"], "test": [{"input": raw["test"][0]["input"]}]}}
        solutions = {"c8f0f002": [raw["test"][0]["output"]]}
        with tempfile.TemporaryDirectory() as d:
            json.dump(challenges, open(os.path.join(d, "arc-agi_evaluation_challenges.json"), "w"))
            json.dump(solutions, open(os.path.join(d, "arc-agi_evaluation_solutions.json"), "w"))
            tasks = load_kaggle_split(d, "evaluation")
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].edge_cases[0]["expected"], raw["test"][0]["output"])

        # Without solutions, unscorable test pairs are dropped from edge_cases
        t = arc_task_to_benchmark("x", {"train": raw["train"], "test": [{"input": [[1]]}]})
        self.assertEqual(t.edge_cases, [])
        self.assertEqual(len(t.test_cases), 3)


if __name__ == "__main__":
    unittest.main()

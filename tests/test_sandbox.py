"""Candidate code runs in a child process with a per-call time limit (evaluation/sandbox.py)."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from benchmarks.arc_challenge.scoring import score_arc_cases, to_grid
from evaluation import sandbox
from evaluation.runner import CodeExecutionRunner
from evaluation.sandbox import run_program

# Loops forever on a grid whose first cell is 1, answers instantly otherwise
LOOPS_ON_ONES = "def solve(grid):\n    while grid[0][0] == 1:\n        pass\n    return grid\n"


class TestSandbox(unittest.TestCase):
    def test_infinite_loop_times_out_and_later_cases_still_run(self):
        t0 = time.monotonic()
        run = run_program(LOOPS_ON_ONES, [("solve",)], [([[0]],), ([[1]],), ([[2]],)], timeout=1.0)
        self.assertLess(time.monotonic() - t0, 10)
        self.assertEqual(run.functions, ["solve"])
        self.assertEqual(run.results[0], [("ok", [[0]])])
        self.assertTrue(run.results[1][0][1].startswith("TimeoutError"))
        self.assertEqual(run.results[2], [("ok", [[2]])])

    def test_load_errors_and_name_groups(self):
        self.assertIn("SyntaxError", run_program("def solve(:", [("solve",)], [()]).load_error)
        self.assertIn("TimeoutError", run_program("while True: pass", [("solve",)], [()], timeout=1.0).load_error)
        two = "def a(x): return x + 1\ndef b(x): return x * 10\ndef solve(x): return 0\n"
        run = run_program(two, [("a", "b"), ("solve",)], [(1,), (2,)])
        self.assertEqual(run.functions, ["a", "b"])
        self.assertEqual(run.results, [[("ok", 2), ("ok", 10)], [("ok", 3), ("ok", 20)]])
        self.assertEqual(run_program("x = 1", [("solve",)], [(1,)]).functions, [])

    def test_exceptions_and_unpicklable_results_are_errors(self):
        code = "def solve(x):\n    if x: raise ValueError('bad')\n    return lambda: 0\n"
        run = run_program(code, [("solve",)], [(1,), (0,)])
        self.assertEqual(run.results[0][0], ("error", "ValueError: bad"))
        self.assertEqual(run.results[1][0][0], "error")

    def test_arc_scoring_survives_a_looping_program(self):
        cases = [{"args": ([[0]],), "expected": [[0]]}, {"args": ([[1]],), "expected": [[1]]}]
        m = score_arc_cases(cases, LOOPS_ON_ONES)
        self.assertEqual(m["runs_successfully"], 1.0)
        self.assertEqual(m["combined_score"], 0.5)
        self.assertTrue(any("TimeoutError" in e for e in m["errors"]))

    def test_repeat_runs_are_cached_and_copies_are_independent(self):
        sandbox.clear_cache()
        code = "def solve(grid):\n    return [row[::-1] for row in grid]\n"
        first = run_program(code, [("solve",)], [([[1, 2]],)])
        first.results[0][0][1][0][0] = 99  # a caller mutating its copy must not change the cache
        t0 = time.monotonic()
        again = run_program(code, [("solve",)], [([[1, 2]],)])
        self.assertLess(time.monotonic() - t0, 0.05)  # no child process
        self.assertEqual(again.results, [[("ok", [[2, 1]])]])
        self.assertEqual(sandbox.cache_stats, {"hits": 1, "misses": 1})
        run_program(code, [("solve",)], [([[3, 4]],)])  # other inputs: a new run
        self.assertEqual(sandbox.cache_stats["misses"], 2)

    def test_arc_cells_must_be_integers(self):
        self.assertEqual(to_grid([[1.0, 2]]), [[1, 2]])
        self.assertEqual(to_grid(np.array([[3, 4]])), [[3, 4]])
        self.assertIsNone(to_grid([["3", 4]]))
        self.assertIsNone(to_grid([[2.9, 4]]))

    def test_runner_reports_timeout_as_failed_case(self):
        report = CodeExecutionRunner.run_task(
            code=LOOPS_ON_ONES, entry_point="solve",
            test_cases=[{"args": ([[0]],), "expected": [[0]]}, {"args": ([[1]],), "expected": [[1]]},
                        {"args": ([[1]],), "expect_exception": True}],
            edge_cases=[],
        )
        self.assertEqual([c.passed for c in report.case_results], [True, False, False])
        self.assertIn("TimeoutError", report.case_results[1].exception_raised)


if __name__ == "__main__":
    unittest.main()

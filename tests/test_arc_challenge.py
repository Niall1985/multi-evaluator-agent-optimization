import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.arc_challenge import (
    AI_GENERATED_HEADER,
    ARC_TASKS,
    DEFAULT_DATA_ROOT,
    PARTIAL_ARC_EVALUATORS,
    REAL_ARC_EVALUATORS,
    ArcColorPaletteEvaluator,
    ArcHeldOutPassAt2Evaluator,
    ArcPassAt2Evaluator,
    ArcPixelAccuracyEvaluator,
    ArcRunsSuccessfullyEvaluator,
    ArcShapeMatchEvaluator,
    arc_task_to_benchmark,
    available_splits,
    build_arc_evaluator_pool,
    get_arc_task,
    load_arc_tasks,
    load_kaggle_split,
    pass_at_2_single,
    pixel_accuracy,
    score_arc_program,
)
from controller import EvolutionController
from core.groq_client import GroqLLMClient
from evaluation.runner import CodeExecutionRunner
from benchmarks.benchmarks_tasks import BENCHMARK_TASKS, get_benchmark_task


class TestArcChallengeBenchmark(unittest.TestCase):
    """Tests for the ARC-AGI benchmark adapter in benchmarks/arc_challenge."""

    def setUp(self):
        self.llm_client = GroqLLMClient(is_mock=True, model="openai/gpt-oss-120b")

    @staticmethod
    def _raw(split):
        root = DEFAULT_DATA_ROOT
        challenges = json.load(open(root / f"arc-agi_{split}_challenges.json"))
        solutions = json.load(open(root / f"arc-agi_{split}_solutions.json"))
        return challenges, solutions

    def test_bundled_split_files_load_with_correct_mapping(self):
        # Kaggle / OpenEvolve layout: arc-agi_{split}_challenges.json + arc-agi_{split}_solutions.json
        self.assertEqual(available_splits(DEFAULT_DATA_ROOT), ["training", "evaluation"])
        raw = {}
        for split in available_splits(DEFAULT_DATA_ROOT):
            challenges, solutions = self._raw(split)
            self.assertEqual(set(challenges), set(solutions))
            for tid, ch in challenges.items():
                for pair in ch["test"]:
                    self.assertNotIn("output", pair)  # test outputs live only in the solutions file
                raw[tid] = (ch, solutions[tid])
        self.assertEqual(len(ARC_TASKS), len(raw))
        self.assertGreaterEqual(len(ARC_TASKS), 4)
        for task in ARC_TASKS:
            ch, sol = raw[task.id.removeprefix("arc_")]
            self.assertTrue(task.id.startswith("arc_"))
            self.assertEqual(task.entry_point, "solve")
            # train pairs -> test_cases, test inputs + solutions -> edge_cases
            self.assertEqual(len(task.test_cases), len(ch["train"]))
            self.assertEqual(len(task.edge_cases), len(sol))
            self.assertEqual(task.test_cases[0]["args"][0], ch["train"][0]["input"])
            self.assertEqual(task.test_cases[0]["expected"], ch["train"][0]["output"])
            self.assertEqual(task.edge_cases[0]["args"][0], ch["test"][0]["input"])
            self.assertEqual(task.edge_cases[0]["expected"], sol[0])
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

    def test_split_loader_limit_order_and_missing_solutions(self):
        challenges, solutions = self._raw("training")
        with tempfile.TemporaryDirectory() as d:
            json.dump(challenges, open(os.path.join(d, "arc-agi_test_challenges.json"), "w"))
            # No solutions file: like the Kaggle hidden test split -> no scorable held-out pairs
            tasks = load_kaggle_split(d, "test")
            self.assertEqual([t.id for t in tasks], [f"arc_{k}" for k in challenges])  # order preserved
            self.assertTrue(all(t.edge_cases == [] for t in tasks))
            self.assertTrue(all(len(t.test_cases) == len(challenges[t.id[4:]]["train"]) for t in tasks))
            self.assertEqual(len(load_kaggle_split(d, "test", limit=2)), 2)
            self.assertEqual(available_splits(d), ["test"])
            self.assertEqual(len(load_arc_tasks(d)), len(challenges))

        # Solutions merged in, and an unscorable test pair without a solution is dropped
        tid = next(iter(challenges))
        t = arc_task_to_benchmark(tid, challenges[tid], test_solutions=solutions[tid])
        self.assertEqual([c["expected"] for c in t.edge_cases], solutions[tid])
        t = arc_task_to_benchmark("x", {"train": challenges[tid]["train"], "test": [{"input": [[1]]}]})
        self.assertEqual(t.edge_cases, [])

    # ------------------------------------------------------------------ evaluators

    def test_real_evaluators_report_official_metrics(self):
        task = get_arc_task("25ff71a9")
        d = {"entry_point": "solve", "test_cases": task.test_cases, "edge_cases": task.edge_cases}
        good = self.llm_client.generate("You are a coding agent.", task.description)
        wrong = "def solve(grid):\n    return [list(r) for r in grid]\n"
        crash = "def solve(grid):\n    return grid[99]\n"

        for E in REAL_ARC_EVALUATORS:
            self.assertEqual(E().evaluate(None, d, good).score, 1.0, E.__name__)
        self.assertEqual(ArcPassAt2Evaluator().evaluate(None, d, wrong).score, 0.0)
        self.assertEqual(ArcHeldOutPassAt2Evaluator().evaluate(None, d, wrong).score, 0.0)
        self.assertEqual(ArcRunsSuccessfullyEvaluator().evaluate(None, d, wrong).score, 1.0)  # runs, just wrong
        self.assertEqual(ArcRunsSuccessfullyEvaluator().evaluate(None, d, crash).score, 0.0)
        self.assertEqual(ArcRunsSuccessfullyEvaluator().evaluate(None, d, "x = (").score, 0.0)

        # pass@2: second attempt rescues the first
        two = (
            "def transform_grid_attempt_1(grid):\n    return grid\n"
            "def transform_grid_attempt_2(grid):\n    return [[0] * len(grid[0])] + [list(r) for r in grid[:-1]]\n"
        )
        res = ArcHeldOutPassAt2Evaluator().evaluate(None, d, two)
        self.assertEqual(res.score, 1.0)
        self.assertFalse(res.details["test_example_0_attempt_0"])
        self.assertTrue(res.details["test_example_0_attempt_1"])
        self.assertNotIn("outputs", res.details)

    def test_partial_evaluators_give_gradient_below_exact_match(self):
        task = get_arc_task("c8f0f002")  # recolour 7 -> 5
        d = {"entry_point": "solve", "test_cases": task.test_cases, "edge_cases": task.edge_cases}
        # Correct rule applied to every row except the last one: wrong, but close
        close = (
            "def solve(grid):\n"
            "    return [[5 if v == 7 else v for v in row] for row in grid[:-1]] + [list(grid[-1])]\n"
        )
        self.assertEqual(ArcPassAt2Evaluator().evaluate(None, d, close).score, 0.0)
        pix = ArcPixelAccuracyEvaluator().evaluate(None, d, close)
        self.assertTrue(0.5 < pix.score < 1.0, pix.score)
        self.assertTrue(pix.details["ai_generated"])
        self.assertEqual(ArcShapeMatchEvaluator().evaluate(None, d, close).score, 1.0)
        self.assertTrue(0.5 < ArcColorPaletteEvaluator().evaluate(None, d, close).score < 1.0)

        # Wrong shape: pixel/shape collapse to 0, palette (shape-agnostic) stays > 0
        transposed = "def solve(grid):\n    return [list(c) for c in zip(*grid)]\n"
        self.assertEqual(ArcShapeMatchEvaluator().evaluate(None, d, transposed).score, 0.0)
        self.assertEqual(ArcPixelAccuracyEvaluator().evaluate(None, d, transposed).score, 0.0)
        self.assertGreater(ArcColorPaletteEvaluator().evaluate(None, d, transposed).score, 0.0)
        # Broken program: everything 0
        for E in PARTIAL_ARC_EVALUATORS:
            self.assertEqual(E().evaluate(None, d, "x = (").score, 0.0)

    def test_partial_evaluators_are_marked_ai_generated(self):
        src_path = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "arc_challenge", "partial_evaluators.py")
        with open(src_path) as f:
            head = [next(f).rstrip("\n") for _ in range(4)]
        # The constant is the literal 4-line header at the top of the file
        self.assertEqual([h.removeprefix("# ") for h in head], AI_GENERATED_HEADER.split("\n"))
        self.assertEqual(len(AI_GENERATED_HEADER.split("\n")), 4)
        self.assertIn("AI-GENERATED", AI_GENERATED_HEADER)

    def test_arc_pool_and_controller_factory(self):
        pool = build_arc_evaluator_pool(self.llm_client)
        self.assertEqual(
            pool.get_evaluator_names(),
            ["arc_runs_successfully", "arc_pass_at_2_train", "arc_pass_at_2_test",
             "arc_pixel_accuracy", "arc_shape_match", "arc_color_palette"],
        )
        self.assertEqual(len(pool.get_evaluators_by_tier("core")), 4)
        self.assertEqual(len(pool.get_evaluators_by_tier("deep")), 2)
        self.assertEqual(len(build_arc_evaluator_pool(self.llm_client, include_partial=False).evaluators), 3)
        self.assertEqual(len(build_arc_evaluator_pool(self.llm_client, include_llm_judges=True).evaluators), 8)

        ctrl = EvolutionController(
            is_mock=True, inter_call_delay=0.0, selected_task_id="arc_3c9b0459",
            evaluator_pool_factory=build_arc_evaluator_pool,
        )
        self.assertEqual(ctrl.metric_names, pool.get_evaluator_names())
        rec = ctrl.run_generation(strategy="full_adaptive")
        self.assertEqual(rec["task_id"], "arc_3c9b0459")
        self.assertEqual(rec["metrics"]["arc_pass_at_2_train"], 1.0)
        self.assertTrue(set(rec["metrics"]) <= set(pool.get_evaluator_names()))
        self.assertIn("arc_pass_at_2_train", ctrl.archive.to_detailed_dataframe().columns)


if __name__ == "__main__":
    unittest.main()


class TestArcVisualisation(unittest.TestCase):
    """Plotly figures / tables for ARC results and the Streamlit ARC flow."""

    def test_gallery_figure_layout_and_orientation(self):
        from benchmarks.arc_challenge import ARC_PALETTE, task_gallery_figure
        from benchmarks.arc_challenge.viz import ARC_COLORSCALE

        task = get_arc_task("c8f0f002")  # 3 train + 1 test, non-square grids
        prog = "def solve(grid):\n    return [[5 if v == 7 else v for v in row] for row in grid[:-1]] + [list(grid[-1])]\n"
        fig = task_gallery_figure(task, prog)
        n_rows = len(task.test_cases) + len(task.edge_cases)
        self.assertEqual(len(fig.data), n_rows * 3)  # Input | Expected | Predicted per pair

        # Heatmap rows are reversed so grid row 0 is drawn on top (plotly's y points up)
        first_input = task.test_cases[0]["args"][0]
        z = [list(r) for r in fig.data[0].z]
        self.assertEqual(z, first_input[::-1])
        self.assertEqual(fig.data[0].customdata[-1][0], 0)  # top plotted row labelled as row 0
        self.assertEqual(len(ARC_PALETTE), 10)
        self.assertEqual(len(ARC_COLORSCALE), 20)
        self.assertEqual(fig.data[0].zmin, -0.5)
        self.assertEqual(fig.data[0].zmax, 9.5)

        titles = [a.text for a in fig.layout.annotations]
        self.assertEqual(len(titles), n_rows * 3)
        self.assertIn("Held-out 1 — Input", titles)
        predicted_titles = titles[2::3]
        self.assertTrue(all("✗" in t and "% px" in t for t in predicted_titles), predicted_titles)

        # Exact solution -> ticks; second attempt shown when only it is right
        two = (
            "def transform_grid_attempt_1(grid):\n    return grid\n"
            "def transform_grid_attempt_2(grid):\n    return [[5 if v == 7 else v for v in row] for row in grid]\n"
        )
        fig2 = task_gallery_figure(task, two)
        self.assertTrue(all("✓" in a.text for a in fig2.layout.annotations[2::3]))
        expected0 = task.test_cases[0]["expected"]
        self.assertEqual([list(r) for r in fig2.data[2].z], expected0[::-1])

        # Broken program still yields a figure with placeholder predicted cells
        fig3 = task_gallery_figure(task, "x = (")
        self.assertEqual(len(fig3.data), n_rows * 3)
        self.assertTrue(all("no grid" in a.text for a in fig3.layout.annotations[2::3]))

    def test_trajectory_and_summary(self):
        from benchmarks.arc_challenge import benchmark_summary_frame, build_arc_evaluator_pool, pass_at_2_trajectory_figure

        ctrl = EvolutionController(
            is_mock=True, inter_call_delay=0.0, selected_task_id="arc_25ff71a9",
            evaluator_pool_factory=build_arc_evaluator_pool,
        )
        ctrl.run_n_generations(2)
        fig = pass_at_2_trajectory_figure(ctrl.history_records)
        names = [tr.name for tr in fig.data]
        self.assertTrue(any("held-out" in n for n in names))
        self.assertTrue(any("AI-gen" in n for n in names))
        self.assertEqual(sorted(set(tr.line.dash for tr in fig.data)), ["dash", "dot", "solid"])
        # Records without ARC metrics produce an empty figure rather than an error
        self.assertEqual(len(pass_at_2_trajectory_figure([{"generation": 0, "metrics": {"mu_1_correctness": 1.0}}]).data), 0)

        df = benchmark_summary_frame(ctrl.archive.get_all_agents())
        self.assertEqual(list(df["Task"]), ["arc_25ff71a9"])
        self.assertEqual(df.loc[0, "arc_pass_at_2_test"], 1.0)
        self.assertTrue(bool(df.loc[0, "Solved (official)"]))
        self.assertTrue(benchmark_summary_frame([]).empty)

    def test_streamlit_arc_flow(self):
        from streamlit.testing.v1 import AppTest

        os.environ.pop("GROQ_API_KEY", None)  # force mock mode
        at = AppTest.from_file(os.path.join(os.path.dirname(__file__), "..", "app.py"), default_timeout=120)
        at.run()
        self.assertFalse(at.exception, at.exception)

        # Switch to the ARC suite: sidebar tier captions and metric names follow the pool
        at.sidebar.radio[0].set_value("ARC-AGI (3 real + 3 partial)").run()
        self.assertFalse(at.exception, at.exception)
        ctrl = at.session_state["controller"]
        self.assertEqual(ctrl.metric_names[:3], ["arc_runs_successfully", "arc_pass_at_2_train", "arc_pass_at_2_test"])
        self.assertTrue(any("arc_pass_at_2_test" in c.value for c in at.sidebar.caption))

        task_box = next(s for s in at.sidebar.selectbox if "Benchmark Problem" in s.label)
        task_box.set_value(next(o for o in task_box.options if o.startswith("ARC 00576224"))).run()
        at.sidebar.button[0].click().run()  # Run 1 Gen
        self.assertFalse(at.exception, at.exception)

        ctrl = at.session_state["controller"]
        self.assertEqual(ctrl.current_generation, 1)
        self.assertEqual(ctrl.history_records[-1]["task_id"], "arc_00576224")
        self.assertIn("Candidate to visualise", [s.label for s in at.selectbox])
        shown = {m.label: m.value for m in at.metric}
        self.assertEqual(shown.get("Pass@2 — held-out (official)"), "1.00")
        self.assertEqual(at.error, [])
        self.assertEqual(at.warning, [])

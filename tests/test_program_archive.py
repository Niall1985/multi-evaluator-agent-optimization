"""Program-archive mode: per-task MAP-Elites archive, TPE with an archived-performance prior, UCB1 over tasks,
repair calls, held-out pass@2 from archive picks, and the JSON checkpoint. Offline: a scripted client."""
import contextlib
import os
import random
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from benchmarks.arc_challenge import ARC_BENCHMARK_ID, build_arc_evaluator_pool, get_arc_task
from benchmarks.arc_challenge.repair import archive_pass_at_2, build_repair_prompt, inspect_program
from controller import EvolutionController
from core.archive import PopulationArchive, ProgramRecord
from core.groq_client import GroqLLMClient
from optimization.bandit import UCB1TaskSelector
from optimization.tpe import categorical_tpe_choice

IDENTITY = "```python\ndef solve(grid):\n    return [list(row) for row in grid]\n```"


class RepairOnlyClient(GroqLLMClient):
    """Fresh ARC attempts get the identity program; a repair prompt gets the mock's correct program. Everything
    else (the mutator) goes to the offline mock."""

    def __init__(self):
        super().__init__(is_mock=True)
        self.prompts = []

    def generate(self, system_prompt, user_prompt, temperature=0.7, top_p=0.95, max_tokens=1024):
        if "ARC-AGI task" in user_prompt:
            self.prompts.append(user_prompt)
            if "Current program:" not in user_prompt:
                return IDENTITY
        return super().generate(system_prompt, user_prompt, temperature, top_p, max_tokens)


class OptionsRecordingClient(RepairOnlyClient):
    """Also records which call_options were active for each ARC call (fresh or repair)."""

    def __init__(self):
        super().__init__()
        self.active = {}
        self.calls = []

    @contextlib.contextmanager
    def call_options(self, **options):
        self.active = options
        try:
            yield
        finally:
            self.active = {}

    def generate(self, system_prompt, user_prompt, temperature=0.7, top_p=0.95, max_tokens=1024):
        if "ARC-AGI task" in user_prompt:
            self.calls.append(("Current program:" in user_prompt, dict(self.active)))
        return super().generate(system_prompt, user_prompt, temperature, top_p, max_tokens)


def make_controller(client, **kwargs):
    return EvolutionController(
        selected_task_id=ARC_BENCHMARK_ID,
        evaluator_pool_factory=lambda llm: build_arc_evaluator_pool(llm, include_held_out=False),
        llm_client=client,
        initial_archetype="ARC Code-First Solver",
        evolution_mode="program_archive",
        n_islands=2,
        tasks_per_child=2,
        knob_bounds={"temperature": (0.1, 0.7), "top_p": (0.6, 1.0)},
        **kwargs,
    )


class TestSelectors(unittest.TestCase):
    def test_tpe_prior_without_trials_follows_prior(self):
        rng = random.Random(0)
        picks = [categorical_tpe_choice(["a", "b"], [], [], rng=rng, prior=[9, 1]) for _ in range(2000)]
        self.assertGreater(picks.count("a") / len(picks), 0.85)

    def test_tpe_trials_override_prior(self):
        rng = random.Random(0)
        observed, scores = ["b"] * 5 + ["a"] * 30, [1.0] * 5 + [0.0] * 30
        picks = [categorical_tpe_choice(["a", "b"], observed, scores, rng=rng, prior=[9, 1]) for _ in range(200)]
        self.assertEqual(picks.count("b"), 200)

    def test_tpe_rejects_bad_prior(self):
        with self.assertRaises(ValueError):
            categorical_tpe_choice(["a", "b"], [], [], prior=[1.0])

    def test_ucb_tasks_unpulled_first_then_index_and_retire(self):
        sel = UCB1TaskSelector()
        for task, reward in [("a", 0.9), ("b", 0.1)]:
            sel.update(task, reward)
        self.assertEqual(sel.select(["a", "b", "c"], 1), ["c"])
        self.assertEqual(sel.select(["a", "b"], 1), ["a"])
        sel.retire("a")
        self.assertEqual(sel.select(["a", "b"], 2), ["b"])
        self.assertEqual(UCB1TaskSelector.from_state(sel.state()).select(["a", "b"], 2), ["b"])


class TestArchiveAndHooks(unittest.TestCase):
    def setUp(self):
        self.task = get_arc_task("arc_25ff71a9")
        self.correct = GroqLLMClient._mock_arc_solution("arc-agi task 25ff71a9")

    def test_inspection_cells_and_feedback(self):
        wrong = inspect_program(self.task, IDENTITY)
        right = inspect_program(self.task, self.correct)
        broken = inspect_program(self.task, "```python\ndef solve(grid):\n    return 1/0\n```")
        self.assertFalse(wrong.solved)
        self.assertTrue(right.solved)
        self.assertEqual(right.cell, "1" * len(self.task.test_cases) + f"|s{len(self.task.test_cases)}")
        self.assertIn("expected", wrong.feedback)
        self.assertIn("ZeroDivisionError", broken.feedback)

    def test_repair_prompt_starts_with_task_and_never_shows_held_out_outputs(self):
        feedback = inspect_program(self.task, IDENTITY).feedback
        prompt = build_repair_prompt(self.task, IDENTITY, feedback, [self.correct])
        self.assertTrue(prompt.startswith(self.task.description))
        self.assertIn("Current program:", prompt)
        self.assertEqual(build_repair_prompt(self.task, None), self.task.description)

        def render(grid):
            return "\n".join("".join(str(v) for v in row) for row in grid)

        # Every grid the prompt may legitimately contain: demonstration inputs/outputs and held-out inputs
        visible = {render(c["args"][0]) for c in self.task.test_cases + self.task.edge_cases}
        visible |= {render(c["expected"]) for c in self.task.test_cases}
        hidden = [render(c["expected"]) for c in self.task.edge_cases if render(c["expected"]) not in visible]
        self.assertTrue(hidden)
        for grid in hidden:
            self.assertNotIn(grid, prompt)

    def test_map_elites_cell_keeps_higher_weighted_score(self):
        archive = PopulationArchive()
        w = {"m": 1.0}
        low = ProgramRecord(task_id="t", solution="a", agent_id="x", generation=0, scores={"m": 0.2}, cell="0|s1")
        high = ProgramRecord(task_id="t", solution="b", agent_id="y", generation=1, scores={"m": 0.6}, cell="0|s1")
        other = ProgramRecord(task_id="t", solution="c", agent_id="z", generation=1, scores={"m": 0.1}, cell="1|s1")
        self.assertTrue(archive.add_program(low, w))
        self.assertTrue(archive.add_program(high, w))
        self.assertFalse(archive.add_program(ProgramRecord(task_id="t", solution="d", agent_id="q", generation=2,
                                                           scores={"m": 0.5}, cell="0|s1"), w))
        archive.add_program(other, w)
        self.assertEqual({p.id for p in archive.task_elites("t")}, {high.id, other.id})
        self.assertEqual(archive.task_best("t", w).id, high.id)
        restored = PopulationArchive.from_state(archive.to_state())
        self.assertEqual(restored.task_cells, archive.task_cells)

    def test_archive_pass_at_2_uses_two_distinct_programs(self):
        ranked = [("p_wrong", IDENTITY), ("p_dup", IDENTITY), ("p_right", self.correct)]
        result = archive_pass_at_2(self.task, ranked)
        self.assertEqual(result["attempt_program_ids"], ["p_wrong", "p_right"])
        self.assertTrue(result["solved"])
        self.assertFalse(archive_pass_at_2(self.task, ranked[:2])["solved"])


class TestProgramArchiveMode(unittest.TestCase):
    def setUp(self):
        random.seed(0)
        np.random.seed(0)

    def test_held_out_evaluator_left_out_of_pool(self):
        ctrl = make_controller(RepairOnlyClient())
        self.assertNotIn("arc_pass_at_2_test", ctrl.metric_names)

    def test_repairs_solve_tasks_and_retire_them(self):
        client = RepairOnlyClient()
        ctrl = make_controller(client)
        n_tasks = len(ctrl._resolve_tasks()[1])
        self.assertEqual(len(ctrl.archive.programs), n_tasks)
        self.assertFalse(ctrl.archive.solved_tasks)
        self.assertEqual(len(ctrl.archive.islands), 2)

        for _ in range(15):
            r = ctrl.run_generation()
            if r.get("all_solved"):
                break
            self.assertLessEqual(r["calls"], 2 * 2 + 2)
        self.assertEqual(len(ctrl.archive.solved_tasks), n_tasks)
        self.assertTrue(any("Current program:" in p for p in client.prompts))
        self.assertEqual(ctrl.task_selector.retired, set(ctrl.archive.solved_tasks))
        # a solved task is never picked again
        self.assertTrue(ctrl.run_generation()["all_solved"])

        held_out = ctrl.archive_held_out()
        self.assertTrue(all(r["solved"] for r in held_out.values()))
        children = [rec for rec in ctrl.history_records if rec["generation"] > 0]
        self.assertTrue(children)
        self.assertTrue(all(rec["island"] in (0, 1) for rec in children))

    def test_repair_calls_get_repair_options_and_fresh_calls_do_not(self):
        client = OptionsRecordingClient()  # sequential (max_concurrency=1), so `active` is unambiguous
        options = {"thinking_budget": 2048, "reasoning_effort": "low"}
        ctrl = make_controller(client, repair_call_options=options)
        for _ in range(3):
            ctrl.run_generation()
        repairs = [opts for is_repair, opts in client.calls if is_repair]
        fresh = [opts for is_repair, opts in client.calls if not is_repair]
        self.assertTrue(repairs and fresh)
        self.assertTrue(all(opts == options for opts in repairs))
        self.assertTrue(all(opts == {} for opts in fresh))

    def test_checkpoint_written_after_seed_and_each_generation_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            ctrl = make_controller(RepairOnlyClient(), state_path=path)
            self.assertTrue(os.path.exists(path))  # seed checkpoint, before any generation
            ctrl.run_generation()
            ctrl.fanova.below_threshold_windows["arc_color_palette"] = 2
            ctrl.save_state()
            expected_draw = random.random(), float(np.random.random())

            # A crash mid-write leaves only a stray temp file; the checkpoint itself stays readable
            with open(path + ".tmp", "w") as f:
                f.write("{torn")
            resumed = make_controller(RepairOnlyClient(), evaluate_seed=False, state_path=path)
            self.assertEqual(resumed.load_state(), 1)
            self.assertEqual((random.random(), float(np.random.random())), expected_draw)
            self.assertEqual(set(resumed.archive.programs), set(ctrl.archive.programs))
            self.assertEqual(resumed.archive.task_cells, ctrl.archive.task_cells)
            self.assertEqual(resumed.archive.islands, ctrl.archive.islands)
            self.assertEqual(resumed.task_selector.counts, ctrl.task_selector.counts)
            self.assertEqual(len(resumed.sampler.trials), len(ctrl.sampler.trials))
            self.assertEqual(len(resumed.sampler.idea_study.trials), len(ctrl.sampler.trials))
            self.assertEqual(resumed.bayesian_optimizer.y_history, ctrl.bayesian_optimizer.y_history)
            self.assertEqual(resumed.fanova.below_threshold_windows["arc_color_palette"], 2)
            self.assertEqual(len(resumed.history_records), len(ctrl.history_records))

            self.assertEqual(resumed.run_generation()["generation"], 2)
            again = make_controller(RepairOnlyClient(), evaluate_seed=False)
            self.assertEqual(again.load_state(path), 2)  # the resumed run kept checkpointing

    def test_checkpoint_rejects_a_different_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            make_controller(RepairOnlyClient(), state_path=path)
            other = EvolutionController(selected_task_id=ARC_BENCHMARK_ID, evaluator_pool_factory=build_arc_evaluator_pool,
                                        llm_client=RepairOnlyClient(), evolution_mode="program_archive",
                                        evaluate_seed=False)
            with self.assertRaises(ValueError):
                other.load_state(path)  # its pool still has arc_pass_at_2_test


if __name__ == "__main__":
    unittest.main()

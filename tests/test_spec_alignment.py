"""MESA spec alignment: Delta_F target, density-ratio choices, joint (w, theta_H) GP, windowed pruning,
and the §5.2 / Algorithm 8 synthetic pruning validation."""
import os
import random
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.synthetic_pruning import run_synthetic_pruning, summarize
from controller import EvolutionController
from optimization.bayesian_weights import BayesianWeightOptimizer
from optimization.fanova import FunctionalANOVA
from optimization.scoring import calculate_delta_f
from optimization.tpe import categorical_tpe_choice


class TestDeltaF(unittest.TestCase):
    def test_parent_rescalarized_under_current_weights_without_cost(self):
        w = {"a": 0.8, "b": 0.2}
        child = {"a": 0.5, "b": 1.0}
        parent = {"a": 0.25, "b": 0.0, "c": 0.9}  # "c" not observed on the child: ignored
        self.assertAlmostEqual(calculate_delta_f(child, parent, w), 0.8 * 0.25 + 0.2 * 1.0)

    def test_evaluator_missing_on_parent_is_not_compared(self):
        self.assertAlmostEqual(calculate_delta_f({"a": 1.0, "b": 1.0}, {"a": 0.5}, {"a": 0.5, "b": 0.5}), 0.25)


class TestCategoricalTPE(unittest.TestCase):
    def test_prefers_the_choice_concentrated_in_the_good_set(self):
        rng = random.Random(0)
        observed = ["x"] * 10 + ["y"] * 10
        scores = [1.0] * 10 + [0.0] * 10
        picks = [categorical_tpe_choice(["x", "y"], observed, scores, rng=rng) for _ in range(50)]
        self.assertEqual(picks.count("x"), 50)

    def test_untried_choice_beats_one_seen_only_in_the_bad_set(self):
        # Bergstra et al. §4.2 pseudo-counts: gamma(20) = 2 good trials (both "x"), so
        # l = (3, 1, 1)/5 and g = (9, 11, 1)/21 for (x, y, z); l/g is 1.4, 0.38 and 4.2. The untried "z" has
        # the highest ratio because nothing in the bad set lands on it: TPE's exploration term.
        rng = random.Random(0)
        observed = ["x"] * 10 + ["y"] * 10
        scores = [1.0] * 10 + [0.0] * 10
        picks = [categorical_tpe_choice(["x", "y", "z"], observed, scores, rng=rng) for _ in range(50)]
        self.assertEqual(picks.count("y"), 0)
        self.assertGreater(picks.count("z"), picks.count("x"))

    def test_without_observations_falls_back_to_uniform(self):
        self.assertIn(categorical_tpe_choice(["p", "q"], [], [], rng=random.Random(1)), ["p", "q"])

    def test_idea_goes_through_optuna_ask_and_tell(self):
        from optuna.trial import TrialState

        ctrl = EvolutionController(is_mock=True, inter_call_delay=0.0)
        for _ in range(3):
            ctrl.run_generation(strategy="full_adaptive")
        trials = ctrl.sampler.idea_study.trials
        self.assertEqual([t.state for t in trials], [TrialState.COMPLETE] * 3)
        self.assertEqual([t.params["idea"] for t in trials], [r["mutation_type"] for r in ctrl.history_records[1:]])
        self.assertEqual([t.value for t in trials], [r["delta_f"] for r in ctrl.history_records[1:]])

    def test_baselines_do_not_touch_the_idea_study(self):
        ctrl = EvolutionController(is_mock=True, inter_call_delay=0.0)
        ctrl.run_generation(strategy="static_cascade")
        self.assertIsNone(ctrl.sampler.idea_study)


class TestJointContinuousGP(unittest.TestCase):
    def test_proposes_weights_on_simplex_and_knobs_in_bounds(self):
        np.random.seed(0)
        names = ["m1", "m2", "m3"]
        opt = BayesianWeightOptimizer(metric_names=names)
        for i in range(6):
            w, knobs = opt.propose_next_config()
            opt.add_observation(w, delta_f=w["m1"] - 0.1 * knobs["temperature"], knobs=knobs)
        w, knobs = opt.propose_next_config()
        self.assertAlmostEqual(sum(w.values()), 1.0, places=6)
        for name, (lo, hi) in opt.knob_bounds.items():
            self.assertTrue(lo <= knobs[name] <= hi)
        self.assertEqual(opt.training_matrix().shape, (6, len(names) + len(opt.knob_bounds)))


class TestWindowedPruning(unittest.TestCase):
    def test_noise_redundant_and_protected(self):
        rng = np.random.RandomState(0)
        q = rng.uniform(0, 1, 40)
        inf0 = q + rng.normal(0, 0.1, 40)
        scores = np.column_stack([
            inf0,
            q + rng.normal(0, 0.1, 40),
            inf0 + rng.normal(0, 0.01, 40),  # near-copy of inf0
            rng.uniform(0, 1, 40),
        ])
        names = ["inf0", "inf1", "red", "noise"]
        fanova = FunctionalANOVA(evaluator_names=names)
        fanova.fit_informativeness(scores, costs=[0.0] * 4)
        self.assertEqual(fanova.redundant_with, {"red": "inf0"})
        self.assertEqual(sorted(fanova.identify_redundant_evaluators()), ["noise", "red"])

        # Protecting the copy makes the redundancy rule drop its correlate instead
        fanova.fit_informativeness(scores, costs=[0.0] * 4, protected=["red"])
        self.assertEqual(fanova.redundant_with, {"inf0": "red"})

    def test_prunes_only_after_patience_consecutive_windows(self):
        fanova = FunctionalANOVA(evaluator_names=["a", "b"])
        fanova.importance_scores = {"a": 0.01, "b": 0.5}
        fanova.close_window()
        fanova.close_window()
        self.assertEqual(fanova.get_pruned_evaluators(patience=3), [])
        fanova.importance_scores = {"a": 0.5, "b": 0.5}  # one good window resets the streak
        fanova.close_window()
        fanova.importance_scores = {"a": 0.01, "b": 0.5}
        for _ in range(3):
            fanova.close_window()
        self.assertEqual(fanova.get_pruned_evaluators(patience=3), ["a"])


class TestSyntheticPruningValidation(unittest.TestCase):
    """Spec §5.2: pruning results on real tasks only count once this passes."""

    def test_mixed_task_precision_and_recall(self):
        reports = [run_synthetic_pruning(["inf", "inf", "red", "noise", "noise"], generations=50, seed=s) for s in range(30)]
        s = summarize(reports)
        self.assertGreaterEqual(s["precision"], 0.9)
        self.assertGreaterEqual(s["recall"], 0.7)
        self.assertLessEqual(s["false_prune_rate"], 0.1)

    def test_control_task_false_prune_rate(self):
        reports = [run_synthetic_pruning(["inf"] * 5, generations=50, seed=s) for s in range(30)]
        self.assertLessEqual(summarize(reports)["false_prune_rate"], 0.06)


class TestControllerLoop(unittest.TestCase):
    def test_cascade_runs_each_core_evaluator_once_per_generation(self):
        ctrl = EvolutionController(is_mock=True, inter_call_delay=0.0)
        calls = []
        original = ctrl.evaluator_pool.evaluate_subset

        def counting(*args, **kwargs):
            calls.extend(kwargs["active_subset"])
            return original(*args, **kwargs)

        ctrl.evaluator_pool.evaluate_subset = counting
        for strategy in ("static_cascade", "full_adaptive"):
            calls.clear()
            rec = ctrl.run_generation(strategy=strategy)
            self.assertEqual(len(calls), len(set(calls)), f"{strategy} re-ran an evaluator: {calls}")
            self.assertEqual(sorted(calls), sorted(rec["active_evaluators"]))

    def test_joint_conditions_feed_gp_and_trials(self):
        ctrl = EvolutionController(is_mock=True, inter_call_delay=0.0, fanova_window=2)
        for _ in range(4):
            rec = ctrl.run_generation(strategy="full_adaptive")
        self.assertEqual(len(ctrl.bayesian_optimizer.y_history), 4)
        self.assertEqual(len(ctrl.sampler.trials), 4)
        parent = next(a for a in ctrl.archive.get_all_agents() if a.id == rec["parent_id"])
        self.assertAlmostEqual(rec["delta_f"], calculate_delta_f(rec["metrics"], parent.metrics, rec["weights"]))
        self.assertNotAlmostEqual(rec["delta_f"], rec["fitness"] - parent.fitness)  # the old, cost-leaking target
        self.assertEqual(ctrl.fanova.windows_measured, 1)  # windows close at gens 2 and 4; gen 2 has < 4 agents


if __name__ == "__main__":
    unittest.main()

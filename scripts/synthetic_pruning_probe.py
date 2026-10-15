"""Spec §5.2 / Algorithm 8 probe: how well does each fANOVA formulation separate evaluator types?

Synthetic evaluators, no LLM calls. Each generation a child of quality q is scored by
  - Type 1 informative: q + small noise
  - Type 2 redundant:   copy of informative[0] + tiny noise
  - Type 3 noise:       uniform, independent of q
and the importance of every evaluator is printed under two surrogates:
  A) Random Forest over the evaluator *scores* -> fitness  (pre-spec-alignment implementation)
  B) GP over the evaluator *weights* (+theta_H)  -> Delta_F (spec §3.3 literal: I_j of w_j)

Run: uv run python scripts/synthetic_pruning_probe.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.bayesian_weights import BayesianWeightOptimizer  # noqa: E402
from optimization.fanova import FunctionalANOVA  # noqa: E402
from optimization.scoring import calculate_delta_f  # noqa: E402

TYPES = ["inf", "inf", "red", "noise", "noise"]
NAMES = [f"{t}{i}" for i, t in enumerate(TYPES)]


def scores_for(q: float, rng: np.random.RandomState) -> dict:
    out = {}
    for name, t in zip(NAMES, TYPES):
        if t == "inf":
            out[name] = float(np.clip(q + rng.normal(0, 0.05), 0, 1))
        elif t == "noise":
            out[name] = float(rng.uniform(0, 1))
    out[NAMES[2]] = float(np.clip(out[NAMES[0]] + rng.normal(0, 0.01), 0, 1))
    return out


def run(seed: int, generations: int = 30):
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    opt = BayesianWeightOptimizer(NAMES)
    parent_q = 0.4
    parent = scores_for(parent_q, rng)
    metrics_X, fitness_y = [], []
    for _ in range(generations):
        w, knobs = opt.propose_next_config()
        q = float(np.clip(parent_q + rng.normal(0.02, 0.08), 0, 1))
        child = scores_for(q, rng)
        opt.add_observation(w, calculate_delta_f(child, parent, w), knobs=knobs)
        metrics_X.append([child[n] for n in NAMES])
        fitness_y.append(sum(w[n] * child[n] for n in NAMES))
        if q > parent_q:
            parent_q, parent = q, child

    a = FunctionalANOVA(NAMES).fit(np.array(metrics_X), np.array(fitness_y)).get_importance_scores()
    b = FunctionalANOVA(NAMES).fit(opt.training_matrix(), np.array(opt.y_history), surrogate=opt.gp).get_importance_scores()
    # C) leave-one-out: share of the *other* evaluators' consensus that mu_j alone explains (R^2)
    M = np.array(metrics_X)
    c = {}
    for j, n in enumerate(NAMES):
        others = np.delete(M, j, axis=1).mean(axis=1)
        r = np.corrcoef(M[:, j], others)[0, 1]
        c[n] = float(r * r) if np.isfinite(r) else 0.0
    return a, b, c


if __name__ == "__main__":
    agg = {"A rf(scores->F)": [], "B gp(w->dF)": [], "C loo consensus": []}
    for seed in range(5):
        for key, imp in zip(agg, run(seed)):
            agg[key].append([imp[n] for n in NAMES])
    print("evaluator        " + "  ".join(f"{n:>7}" for n in NAMES))
    for key, rows in agg.items():
        print(f"{key:<17}" + "  ".join(f"{v:7.3f}" for v in np.mean(rows, axis=0)))

    # Algorithm 8 end-to-end on the real windowed pruning (fit_informativeness + patience + epsilon-probe)
    from benchmarks.synthetic_pruning import run_synthetic_pruning, summarize

    seeds = range(20)
    mixed = summarize([run_synthetic_pruning(TYPES, seed=s) for s in seeds])
    control = summarize([run_synthetic_pruning(["inf"] * 5, seed=s) for s in seeds])
    print("\nAlgorithm 8, 30 generations, W=3, 20 seeds")
    print("  mixed   (inf,inf,red,noise,noise):", {k: round(v, 3) for k, v in mixed.items()})
    print("  control (all informative):        ", {k: round(v, 3) for k, v in control.items()})

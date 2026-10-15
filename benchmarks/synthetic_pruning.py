"""Spec §5.2 / Algorithm 8: synthetic pruning validation task (no LLM calls).

Each generation produces a child of latent quality q, scored by k synthetic evaluators of known type:
  - "inf"   Type 1, tracks true fitness:            q + N(0, inf_noise)
  - "red"   Type 2, near-copy of the first "inf":   inf_0 + N(0, red_noise)
  - "noise" Type 3, independent of fitness:         U(0, 1)
The real pruning machinery (FunctionalANOVA.fit_informativeness, windows of W generations, W-window
patience, epsilon-probing of pruned evaluators) runs on the growing archive of scores, and the report gives
pruning precision/recall and generations-to-convergence. A control run with only "inf" evaluators measures
the false-prune rate.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from optimization.fanova import FunctionalANOVA


@dataclass
class PruningReport:
    pruned: List[str]
    should_prune: List[str]
    should_keep: List[str]
    precision: float
    recall: float
    false_prune_rate: float
    generations_to_convergence: Optional[int]  # first gen after which the pruned set never changed again


def run_synthetic_pruning(
    types: Sequence[str],
    generations: int = 30,
    window: int = 3,
    epsilon: float = 0.1,
    inf_noise: float = 0.1,
    red_noise: float = 0.01,
    tau: float = 0.05,
    seed: int = 0,
) -> PruningReport:
    rng = np.random.RandomState(seed)
    names = [f"{t}{i}" for i, t in enumerate(types)]
    first_inf = next((i for i, t in enumerate(types) if t == "inf"), None)
    fanova = FunctionalANOVA(evaluator_names=names, redundancy_threshold=tau)

    rows: List[List[float]] = []
    q = 0.5
    pruned_history: List[frozenset] = []
    for gen in range(1, generations + 1):
        q = float(np.clip(q + rng.normal(0.0, 0.15), 0.0, 1.0))
        pruned = set(fanova.get_pruned_evaluators(window))
        row: List[float] = []
        for i, t in enumerate(types):
            if t == "inf":
                val = q + rng.normal(0.0, inf_noise)
            elif t == "red":
                base = row[first_inf] if first_inf is not None and first_inf < i and not np.isnan(row[first_inf]) else q
                val = base + rng.normal(0.0, red_noise)
            else:
                val = rng.uniform(0.0, 1.0)
            # Pruned evaluators are only run on an epsilon-probe
            row.append(float("nan") if names[i] in pruned and rng.rand() >= epsilon else float(val))
        rows.append(row)
        if gen % window == 0 and len(rows) >= 4:
            fanova.fit_informativeness(np.array(rows), costs=[0.0] * len(types))
            fanova.close_window()
        pruned_history.append(frozenset(fanova.get_pruned_evaluators(window)))

    final = pruned_history[-1]
    convergence = None
    for g in range(len(pruned_history)):
        if all(p == final for p in pruned_history[g:]):
            convergence = g + 1
            break

    should_prune = [n for n, t in zip(names, types) if t in ("red", "noise")]
    should_keep = [n for n, t in zip(names, types) if t == "inf"]
    true_pos = len([n for n in final if n in should_prune])
    precision = true_pos / len(final) if final else 1.0
    recall = true_pos / len(should_prune) if should_prune else 1.0
    false_prune = len([n for n in final if n in should_keep]) / len(should_keep) if should_keep else 0.0
    return PruningReport(
        pruned=sorted(final),
        should_prune=should_prune,
        should_keep=should_keep,
        precision=precision,
        recall=recall,
        false_prune_rate=false_prune,
        generations_to_convergence=convergence,
    )


def summarize(reports: Sequence[PruningReport]) -> Dict[str, float]:
    conv = [r.generations_to_convergence for r in reports if r.generations_to_convergence is not None]
    return {
        "precision": float(np.mean([r.precision for r in reports])),
        "recall": float(np.mean([r.recall for r in reports])),
        "false_prune_rate": float(np.mean([r.false_prune_rate for r in reports])),
        "generations_to_convergence": float(np.mean(conv)) if conv else float("nan"),
    }

"""Tree-structured Parzen Estimator for one categorical variable whose choice set changes between trials.

Used for the parent pi, whose candidates (the current Pareto front) change every generation. Optuna's
TPESampler cannot model that: its CategoricalDistribution "does not support dynamic value space" and raises
ValueError if the choices differ from earlier trials. The mutation idea has a fixed menu and goes through
Optuna directly (see JointSearchSampler.sample_idea).

Math, from Bergstra, Bardenet, Bengio & Kegl, "Algorithms for Hyper-Parameter Optimization", NeurIPS 2011:
  §4, Eq. 2      p(x | y) = l(x) if y is in the better gamma-quantile, g(x) otherwise
  §4.1, Eq. 3    EI(x) is proportional to (gamma + (g(x) / l(x)) * (1 - gamma))^-1, so EI is maximised by
                 maximising l(x) / g(x); candidates are drawn from l and the best ratio is returned
  §4.2           categorical: "supposing the prior was a vector of N probabilities p_i, the posterior vector
                 elements were proportional to N p_i + C_i where C_i counts the occurrences of choice i"
With a uniform prior p_i = 1/N this is one pseudo-count per choice, which keeps a choice never seen in the
better group (e.g. an agent that just joined the frontier) reachable.

The split size gamma(n) = min(ceil(0.1 n), 25) and the 24 candidate draws are Optuna's TPESampler defaults
(`default_gamma`, `n_ei_candidates`), so the parent and idea choices use the same split rule.
"""
import math
import random
from typing import Hashable, List, Optional, Sequence


def default_gamma(n: int) -> int:
    """Size of the better group for n observations (Optuna TPESampler default)."""
    return min(math.ceil(0.1 * n), 25)


def categorical_tpe_choice(
    candidates: Sequence[Hashable],
    observed: Sequence[Hashable],
    scores: Sequence[float],
    n_ei_candidates: int = 24,
    rng: Optional[random.Random] = None,
) -> Hashable:
    """Returns the candidate maximising l(x) / g(x) among `n_ei_candidates` draws from l(x).

    `observed[t]` is the value the variable took in trial t and `scores[t]` that trial's score (higher is
    better). Trials whose value is not a current candidate are dropped before the split.
    """
    rng = rng or random
    cands = list(candidates)
    if not cands:
        raise ValueError("categorical_tpe_choice needs at least one candidate")
    index = {c: i for i, c in enumerate(cands)}
    trials = [(index[x], s) for x, s in zip(observed, scores) if x in index]
    if not trials:
        return rng.choice(cands)

    trials.sort(key=lambda t: t[1], reverse=True)
    n_good = max(1, default_gamma(len(trials)))
    k = len(cands)

    def posterior(split: List[tuple]) -> List[float]:
        # Bergstra et al. §4.2 with uniform prior p_i = 1/k: proportional to k * (1/k) + C_i = 1 + C_i
        counts = [1.0] * k
        for i, _ in split:
            counts[i] += 1.0
        total = sum(counts)
        return [c / total for c in counts]

    l_x, g_x = posterior(trials[:n_good]), posterior(trials[n_good:])
    draws = rng.choices(range(k), weights=l_x, k=n_ei_candidates)
    return cands[max(draws, key=lambda i: l_x[i] / g_x[i])]

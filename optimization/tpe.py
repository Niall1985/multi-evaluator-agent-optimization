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
better group (e.g. an agent that just joined the frontier) reachable. A non-uniform prior (the program-archive
mode passes archived performance) keeps the same N pseudo-counts but spreads them by p_i, so a strong archived
candidate starts favoured and the observed Delta_F splits then move it.

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
    prior: Optional[Sequence[float]] = None,
) -> Hashable:
    """Returns the candidate maximising l(x) / g(x) among `n_ei_candidates` draws from l(x).

    `observed[t]` is the value the variable took in trial t and `scores[t]` that trial's score (higher is
    better). Trials whose value is not a current candidate are dropped before the split. `prior` (one
    non-negative weight per candidate, normalised here) is Bergstra et al.'s p_i; None is uniform. With no
    usable trials the draw is from the prior alone.
    """
    rng = rng or random
    cands = list(candidates)
    if not cands:
        raise ValueError("categorical_tpe_choice needs at least one candidate")
    k = len(cands)
    if prior is None:
        p = [1.0 / k] * k
    else:
        if len(prior) != k or any(w < 0 for w in prior):
            raise ValueError("prior needs one non-negative weight per candidate")
        total = sum(prior)
        p = [w / total for w in prior] if total > 0 else [1.0 / k] * k
    index = {c: i for i, c in enumerate(cands)}
    trials = [(index[x], s) for x, s in zip(observed, scores) if x in index]
    if not trials:
        return cands[rng.choices(range(k), weights=p, k=1)[0]]

    trials.sort(key=lambda t: t[1], reverse=True)
    n_good = max(1, default_gamma(len(trials)))

    def posterior(split: List[tuple]) -> List[float]:
        # Bergstra et al. §4.2: proportional to k * p_i + C_i (uniform p_i = 1/k gives 1 + C_i)
        counts = [k * p_i for p_i in p]
        for i, _ in split:
            counts[i] += 1.0
        total = sum(counts)
        return [c / total for c in counts]

    l_x, g_x = posterior(trials[:n_good]), posterior(trials[n_good:])
    draws = rng.choices(range(k), weights=l_x, k=n_ei_candidates)
    return cands[max(draws, key=lambda i: l_x[i] / g_x[i])]

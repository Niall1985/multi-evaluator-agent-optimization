import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

warnings.filterwarnings("ignore")


class FunctionalANOVA:
    """Gap 4: Functional ANOVA (fANOVA) for Evaluator Redundancy & Importance Detection.
    
    Decomposes the total variance of the surrogate model into main effect contributions
    attributable to each individual evaluation dimension mu_j:
        V(f) = sum_j V_j + sum_{j < k} V_{jk} + ...
    
    The Sobol first-order sensitivity index (fANOVA importance) is:
        S_j = V_j / V_total in [0, 1]
    
    Evaluators with S_j < threshold contribute negligible variance to the optimization objective
    and are identified as redundant/uninformative candidates for automated pruning.
    """

    def __init__(
        self,
        evaluator_names: List[str],
        redundancy_threshold: float = 0.05,
        n_mc_samples: int = 500,
        random_state: int = 42,
    ):
        self.evaluator_names = evaluator_names
        self.dim = len(evaluator_names)
        self.redundancy_threshold = redundancy_threshold
        self.n_mc_samples = n_mc_samples
        self.random_state = random_state
        self.surrogate: Optional[Any] = None  # anything with predict(X); RandomForestRegressor by default
        self.importance_scores: Dict[str, float] = {name: 1.0 / self.dim for name in evaluator_names}
        self.pairwise_interactions: Dict[str, float] = {}
        self.total_variance: float = 0.0
        self.history_X: List[List[float]] = []
        self.history_y: List[float] = []
        # Spec §3.3: number of consecutive windows each evaluator has spent below tau
        self.below_threshold_windows: Dict[str, int] = {name: 0 for name in evaluator_names}
        self.windows_measured: int = 0
        # Leave-one-out mode (fit_informativeness): Type-2 victims {dropped: kept correlate}, and evaluators
        # that are never reported as prunable (the benchmark's headline metric)
        self.redundant_with: Dict[str, str] = {}
        self.protected: set = set()

    def fit_informativeness(
        self,
        scores: np.ndarray,
        costs: Optional[List[float]] = None,
        protected: Optional[List[str]] = None,
        redundancy_corr: float = 0.95,
        min_samples: int = 4,
        adjust_r2: bool = False,
    ):
        """Non-circular evaluator informativeness for pruning (spec §5.2 evaluator types).

        `scores` is N agents x K evaluators, NaN where an evaluator was not run. Two tests:

        Type 3 (noise), leave-one-out main effect. Each column is z-scored, and for evaluator j the
        consensus of the *other* evaluators c_{-j} = mean_{i != j} z_i is the target. I_j is the share of
        Var(c_{-j}) explained by mu_j, i.e. the fANOVA main-effect fraction V_j / V_total of a one-input
        linear model, which is R^2 = corr(mu_j, c_{-j})^2 (psychometrics' corrected item-total correlation,
        squared). Leaving j out of its own target is the point: regressing an objective that contains
        w_j * mu_j on mu_j credits pure noise with the variance it injects into that objective.

        Type 2 (redundant), pairwise. Main effects cannot see redundancy (a near-copy of an informative
        evaluator has the same main effect), and Sobol indices are not well defined for dependent inputs,
        so near-duplicates are found directly: |corr(mu_i, mu_j)| >= `redundancy_corr` marks the costlier
        one (ties: the later one) as redundant with the one kept. Protected evaluators are always kept.

        An evaluator with fewer than `min_samples` usable rows gets I_j = 1 (unknown is kept, not pruned).
        """
        M = np.asarray(scores, dtype=float)
        k = self.dim
        costs = list(costs) if costs is not None else [0.0] * k
        self.protected = set(protected or [])

        observed = ~np.isnan(M)
        mean = np.nanmean(np.where(observed, M, np.nan), axis=0) if len(M) else np.zeros(k)
        std = np.nanstd(np.where(observed, M, np.nan), axis=0) if len(M) else np.zeros(k)
        with np.errstate(invalid="ignore", divide="ignore"):
            Z = np.where(std > 1e-12, (M - mean) / std, 0.0)
        Z = np.where(observed, Z, np.nan)

        def r2(a: np.ndarray, b: np.ndarray) -> Optional[float]:
            ok = ~np.isnan(a) & ~np.isnan(b)
            if ok.sum() < min_samples or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
                return None
            r = float(np.corrcoef(a[ok], b[ok])[0, 1])
            if not np.isfinite(r):
                return None
            # The raw r^2 of two independent series has mean 1/(n-1), which at 10-30 archived agents sits on
            # tau, so noise often resets its below-tau streak. Adjusted R^2 removes that chance level and
            # prunes noise sooner, but on the Algorithm 8 task it also more than doubles false prunes of weak
            # informative evaluators (100 seeds, 30 gens: recall .72 -> .89, false-prune .04 -> .09), so it
            # is opt-in: a missed prune costs evaluator calls, a false prune costs signal.
            if not adjust_r2:
                return r * r
            n = int(ok.sum())
            return max(0.0, 1.0 - (1.0 - r * r) * (n - 1) / (n - 2))

        def loo_importance(pool: List[int]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for j, name in enumerate(self.evaluator_names):
                others = Z[:, [i for i in pool if i != j]]
                if std[j] <= 1e-12 and observed[:, j].sum() >= min_samples:
                    out[name] = 0.0  # constant evaluator: carries no information about anything
                    continue
                if others.shape[1] == 0:
                    out[name] = 1.0
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows -> NaN consensus
                    consensus = np.nanmean(others, axis=1)
                val = r2(Z[:, j], consensus)
                out[name] = 1.0 if val is None else val
            return out

        # Redundancy: cheaper (then earlier, then protected-first) evaluators are kept preferentially
        order = sorted(range(k), key=lambda i: (self.evaluator_names[i] not in self.protected, costs[i], i))
        kept: List[int] = []
        self.redundant_with = {}
        for j in order:
            twin = next(
                (i for i in kept if (val := r2(Z[:, i], Z[:, j])) is not None and val >= redundancy_corr ** 2),
                None,
            )
            if twin is not None and self.evaluator_names[j] not in self.protected:
                self.redundant_with[self.evaluator_names[j]] = self.evaluator_names[twin]
            else:
                kept.append(j)

        # Scale purification: the first pass's consensus is diluted by noise columns and double-counts a
        # signal that has a near-copy, so the second pass rebuilds each consensus from the evaluators that
        # survived the first (above tau and not redundant). Falls back to the first pass if too few survive.
        importance = loo_importance(list(range(k)))
        core = [
            i for i in kept
            if importance[self.evaluator_names[i]] >= self.redundancy_threshold
            or self.evaluator_names[i] in self.protected
        ]
        if len(core) >= 2:
            importance = loo_importance(core)

        self.importance_scores = importance
        total = sum(importance.values())
        self.normalized_importance = {
            n: (v / total if total > 0 else 1.0 / k) for n, v in importance.items()
        }
        self.total_variance = 1.0  # importances are R^2 shares of a z-scored consensus
        self.history_X = np.where(observed, M, 0.0).tolist()
        self.history_y = []
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, surrogate: Optional[Any] = None):
        """Decomposes a surrogate over configurations X (N x D) with targets y (N).

        The first K columns of X are the per-evaluator variables (named by `evaluator_names`); any further
        columns (e.g. theta_H) are marginalised over and count towards V_total but get no importance.
        `surrogate` is an already-fitted model with `predict` (spec §3.3 decomposes "the fitted predictive
        model", i.e. the GP); without one a Random Forest is fitted here.
        """
        if len(y) < 4:
            # Not enough data for variance decomposition; return uniform importance
            self.importance_scores = {name: 1.0 / self.dim for name in self.evaluator_names}
            return self

        self.history_X = X.tolist()
        self.history_y = y.tolist()

        if surrogate is not None:
            self.surrogate = surrogate
            self._compute_variance_decomposition(X)
            return self

        # Train Random Forest surrogate for robust non-linear variance decomposition
        self.surrogate = RandomForestRegressor(
            n_estimators=100,
            max_depth=6,
            min_samples_leaf=2,
            random_state=self.random_state,
        )
        self.surrogate.fit(X, y)
        self._compute_variance_decomposition(X)
        return self

    def _compute_variance_decomposition(self, X: np.ndarray):
        """Computes Sobol first-order sensitivity indices via Monte Carlo marginal integration."""
        if self.surrogate is None or len(X) < 4:
            return

        # Background distribution: resample the observed configurations, so V_total is the variance of f
        # "over the observed configuration space" (spec §3.3) rather than over an axis-aligned box, which for
        # simplex-constrained weights is mostly infeasible points.
        rng = np.random.RandomState(self.random_state)
        n_samples = min(self.n_mc_samples, max(200, len(X) * 20))
        X_bg = X[rng.randint(0, len(X), size=n_samples)]

        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)

        y_pred = self.surrogate.predict(X_bg)
        total_var = float(np.var(y_pred))
        self.total_variance = max(total_var, 1e-8)

        # 1. Main effects V_j: Var_{x_j}( E_{X_{-j}}[ f(X) | x_j ] )
        # One batched predict per dimension (grid_evals * n_samples rows) instead of grid_evals small ones;
        # forest predict cost is dominated by per-call overhead at these sizes.
        main_effects = {}
        grid_evals = 20

        for j in range(self.dim):
            if maxs[j] <= mins[j]:
                # Constant column: conditioning on it cannot move the expectation
                main_effects[self.evaluator_names[j]] = 0.0
                continue
            # Grid on x_j's empirical quantiles, so Var over the grid is Var over x_j's observed marginal,
            # the same measure V_total uses; an even linspace over [min, max] over-weights the tails and can
            # push V_j above V_total.
            x_j_grid = np.quantile(X[:, j], (np.arange(grid_evals) + 0.5) / grid_evals)
            X_stack = np.tile(X_bg, (grid_evals, 1))
            X_stack[:, j] = np.repeat(x_j_grid, n_samples)
            marginal_means = self.surrogate.predict(X_stack).reshape(grid_evals, n_samples).mean(axis=1)
            main_effects[self.evaluator_names[j]] = max(0.0, float(np.var(marginal_means)))

        # Normalize to Sobol First-Order Index S_j = V_j / V_total
        sum_v = sum(main_effects.values())
        if sum_v > 0 and self.total_variance > 0:
            # Compute fraction of variance explained
            self.importance_scores = {
                name: float(np.clip(v / self.total_variance, 0.0, 1.0))
                for name, v in main_effects.items()
            }
            # Also calculate normalized share
            tot_share = sum(self.importance_scores.values())
            if tot_share > 0:
                self.normalized_importance = {
                    name: float(score / tot_share)
                    for name, score in self.importance_scores.items()
                }
            else:
                self.normalized_importance = {name: 1.0 / self.dim for name in self.evaluator_names}
        else:
            # Fallback to feature importances from random forest
            if hasattr(self.surrogate, "feature_importances_"):
                fi = self.surrogate.feature_importances_
                self.importance_scores = {
                    self.evaluator_names[i]: float(fi[i]) for i in range(self.dim)
                }
                self.normalized_importance = dict(self.importance_scores)
            else:
                self.importance_scores = {name: 1.0 / self.dim for name in self.evaluator_names}
                self.normalized_importance = {name: 1.0 / self.dim for name in self.evaluator_names}

    def get_importance_scores(self) -> Dict[str, float]:
        """Returns the fANOVA importance score (variance contribution) for each evaluator."""
        return dict(self.importance_scores)

    def identify_redundant_evaluators(self, threshold: Optional[float] = None) -> List[str]:
        """Returns the list of evaluators whose variance contribution falls below the threshold."""
        thresh = threshold if threshold is not None else self.redundancy_threshold
        redundant = []
        for name, score in self.importance_scores.items():
            if name in self.protected:
                continue
            if score < thresh or name in self.redundant_with:
                redundant.append(name)
        return redundant

    def close_window(self) -> None:
        """Records one measurement window: evaluators below tau extend their streak, the rest reset to 0."""
        below = set(self.identify_redundant_evaluators())
        for name in self.evaluator_names:
            self.below_threshold_windows[name] = self.below_threshold_windows[name] + 1 if name in below else 0
        self.windows_measured += 1

    def get_pruned_evaluators(self, patience: int) -> List[str]:
        """Evaluators whose importance has been below tau for `patience` consecutive windows (spec §3.3)."""
        return [n for n in self.evaluator_names if self.below_threshold_windows[n] >= patience]

    def get_fanova_summary(self) -> Dict[str, Any]:
        """Returns a comprehensive diagnostic summary of fANOVA analysis for UI cards and charts."""
        redundant = self.identify_redundant_evaluators()
        return {
            "total_variance": float(self.total_variance),
            "evaluator_importances": {
                name: round(self.importance_scores.get(name, 0.0), 4)
                for name in self.evaluator_names
            },
            "normalized_shares": {
                name: round(getattr(self, "normalized_importance", {}).get(name, 1.0 / self.dim), 4)
                for name in self.evaluator_names
            },
            "redundant_evaluators": redundant,
            "active_informative_count": len(self.evaluator_names) - len(redundant),
            "threshold_used": self.redundancy_threshold,
            "sample_count": len(self.history_y) or len(self.history_X),
            "redundant_with": dict(self.redundant_with),
            "protected": sorted(self.protected),
            "below_threshold_windows": dict(self.below_threshold_windows),
            "windows_measured": self.windows_measured,
        }

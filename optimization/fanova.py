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
        self.surrogate: Optional[RandomForestRegressor] = None
        self.importance_scores: Dict[str, float] = {name: 1.0 / self.dim for name in evaluator_names}
        self.pairwise_interactions: Dict[str, float] = {}
        self.total_variance: float = 0.0
        self.history_X: List[List[float]] = []
        self.history_y: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fits the surrogate model on evaluator observations X (N x K) and scalar fitness/delta y (N)."""
        if len(y) < 4:
            # Not enough data for variance decomposition; return uniform importance
            self.importance_scores = {name: 1.0 / self.dim for name in self.evaluator_names}
            return self

        self.history_X = X.tolist()
        self.history_y = y.tolist()

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

        # Generate background sample distribution from empirical marginals
        rng = np.random.RandomState(self.random_state)
        n_samples = min(self.n_mc_samples, max(200, len(X) * 20))
        
        # Sample background points within empirical bounds [min, max] per feature
        mins = np.min(X, axis=0)
        maxs = np.max(X, axis=0)
        # Avoid zero range
        maxs = np.where(maxs > mins, maxs, mins + 1.0)
        
        X_bg = rng.uniform(mins, maxs, size=(n_samples, self.dim))
        y_pred = self.surrogate.predict(X_bg)
        total_var = float(np.var(y_pred))
        self.total_variance = max(total_var, 1e-8)

        # 1. Main effects V_j: Var_{x_j}( E_{X_{-j}}[ f(X) | x_j ] )
        main_effects = {}
        grid_evals = 20

        for j in range(self.dim):
            x_j_grid = np.linspace(mins[j], maxs[j], grid_evals)
            marginal_means = []

            for val in x_j_grid:
                # Replace column j with val across all background samples
                X_temp = X_bg.copy()
                X_temp[:, j] = val
                preds = self.surrogate.predict(X_temp)
                marginal_means.append(np.mean(preds))

            v_j = float(np.var(marginal_means))
            main_effects[self.evaluator_names[j]] = max(0.0, v_j)

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
            if score < thresh:
                redundant.append(name)
        return redundant

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
            "sample_count": len(self.history_y),
        }

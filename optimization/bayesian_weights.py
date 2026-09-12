import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BayesianWeightOptimizer:
    """Gap 2: Bayesian Weight Optimization over metric weights w in R^k via Gaussian Process + Expected Improvement (EI)."""

    def __init__(self, metric_names: List[str], xi: float = 0.01):
        self.metric_names = metric_names
        self.dim = len(metric_names)
        self.xi = xi
        self.X_history: List[List[float]] = []  # Weight vectors w (sum=1)
        self.y_history: List[float] = []        # Delta_F (relative improvements)
        self.weights_log: List[Dict[str, float]] = []

        # Kernel: Matern 5/2 with constant scaling
        self.kernel = C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(self.dim),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-3,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=42,
        )

    def add_observation(self, weights: Dict[str, float], delta_f: float):
        """Records an observed weight vector and resulting relative fitness improvement Delta_F."""
        w_vec = [float(weights.get(m, 0.0)) for m in self.metric_names]
        # Normalize just in case
        total = sum(w_vec)
        if total > 0:
            w_vec = [x / total for x in w_vec]
        else:
            w_vec = [1.0 / self.dim] * self.dim

        self.X_history.append(w_vec)
        self.y_history.append(float(delta_f))
        self.weights_log.append({name: w_vec[i] for i, name in enumerate(self.metric_names)})

        if len(self.X_history) >= 3:
            X = np.array(self.X_history)
            y = np.array(self.y_history)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(X, y)

    def propose_next_weights(self) -> Dict[str, float]:
        """Proposes the next continuous weight vector w maximizing Expected Improvement."""
        if len(self.X_history) < 3:
            # Initial exploration via Dirichlet sampling
            raw_w = np.random.dirichlet(np.ones(self.dim))
        else:
            # Generate Dirichlet candidate pool on the simplex
            candidates = np.random.dirichlet(np.ones(self.dim), size=300)
            mu, sigma = self.gp.predict(candidates, return_std=True)
            f_best = np.max(self.y_history)

            # Expected Improvement (EI)
            improvement = mu - f_best - self.xi
            sigma_safe = np.maximum(sigma, 1e-9)
            Z = improvement / sigma_safe
            ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)

            best_idx = int(np.argmax(ei))
            raw_w = candidates[best_idx]

        # Ensure simplex constraint sum(w_j) = 1
        w_norm = raw_w / np.sum(raw_w)
        return {name: float(w_norm[i]) for i, name in enumerate(self.metric_names)}

    def get_1d_gp_slice(
        self, target_idx: int = 0, resolution: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Returns 1D slice of GP predictions, uncertainty bounds, and EI curve for live Streamlit plotting.
        
        Returns:
            w_grid: 1D grid values for target metric weight (0 to 1)
            mu: Mean predicted Delta_F
            sigma: Standard deviation
            ei: Expected Improvement acquisition values
            sampled_x: Sampled trial points for this target metric
            sampled_y: Observed Delta_F values
        """
        w_grid = np.linspace(0.01, 0.99, resolution)
        grid_points = np.zeros((resolution, self.dim))
        grid_points[:, target_idx] = w_grid

        # Distribute remaining weight across other metrics
        remainder = (1.0 - w_grid) / max(1, (self.dim - 1))
        for j in range(self.dim):
            if j != target_idx:
                grid_points[:, j] = remainder

        if len(self.X_history) >= 3:
            mu, sigma = self.gp.predict(grid_points, return_std=True)
            f_best = np.max(self.y_history)
            improvement = mu - f_best - self.xi
            sigma_safe = np.maximum(sigma, 1e-9)
            Z = improvement / sigma_safe
            ei = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
        else:
            mu = np.zeros(resolution)
            sigma = np.ones(resolution) * 0.2
            ei = np.zeros(resolution)

        sampled_x = np.array([x[target_idx] for x in self.X_history]) if self.X_history else np.array([])
        sampled_y = np.array(self.y_history) if self.y_history else np.array([])

        return w_grid, mu, sigma, ei, sampled_x, sampled_y

    def get_weights_history_df(self) -> pd.DataFrame:
        """Returns a DataFrame of weight evolution over generations for stacked area charts."""
        if not self.weights_log:
            return pd.DataFrame()
        df = pd.DataFrame(self.weights_log)
        df["Generation"] = range(1, len(df) + 1)
        return df

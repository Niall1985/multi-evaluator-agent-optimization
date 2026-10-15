import warnings
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C
from sklearn.exceptions import ConvergenceWarning
from scipy.stats import norm

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BayesianWeightOptimizer:
    """Gap 2 & Gap 6: Bayesian Weight Optimization over metric weights w on the simplex Delta^(K-1)
    using Gaussian Process with Expected Improvement (EI) or GP-UCB with formal sublinear regret bound.
    """

    # Spec §3.2: theta_H shares the continuous GP with w. Bounds match the harness's sampling range.
    DEFAULT_KNOB_BOUNDS: Dict[str, Tuple[float, float]] = {
        "temperature": (0.1, 1.0),
        "top_p": (0.6, 1.0),
    }

    def __init__(
        self,
        metric_names: List[str],
        xi: float = 0.01,
        acquisition_type: str = "gp_ucb",
        beta_scale: float = 1.0,
        knob_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        n_candidates: int = 1000,
    ):
        self.metric_names = metric_names
        self.dim = len(metric_names)
        self.xi = xi
        self.acquisition_type = acquisition_type  # 'gp_ucb' or 'expected_improvement'
        self.beta_scale = beta_scale
        self.knob_bounds = dict(self.DEFAULT_KNOB_BOUNDS if knob_bounds is None else knob_bounds)
        self.knob_names = list(self.knob_bounds)
        self.n_candidates = n_candidates
        self.X_history: List[List[float]] = []  # Weight vectors w (sum=1)
        self.knob_history: List[List[float]] = []  # theta_H, scaled to [0, 1] per knob
        self.y_history: List[float] = []        # Delta_F (relative improvements)
        self.weights_log: List[Dict[str, float]] = []

        # Kernel: Matern 5/2 with constant scaling, one length scale per weight and per knob
        self.kernel = C(1.0, (1e-2, 1e2)) * Matern(
            length_scale=np.ones(self.dim + len(self.knob_names)),
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

    def _scale_knobs(self, knobs: Optional[Dict[str, float]]) -> List[float]:
        """Maps theta_H into [0, 1] per knob; a missing knob sits at the middle of its range."""
        out = []
        for name in self.knob_names:
            lo, hi = self.knob_bounds[name]
            val = (knobs or {}).get(name)
            out.append(0.5 if val is None else float(np.clip((float(val) - lo) / (hi - lo), 0.0, 1.0)))
        return out

    def _unscale_knobs(self, scaled: np.ndarray) -> Dict[str, float]:
        return {
            name: float(lo + scaled[i] * (hi - lo))
            for i, (name, (lo, hi)) in enumerate(self.knob_bounds.items())
        }

    def training_matrix(self) -> np.ndarray:
        return np.hstack([np.array(self.X_history), np.array(self.knob_history).reshape(len(self.X_history), len(self.knob_names))])

    def add_observation(self, weights: Dict[str, float], delta_f: float, knobs: Optional[Dict[str, float]] = None):
        """Records an observed (w, theta_H) configuration and resulting relative fitness improvement Delta_F."""
        w_vec = [float(weights.get(m, 0.0)) for m in self.metric_names]
        # Normalize just in case
        total = sum(w_vec)
        if total > 0:
            w_vec = [x / total for x in w_vec]
        else:
            w_vec = [1.0 / self.dim] * self.dim

        self.X_history.append(w_vec)
        self.knob_history.append(self._scale_knobs(knobs))
        self.y_history.append(float(delta_f))
        self.weights_log.append({name: w_vec[i] for i, name in enumerate(self.metric_names)})

        if len(self.X_history) >= 3:
            y = np.array(self.y_history)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.gp.fit(self.training_matrix(), y)

    def compute_beta_t(self, t: int, delta: float = 0.1) -> float:
        """Gap 6: Computes Srinivas et al. theoretical beta_t for GP-UCB sublinear regret bound O(sqrt(T * gamma_T))."""
        # beta_t = sqrt(2 * ln(t^2 * pi^2 / (6 * delta)))
        safe_t = max(1, t)
        val = 2.0 * np.log((safe_t ** 2) * (np.pi ** 2) / (6.0 * delta))
        return self.beta_scale * float(np.sqrt(max(0.1, val)))

    def propose_next_weights(self, acquisition: Optional[str] = None) -> Dict[str, float]:
        """Proposes the next continuous weight vector w on simplex Delta^(K-1) via GP-UCB or EI."""
        return self.propose_next_config(acquisition=acquisition)[0]

    def propose_next_config(self, acquisition: Optional[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Proposes (w, theta_H) jointly by maximising GP-UCB or EI over Dirichlet x uniform-knob candidates."""
        acq = acquisition or self.acquisition_type
        t = len(self.X_history) + 1
        n_knobs = len(self.knob_names)

        if len(self.X_history) < 3:
            # Initial exploration via Dirichlet sampling
            raw_w = np.random.dirichlet(np.ones(self.dim))
            raw_k = np.random.uniform(0.0, 1.0, size=n_knobs)
        else:
            # Candidate pool: weights on the simplex, knobs uniform on their (scaled) box
            cand_w = np.random.dirichlet(np.ones(self.dim), size=self.n_candidates)
            cand_k = np.random.uniform(0.0, 1.0, size=(self.n_candidates, n_knobs))
            candidates = np.hstack([cand_w, cand_k])
            mu, sigma = self.gp.predict(candidates, return_std=True)
            f_best = np.max(self.y_history)

            if acq == "gp_ucb":
                # Gap 6: GP-UCB acquisition with theoretical sublinear regret parameter beta_t
                beta_t = self.compute_beta_t(t=t)
                acq_values = mu + beta_t * sigma
            else:
                # Expected Improvement (EI)
                improvement = mu - f_best - self.xi
                sigma_safe = np.maximum(sigma, 1e-9)
                Z = improvement / sigma_safe
                acq_values = improvement * norm.cdf(Z) + sigma_safe * norm.pdf(Z)

            best_idx = int(np.argmax(acq_values))
            raw_w = cand_w[best_idx]
            raw_k = cand_k[best_idx]

        # Ensure simplex constraint sum(w_j) = 1
        w_norm = raw_w / np.sum(raw_w)
        weights = {name: float(w_norm[i]) for i, name in enumerate(self.metric_names)}
        return weights, self._unscale_knobs(raw_k)

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
            # Slice through the knob dimensions at their observed mean
            knob_mean = np.array(self.knob_history).reshape(len(self.X_history), len(self.knob_names)).mean(axis=0)
            knob_fill = np.tile(knob_mean, (resolution, 1))
            mu, sigma = self.gp.predict(np.hstack([grid_points, knob_fill]), return_std=True)
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

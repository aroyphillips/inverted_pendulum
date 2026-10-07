from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from models.dataset_views import stratified_transition_sample
from models.lqr_utils import finite_difference_linearization, lqr_diagnostics


@dataclass
class GPDynamicsConfig:
    max_train_points: int = 5000
    seed: int = 0
    alpha: float = 1e-8
    n_restarts_optimizer: int = 0


class GPDynamicsModel:
    def __init__(self, config: GPDynamicsConfig | None = None) -> None:
        self.config = config or GPDynamicsConfig()
        self.models = []
        self.input_scaler = None
        self.target_scaler = None
        self.metadata: dict = {}

    def fit(self, x: np.ndarray, delta_x: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "GPDynamicsModel":
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ImportError(
                "GP dynamics require scikit-learn. Install it with `pip install scikit-learn`."
            ) from exc

        x_train, y_train = stratified_transition_sample(
            np.asarray(x, dtype=float),
            np.asarray(delta_x, dtype=float),
            max_points=self.config.max_train_points,
            seed=self.config.seed,
        )
        self.input_scaler = StandardScaler().fit(x_train)
        self.target_scaler = StandardScaler().fit(y_train)
        xs = self.input_scaler.transform(x_train)
        ys = self.target_scaler.transform(y_train)

        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=np.ones(xs.shape[1])) + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-8, 1e0),
        )
        self.models = []
        for j in range(ys.shape[1]):
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.config.alpha,
                normalize_y=False,
                n_restarts_optimizer=self.config.n_restarts_optimizer,
                random_state=self.config.seed + j,
            )
            gp.fit(xs, ys[:, j])
            self.models.append(gp)

        self.metadata = {
            "config": self.config.__dict__,
            "train_points": int(x_train.shape[0]),
            "input_dim": int(x_train.shape[1]),
            "state_dim": int(y_train.shape[1]),
            "kernels": [str(model.kernel_) for model in self.models],
        }
        if x_val is not None and y_val is not None:
            pred, std = self.predict_delta(x_val, return_std=True)
            err = pred - y_val
            self.metadata["validation_rmse_by_state"] = np.sqrt(np.mean(err**2, axis=0)).tolist()
            self.metadata["validation_rmse_mean"] = float(np.sqrt(np.mean(err**2)))
            self.metadata["predictive_std_mean_by_state"] = np.mean(std, axis=0).tolist()
        return self

    def predict_delta(self, x_u: np.ndarray, return_std: bool = False):
        if self.input_scaler is None or self.target_scaler is None or not self.models:
            raise RuntimeError("GPDynamicsModel has not been fit.")
        x_u = np.asarray(x_u, dtype=float)
        one_dim = x_u.ndim == 1
        if one_dim:
            x_u = x_u[None, :]
        xs = self.input_scaler.transform(x_u)
        means = []
        stds = []
        for model in self.models:
            if return_std:
                mean_j, std_j = model.predict(xs, return_std=True)
                stds.append(std_j)
            else:
                mean_j = model.predict(xs)
            means.append(mean_j)
        mean_scaled = np.stack(means, axis=1)
        mean = self.target_scaler.inverse_transform(mean_scaled)
        if one_dim:
            mean = mean[0]
        if not return_std:
            return mean
        std_scaled = np.stack(stds, axis=1)
        std = std_scaled * self.target_scaler.scale_[None, :]
        if one_dim:
            std = std[0]
        return mean, std

    def predict_next(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float).reshape(-1)
        delta = self.predict_delta(np.concatenate([x, u]))
        return x + delta

    def linearize(self, x_eq: np.ndarray | None = None, u_eq: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        a, b, _ = self.linearize_with_diagnostics(x_eq=x_eq, u_eq=u_eq)
        return a, b

    def linearize_with_diagnostics(
        self,
        x_eq: np.ndarray | None = None,
        u_eq: np.ndarray | None = None,
        eps_x: float = 1e-4,
        eps_u: float = 1e-4,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Linearize the GP mean at equilibrium and return (A, B, diagnostics).

        Diagnostics include controllability rank, A eigenvalues, norm(B), GP predictive
        std at the equilibrium point, and whether the linearization looks reliable.
        """
        x_eq = np.zeros(4, dtype=float) if x_eq is None else np.asarray(x_eq, dtype=float)
        u_eq = np.zeros(1, dtype=float) if u_eq is None else np.asarray(u_eq, dtype=float)
        a, b = finite_difference_linearization(self.predict_next, x_eq, u_eq, eps_x=eps_x, eps_u=eps_u)

        ctrl_diag = lqr_diagnostics(a, b)

        xu_eq = np.concatenate([x_eq, u_eq])
        _, std_eq = self.predict_delta(xu_eq, return_std=True)

        diag: Dict = {
            **ctrl_diag,
            "gp_std_at_eq": std_eq.tolist(),
            "eps_x": eps_x,
            "eps_u": eps_u,
        }
        return a, b, diag

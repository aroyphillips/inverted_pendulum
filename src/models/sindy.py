from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from models.dataset_views import stratified_transition_sample
from models.lqr_utils import finite_difference_linearization


@dataclass
class SINDyConfig:
    threshold: float = 1e-3
    ridge: float = 1e-6
    max_iter: int = 12
    max_train_points: int = 100000
    seed: int = 0


def sindy_library(x_u: np.ndarray) -> Tuple[np.ndarray, list[str]]:
    arr = np.asarray(x_u, dtype=float)
    one_dim = arr.ndim == 1
    if one_dim:
        arr = arr[None, :]
    x = arr[:, 0]
    x_dot = arr[:, 1]
    theta = arr[:, 2]
    theta_dot = arr[:, 3]
    u = arr[:, 4]
    cols = [
        np.ones_like(x),
        x,
        x_dot,
        theta,
        theta_dot,
        u,
        x**2,
        x_dot**2,
        theta**2,
        theta_dot**2,
        u**2,
        x * theta,
        x_dot * theta_dot,
        theta * theta_dot,
        x * u,
        theta * u,
        np.sin(theta),
        np.cos(theta) - 1.0,
        u * np.cos(theta),
        theta_dot**2 * np.sin(theta),
    ]
    names = [
        "1",
        "x",
        "x_dot",
        "theta",
        "theta_dot",
        "u",
        "x^2",
        "x_dot^2",
        "theta^2",
        "theta_dot^2",
        "u^2",
        "x*theta",
        "x_dot*theta_dot",
        "theta*theta_dot",
        "x*u",
        "theta*u",
        "sin(theta)",
        "cos(theta)-1",
        "u*cos(theta)",
        "theta_dot^2*sin(theta)",
    ]
    theta_mat = np.stack(cols, axis=1)
    if one_dim:
        theta_mat = theta_mat[0]
    return theta_mat, names


def stlsq(theta: np.ndarray, y: np.ndarray, config: SINDyConfig) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    y = np.asarray(y, dtype=float)
    eye = np.eye(theta.shape[1])
    xi = np.linalg.solve(theta.T @ theta + config.ridge * eye, theta.T @ y)
    for _ in range(config.max_iter):
        small = np.abs(xi) < config.threshold
        xi[small] = 0.0
        for j in range(y.shape[1]):
            active = ~small[:, j]
            if not np.any(active):
                continue
            lhs = theta[:, active].T @ theta[:, active] + config.ridge * np.eye(np.sum(active))
            xi[active, j] = np.linalg.solve(lhs, theta[:, active].T @ y[:, j])
    return xi


class SINDyDynamicsModel:
    def __init__(self, config: SINDyConfig | None = None) -> None:
        self.config = config or SINDyConfig()
        self.coef_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.metadata: dict = {}

    def fit(self, x_u: np.ndarray, x_next: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "SINDyDynamicsModel":
        x_train, y_train = stratified_transition_sample(
            np.asarray(x_u, dtype=float),
            np.asarray(x_next, dtype=float),
            max_points=self.config.max_train_points,
            seed=self.config.seed,
        )
        theta, names = sindy_library(x_train)
        self.coef_ = stlsq(theta, y_train, self.config)
        self.feature_names_ = names
        pred = theta @ self.coef_
        train_err = pred - y_train
        self.metadata = {
            "config": self.config.__dict__,
            "train_points": int(x_train.shape[0]),
            "num_features": int(theta.shape[1]),
            "active_terms_per_state": np.sum(np.abs(self.coef_) > 0.0, axis=0).astype(int).tolist(),
            "train_rmse_by_state": np.sqrt(np.mean(train_err**2, axis=0)).tolist(),
            "equations": self.equations(),
        }
        if x_val is not None and y_val is not None:
            pred_val = self.predict_next_from_xu(x_val)
            err = pred_val - y_val
            self.metadata["validation_rmse_by_state"] = np.sqrt(np.mean(err**2, axis=0)).tolist()
            self.metadata["validation_rmse_mean"] = float(np.sqrt(np.mean(err**2)))
        return self

    def predict_next_from_xu(self, x_u: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("SINDyDynamicsModel has not been fit.")
        theta, _ = sindy_library(x_u)
        return theta @ self.coef_

    def predict_next(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        x_u = np.concatenate([np.asarray(x, dtype=float), np.asarray(u, dtype=float).reshape(-1)])
        return np.asarray(self.predict_next_from_xu(x_u), dtype=float)

    def linearize(self, x_eq: np.ndarray | None = None, u_eq: np.ndarray | None = None) -> Tuple[np.ndarray, np.ndarray]:
        x_eq = np.zeros(4, dtype=float) if x_eq is None else np.asarray(x_eq, dtype=float)
        u_eq = np.zeros(1, dtype=float) if u_eq is None else np.asarray(u_eq, dtype=float)
        return finite_difference_linearization(self.predict_next, x_eq, u_eq)

    def equations(self, precision: int = 4) -> list[str]:
        if self.coef_ is None:
            return []
        state_names = ["x_next", "x_dot_next", "theta_next", "theta_dot_next"]
        rows = []
        for j, state_name in enumerate(state_names):
            terms = []
            for coef, fname in zip(self.coef_[:, j], self.feature_names_):
                if abs(coef) > 0.0:
                    terms.append(f"{coef:.{precision}g}*{fname}")
            rows.append(f"{state_name} = " + (" + ".join(terms) if terms else "0"))
        return rows


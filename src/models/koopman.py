from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.scaling import NumpyStandardScaler


@dataclass
class KoopmanLift:
    standardize: bool = True
    feature_set: str = "standard"   # "standard" (13) | "rich" (17)
    scaler: NumpyStandardScaler | None = None

    @property
    def feature_names(self) -> list[str]:
        base = [
            "x", "x_dot", "theta", "theta_dot",
            "sin(theta)", "cos(theta)-1",
            "x^2", "x_dot^2", "theta^2", "theta_dot^2",
            "x*theta", "x_dot*theta_dot", "theta*theta_dot",
        ]
        if self.feature_set == "rich":
            base += ["sin(2*theta)", "cos(2*theta)", "x*sin(theta)", "x_dot*cos(theta)"]
        return base

    def raw_features(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        x_pos = arr[..., 0]
        x_dot = arr[..., 1]
        theta = arr[..., 2]
        theta_dot = arr[..., 3]
        cols = [
            x_pos, x_dot, theta, theta_dot,
            np.sin(theta), np.cos(theta) - 1.0,
            x_pos**2, x_dot**2, theta**2, theta_dot**2,
            x_pos * theta, x_dot * theta_dot, theta * theta_dot,
        ]
        if self.feature_set == "rich":
            cols += [
                np.sin(2.0 * theta),
                np.cos(2.0 * theta),
                x_pos * np.sin(theta),
                x_dot * np.cos(theta),
            ]
        return np.stack(cols, axis=-1)

    def fit(self, x: np.ndarray) -> "KoopmanLift":
        z = self.raw_features(x)
        if self.standardize:
            self.scaler = NumpyStandardScaler().fit(z.reshape((-1, z.shape[-1])))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = self.raw_features(x)
        if self.standardize:
            if self.scaler is None:
                raise RuntimeError("KoopmanLift has not been fit.")
            flat = z.reshape((-1, z.shape[-1]))
            return self.scaler.transform(flat).reshape(z.shape)
        return z

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)


def edmdc_diagnostics(z: np.ndarray, u: np.ndarray, ridge: float = 1e-8) -> dict:
    zk = z[:, :-1, :].reshape((-1, z.shape[-1]))
    zkp1 = z[:, 1:, :].reshape((-1, z.shape[-1]))
    uk = u[:, :-1, :].reshape((-1, u.shape[-1]))
    xu = np.concatenate([zk, uk], axis=1)
    lhs = xu.T @ xu + ridge * np.eye(xu.shape[1])
    coef = np.linalg.solve(lhs, xu.T @ zkp1).T
    pred = xu @ coef.T
    rmse = np.sqrt(np.mean((pred - zkp1) ** 2, axis=0))
    return {
        "lift_dim": int(z.shape[-1]),
        "rmse_mean": float(np.mean(rmse)),
        "rmse_max": float(np.max(rmse)),
        "coef_shape": list(coef.shape),
    }


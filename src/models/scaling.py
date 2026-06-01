from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NumpyStandardScaler:
    mean_: np.ndarray | None = None
    scale_: np.ndarray | None = None
    eps: float = 1e-12

    def fit(self, x: np.ndarray) -> "NumpyStandardScaler":
        x = np.asarray(x, dtype=float)
        self.mean_ = np.mean(x, axis=0)
        scale = np.std(x, axis=0)
        self.scale_ = np.where(scale < self.eps, 1.0, scale)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit.")
        return (np.asarray(x, dtype=float) - self.mean_) / self.scale_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler has not been fit.")
        return np.asarray(x, dtype=float) * self.scale_ + self.mean_

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


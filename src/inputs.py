from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class ZeroInput:
    def __call__(self, t: float, y: np.ndarray) -> float:
        return 0.0


@dataclass
class ConstantInput:
    force: float

    def __call__(self, t: float, y: np.ndarray) -> float:
        return float(self.force)


@dataclass
class ImpulseThenZeroInput:
    impulse_force: float = 0.01
    impulse_duration: float = 1e-9

    def __call__(self, t: float, y: np.ndarray) -> float:
        if t <= self.impulse_duration:
            return float(self.impulse_force)
        return 0.0


@dataclass
class PiecewiseLinearInput:
    t_grid: np.ndarray
    u_grid: np.ndarray

    def __post_init__(self) -> None:
        self.t_grid = np.asarray(self.t_grid, dtype=float)
        self.u_grid = np.asarray(self.u_grid, dtype=float)
        if self.t_grid.ndim != 1 or self.u_grid.ndim != 1:
            raise ValueError("t_grid and u_grid must be 1D arrays")
        if self.t_grid.size != self.u_grid.size:
            raise ValueError("t_grid and u_grid must have the same length")
        if self.t_grid.size < 2:
            raise ValueError("t_grid must have at least 2 elements")

    def __call__(self, t: float, y: np.ndarray) -> float:
        return float(np.interp(t, self.t_grid, self.u_grid))


@dataclass
class GaussianRandomWalkInput:
    t_end: float
    dt_force: float = 0.1
    sigma_force: float = 0.08
    force_max: float = 1.0
    seed: Optional[int] = None
    enforce_sign_balance: bool = True

    t_grid: np.ndarray = None
    u_grid: np.ndarray = None

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        t_grid = np.arange(0.0, self.t_end + self.dt_force, self.dt_force, dtype=float)

        while True:
            steps = rng.normal(loc=0.0, scale=self.sigma_force, size=t_grid.size)
            u = np.cumsum(steps)
            u = np.clip(u, -self.force_max, self.force_max)
            if not self.enforce_sign_balance:
                break
            if np.any(u > 0.0) and np.any(u < 0.0):
                break

        self.t_grid = t_grid
        self.u_grid = u

    def __call__(self, t: float, y: np.ndarray) -> float:
        return float(np.interp(t, self.t_grid, self.u_grid))


def to_force_fn(force_like: Callable[[float, np.ndarray], float]) -> Callable[[float, np.ndarray], float]:
    return force_like

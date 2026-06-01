from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.linalg import solve_discrete_are


def discrete_lqr_gain(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    p = solve_discrete_are(a, b, q, r)
    k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    return k, p


def finite_difference_linearization(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_eq: np.ndarray,
    u_eq: np.ndarray,
    eps_x: float = 1e-5,
    eps_u: float = 1e-5,
) -> Tuple[np.ndarray, np.ndarray]:
    x_eq = np.asarray(x_eq, dtype=float)
    u_eq = np.asarray(u_eq, dtype=float)
    base = np.asarray(f(x_eq, u_eq), dtype=float)
    a = np.empty((base.size, x_eq.size), dtype=float)
    b = np.empty((base.size, u_eq.size), dtype=float)

    for i in range(x_eq.size):
        dx = np.zeros_like(x_eq)
        dx[i] = eps_x
        a[:, i] = (f(x_eq + dx, u_eq) - f(x_eq - dx, u_eq)) / (2.0 * eps_x)

    for i in range(u_eq.size):
        du = np.zeros_like(u_eq)
        du[i] = eps_u
        b[:, i] = (f(x_eq, u_eq + du) - f(x_eq, u_eq - du)) / (2.0 * eps_u)

    return a, b


from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.linalg import LinAlgError, solve_discrete_are


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


def lqr_diagnostics(a: np.ndarray, b: np.ndarray) -> Dict:
    """Controllability and spectral diagnostics for an (A, B) pair."""
    n = a.shape[0]
    a_finite = bool(np.all(np.isfinite(a)))
    b_finite = bool(np.all(np.isfinite(b)))

    ctrl_rank = 0
    if a_finite and b_finite:
        ab = b.copy()
        c_cols = [b]
        for _ in range(n - 1):
            ab = a @ ab
            c_cols.append(ab)
        ctrl_rank = int(np.linalg.matrix_rank(np.hstack(c_cols)))

    return {
        "controllability_rank": ctrl_rank,
        "state_dim": n,
        "controllable": ctrl_rank == n,
        "eigenvalues_A": np.linalg.eigvals(a).tolist() if a_finite else None,
        "norm_B": float(np.linalg.norm(b)) if b_finite else None,
        "A_finite": a_finite,
        "B_finite": b_finite,
    }


def safe_discrete_lqr_gain(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """Attempt DARE solve with diagnostics.

    Returns (K, P, diagnostics). K and P are None when the solve fails.
    A damped fallback (A *= 0.99) is tried if the first solve fails, to handle
    near-unit-circle eigenvalues that cause ill-conditioning.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    q = np.asarray(q, dtype=float)
    r = np.asarray(r, dtype=float)
    diag = lqr_diagnostics(a, b)

    if not diag["A_finite"] or not diag["B_finite"]:
        diag["failure_reason"] = "non_finite_AB"
        return None, None, diag

    if not diag["controllable"]:
        diag["failure_reason"] = (
            f"uncontrollable_rank_{diag['controllability_rank']}_of_{diag['state_dim']}"
        )
        return None, None, diag

    for attempt, a_try in enumerate([a, a * 0.99]):
        try:
            p = solve_discrete_are(a_try, b, q, r)
            k = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a_try)
            diag["solver_status"] = "success"
            if attempt > 0:
                diag["solver_note"] = "damped_A_fallback_used"
            return k, p, diag
        except (LinAlgError, ValueError):
            pass

    diag["failure_reason"] = "DARE_solver_failed"
    diag["solver_status"] = "failed"
    return None, None, diag


def finite_difference_linearization(
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x_eq: np.ndarray,
    u_eq: np.ndarray,
    eps_x: float = 1e-4,
    eps_u: float = 1e-4,
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

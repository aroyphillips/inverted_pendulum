from __future__ import annotations

from typing import Dict

import numpy as np


DEFAULT_Q = np.diag([1.0, 0.2, 20.0, 1.0])
DEFAULT_R = np.array([[0.05]])

THETA_TOL = np.deg2rad(2.0)   # 2 degrees
THETADOT_TOL = 0.1             # rad/s
X_TOL = 0.25                   # m
XDOT_TOL = 0.25                # m/s


def settling_time(t: np.ndarray, values: np.ndarray, threshold: float) -> float | None:
    """First time all subsequent values stay within threshold (single dimension)."""
    mask = np.abs(values) <= threshold
    for i in range(mask.size):
        if bool(np.all(mask[i:])):
            return float(t[i])
    return None


def settle_and_remain_time(
    t: np.ndarray,
    y: np.ndarray,
    theta_tol: float = THETA_TOL,
    thetadot_tol: float = THETADOT_TOL,
    x_tol: float = X_TOL,
    xdot_tol: float = XDOT_TOL,
) -> float | None:
    """Return the first time all 4 state tolerances hold for the remainder of the trajectory.

    All tolerances must be simultaneously satisfied from the returned time onward.
    Returns None if the system never settles and remains settled.
    """
    mask = (
        (np.abs(y[0, :]) <= x_tol)
        & (np.abs(y[1, :]) <= xdot_tol)
        & (np.abs(y[2, :]) <= theta_tol)
        & (np.abs(y[3, :]) <= thetadot_tol)
    )
    for i in range(mask.size):
        if bool(np.all(mask[i:])):
            return float(t[i])
    return None


def rollout_metrics(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    control_times_sec: np.ndarray | None = None,
    q: np.ndarray = DEFAULT_Q,
    r: np.ndarray = DEFAULT_R,
) -> Dict[str, float | bool | None]:
    dt = float(np.mean(np.diff(t))) if t.size > 1 else 0.0
    finite = bool(np.all(np.isfinite(y)) and np.all(np.isfinite(u)))
    if not finite:
        return {"success": False, "finite": False}

    state_cost = np.einsum("it,ij,jt->t", y, q, y)
    control_cost = float(r[0, 0]) * u**2
    cost = float(np.sum(state_cost + control_cost) * dt)

    t_end = float(t[-1])
    t_settle = settle_and_remain_time(t, y)

    # Fraction of time steps where all 4 tolerances are simultaneously satisfied.
    all_in_tol = (
        (np.abs(y[0, :]) <= X_TOL)
        & (np.abs(y[1, :]) <= XDOT_TOL)
        & (np.abs(y[2, :]) <= THETA_TOL)
        & (np.abs(y[3, :]) <= THETADOT_TOL)
    )
    settled_fraction = float(np.mean(all_in_tol))

    # max |theta| after settling time (or overall max if never settled).
    if t_settle is not None:
        settle_idx = int(np.searchsorted(t, t_settle))
        post_settle_max_theta = float(np.max(np.abs(y[2, settle_idx:])))
    else:
        post_settle_max_theta = float(np.max(np.abs(y[2, :])))

    metrics: Dict[str, float | bool | None] = {
        "finite": True,
        "final_abs_theta": float(abs(y[2, -1])),
        "final_abs_theta_dot": float(abs(y[3, -1])),
        "final_abs_x": float(abs(y[0, -1])),
        "final_abs_x_dot": float(abs(y[1, -1])),
        "max_abs_theta": float(np.max(np.abs(y[2, :]))),
        "max_abs_x": float(np.max(np.abs(y[0, :]))),
        "control_energy": float(np.sum(u**2) * dt),
        "mean_abs_force": float(np.mean(np.abs(u))),
        "cost": cost,
        # Legacy single-dim settling time (kept for backward compat)
        "settling_time_theta_2deg": settling_time(t, y[2, :], THETA_TOL),
        # Settle-and-remain metrics
        "settle_and_remain_time": t_settle,
        "post_settle_duration": float(t_end - t_settle) if t_settle is not None else None,
        "settled_fraction": settled_fraction,
        "post_settle_max_theta": post_settle_max_theta,
    }
    metrics["no_fall"] = bool(metrics["max_abs_theta"] < np.pi / 2.0)

    # Success: all 4 dims settle and remain, and settled for at least as long as it took to settle.
    # Requires t_settle <= t_end / 2 so "remain" time >= settle time.
    metrics["success"] = bool(
        t_settle is not None
        and t_settle <= t_end / 2.0
        and metrics["no_fall"]
    )

    if control_times_sec is not None and control_times_sec.size:
        metrics["runtime_control_ms_per_step"] = float(np.mean(control_times_sec) * 1000.0)
    return metrics


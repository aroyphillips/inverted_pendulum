from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import time

import numpy as np
from scipy.integrate import solve_ivp

from config import PendulumParams
from controllers.base import Controller, clip_force
from dynamics import inverted_pendulum_dynamics


@dataclass
class RolloutResult:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    success: bool
    message: str
    control_times_sec: np.ndarray


def rollout_controller(
    controller: Controller | Callable[[float, np.ndarray], float],
    x0: np.ndarray,
    t_end: float = 20.0,
    dt: float = 0.02,
    force_limit: float = 1.0,
    params: PendulumParams = PendulumParams(),
) -> RolloutResult:
    t = np.arange(0.0, t_end + 0.5 * dt, dt, dtype=float)
    y = np.empty((4, t.size), dtype=float)
    u = np.empty(t.size, dtype=float)
    control_times = np.zeros(t.size, dtype=float)
    y[:, 0] = np.asarray(x0, dtype=float)

    if isinstance(controller, Controller):
        controller.reset(y[:, 0])

    for k in range(t.size - 1):
        tic = time.perf_counter()
        if isinstance(controller, Controller):
            force = controller.act(float(t[k]), y[:, k].copy())
        else:
            force = float(controller(float(t[k]), y[:, k].copy()))
        control_times[k] = time.perf_counter() - tic
        u[k] = clip_force(force, force_limit)

        constant_force = float(u[k])
        sol = solve_ivp(
            fun=lambda tau, state: inverted_pendulum_dynamics(
                tau, state, params, lambda _t, _x: constant_force
            ),
            t_span=(float(t[k]), float(t[k + 1])),
            y0=y[:, k],
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
            t_eval=[float(t[k + 1])],
        )
        if not sol.success or not np.all(np.isfinite(sol.y)):
            y[:, k + 1 :] = np.nan
            u[k + 1 :] = np.nan
            return RolloutResult(t, y, u, False, sol.message, control_times)
        y[:, k + 1] = sol.y[:, -1]

    tic = time.perf_counter()
    if isinstance(controller, Controller):
        u[-1] = clip_force(controller.act(float(t[-1]), y[:, -1].copy()), force_limit)
    else:
        u[-1] = clip_force(controller(float(t[-1]), y[:, -1].copy()), force_limit)
    control_times[-1] = time.perf_counter() - tic

    return RolloutResult(t, y, u, True, "ok", control_times)


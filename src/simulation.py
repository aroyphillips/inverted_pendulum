from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from config import PendulumParams, SimulationSettings
from dynamics import inverted_pendulum_dynamics


@dataclass
class SimulationResult:
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray
    success: bool
    message: str


def run_simulation(
    y0: np.ndarray,
    controller: Callable[[float, np.ndarray], float],
    params: PendulumParams,
    settings: SimulationSettings,
) -> SimulationResult:
    t_eval = np.linspace(0.0, settings.t_end, settings.num_samples, dtype=float)

    sol = solve_ivp(
        fun=lambda t, y: inverted_pendulum_dynamics(t, y, params, controller),
        t_span=(0.0, settings.t_end),
        y0=np.asarray(y0, dtype=float),
        method=settings.solver_method,
        t_eval=t_eval,
        rtol=settings.rtol,
        atol=settings.atol,
    )

    if not sol.success:
        return SimulationResult(
            t=t_eval,
            y=np.full((4, t_eval.size), np.nan),
            u=np.full(t_eval.size, np.nan),
            success=False,
            message=sol.message,
        )

    u = np.array([float(controller(ti, sol.y[:, i])) for i, ti in enumerate(sol.t)], dtype=float)

    if np.any(~np.isfinite(sol.y)) or np.any(~np.isfinite(u)):
        return SimulationResult(
            t=sol.t,
            y=np.full_like(sol.y, np.nan),
            u=np.full(sol.t.size, np.nan),
            success=False,
            message="Non-finite values detected.",
        )

    return SimulationResult(t=sol.t, y=sol.y, u=u, success=True, message=sol.message)

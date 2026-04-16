from dataclasses import dataclass
from typing import Optional

import numpy as np

from config import PendulumParams, SimulationSettings
from inputs import ImpulseThenZeroInput
from simulation import run_simulation


@dataclass
class InitialStateResult:
    initial_state: Optional[np.ndarray]
    crossing_time: Optional[float]
    target_theta_rad: float
    success: bool
    diagnostic: str
    full_t: np.ndarray
    full_y: np.ndarray
    full_u: np.ndarray


def derive_initial_state_from_impulse_crossing(
    theta_target_deg: float,
    params: PendulumParams,
    settings: SimulationSettings,
    impulse_force: float = 0.01,
    equilibrium_state: Optional[np.ndarray] = None,
) -> InitialStateResult:
    if equilibrium_state is None:
        equilibrium_state = np.zeros(4, dtype=float)

    theta_target_rad = np.deg2rad(theta_target_deg)
    controller = ImpulseThenZeroInput(impulse_force=impulse_force)

    sim = run_simulation(equilibrium_state, controller, params, settings)
    if not sim.success:
        return InitialStateResult(
            initial_state=None,
            crossing_time=None,
            target_theta_rad=theta_target_rad,
            success=False,
            diagnostic=f"Simulation failed: {sim.message}",
            full_t=sim.t,
            full_y=sim.y,
            full_u=sim.u,
        )

    theta = sim.y[2, :]
    idx = np.where(theta >= theta_target_rad)[0]
    if idx.size == 0:
        return InitialStateResult(
            initial_state=None,
            crossing_time=None,
            target_theta_rad=theta_target_rad,
            success=False,
            diagnostic="Theta target not reached in configured horizon.",
            full_t=sim.t,
            full_y=sim.y,
            full_u=sim.u,
        )

    i = int(idx[0])
    if i == 0:
        crossing_state = sim.y[:, 0].copy()
        crossing_t = float(sim.t[0])
    else:
        t0, t1 = float(sim.t[i - 1]), float(sim.t[i])
        th0, th1 = float(theta[i - 1]), float(theta[i])
        alpha = (theta_target_rad - th0) / (th1 - th0)
        alpha = float(np.clip(alpha, 0.0, 1.0))
        crossing_t = t0 + alpha * (t1 - t0)
        crossing_state = sim.y[:, i - 1] + alpha * (sim.y[:, i] - sim.y[:, i - 1])

    return InitialStateResult(
        initial_state=crossing_state,
        crossing_time=crossing_t,
        target_theta_rad=theta_target_rad,
        success=True,
        diagnostic="Crossing detected.",
        full_t=sim.t,
        full_y=sim.y,
        full_u=sim.u,
    )

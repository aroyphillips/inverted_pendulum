from typing import Callable

import numpy as np

from config import PendulumParams


State = np.ndarray
ForceFn = Callable[[float, State], float]


def inverted_pendulum_dynamics(t: float, y: State, params: PendulumParams, controller: ForceFn) -> np.ndarray:
    """Nonlinear dynamics matching notebook equations.

    State convention: y = [x, x_dot, theta, theta_dot].
    """
    x, x_dot, theta, theta_dot = y
    force = float(controller(t, y))

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    denom = params.M + params.m * sin_t**2

    x_ddot = (
        force
        + params.m * params.g * sin_t * cos_t
        - params.m * params.l * theta_dot**2 * sin_t
    ) / denom

    theta_ddot = (
        force * cos_t
        + (params.M + params.m) * params.g * sin_t
        - params.m * params.l * theta_dot**2 * sin_t * cos_t
    ) / (params.l * denom)

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot], dtype=float)

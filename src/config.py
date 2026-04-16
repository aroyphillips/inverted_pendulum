from dataclasses import dataclass


@dataclass(frozen=True)
class PendulumParams:
    """Physical parameters for the cart-pendulum system."""

    M: float = 1.0
    m: float = 0.2
    l: float = 0.5
    g: float = 9.81


@dataclass(frozen=True)
class SimulationSettings:
    """Numerical integration and sampling configuration."""

    t_end: float = 10.0
    num_samples: int = 1000
    solver_method: str = "DOP853"
    rtol: float = 1e-8
    atol: float = 1e-10

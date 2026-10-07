from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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


@dataclass
class ValidationSpec:
    """Sampled validation cases for stress testing and cross-validation.

    Each case is defined by an impulse force and a target crossing angle.
    Infeasible cases (where the impulse cannot reach the target angle within the
    simulation horizon) have NaN initial_states and crossing_times entries.
    """

    forces: np.ndarray        # (n,) impulse forces, N
    angles_deg: np.ndarray    # (n,) target crossing angles, degrees
    seed: int
    initial_states: np.ndarray   # (n, 4), NaN rows for infeasible cases
    crossing_times: np.ndarray   # (n,), NaN for infeasible cases
    feasible: np.ndarray         # (n,) bool

    @property
    def n(self) -> int:
        return int(self.forces.size)

    @property
    def n_feasible(self) -> int:
        return int(np.sum(self.feasible))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            forces=self.forces,
            angles_deg=self.angles_deg,
            seed=np.array(self.seed),
            initial_states=self.initial_states,
            crossing_times=self.crossing_times,
            feasible=self.feasible,
        )

    @classmethod
    def load(cls, path: Path) -> "ValidationSpec":
        data = np.load(path)
        return cls(
            forces=data["forces"],
            angles_deg=data["angles_deg"],
            seed=int(data["seed"]),
            initial_states=data["initial_states"],
            crossing_times=data["crossing_times"],
            feasible=data["feasible"].astype(bool),
        )


def sample_validation_batch(
    n_cases: int = 100,
    seed: int = 42,
    force_range: Tuple[float, float] = (0.5, 5.0),
    angle_range_deg: Tuple[float, float] = (5.0, 90.0),
    params: Optional[PendulumParams] = None,
    settings: Optional[SimulationSettings] = None,
) -> ValidationSpec:
    """Sample n_cases validation initial states from random (force, angle) pairs.

    Each case draws an impulse force uniformly from force_range and a target
    crossing angle uniformly from angle_range_deg, then derives the initial state
    at the moment the pendulum first crosses that angle under the impulse.

    Cases where the impulse cannot reach the target angle within the simulation
    horizon are marked infeasible (NaN in initial_states).
    """
    if params is None:
        params = PendulumParams()
    if settings is None:
        # Generous horizon so high-angle, low-force cases still have a chance.
        settings = SimulationSettings(t_end=20.0, num_samples=2000)

    rng = np.random.default_rng(seed)
    forces = rng.uniform(force_range[0], force_range[1], size=n_cases)
    angles = rng.uniform(angle_range_deg[0], angle_range_deg[1], size=n_cases)

    initial_states = np.full((n_cases, 4), np.nan)
    crossing_times = np.full(n_cases, np.nan)
    feasible = np.zeros(n_cases, dtype=bool)

    for i in range(n_cases):
        result = derive_initial_state_from_impulse_crossing(
            theta_target_deg=float(angles[i]),
            params=params,
            settings=settings,
            impulse_force=float(forces[i]),
        )
        if result.success:
            initial_states[i] = result.initial_state
            crossing_times[i] = result.crossing_time
            feasible[i] = True

    return ValidationSpec(
        forces=forces,
        angles_deg=angles,
        seed=seed,
        initial_states=initial_states,
        crossing_times=crossing_times,
        feasible=feasible,
    )

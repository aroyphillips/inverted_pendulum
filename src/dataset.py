from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from config import PendulumParams, SimulationSettings
from inputs import GaussianRandomWalkInput
from simulation import run_simulation


@dataclass
class DatasetGenerationResult:
    states: np.ndarray
    inputs: np.ndarray
    t: np.ndarray
    seeds: np.ndarray
    num_attempts: int


@dataclass
class DatasetConfig:
    num_trajectories: int = 500
    dt_force: float = 0.1
    sigma_force: float = 0.08
    force_max: float = 1.0
    enforce_sign_balance: bool = True
    max_attempts: int = 5000
    base_seed: int = 12345
    num_workers: int = 0  # 0 -> auto


def _simulate_one(
    seed: int,
    y0: np.ndarray,
    params: PendulumParams,
    settings: SimulationSettings,
    dt_force: float,
    sigma_force: float,
    force_max: float,
    enforce_sign_balance: bool,
) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    controller = GaussianRandomWalkInput(
        t_end=settings.t_end,
        dt_force=dt_force,
        sigma_force=sigma_force,
        force_max=force_max,
        seed=seed,
        enforce_sign_balance=enforce_sign_balance,
    )
    sim = run_simulation(y0, controller, params, settings)
    if not sim.success:
        return False, np.array([]), np.array([]), np.array([])
    return True, sim.y.T, sim.u, sim.t


def _collect_serial(
    y0: np.ndarray,
    params: PendulumParams,
    settings: SimulationSettings,
    cfg: DatasetConfig,
) -> DatasetGenerationResult:
    states_list: List[np.ndarray] = []
    inputs_list: List[np.ndarray] = []
    seeds: List[int] = []
    attempts = 0

    while len(states_list) < cfg.num_trajectories and attempts < cfg.max_attempts:
        seed = cfg.base_seed + attempts
        ok, y, u, t = _simulate_one(
            seed,
            y0,
            params,
            settings,
            cfg.dt_force,
            cfg.sigma_force,
            cfg.force_max,
            cfg.enforce_sign_balance,
        )
        attempts += 1
        if not ok:
            continue
        states_list.append(y)
        inputs_list.append(u)
        seeds.append(seed)

    if len(states_list) != cfg.num_trajectories:
        raise RuntimeError(
            f"Collected {len(states_list)} trajectories before max_attempts={cfg.max_attempts}."
        )

    return DatasetGenerationResult(
        states=np.stack(states_list, axis=0),
        inputs=np.stack(inputs_list, axis=0),
        t=t,
        seeds=np.asarray(seeds, dtype=int),
        num_attempts=attempts,
    )


def _collect_parallel(
    y0: np.ndarray,
    params: PendulumParams,
    settings: SimulationSettings,
    cfg: DatasetConfig,
) -> DatasetGenerationResult:
    states_list: List[np.ndarray] = []
    inputs_list: List[np.ndarray] = []
    seeds: List[int] = []
    attempts = 0
    max_workers = cfg.num_workers if cfg.num_workers and cfg.num_workers > 0 else None

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {}

        while len(states_list) < cfg.num_trajectories and attempts < cfg.max_attempts:
            while (
                len(futures) < (max_workers or 4)
                and attempts < cfg.max_attempts
                and len(states_list) + len(futures) < cfg.num_trajectories
            ):
                seed = cfg.base_seed + attempts
                fut = ex.submit(
                    _simulate_one,
                    seed,
                    y0,
                    params,
                    settings,
                    cfg.dt_force,
                    cfg.sigma_force,
                    cfg.force_max,
                    cfg.enforce_sign_balance,
                )
                futures[fut] = seed
                attempts += 1

            if not futures:
                break

            for fut in as_completed(list(futures.keys())):
                seed = futures.pop(fut)
                ok, y, u, t = fut.result()
                if not ok:
                    continue
                states_list.append(y)
                inputs_list.append(u)
                seeds.append(seed)
                if len(states_list) >= cfg.num_trajectories:
                    for pending in futures:
                        pending.cancel()
                    futures.clear()
                    break

    if len(states_list) != cfg.num_trajectories:
        raise RuntimeError(
            f"Collected {len(states_list)} trajectories before max_attempts={cfg.max_attempts}."
        )

    return DatasetGenerationResult(
        states=np.stack(states_list, axis=0),
        inputs=np.stack(inputs_list, axis=0),
        t=t,
        seeds=np.asarray(seeds, dtype=int),
        num_attempts=attempts,
    )


def generate_random_walk_dataset(
    y0: np.ndarray,
    params: PendulumParams,
    settings: SimulationSettings,
    cfg: Optional[DatasetConfig] = None,
) -> DatasetGenerationResult:
    if cfg is None:
        cfg = DatasetConfig()

    if cfg.num_workers == 1:
        return _collect_serial(y0, params, settings, cfg)
    return _collect_parallel(y0, params, settings, cfg)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import json

import numpy as np

from config import PendulumParams, SimulationSettings
from inputs import PiecewiseConstantInput, PiecewiseLinearInput
from simulation import SimulationResult, run_simulation


@dataclass
class InputTrajectory:
    t: np.ndarray
    u: np.ndarray
    source: str = "generated"

    @property
    def t_end(self) -> float:
        return float(self.t[-1])

    @property
    def dt_nominal(self) -> float:
        return float(np.median(np.diff(self.t)))


def _as_1d_force(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0, :]
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(
        "Expected one input trajectory with shape (T,), (T, 1), or (1, T). "
        f"Got shape {arr.shape}."
    )


def _from_array(
    arr: np.ndarray,
    dt: float,
    source: str,
    force_limit: Optional[float] = 1.0,
) -> InputTrajectory:
    data = np.asarray(arr, dtype=float)
    if data.ndim == 2 and data.shape[1] >= 2:
        t = np.asarray(data[:, 0], dtype=float)
        u = np.asarray(data[:, 1], dtype=float)
        return normalize_input_trajectory(t, u, dt=dt, source=source, force_limit=force_limit)
    u = _as_1d_force(data)
    t = np.arange(u.size, dtype=float) * float(dt)
    return normalize_input_trajectory(t, u, dt=dt, source=source, force_limit=force_limit)


def normalize_input_trajectory(
    t: np.ndarray,
    u: np.ndarray,
    dt: float,
    source: str = "user",
    force_limit: Optional[float] = 1.0,
) -> InputTrajectory:
    t = np.asarray(t, dtype=float).reshape(-1)
    u = _as_1d_force(u)
    if t.size != u.size:
        raise ValueError(f"t and u must have the same length. Got {t.size} and {u.size}.")
    if t.size == 0:
        raise ValueError("Input trajectory must contain at least one force value.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(u)):
        raise ValueError("Input trajectory contains non-finite time or force values.")

    if t.size == 1:
        t = np.array([0.0, float(dt)], dtype=float)
        u = np.array([u[0], u[0]], dtype=float)
    else:
        t = t - float(t[0])
        if np.any(np.diff(t) <= 0.0):
            raise ValueError("Input times must be strictly increasing.")

    if force_limit is not None:
        u = np.clip(u, -float(force_limit), float(force_limit))
    return InputTrajectory(t=t, u=u, source=source)


def load_input_trajectory(
    path: Optional[Path | str] = None,
    default_num_points: int = 100,
    dt: float = 0.01,
    force_limit: Optional[float] = 1.0,
) -> InputTrajectory:
    if path is None or str(path).strip() == "":
        u = np.zeros(int(default_num_points), dtype=float)
        t = np.arange(u.size, dtype=float) * float(dt)
        return normalize_input_trajectory(t, u, dt=dt, source="default_zero", force_limit=force_limit)

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return _from_array(np.load(path, allow_pickle=False), dt=dt, source=str(path), force_limit=force_limit)
    if suffix == ".npz":
        arr = np.load(path, allow_pickle=False)
        force_key = next((key for key in ("u", "inputs", "force", "forces") if key in arr), None)
        if force_key is None:
            raise ValueError(f"{path} must contain one of: u, inputs, force, forces.")
        u = _as_1d_force(arr[force_key])
        if "t" in arr:
            t = np.asarray(arr["t"], dtype=float)
        else:
            t = np.arange(u.size, dtype=float) * float(dt)
        return normalize_input_trajectory(t, u, dt=dt, source=str(path), force_limit=force_limit)
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        raw = np.genfromtxt(path, delimiter=delimiter, comments="#")
        raw = np.asarray(raw, dtype=float)
        if raw.ndim == 2:
            raw = raw[np.all(np.isfinite(raw), axis=1)]
        elif raw.ndim == 1:
            raw = raw[np.isfinite(raw)]
        return _from_array(raw, dt=dt, source=str(path), force_limit=force_limit)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            force_key = next((key for key in ("u", "inputs", "force", "forces") if key in payload), None)
            if force_key is None:
                raise ValueError(f"{path} must contain one of: u, inputs, force, forces.")
            u = _as_1d_force(payload[force_key])
            t = np.asarray(payload.get("t", np.arange(u.size) * float(dt)), dtype=float)
            return normalize_input_trajectory(t, u, dt=dt, source=str(path), force_limit=force_limit)
        return _from_array(np.asarray(payload, dtype=float), dt=dt, source=str(path), force_limit=force_limit)
    raise ValueError(f"Unsupported trajectory file type: {suffix}")


def make_input_controller(trajectory: InputTrajectory, interpolation: str = "previous"):
    if interpolation == "previous":
        return PiecewiseConstantInput(trajectory.t, trajectory.u)
    if interpolation == "linear":
        return PiecewiseLinearInput(trajectory.t, trajectory.u)
    raise ValueError("interpolation must be 'previous' or 'linear'.")


def simulate_input_trajectory(
    trajectory: InputTrajectory,
    initial_state: np.ndarray,
    interpolation: str = "previous",
    params: PendulumParams = PendulumParams(),
    num_samples: Optional[int] = None,
) -> SimulationResult:
    controller = make_input_controller(trajectory, interpolation=interpolation)
    samples = int(num_samples) if num_samples is not None else int(max(2, trajectory.t.size))
    settings = SimulationSettings(t_end=trajectory.t_end, num_samples=samples)
    return run_simulation(np.asarray(initial_state, dtype=float), controller, params, settings)

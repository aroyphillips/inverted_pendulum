from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class PEResult:
    order: int
    rank: int
    min_dim: int
    is_persistently_exciting: bool
    singular_values: np.ndarray
    condition_number: float


def hankel_matrix(signal: np.ndarray, rows: int) -> np.ndarray:
    u = np.asarray(signal, dtype=float).reshape(-1)
    if rows <= 0:
        raise ValueError("rows must be positive")
    cols = u.size - rows + 1
    if cols <= 0:
        raise ValueError("rows too large for signal length")

    h = np.empty((rows, cols), dtype=float)
    for i in range(rows):
        h[i, :] = u[i : i + cols]
    return h


def persistent_excitation_check(signal: np.ndarray, order: int, tol: float = 1e-10) -> PEResult:
    h = hankel_matrix(signal, order)
    svals = np.linalg.svd(h, full_matrices=False, compute_uv=False)
    rank = int(np.sum(svals > tol))
    min_dim = min(h.shape)
    cond = float(svals[0] / svals[-1]) if svals[-1] > tol else float("inf")

    return PEResult(
        order=order,
        rank=rank,
        min_dim=min_dim,
        is_persistently_exciting=(rank >= order),
        singular_values=svals,
        condition_number=cond,
    )


def readiness_checks(states: np.ndarray, inputs: np.ndarray, t: np.ndarray) -> Dict[str, bool]:
    dt = np.diff(t)
    return {
        "shape_match": states.shape[0] == inputs.shape[0] and states.shape[1] == inputs.shape[1],
        "finite_states": bool(np.all(np.isfinite(states))),
        "finite_inputs": bool(np.all(np.isfinite(inputs))),
        "monotonic_time": bool(np.all(dt > 0.0)),
        "uniform_sampling": bool(np.allclose(dt, dt[0], rtol=1e-8, atol=1e-10)),
        "nontrivial_input_variance": bool(np.any(np.var(inputs, axis=1) > 1e-12)),
    }

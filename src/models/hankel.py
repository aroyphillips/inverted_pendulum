from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class HankelRankReport:
    rows: int
    cols: int
    rank: int
    condition_number: float
    min_singular_value: float
    max_singular_value: float


def block_hankel(signal: np.ndarray, horizon: int) -> np.ndarray:
    data = np.asarray(signal, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    t_steps, dim = data.shape
    cols = t_steps - horizon + 1
    if cols <= 0:
        raise ValueError("horizon is longer than the signal")

    h = np.empty((dim * horizon, cols), dtype=float)
    for i in range(horizon):
        h[i * dim : (i + 1) * dim, :] = data[i : i + cols, :].T
    return h


def build_block_hankel_trajectories(
    signals: np.ndarray,
    horizon: int,
    max_columns: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    data = np.asarray(signals, dtype=float)
    if data.ndim == 2:
        data = data[:, :, None]
    blocks = [block_hankel(traj, horizon) for traj in data]
    h = np.concatenate(blocks, axis=1)
    if max_columns is not None and 0 < max_columns < h.shape[1]:
        rng = np.random.default_rng(seed)
        cols = rng.choice(h.shape[1], size=max_columns, replace=False)
        cols.sort()
        h = h[:, cols]
    return h


def deepc_partitions(
    inputs: np.ndarray,
    outputs: np.ndarray,
    t_ini: int,
    horizon: int,
    max_columns: Optional[int] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    total = int(t_ini + horizon)
    hu = build_block_hankel_trajectories(inputs, total, max_columns=max_columns, seed=seed)
    hy = build_block_hankel_trajectories(outputs, total, max_columns=max_columns, seed=seed)

    u_dim = inputs.shape[-1] if inputs.ndim == 3 else 1
    y_dim = outputs.shape[-1]
    up_rows = t_ini * u_dim
    yp_rows = t_ini * y_dim
    return (
        hu[:up_rows, :],
        hu[up_rows:, :],
        hy[:yp_rows, :],
        hy[yp_rows:, :],
    )


def rank_report(matrix: np.ndarray, tol: float = 1e-9) -> HankelRankReport:
    svals = np.linalg.svd(np.asarray(matrix, dtype=float), full_matrices=False, compute_uv=False)
    rank = int(np.sum(svals > tol))
    cond = float(svals[0] / svals[-1]) if svals.size and svals[-1] > tol else float("inf")
    return HankelRankReport(
        rows=int(matrix.shape[0]),
        cols=int(matrix.shape[1]),
        rank=rank,
        condition_number=cond,
        min_singular_value=float(svals[-1]) if svals.size else 0.0,
        max_singular_value=float(svals[0]) if svals.size else 0.0,
    )


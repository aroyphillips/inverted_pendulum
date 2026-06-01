from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from io_utils import load_dataset_npz


@dataclass
class ControlDataset:
    states: np.ndarray
    inputs: np.ndarray
    t: np.ndarray
    initial_state: np.ndarray

    @property
    def dt(self) -> float:
        return float(np.mean(np.diff(self.t)))

    @property
    def num_trajectories(self) -> int:
        return int(self.states.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.states.shape[1])


def load_control_dataset(path: Path) -> ControlDataset:
    data = load_dataset_npz(path)
    states = np.asarray(data["states"], dtype=float)
    inputs = np.asarray(data["inputs"], dtype=float)
    if inputs.ndim == 2:
        inputs = inputs[:, :, None]
    return ControlDataset(
        states=states,
        inputs=inputs,
        t=np.asarray(data["t"], dtype=float),
        initial_state=np.asarray(data["initial_state"], dtype=float),
    )


def supervised_pairs(dataset: ControlDataset) -> Dict[str, np.ndarray]:
    xk = dataset.states[:, :-1, :]
    uk = dataset.inputs[:, :-1, :]
    xkp1 = dataset.states[:, 1:, :]
    return {
        "Xk": xk,
        "Uk": uk,
        "Xkp1": xkp1,
        "DeltaX": xkp1 - xk,
        "ZU": np.concatenate([xk, uk], axis=-1),
    }


def split_by_trajectory(
    dataset: ControlDataset,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = dataset.num_trajectories
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    n_val = int(round(val_fraction * n))
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


def flatten_time_major(arr: np.ndarray, trajectory_indices: np.ndarray) -> np.ndarray:
    subset = np.asarray(arr)[np.asarray(trajectory_indices)]
    return subset.reshape((-1, subset.shape[-1]))


def flatten_supervised(
    dataset: ControlDataset,
    trajectory_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = supervised_pairs(dataset)
    x = flatten_time_major(pairs["ZU"], trajectory_indices)
    y_delta = flatten_time_major(pairs["DeltaX"], trajectory_indices)
    y_next = flatten_time_major(pairs["Xkp1"], trajectory_indices)
    return x, y_delta, y_next


def stratified_transition_sample(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    seed: int = 0,
    theta_bins: int = 12,
    theta_dot_bins: int = 12,
) -> Tuple[np.ndarray, np.ndarray]:
    if max_points <= 0 or x.shape[0] <= max_points:
        return x, y

    rng = np.random.default_rng(seed)
    theta = x[:, 2]
    theta_dot = x[:, 3]
    theta_edges = np.quantile(theta, np.linspace(0.0, 1.0, theta_bins + 1))
    theta_dot_edges = np.quantile(theta_dot, np.linspace(0.0, 1.0, theta_dot_bins + 1))
    theta_edges = np.unique(theta_edges)
    theta_dot_edges = np.unique(theta_dot_edges)

    if theta_edges.size < 3 or theta_dot_edges.size < 3:
        idx = rng.choice(x.shape[0], size=max_points, replace=False)
        return x[idx], y[idx]

    ti = np.digitize(theta, theta_edges[1:-1], right=False)
    wi = np.digitize(theta_dot, theta_dot_edges[1:-1], right=False)
    groups = ti * max(1, theta_dot_edges.size - 1) + wi
    unique_groups = np.unique(groups)
    per_group = max(1, int(np.ceil(max_points / unique_groups.size)))

    chosen = []
    for group in unique_groups:
        members = np.flatnonzero(groups == group)
        take = min(per_group, members.size)
        chosen.extend(rng.choice(members, size=take, replace=False).tolist())

    chosen = np.asarray(chosen, dtype=int)
    if chosen.size > max_points:
        chosen = rng.choice(chosen, size=max_points, replace=False)
    elif chosen.size < max_points:
        remaining = np.setdiff1d(np.arange(x.shape[0]), chosen, assume_unique=False)
        extra = rng.choice(remaining, size=max_points - chosen.size, replace=False)
        chosen = np.concatenate([chosen, extra])

    rng.shuffle(chosen)
    return x[chosen], y[chosen]


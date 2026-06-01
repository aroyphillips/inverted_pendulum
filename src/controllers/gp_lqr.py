from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pickle
import time

import numpy as np

from controllers.base import Controller
from eval.metrics import rollout_metrics
from eval.rollout import rollout_controller
from models.dataset_views import ControlDataset, flatten_supervised, split_by_trajectory
from models.gp_dynamics import GPDynamicsConfig, GPDynamicsModel
from models.lqr_utils import discrete_lqr_gain


# Default Q/R grid to try during closed-loop tuning.
_DEFAULT_QR_GRID: List[Tuple[np.ndarray, np.ndarray]] = [
    (np.diag([1.0, 0.2,  10.0,  1.0]), np.array([[0.10]])),
    (np.diag([1.0, 0.2,  20.0,  1.0]), np.array([[0.05]])),
    (np.diag([1.0, 0.2,  50.0,  5.0]), np.array([[0.05]])),
    (np.diag([1.0, 0.2, 100.0, 10.0]), np.array([[0.05]])),
    (np.diag([1.0, 0.2, 100.0, 10.0]), np.array([[0.01]])),
]
_DEFAULT_ALPHA_RANGE: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


class GPLQRController(Controller):
    def __init__(
        self,
        model: GPDynamicsModel | None = None,
        k: np.ndarray | None = None,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        force_limit: float = 5.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(name="gp_lqr", force_limit=force_limit)
        self.model = model
        self.k = None if k is None else np.asarray(k, dtype=float)
        self.q = np.diag([1.0, 0.2, 20.0, 1.0]) if q is None else np.asarray(q, dtype=float)
        self.r = np.array([[0.05]]) if r is None else np.asarray(r, dtype=float)
        self.alpha = float(alpha)

    def fit(self, dataset: ControlDataset, config: GPDynamicsConfig | None = None, seed: int = 0) -> "GPLQRController":
        train_idx, val_idx, _ = split_by_trajectory(dataset, seed=seed)
        x_train, y_train, _ = flatten_supervised(dataset, train_idx)
        x_val, y_val, _ = flatten_supervised(dataset, val_idx)

        # Near-upright validation subset (|theta| < 0.1 rad)
        near_upright_mask = np.abs(x_val[:, 2]) < 0.1
        x_val_upright = x_val[near_upright_mask]
        y_val_upright = y_val[near_upright_mask] if near_upright_mask.any() else None

        tic = time.perf_counter()
        self.model = GPDynamicsModel(config=config).fit(x_train, y_train, x_val=x_val, y_val=y_val)
        a, b = self.model.linearize()
        self.k, p = discrete_lqr_gain(a, b, self.q, self.r)
        runtime = time.perf_counter() - tic

        # Near-upright RMSE
        near_upright_rmse = None
        if y_val_upright is not None and x_val_upright.shape[0] > 0:
            pred_up = self.model.predict_delta(x_val_upright)
            near_upright_rmse = float(np.sqrt(np.mean((pred_up - y_val_upright) ** 2)))

        self.metadata.update(
            {
                "runtime_train_sec": runtime,
                "A": a.tolist(),
                "B": b.tolist(),
                "K": self.k.tolist(),
                "alpha": self.alpha,
                "P": p.tolist(),
                "Q": self.q.tolist(),
                "R": self.r.tolist(),
                "model": self.model.metadata,
                "near_upright_rmse": near_upright_rmse,
                "near_upright_val_points": int(x_val_upright.shape[0]) if x_val_upright is not None else 0,
            }
        )
        return self

    def act(self, t: float, x: np.ndarray) -> float:
        if self.k is None:
            raise RuntimeError("GPLQRController has not been fit.")
        u = -self.alpha * float(np.ravel(self.k @ np.asarray(x, dtype=float))[0])
        return self.clipped(u)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "GPLQRController":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def train_gp_lqr(
    dataset: ControlDataset,
    max_train_points: int = 5000,
    seed: int = 0,
    force_limit: float = 5.0,
    initial_state: Optional[np.ndarray] = None,
    qr_grid: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
    alpha_range: Optional[List[float]] = None,
    t_end: float = 20.0,
    dt: float = 0.02,
) -> GPLQRController:
    """Fit GP dynamics, then select Q/R and alpha by closed-loop validation rollouts."""
    if qr_grid is None:
        qr_grid = _DEFAULT_QR_GRID
    if alpha_range is None:
        alpha_range = _DEFAULT_ALPHA_RANGE

    x0 = dataset.initial_state if initial_state is None else np.asarray(initial_state, dtype=float)

    # Fit the GP dynamics model once (shared across all Q/R/alpha candidates).
    base_ctrl = GPLQRController(force_limit=force_limit)
    base_ctrl.fit(dataset, config=GPDynamicsConfig(max_train_points=max_train_points, seed=seed), seed=seed)
    a = np.asarray(base_ctrl.metadata["A"])
    b = np.asarray(base_ctrl.metadata["B"])

    best_score = float("inf")
    best_q = qr_grid[0][0]
    best_r = qr_grid[0][1]
    best_alpha = 1.0
    best_k = base_ctrl.k

    tune_results = []
    for q_cand, r_cand in qr_grid:
        try:
            k_cand, _ = discrete_lqr_gain(a, b, q_cand, r_cand)
        except Exception:
            continue
        for alpha in alpha_range:
            ctrl = GPLQRController(
                model=base_ctrl.model,
                k=k_cand,
                q=q_cand,
                r=r_cand,
                force_limit=force_limit,
                alpha=alpha,
            )
            result = rollout_controller(ctrl, x0, t_end=t_end, dt=dt, force_limit=force_limit)
            m = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
            success = bool(m.get("success", False))
            cost = float(m.get("cost", 1e12))
            settle_t = m.get("settle_and_remain_time")
            score = cost + (0.0 if success else 2000.0)
            tune_results.append({
                "Q_diag": np.diag(q_cand).tolist(),
                "R": float(r_cand[0, 0]),
                "alpha": alpha,
                "success": success,
                "settle_and_remain_time": settle_t,
                "cost": cost,
            })
            if score < best_score:
                best_score = score
                best_q = q_cand
                best_r = r_cand
                best_alpha = alpha
                best_k = k_cand

    best_ctrl = GPLQRController(
        model=base_ctrl.model,
        k=best_k,
        q=best_q,
        r=best_r,
        force_limit=force_limit,
        alpha=best_alpha,
    )
    best_ctrl.metadata.update(base_ctrl.metadata)
    best_ctrl.metadata.update({
        "K": best_k.tolist(),
        "Q": best_q.tolist(),
        "R": best_r.tolist(),
        "alpha": best_alpha,
        "tuning_results": tune_results,
        "best_tuning_score": best_score,
    })
    return best_ctrl


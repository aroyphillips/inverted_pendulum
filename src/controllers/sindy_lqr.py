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
from models.lqr_utils import discrete_lqr_gain
from models.sindy import SINDyConfig, SINDyDynamicsModel


# Default Q to use when computing LQR from SINDy model (heavy theta weight).
_DEFAULT_Q = np.diag([1.0, 0.2, 100.0, 10.0])
_DEFAULT_R = np.array([[0.05]])

# Grid of (threshold, ridge) pairs to try during tuning.
_DEFAULT_THRESHOLD_RIDGE_GRID: List[Tuple[float, float]] = [
    (1e-4, 1e-6),
    (1e-3, 1e-6),
    (5e-3, 1e-6),
    (1e-3, 1e-4),
    (5e-3, 1e-4),
]
_DEFAULT_ALPHA_RANGE: List[float] = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]


class SINDyLQRController(Controller):
    def __init__(
        self,
        model: SINDyDynamicsModel | None = None,
        k: np.ndarray | None = None,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        force_limit: float = 5.0,
        alpha: float = 1.0,
    ) -> None:
        super().__init__(name="sindy_lqr", force_limit=force_limit)
        self.model = model
        self.k = None if k is None else np.asarray(k, dtype=float)
        self.q = _DEFAULT_Q.copy() if q is None else np.asarray(q, dtype=float)
        self.r = _DEFAULT_R.copy() if r is None else np.asarray(r, dtype=float)
        self.alpha = float(alpha)

    def fit(self, dataset: ControlDataset, config: SINDyConfig | None = None, seed: int = 0) -> "SINDyLQRController":
        train_idx, val_idx, _ = split_by_trajectory(dataset, seed=seed)
        x_train, _, x_next_train = flatten_supervised(dataset, train_idx)
        x_val, _, x_next_val = flatten_supervised(dataset, val_idx)
        tic = time.perf_counter()
        self.model = SINDyDynamicsModel(config=config).fit(x_train, x_next_train, x_val=x_val, y_val=x_next_val)
        a, b = self.model.linearize()
        self.k, p = discrete_lqr_gain(a, b, self.q, self.r)
        runtime = time.perf_counter() - tic
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
            }
        )
        return self

    def act(self, t: float, x: np.ndarray) -> float:
        if self.k is None:
            raise RuntimeError("SINDyLQRController has not been fit.")
        u = -self.alpha * float(np.ravel(self.k @ np.asarray(x, dtype=float))[0])
        return self.clipped(u)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SINDyLQRController":
        with Path(path).open("rb") as f:
            return pickle.load(f)


def train_sindy_lqr(
    dataset: ControlDataset,
    threshold: float = 1e-3,
    max_train_points: int = 100000,
    seed: int = 0,
    force_limit: float = 5.0,
    initial_state: Optional[np.ndarray] = None,
    threshold_ridge_grid: Optional[List[Tuple[float, float]]] = None,
    alpha_range: Optional[List[float]] = None,
    t_end: float = 20.0,
    dt: float = 0.02,
) -> SINDyLQRController:
    """Fit SINDy dynamics, then select threshold/ridge and alpha by closed-loop validation.

    Also reports A/B matrix sanity: θ̈ at the agreed initial state under max force.
    """
    if threshold_ridge_grid is None:
        threshold_ridge_grid = _DEFAULT_THRESHOLD_RIDGE_GRID
    if alpha_range is None:
        alpha_range = _DEFAULT_ALPHA_RANGE

    x0 = dataset.initial_state if initial_state is None else np.asarray(initial_state, dtype=float)

    best_score = float("inf")
    best_ctrl: Optional[SINDyLQRController] = None
    tune_results = []

    for thr, ridge in threshold_ridge_grid:
        config = SINDyConfig(threshold=thr, ridge=ridge, max_train_points=max_train_points, seed=seed)
        ctrl = SINDyLQRController(q=_DEFAULT_Q, r=_DEFAULT_R, force_limit=force_limit)
        try:
            ctrl.fit(dataset, config=config, seed=seed)
        except Exception as exc:
            tune_results.append({"threshold": thr, "ridge": ridge, "error": str(exc)})
            continue

        a = np.asarray(ctrl.metadata["A"])
        b = np.asarray(ctrl.metadata["B"])

        # Sanity: θ̈ predicted by learned model at initial state under max force.
        theta_ddot_pred = float(np.ravel(a @ x0 + b.ravel() * force_limit - x0)[3])

        for alpha in alpha_range:
            test_ctrl = SINDyLQRController(
                model=ctrl.model, k=ctrl.k, q=ctrl.q, r=ctrl.r,
                force_limit=force_limit, alpha=alpha,
            )
            result = rollout_controller(test_ctrl, x0, t_end=t_end, dt=dt, force_limit=force_limit)
            m = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
            success = bool(m.get("success", False))
            cost = float(m.get("cost", 1e12))
            settle_t = m.get("settle_and_remain_time")
            score = cost + (0.0 if success else 2000.0)
            tune_results.append({
                "threshold": thr,
                "ridge": ridge,
                "alpha": alpha,
                "success": success,
                "settle_and_remain_time": settle_t,
                "cost": cost,
                "theta_ddot_at_x0_max_force": theta_ddot_pred,
            })
            if score < best_score:
                best_score = score
                best_ctrl = SINDyLQRController(
                    model=ctrl.model, k=ctrl.k, q=ctrl.q, r=ctrl.r,
                    force_limit=force_limit, alpha=alpha,
                )
                best_ctrl.metadata.update(ctrl.metadata)
                best_ctrl.metadata.update({
                    "alpha": alpha,
                    "K": ctrl.k.tolist(),
                    "tuning_results": None,  # will be filled below
                    "best_tuning_score": best_score,
                    "theta_ddot_at_x0_max_force": theta_ddot_pred,
                })

    if best_ctrl is None:
        raise RuntimeError("SINDy tuning failed for all threshold/ridge combinations.")

    best_ctrl.metadata["tuning_results"] = tune_results
    return best_ctrl


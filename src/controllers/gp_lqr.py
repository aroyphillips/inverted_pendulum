from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pickle
import time
import warnings

import numpy as np

from controllers.base import Controller
from eval.metrics import THETA_TOL, THETADOT_TOL, X_TOL, XDOT_TOL, rollout_metrics
from eval.rollout import rollout_controller
from models.dataset_views import ControlDataset, flatten_supervised, split_by_trajectory
from models.gp_dynamics import GPDynamicsConfig, GPDynamicsModel
from models.lqr_utils import safe_discrete_lqr_gain

# Minimum near-upright (|theta| < 0.1 rad) transitions required in the training set
# for the GP linearization to be considered reliable.
_MIN_NEAR_UPRIGHT_TRAIN = 10
_NEAR_UPRIGHT_THETA = 0.1  # rad
# RMSE threshold above which we warn that near-upright GP accuracy is poor.
_MAX_NEAR_UPRIGHT_RMSE = 1.0

# Q/R candidate grid for closed-loop tuning.
#
# The grid has two parts:
#   1. Bryson-rule anchor (Q_i = 1/tol_i^2, R = 1/u_max^2) and four scaled variants —
#      these encode the task settling tolerances with no ODE knowledge.
#   2. Additional candidates with weights not anchored to Bryson's rule, giving the
#      rollout search more room to find a gain that works for the learned (A, B).
#
# The best (Q, R, alpha) triple is selected by closed-loop rollout cross-validation.
# This is a heuristic-guided search, not a pure black-box optimizer.
def _bryson_q() -> np.ndarray:
    return np.diag([1.0/X_TOL**2, 1.0/XDOT_TOL**2, 1.0/THETA_TOL**2, 1.0/THETADOT_TOL**2])

def _bryson_r(force_limit: float = 5.0) -> np.ndarray:
    return np.array([[1.0 / force_limit**2]])

_DEFAULT_QR_GRID: List[Tuple[np.ndarray, np.ndarray]] = [
    # Bryson anchor and neighbourhood
    (_bryson_q() * 0.25, _bryson_r()),
    (_bryson_q(),        _bryson_r()),
    (_bryson_q() * 4.0,  _bryson_r()),
    (_bryson_q(),        _bryson_r() * 0.25),
    (_bryson_q(),        _bryson_r() * 4.0),
    # Non-Bryson alternatives: pole-priority, balanced, and high-authority
    (np.diag([4.0,  1.0, 400.0,  4.0]),   np.array([[0.04]])),
    (np.diag([10.0, 10.0, 200.0, 20.0]),  np.array([[0.1]])),
    (np.diag([1.0,  1.0, 4000.0, 50.0]),  np.array([[0.5]])),
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

        # Near-upright data quality check
        near_upright_train_mask = np.abs(x_train[:, 2]) < _NEAR_UPRIGHT_THETA
        n_near_upright_train = int(near_upright_train_mask.sum())
        if n_near_upright_train < _MIN_NEAR_UPRIGHT_TRAIN:
            warnings.warn(
                f"[GP-LQR] Only {n_near_upright_train} near-upright training points "
                f"(|theta|<{_NEAR_UPRIGHT_THETA} rad). GP linearization near the upright "
                "equilibrium may be unreliable. Consider collecting more data near upright.",
                stacklevel=2,
            )

        near_upright_val_mask = np.abs(x_val[:, 2]) < _NEAR_UPRIGHT_THETA
        x_val_upright = x_val[near_upright_val_mask]
        y_val_upright = y_val[near_upright_val_mask] if near_upright_val_mask.any() else None

        tic = time.perf_counter()
        self.model = GPDynamicsModel(config=config).fit(x_train, y_train, x_val=x_val, y_val=y_val)
        a, b, lin_diag = self.model.linearize_with_diagnostics()
        k, p, lqr_diag = safe_discrete_lqr_gain(a, b, self.q, self.r)
        self.k = k
        runtime = time.perf_counter() - tic

        if k is None:
            warnings.warn(
                f"[GP-LQR] DARE solve failed: {lqr_diag.get('failure_reason')}. "
                "Controller cannot act until a valid gain is found. "
                "Check metadata['lqr_diagnostics'] for details.",
                stacklevel=2,
            )

        # Near-upright RMSE and quality warning
        near_upright_rmse = None
        n_val_upright = int(x_val_upright.shape[0]) if x_val_upright is not None else 0
        if y_val_upright is not None and n_val_upright > 0:
            pred_up = self.model.predict_delta(x_val_upright)
            near_upright_rmse = float(np.sqrt(np.mean((pred_up - y_val_upright) ** 2)))
            if near_upright_rmse > _MAX_NEAR_UPRIGHT_RMSE:
                warnings.warn(
                    f"[GP-LQR] Near-upright validation RMSE = {near_upright_rmse:.4f} "
                    f"(threshold {_MAX_NEAR_UPRIGHT_RMSE}). GP accuracy near equilibrium "
                    "may be insufficient for reliable LQR.",
                    stacklevel=2,
                )

        self.metadata.update({
            "runtime_train_sec": runtime,
            "A": a.tolist(),
            "B": b.tolist(),
            "K": k.tolist() if k is not None else None,
            "alpha": self.alpha,
            "P": p.tolist() if p is not None else None,
            "Q": self.q.tolist(),
            "R": self.r.tolist(),
            "model": self.model.metadata,
            "near_upright_rmse": near_upright_rmse,
            "near_upright_val_points": n_val_upright,
            "near_upright_train_points": n_near_upright_train,
            "linearization_diagnostics": lin_diag,
            "lqr_diagnostics": lqr_diag,
        })
        return self

    def act(self, t: float, x: np.ndarray) -> float:
        if self.k is None:
            raise RuntimeError(
                "GPLQRController has no valid gain (DARE solve failed during fit). "
                "Check metadata['lqr_diagnostics'] for the failure reason."
            )
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
    """Fit GP dynamics once, then select Q/R and alpha by closed-loop rollout validation.

    Q/R candidates combine Bryson-rule anchors (derived from task tolerances) with
    additional non-Bryson alternatives. Selection is by rollout score, not by physics
    equations. The result is a heuristic-guided search over the gain space.
    """
    if qr_grid is None:
        qr_grid = _DEFAULT_QR_GRID
    if alpha_range is None:
        alpha_range = _DEFAULT_ALPHA_RANGE

    x0 = dataset.initial_state if initial_state is None else np.asarray(initial_state, dtype=float)

    base_ctrl = GPLQRController(force_limit=force_limit)
    base_ctrl.fit(dataset, config=GPDynamicsConfig(max_train_points=max_train_points, seed=seed), seed=seed)
    a = np.asarray(base_ctrl.metadata["A"])
    b = np.asarray(base_ctrl.metadata["B"])

    n_near_upright_train = base_ctrl.metadata.get("near_upright_train_points", 0)
    near_upright_rmse = base_ctrl.metadata.get("near_upright_rmse")
    if n_near_upright_train < _MIN_NEAR_UPRIGHT_TRAIN:
        warnings.warn(
            f"[GP-LQR] Only {n_near_upright_train} near-upright training points; "
            "tuning results may not be reliable.",
            stacklevel=2,
        )
    if near_upright_rmse is not None and near_upright_rmse > _MAX_NEAR_UPRIGHT_RMSE:
        warnings.warn(
            f"[GP-LQR] Near-upright RMSE = {near_upright_rmse:.4f}; "
            "GP accuracy near equilibrium is poor — tuning may not improve stability.",
            stacklevel=2,
        )

    best_score = float("inf")
    best_q: Optional[np.ndarray] = None
    best_r: Optional[np.ndarray] = None
    best_alpha: float = 1.0
    best_k: Optional[np.ndarray] = None
    best_p: Optional[np.ndarray] = None

    tune_results = []
    for q_cand, r_cand in qr_grid:
        k_cand, p_cand, lqr_diag = safe_discrete_lqr_gain(a, b, q_cand, r_cand)
        if k_cand is None:
            tune_results.append({
                "Q_diag": np.diag(q_cand).tolist(),
                "R": float(r_cand[0, 0]),
                "lqr_failure": lqr_diag.get("failure_reason"),
            })
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
                best_p = p_cand

    if best_k is None:
        lin_diag = base_ctrl.metadata.get("linearization_diagnostics", {})
        raise RuntimeError(
            "All Q/R candidates failed the DARE solve for the learned (A, B). "
            f"Linearization diagnostics: controllability_rank="
            f"{lin_diag.get('controllability_rank')}, "
            f"norm_B={lin_diag.get('norm_B')}, "
            f"A_finite={lin_diag.get('A_finite')}, B_finite={lin_diag.get('B_finite')}. "
            "Consider: (1) collecting more near-upright training data, "
            "(2) increasing max_train_points, (3) checking dataset for quality issues."
        )

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
        "P": best_p.tolist() if best_p is not None else None,
        "alpha": best_alpha,
        "tuning_results": tune_results,
        "best_tuning_score": best_score,
        "max_train_points": max_train_points,
        "seed": seed,
    })
    return best_ctrl

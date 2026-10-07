from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import cvxpy as cp

from controllers.base import Controller
from eval.metrics import THETA_TOL, THETADOT_TOL, X_TOL, XDOT_TOL, rollout_metrics
from eval.rollout import rollout_controller
from models.dataset_views import ControlDataset, flatten_supervised, split_by_trajectory
from models.lqr_utils import discrete_lqr_gain
from models.sindy import SINDyConfig, SINDyDynamicsModel


_DEFAULT_Q = np.diag([1.0/X_TOL**2, 1.0/XDOT_TOL**2, 1.0/THETA_TOL**2, 1.0/THETADOT_TOL**2])
_DEFAULT_R = np.array([[1.0 / 5.0**2]])

_DEFAULT_THRESHOLD_RIDGE_GRID: List[Tuple[float, float]] = [
    (1e-4, 1e-6),
    (1e-3, 1e-6),
    (5e-3, 1e-6),
    (1e-3, 1e-4),
    (5e-3, 1e-4),
]
_DEFAULT_HORIZON_GRID: List[int] = [20, 30, 50]


@dataclass
class SINDyMPCConfig:
    horizon: int = 30
    threshold: float = 5e-3
    ridge: float = 1e-4
    max_train_points: int = 100_000
    seed: int = 0


class SINDyMPCController(Controller):
    """Receding-horizon MPC using a SINDy-identified nonlinear model.

    At each control step the SINDy model is re-linearised at the current state
    (successive linearisation), yielding time-varying affine dynamics:

        x_{k+1} ≈ A_t x_k + b_t u_k + d_t

    A finite-horizon QP is then solved with explicit force constraints.  The
    terminal cost is the DARE solution at the upright equilibrium, which reduces
    the controller to LQR as the horizon N → ∞ and provides a stability
    certificate near the equilibrium.

    Falls back to a stored LQR gain when OSQP fails to find a feasible solution.
    """

    def __init__(
        self,
        model: SINDyDynamicsModel | None = None,
        config: SINDyMPCConfig | None = None,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
        p_terminal: np.ndarray | None = None,
        k_fallback: np.ndarray | None = None,
        force_limit: float = 5.0,
    ) -> None:
        super().__init__(name="sindy_mpc", force_limit=force_limit)
        self.model = model
        self.config = config or SINDyMPCConfig()
        self.q = _DEFAULT_Q.copy() if q is None else np.asarray(q, dtype=float)
        self.r = _DEFAULT_R.copy() if r is None else np.asarray(r, dtype=float)
        self.p_terminal = p_terminal
        self.k_fallback = None if k_fallback is None else np.asarray(k_fallback, dtype=float)

        # CVXPY problem (rebuilt after unpickling)
        self._problem: cp.Problem | None = None
        self._x0_par: cp.Parameter | None = None
        self._A_par: cp.Parameter | None = None
        self._B_par: cp.Parameter | None = None
        self._d_par: cp.Parameter | None = None
        self._U_var: cp.Variable | None = None

        if model is not None:
            self._build_problem()

    # ------------------------------------------------------------------
    # CVXPY problem construction
    # ------------------------------------------------------------------

    def _build_problem(self) -> None:
        """Build the horizon-N QP once; parameters are updated at each act() call."""
        N = self.config.horizon
        p_mat = self.p_terminal if self.p_terminal is not None else self.q

        self._x0_par = cp.Parameter(4, name="x0")
        self._A_par  = cp.Parameter((4, 4), name="A")
        self._B_par  = cp.Parameter(4, name="B")   # B column, shape (4,)
        self._d_par  = cp.Parameter(4, name="d")

        U = cp.Variable(N, name="U")
        X = cp.Variable((N + 1, 4), name="X")

        obj_terms = []
        constr = [X[0] == self._x0_par]

        for k in range(N):
            constr.append(
                X[k + 1] == self._A_par @ X[k] + U[k] * self._B_par + self._d_par
            )
            constr.append(U[k] >= -self.force_limit)
            constr.append(U[k] <= self.force_limit)
            obj_terms.append(cp.quad_form(X[k], self.q))
            obj_terms.append(float(self.r[0, 0]) * cp.square(U[k]))

        obj_terms.append(cp.quad_form(X[N], p_mat))

        self._U_var = U
        self._problem = cp.Problem(cp.Minimize(sum(obj_terms)), constr)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        dataset: ControlDataset,
        config: SINDyMPCConfig | None = None,
        seed: int = 0,
    ) -> "SINDyMPCController":
        cfg = config or self.config
        self.config = cfg

        train_idx, val_idx, _ = split_by_trajectory(dataset, seed=seed)
        x_train, _, x_next_train = flatten_supervised(dataset, train_idx)
        x_val,   _, x_next_val   = flatten_supervised(dataset, val_idx)

        sindy_cfg = SINDyConfig(
            threshold=cfg.threshold,
            ridge=cfg.ridge,
            max_train_points=cfg.max_train_points,
            seed=seed,
        )
        tic = time.perf_counter()
        self.model = SINDyDynamicsModel(config=sindy_cfg).fit(
            x_train, x_next_train, x_val=x_val, y_val=x_next_val
        )
        a, b = self.model.linearize()
        k, p = discrete_lqr_gain(a, b, self.q, self.r)
        self.p_terminal = p
        self.k_fallback = k
        runtime = time.perf_counter() - tic

        self._build_problem()

        self.metadata.update({
            "runtime_train_sec": runtime,
            "A": a.tolist(),
            "B": b.tolist(),
            "K_fallback": k.tolist(),
            "P": p.tolist(),
            "Q": self.q.tolist(),
            "R": self.r.tolist(),
            "horizon": cfg.horizon,
            "model": self.model.metadata,
        })
        return self

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def act(self, t: float, x: np.ndarray) -> float:
        if self._problem is None:
            raise RuntimeError("SINDyMPCController has not been fit.")

        x = np.asarray(x, dtype=float)

        # Successive linearisation at current state
        A, B = self.model.linearize(x_eq=x, u_eq=np.zeros(1))
        B_col = B[:, 0]                            # (4,)
        d = self.model.predict_next(x, np.zeros(1)) - A @ x  # affine offset (4,)

        self._x0_par.value = x
        self._A_par.value  = A
        self._B_par.value  = B_col
        self._d_par.value  = d

        tic = time.perf_counter()
        try:
            self._problem.solve(solver="OSQP", warm_start=True, verbose=False)
        except Exception:
            pass
        control_time = time.perf_counter() - tic

        status = self._problem.status
        if status in ("optimal", "optimal_inaccurate") and self._U_var.value is not None:
            u = float(self._U_var.value[0])
        else:
            # LQR fallback
            if self.k_fallback is not None:
                u = -float(np.ravel(self.k_fallback @ x)[0])
            else:
                u = 0.0

        return self.clipped(u)

    # ------------------------------------------------------------------
    # Persistence — exclude CVXPY objects from pickle
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in ("_problem", "_x0_par", "_A_par", "_B_par", "_d_par", "_U_var"):
            state.pop(key, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._problem = None
        self._x0_par = None
        self._A_par  = None
        self._B_par  = None
        self._d_par  = None
        self._U_var  = None
        if getattr(self, "model", None) is not None:
            self._build_problem()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "SINDyMPCController":
        with Path(path).open("rb") as f:
            return pickle.load(f)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train_sindy_mpc(
    dataset: ControlDataset,
    max_train_points: int = 100_000,
    seed: int = 0,
    force_limit: float = 5.0,
    initial_state: Optional[np.ndarray] = None,
    threshold_ridge_grid: Optional[List[Tuple[float, float]]] = None,
    horizon_grid: Optional[List[int]] = None,
    t_end: float = 20.0,
    dt: float = 0.02,
) -> SINDyMPCController:
    """Fit SINDy dynamics, then select (threshold, ridge, horizon) by closed-loop rollout.

    Grid searches over the Cartesian product of threshold/ridge pairs and prediction
    horizons, evaluating each candidate with a single closed-loop rollout from
    dataset.initial_state (or the supplied override).  The candidate with the lowest
    score (cost + 2000 if not successful) is returned.
    """
    if threshold_ridge_grid is None:
        threshold_ridge_grid = _DEFAULT_THRESHOLD_RIDGE_GRID
    if horizon_grid is None:
        horizon_grid = _DEFAULT_HORIZON_GRID

    x0 = dataset.initial_state if initial_state is None else np.asarray(initial_state, dtype=float)

    best_score = float("inf")
    best_ctrl: Optional[SINDyMPCController] = None
    tune_results = []

    for thr, ridge in threshold_ridge_grid:
        # Fit SINDy model once per (threshold, ridge)
        cfg_train = SINDyMPCConfig(
            threshold=thr, ridge=ridge,
            max_train_points=max_train_points, seed=seed,
        )
        base_ctrl = SINDyMPCController(
            config=cfg_train,
            q=_DEFAULT_Q, r=_DEFAULT_R,
            force_limit=force_limit,
        )
        try:
            base_ctrl.fit(dataset, config=cfg_train, seed=seed)
        except Exception as exc:
            tune_results.append({"threshold": thr, "ridge": ridge, "error": str(exc)})
            continue

        a = np.asarray(base_ctrl.metadata["A"])
        b = np.asarray(base_ctrl.metadata["B"])
        theta_ddot_pred = float(np.ravel(a @ x0 + b.ravel() * force_limit - x0)[3])

        for horizon in horizon_grid:
            # Rebuild problem with this horizon (keep same fitted model & P)
            test_cfg = SINDyMPCConfig(
                threshold=thr, ridge=ridge,
                max_train_points=max_train_points, seed=seed,
                horizon=horizon,
            )
            test_ctrl = SINDyMPCController(
                model=base_ctrl.model,
                config=test_cfg,
                q=base_ctrl.q,
                r=base_ctrl.r,
                p_terminal=base_ctrl.p_terminal,
                k_fallback=base_ctrl.k_fallback,
                force_limit=force_limit,
            )

            result = rollout_controller(test_ctrl, x0, t_end=t_end, dt=dt, force_limit=force_limit)
            m = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
            success   = bool(m.get("success", False))
            cost      = float(m.get("cost", 1e12))
            settle_t  = m.get("settle_and_remain_time")
            score     = cost + (0.0 if success else 2000.0)

            tune_results.append({
                "threshold": thr,
                "ridge": ridge,
                "horizon": horizon,
                "success": success,
                "settle_and_remain_time": settle_t,
                "cost": cost,
                "theta_ddot_at_x0_max_force": theta_ddot_pred,
            })

            if score < best_score:
                best_score = score
                best_ctrl = SINDyMPCController(
                    model=base_ctrl.model,
                    config=test_cfg,
                    q=base_ctrl.q,
                    r=base_ctrl.r,
                    p_terminal=base_ctrl.p_terminal,
                    k_fallback=base_ctrl.k_fallback,
                    force_limit=force_limit,
                )
                best_ctrl.metadata.update(base_ctrl.metadata)
                best_ctrl.metadata.update({
                    "horizon": horizon,
                    "tuning_results": None,
                    "best_tuning_score": best_score,
                    "theta_ddot_at_x0_max_force": theta_ddot_pred,
                })

    if best_ctrl is None:
        raise RuntimeError("SINDy-MPC tuning failed for all threshold/ridge/horizon combinations.")

    best_ctrl.metadata["tuning_results"] = tune_results
    return best_ctrl

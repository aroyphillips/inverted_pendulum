from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pickle
import time

import numpy as np

from controllers.base import Controller
from models.hankel import deepc_partitions, rank_report


@dataclass
class DeePCConfig:
    t_ini: int = 3
    horizon: int = 50
    lambda_g: float = 1e-4
    lambda_y_ini: float = 1e4
    lambda_u_rate: float = 1e-3    # input-rate smoothness penalty
    lambda_terminal: float = 10.0  # multiplier on terminal state cost
    max_hankel_columns: int = 2000
    seed: int = 0


# Default Q/R with stronger theta and theta_dot penalties.
_DEFAULT_Q = np.diag([1.0, 0.2, 100.0, 10.0])
_DEFAULT_R = np.array([[0.05]])


class DeePCController(Controller):
    def __init__(
        self,
        config: DeePCConfig | None = None,
        force_limit: float = 5.0,
        fallback: Optional[Controller] = None,
        q: np.ndarray | None = None,
        r: np.ndarray | None = None,
    ) -> None:
        super().__init__(name="deepc", force_limit=force_limit)
        self.config = config or DeePCConfig()
        self.fallback = fallback
        self.q = _DEFAULT_Q.copy() if q is None else np.asarray(q, dtype=float)
        self.r = _DEFAULT_R.copy() if r is None else np.asarray(r, dtype=float)
        self.Up: np.ndarray | None = None
        self.Uf: np.ndarray | None = None
        self.Yp: np.ndarray | None = None
        self.Yf: np.ndarray | None = None
        self._problem = None
        self._g = None
        self._slack = None
        self._ini_param = None          # combined [u_ini; y_ini] parameter
        self._last_g = None
        self._u_ini_buf: List[float] = []  # last t_ini control outputs (for past-input constraint)
        self.solve_times: list[float] = []
        self.statuses: list[str] = []

    def fit(self, states: np.ndarray, inputs: np.ndarray) -> "DeePCController":
        states = np.asarray(states, dtype=float)
        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim == 2:
            inputs = inputs[:, :, None]
        self.Up, self.Uf, self.Yp, self.Yf = deepc_partitions(
            inputs,
            states,
            self.config.t_ini,
            self.config.horizon,
            max_columns=self.config.max_hankel_columns,
            seed=self.config.seed,
        )
        self.metadata.update(
            {
                "config": self.config.__dict__,
                "hankel_columns": int(self.Yp.shape[1]),
                "Yp_rank": rank_report(self.Yp).__dict__,
                "Uf_rank": rank_report(self.Uf).__dict__,
                "state_dim": int(states.shape[-1]),
                "input_dim": int(inputs.shape[-1]),
            }
        )
        self._build_problem()
        return self

    def _build_problem(self) -> None:
        try:
            import cvxpy as cp
        except ImportError as exc:
            raise ImportError(
                "DeePC requires cvxpy and osqp. Install them with `pip install cvxpy osqp`."
            ) from exc
        if self.Yp is None or self.Yf is None or self.Uf is None or self.Up is None:
            raise RuntimeError("Call fit before building the DeePC problem.")

        cols = self.Yp.shape[1]
        state_dim = self.q.shape[0]
        input_dim = self.r.shape[0]
        t_ini = self.config.t_ini
        horizon = self.config.horizon

        qbar = np.kron(np.eye(horizon), self.q)
        rbar = np.kron(np.eye(horizon), self.r)

        # Combined past-data constraint: [Up; Yp] @ g = [u_ini; y_ini] + slack
        ini_matrix = np.vstack([self.Up, self.Yp])     # ((input+state)*t_ini, cols)
        ini_dim = ini_matrix.shape[0]

        # Input-rate difference matrix: D @ u_future → consecutive differences
        D_rate = np.zeros((horizon - 1, horizon * input_dim), dtype=float)
        for i in range(horizon - 1):
            D_rate[i, i * input_dim:(i + 1) * input_dim] = -1.0
            D_rate[i, (i + 1) * input_dim:(i + 2) * input_dim] = 1.0

        g = cp.Variable(cols)
        slack = cp.Variable(ini_dim)
        ini_param = cp.Parameter(ini_dim)
        y_future = self.Yf @ g
        u_future = self.Uf @ g

        # Terminal state: last state_dim entries of y_future
        y_terminal = y_future[(horizon - 1) * state_dim:]

        objective = (
            cp.quad_form(y_future, qbar)
            + cp.quad_form(u_future, rbar)
            + self.config.lambda_g * cp.sum_squares(g)
            + self.config.lambda_y_ini * cp.sum_squares(slack)
            + self.config.lambda_terminal * cp.quad_form(y_terminal, self.q)
            + self.config.lambda_u_rate * cp.sum_squares(D_rate @ u_future)
        )
        constraints = [
            ini_matrix @ g == ini_param + slack,
            u_future <= self.force_limit,
            u_future >= -self.force_limit,
        ]
        self._problem = cp.Problem(cp.Minimize(objective), constraints)
        self._g = g
        self._slack = slack
        self._ini_param = ini_param
        self._input_dim = input_dim
        self._state_dim = state_dim
        self.metadata["qp_variables"] = {
            "g": int(cols),
            "ini_dim": ini_dim,
            "state_dim": int(state_dim),
            "input_dim": int(input_dim),
        }

    def reset(self, x0: np.ndarray) -> None:
        super().reset(x0)
        if self.fallback is not None:
            self.fallback.reset(x0)
        self._last_g = None
        # Initialize past-input buffer with zeros
        t_ini = self.config.t_ini
        self._u_ini_buf = [0.0] * t_ini
        self.solve_times.clear()
        self.statuses.clear()

    def _fallback_action(self, t: float, x: np.ndarray) -> float:
        if self.fallback is None:
            return 0.0
        return self.fallback.act(t, x)

    def act(self, t: float, x: np.ndarray) -> float:
        if self._problem is None or self._ini_param is None or self._g is None or self.Uf is None:
            return self.clipped(self._fallback_action(t, x))

        t_ini = self.config.t_ini
        input_dim = getattr(self, "_input_dim", 1)
        state_dim = getattr(self, "_state_dim", 4)

        # Build y_ini from current state (repeat if t_ini > 1)
        y_ini = np.asarray(x, dtype=float).reshape(-1)
        if t_ini > 1:
            y_ini = np.tile(y_ini, t_ini)

        # Build u_ini from history buffer
        u_ini = np.array(self._u_ini_buf[-t_ini:], dtype=float).reshape(-1)
        if u_ini.size < input_dim * t_ini:
            u_ini = np.pad(u_ini, (input_dim * t_ini - u_ini.size, 0))

        self._ini_param.value = np.concatenate([u_ini, y_ini])

        tic = time.perf_counter()
        try:
            self._problem.solve(solver="OSQP", warm_start=True, verbose=False)
            status = str(self._problem.status)
        except Exception as exc:
            status = f"error:{type(exc).__name__}"
        elapsed = time.perf_counter() - tic
        self.solve_times.append(elapsed)
        self.statuses.append(status)

        if status not in {"optimal", "optimal_inaccurate"} or self._g.value is None:
            u_fb = self.clipped(self._fallback_action(t, x))
            self._u_ini_buf.append(u_fb)
            return u_fb

        self._last_g = np.asarray(self._g.value, dtype=float)
        first_u = self.clipped(float((self.Uf @ self._last_g)[0]))
        self._u_ini_buf.append(first_u)
        return first_u

    def diagnostics(self) -> dict:
        feasible = [s in {"optimal", "optimal_inaccurate"} for s in self.statuses]
        return {
            "qp_status_counts": {s: self.statuses.count(s) for s in sorted(set(self.statuses))},
            "qp_feasibility_rate": float(np.mean(feasible)) if feasible else None,
            "avg_solve_time_sec": float(np.mean(self.solve_times)) if self.solve_times else None,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config,
            "force_limit": self.force_limit,
            "q": self.q,
            "r": self.r,
            "Up": self.Up,
            "Uf": self.Uf,
            "Yp": self.Yp,
            "Yf": self.Yf,
            "metadata": self.metadata,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path, fallback: Optional[Controller] = None) -> "DeePCController":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        ctrl = cls(
            config=payload["config"],
            force_limit=payload["force_limit"],
            fallback=fallback,
            q=payload["q"],
            r=payload["r"],
        )
        ctrl.Up = payload["Up"]
        ctrl.Uf = payload["Uf"]
        ctrl.Yp = payload["Yp"]
        ctrl.Yf = payload["Yf"]
        ctrl.metadata.update(payload.get("metadata", {}))
        ctrl._build_problem()
        return ctrl


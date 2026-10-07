from __future__ import annotations

from pathlib import Path
from typing import Optional

import pickle

import numpy as np

from controllers.base import Controller
from controllers.deepc import DeePCConfig, DeePCController
from eval.metrics import THETA_TOL, THETADOT_TOL, X_TOL, XDOT_TOL
from models.koopman import KoopmanLift, edmdc_diagnostics


class KoopmanDeePCController(DeePCController):
    def __init__(
        self,
        config: DeePCConfig | None = None,
        force_limit: float = 5.0,
        fallback: Optional[Controller] = None,
        feature_set: str = "standard",
    ) -> None:
        # Q: Bryson's rule on original 4 states, small regularisation on lifted dims.
        # Original-state weights: Q_i = 1/tol_i^2 from task settling tolerances.
        # Lifted dims 4+ get a uniform weight equal to the mean of the original weights.
        n_extra = 4 if feature_set == "rich" else 0
        n_lifted = 13 + n_extra
        _bryson_diag = [1.0/X_TOL**2, 1.0/XDOT_TOL**2, 1.0/THETA_TOL**2, 1.0/THETADOT_TOL**2]
        _lifted_weight = float(np.mean(_bryson_diag))
        q = np.diag(_bryson_diag + [_lifted_weight] * (n_lifted - 4))
        super().__init__(config=config, force_limit=force_limit, fallback=fallback, q=q, r=np.array([[1.0/force_limit**2]]))
        self.name = "koopman_deepc"
        self.lift = KoopmanLift(standardize=True, feature_set=feature_set)

    def fit(self, states: np.ndarray, inputs: np.ndarray) -> "KoopmanDeePCController":
        z = self.lift.fit_transform(states)
        super().fit(z, inputs)
        self.name = "koopman_deepc"
        self.metadata["lift_feature_names"] = self.lift.feature_names
        self.metadata["edmdc_diagnostics"] = edmdc_diagnostics(z, inputs)
        return self

    def _fallback_action(self, t: float, x: np.ndarray) -> float:
        if self.fallback is None:
            return 0.0
        raw = getattr(self, "_raw_fallback_state", x)
        return self.fallback.act(t, raw)

    def act(self, t: float, x: np.ndarray) -> float:
        self._raw_fallback_state = np.asarray(x, dtype=float)
        z = self.lift.transform(np.asarray(x, dtype=float)[None, :])[0]
        return super().act(t, z)

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
            "lift": self.lift,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: Path, fallback: Optional[Controller] = None) -> "KoopmanDeePCController":
        with Path(path).open("rb") as f:
            payload = pickle.load(f)
        ctrl = cls(config=payload["config"], force_limit=payload["force_limit"], fallback=fallback)
        ctrl.q = payload["q"]
        ctrl.r = payload["r"]
        ctrl.Up = payload["Up"]
        ctrl.Uf = payload["Uf"]
        ctrl.Yp = payload["Yp"]
        ctrl.Yf = payload["Yf"]
        ctrl.metadata.update(payload.get("metadata", {}))
        ctrl.lift = payload["lift"]
        ctrl._build_problem()
        return ctrl

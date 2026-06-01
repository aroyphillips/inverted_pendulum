from __future__ import annotations

from typing import Dict

import numpy as np

from controllers.base import Controller
from eval.metrics import rollout_metrics
from eval.rollout import RolloutResult, rollout_controller


def compare_controllers(
    controllers: Dict[str, Controller],
    initial_state: np.ndarray,
    t_end: float = 10.0,
    dt: float = 0.02,
    force_limit: float = 1.0,
) -> tuple[dict, dict[str, RolloutResult]]:
    results = {}
    rollouts = {}
    for name, controller in controllers.items():
        result = rollout_controller(
            controller,
            initial_state,
            t_end=t_end,
            dt=dt,
            force_limit=force_limit,
        )
        metrics = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
        if hasattr(controller, "diagnostics"):
            metrics.update(getattr(controller, "diagnostics")())
        results[name] = {
            "metrics": metrics,
            "metadata": getattr(controller, "metadata", {}),
            "final_state": result.y[:, -1].tolist(),
        }
        rollouts[name] = result
    return results, rollouts


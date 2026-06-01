from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import json
import time

import numpy as np


def clip_force(u: float, force_limit: float = 1.0) -> float:
    return float(np.clip(float(u), -float(force_limit), float(force_limit)))


@dataclass
class Controller(ABC):
    """Minimal interface shared by all feedback policies."""

    name: str
    force_limit: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reset(self, x0: np.ndarray) -> None:
        self.metadata.setdefault("last_reset_state", np.asarray(x0, dtype=float).tolist())

    @abstractmethod
    def act(self, t: float, x: np.ndarray) -> float:
        """Return the force command at time ``t`` for state ``x``."""

    def __call__(self, t: float, x: np.ndarray) -> float:
        return self.act(t, x)

    def clipped(self, u: float) -> float:
        return clip_force(u, self.force_limit)


class ZeroController(Controller):
    def __init__(self, force_limit: float = 1.0) -> None:
        super().__init__(name="zero", force_limit=force_limit)

    def act(self, t: float, x: np.ndarray) -> float:
        return 0.0


class TrainingTimer:
    def __enter__(self) -> "TrainingTimer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.elapsed = time.perf_counter() - self.start


def save_controller_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_controller_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def optional_import(module_name: str, package_hint: Optional[str] = None):
    try:
        return __import__(module_name)
    except ImportError as exc:
        package = package_hint or module_name
        raise ImportError(
            f"Optional dependency '{package}' is required for this controller. "
            f"Install it with `pip install {package}` or `pip install -r requirements.txt`."
        ) from exc


from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/inverted_pendulum_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/inverted_pendulum_cache")


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_summary_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def artifact_path(name: str) -> Path:
    mapping = {
        "pid": PROJECT_ROOT / "artifacts/controllers/pid.json",
        "deepc": PROJECT_ROOT / "artifacts/controllers/deepc.pkl",
        "koopman_deepc": PROJECT_ROOT / "artifacts/controllers/koopman_deepc.pkl",
        "gp_lqr": PROJECT_ROOT / "artifacts/controllers/gp_lqr.pkl",
        "sindy_lqr": PROJECT_ROOT / "artifacts/controllers/sindy_lqr.pkl",
        "sindy_mpc": PROJECT_ROOT / "artifacts/controllers/sindy_mpc.pkl",
        "rl_sac": PROJECT_ROOT / "artifacts/rl/sac.zip",
    }
    return mapping[name]

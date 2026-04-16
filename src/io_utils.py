import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def save_dataset_npz(output_path: Path, states: np.ndarray, inputs: np.ndarray, t: np.ndarray, initial_state: np.ndarray) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=states,
        inputs=inputs,
        t=t,
        initial_state=initial_state,
    )


def load_dataset_npz(path: Path) -> Dict[str, np.ndarray]:
    arr = np.load(Path(path), allow_pickle=False)
    return {
        "states": arr["states"],
        "inputs": arr["inputs"],
        "t": arr["t"],
        "initial_state": arr["initial_state"],
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

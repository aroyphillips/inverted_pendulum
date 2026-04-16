from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np

from config import PendulumParams, SimulationSettings
from dataset import DatasetConfig, generate_random_walk_dataset
from initial_state import derive_initial_state_from_impulse_crossing
from io_utils import save_dataset_npz, save_json
from pe import persistent_excitation_check, readiness_checks


@dataclass
class PipelineOutputs:
    dataset_path: Path
    metadata_path: Path
    report_path: Path


def run_full_pipeline(
    output_dir: Path,
    theta_target_deg: float = 10.0,
    impulse_force: float = 0.01,
    params: PendulumParams = PendulumParams(),
    sim_settings: SimulationSettings = SimulationSettings(),
    dataset_cfg: DatasetConfig = DatasetConfig(),
    pe_order: int = 20,
) -> PipelineOutputs:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    init = derive_initial_state_from_impulse_crossing(
        theta_target_deg=theta_target_deg,
        params=params,
        settings=sim_settings,
        impulse_force=impulse_force,
    )
    if not init.success:
        raise RuntimeError(f"Failed to derive initial state: {init.diagnostic}")

    ds = generate_random_walk_dataset(
        y0=init.initial_state,
        params=params,
        settings=sim_settings,
        cfg=dataset_cfg,
    )

    pe_checks = [persistent_excitation_check(u, order=pe_order) for u in ds.inputs]
    pass_indices = [i for i, c in enumerate(pe_checks) if c.is_persistently_exciting]
    first_pass = pass_indices[0] if pass_indices else None

    readiness = readiness_checks(ds.states, ds.inputs, ds.t)

    dataset_path = output_dir / "random_walk_dataset.npz"
    metadata_path = output_dir / "random_walk_dataset.metadata.json"
    report_path = output_dir / "random_walk_dataset.report.json"

    save_dataset_npz(dataset_path, ds.states, ds.inputs, ds.t, init.initial_state)

    metadata: Dict[str, Any] = {
        "state_order": ["x", "x_dot", "theta", "theta_dot"],
        "units": {
            "x": "m",
            "x_dot": "m/s",
            "theta": "rad",
            "theta_dot": "rad/s",
            "force": "N",
            "time": "s",
        },
        "num_trajectories": int(ds.states.shape[0]),
        "num_samples": int(ds.states.shape[1]),
        "dt": float(ds.t[1] - ds.t[0]),
        "params": asdict(params),
        "simulation_settings": asdict(sim_settings),
        "dataset_config": asdict(dataset_cfg),
        "derived_initial_state": init.initial_state.tolist(),
        "derived_initial_state_crossing_time": float(init.crossing_time),
        "theta_target_deg": float(theta_target_deg),
        "impulse_force": float(impulse_force),
        "attempts_used": int(ds.num_attempts),
        "seeds": ds.seeds.tolist(),
    }
    save_json(metadata_path, metadata)

    report: Dict[str, Any] = {
        "pe_order": int(pe_order),
        "num_pe_pass": int(len(pass_indices)),
        "first_pe_pass_index": first_pass,
        "first_pe_pass_rank": int(pe_checks[first_pass].rank) if first_pass is not None else None,
        "first_pe_pass_condition_number": float(pe_checks[first_pass].condition_number)
        if first_pass is not None
        else None,
        "readiness_checks": readiness,
    }
    save_json(report_path, report)

    return PipelineOutputs(
        dataset_path=dataset_path,
        metadata_path=metadata_path,
        report_path=report_path,
    )

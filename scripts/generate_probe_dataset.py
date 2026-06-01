#!/usr/bin/env python3
"""Generate a probe dataset with higher-authority inputs for controller training.

Produces two trajectory families and saves them combined into one npz file:
  1. Recovery trajectories: start from the agreed 10-degree initial state,
     using Gaussian random-walk inputs with force_max up to 8 N.  These
     expose the system to recovery manoeuvres that the original ±1 N dataset
     cannot contain.
  2. Near-upright regulation trajectories: start from states jittered near
     the upright equilibrium (|theta| ≤ 0.05 rad), force_max = 2 N.  These
     give data-driven methods fine-grained information about stabilisation
     behaviour close to the goal.

Output npz layout (compatible with load_control_dataset):
  states          (N+M, T, 4)  — all trajectories
  inputs          (N+M, T)     — all forces (1D per trajectory)
  t               (T,)         — shared time grid
  initial_state   (4,)         — the recovery initial state
  recovery_states (N, T, 4)    — recovery trajectories only
  recovery_inputs (N, T)       — recovery inputs only
  regulation_states (M, T, 4)  — near-upright trajectories only
  regulation_inputs (M, T)     — near-upright inputs only
"""
from __future__ import annotations

import argparse

import numpy as np

import common  # noqa: F401 – adds src/ to sys.path

from config import PendulumParams, SimulationSettings
from dataset import DatasetConfig, generate_random_walk_dataset
from initial_state import derive_initial_state_from_impulse_crossing
from models.dataset_views import load_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate probe dataset for controller training.")
    parser.add_argument("--out", default="data/probe/probe_dataset.npz")
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz",
                        help="Source dataset for initial state.")
    parser.add_argument("--num-traj", type=int, default=500,
                        help="Number of recovery trajectories.")
    parser.add_argument("--num-reg-traj", type=int, default=200,
                        help="Number of near-upright regulation trajectories.")
    parser.add_argument("--force-max", type=float, default=8.0,
                        help="Force cap for recovery trajectories.")
    parser.add_argument("--reg-force-max", type=float, default=2.0,
                        help="Force cap for near-upright trajectories.")
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Number of time samples per trajectory (sets dt = t_end/num_samples).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers (1 = serial).")
    args = parser.parse_args()

    params = PendulumParams()
    settings = SimulationSettings(t_end=args.t_end, num_samples=args.num_samples)

    # Load agreed initial state from reference dataset.
    src = load_control_dataset(args.dataset)
    x0_recovery = src.initial_state
    print(f"Recovery initial state: {x0_recovery}")
    print(f"  theta = {np.degrees(x0_recovery[2]):.2f} deg,  theta_dot = {x0_recovery[3]:.4f} rad/s")

    # ------------------------------------------------------------------ #
    # 1. Recovery trajectories (high force)                               #
    # ------------------------------------------------------------------ #
    print(f"\nGenerating {args.num_traj} recovery trajectories "
          f"(force_max={args.force_max} N, t_end={args.t_end} s)…")
    recovery_cfg = DatasetConfig(
        num_trajectories=args.num_traj,
        dt_force=0.05,
        sigma_force=0.5,
        force_max=args.force_max,
        enforce_sign_balance=True,
        max_attempts=args.num_traj * 20,
        base_seed=args.seed,
        num_workers=args.workers,
    )
    rec = generate_random_walk_dataset(x0_recovery, params, settings, recovery_cfg)
    print(f"  Generated {rec.states.shape[0]} trajectories, "
          f"shape states={rec.states.shape}, inputs={rec.inputs.shape}")

    # ------------------------------------------------------------------ #
    # 2. Near-upright regulation trajectories (low force, near zero)      #
    # ------------------------------------------------------------------ #
    print(f"\nGenerating {args.num_reg_traj} near-upright regulation trajectories "
          f"(force_max={args.reg_force_max} N)…")
    rng = np.random.default_rng(args.seed + 1000)
    jitter_scale = np.array([0.02, 0.02, 0.05, 0.05])
    reg_states_list = []
    reg_inputs_list = []

    for i in range(args.num_reg_traj):
        x0_reg = rng.normal(scale=jitter_scale)
        # Ensure theta stays within ±0.05 rad
        x0_reg[2] = float(np.clip(x0_reg[2], -0.05, 0.05))
        reg_cfg = DatasetConfig(
            num_trajectories=1,
            dt_force=0.05,
            sigma_force=0.2,
            force_max=args.reg_force_max,
            enforce_sign_balance=False,
            max_attempts=50,
            base_seed=args.seed + 2000 + i,
            num_workers=1,
        )
        try:
            r = generate_random_walk_dataset(x0_reg, params, settings, reg_cfg)
            reg_states_list.append(r.states[0])
            reg_inputs_list.append(r.inputs[0])
        except RuntimeError:
            pass  # skip if generation fails for this jitter

    if len(reg_states_list) == 0:
        print("  Warning: no regulation trajectories generated; skipping regulation split.")
        reg_states = np.zeros((0, rec.states.shape[1], 4), dtype=float)
        reg_inputs = np.zeros((0, rec.inputs.shape[1]), dtype=float)
    else:
        reg_states = np.stack(reg_states_list, axis=0)
        reg_inputs = np.stack(reg_inputs_list, axis=0)
    print(f"  Generated {reg_states.shape[0]} regulation trajectories.")

    # ------------------------------------------------------------------ #
    # 3. Combine and save                                                  #
    # ------------------------------------------------------------------ #
    all_states = np.concatenate([rec.states, reg_states], axis=0)
    all_inputs = np.concatenate([rec.inputs, reg_inputs], axis=0)
    t = rec.t

    from pathlib import Path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        states=all_states,
        inputs=all_inputs,
        t=t,
        initial_state=x0_recovery,
        recovery_states=rec.states,
        recovery_inputs=rec.inputs,
        regulation_states=reg_states,
        regulation_inputs=reg_inputs,
    )

    import json
    meta = {
        "num_recovery_trajectories": int(rec.states.shape[0]),
        "num_regulation_trajectories": int(reg_states.shape[0]),
        "num_trajectories_total": int(all_states.shape[0]),
        "num_samples": int(t.size),
        "dt": float(np.mean(np.diff(t))),
        "t_end": args.t_end,
        "recovery": {
            "force_max": args.force_max,
            "dt_force": 0.05,
            "sigma_force": 0.5,
        },
        "regulation": {
            "force_max": args.reg_force_max,
            "dt_force": 0.05,
            "sigma_force": 0.2,
        },
        "initial_state": x0_recovery.tolist(),
        "seed": args.seed,
    }
    meta_path = out_path.with_suffix(".metadata.json")
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\nSaved probe dataset to {out_path}")
    print(f"  Total trajectories: {all_states.shape[0]} "
          f"({rec.states.shape[0]} recovery + {reg_states.shape[0]} regulation)")
    print(f"  Shape: states={all_states.shape}, inputs={all_inputs.shape}, t={t.shape}")
    print(f"  dt = {float(np.mean(np.diff(t))):.6f} s")


if __name__ == "__main__":
    main()

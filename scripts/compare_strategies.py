#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import artifact_path, write_json, write_summary_csv
from controllers.deepc import DeePCController
from controllers.gp_lqr import GPLQRController
from controllers.koopman_deepc import KoopmanDeePCController
from controllers.pid import PIDController
from controllers.rl_sac import RLController
from controllers.sindy_lqr import SINDyLQRController
from eval.metrics import rollout_metrics
from eval.rollout import rollout_controller
from models.dataset_views import load_control_dataset


TRAIN_COMMANDS = {
    "pid": "python scripts/train_pid.py --dataset data/full_example/random_walk_dataset.npz",
    "deepc": "python scripts/train_deepc.py --dataset data/full_example/random_walk_dataset.npz",
    "koopman_deepc": "python scripts/train_koopman_deepc.py --dataset data/full_example/random_walk_dataset.npz",
    "gp_lqr": "python scripts/train_gp_lqr.py --dataset data/full_example/random_walk_dataset.npz --max-train-points 5000",
    "sindy_lqr": "python scripts/train_sindy_lqr.py --dataset data/full_example/random_walk_dataset.npz",
    "rl_sac": "python scripts/train_rl_sac.py --timesteps 200000",
}


def load_controller(name: str):
    path = artifact_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact for {name}: {path}. Train it with `{TRAIN_COMMANDS[name]}`.")
    if name == "pid":
        return PIDController.load(path)
    if name == "deepc":
        fallback = PIDController.load(artifact_path("pid")) if artifact_path("pid").exists() else PIDController()
        return DeePCController.load(path, fallback=fallback)
    if name == "koopman_deepc":
        fallback = PIDController.load(artifact_path("pid")) if artifact_path("pid").exists() else PIDController()
        return KoopmanDeePCController.load(path, fallback=fallback)
    if name == "gp_lqr":
        return GPLQRController.load(path)
    if name == "sindy_lqr":
        return SINDyLQRController.load(path)
    if name == "rl_sac":
        return RLController.load_sac(path)
    raise ValueError(f"Unknown controller: {name}")


def plot_state_rollouts(path: Path, rollouts: dict) -> None:
    import matplotlib.pyplot as plt

    labels = ["x (m)", "x_dot (m/s)", "theta (rad)", "theta_dot (rad/s)"]
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for name, result in rollouts.items():
        for i, ax in enumerate(axes):
            ax.plot(result.t, result.y[i, :], label=name, linewidth=1.4)
            ax.set_ylabel(labels[i])
            ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_control_rollouts(path: Path, rollouts: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 3.5))
    for name, result in rollouts.items():
        ax.plot(result.t, result.u, label=name, linewidth=1.4)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("force (N)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained inverted-pendulum control strategies.")
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz")
    parser.add_argument(
        "--controllers",
        nargs="+",
        default=["pid", "koopman_deepc", "gp_lqr", "rl_sac", "sindy_lqr", "deepc"],
        choices=["pid", "koopman_deepc", "gp_lqr", "rl_sac", "sindy_lqr", "deepc"],
    )
    parser.add_argument("--initial-state-from-dataset", action="store_true")
    parser.add_argument("--initial-state", nargs=4, type=float)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--force-limit", type=float, default=5.0)
    parser.add_argument("--out", default="artifacts/comparison/results.json")
    args = parser.parse_args()

    dataset = load_control_dataset(args.dataset)
    if args.initial_state is not None:
        x0 = np.asarray(args.initial_state, dtype=float)
    else:
        x0 = dataset.initial_state if args.initial_state_from_dataset else dataset.states[0, 0, :]

    results = {}
    rows = []
    rollouts = {}
    for name in args.controllers:
        controller = load_controller(name)
        # Use the force_limit baked into the artifact; fall back to CLI arg.
        eff_force_limit = float(getattr(controller, "force_limit", args.force_limit))
        result = rollout_controller(controller, x0, t_end=args.t_end, dt=args.dt, force_limit=eff_force_limit)
        metrics = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
        if hasattr(controller, "diagnostics"):
            metrics.update(getattr(controller, "diagnostics")())
        train_runtime = getattr(controller, "metadata", {}).get("runtime_train_sec")
        if train_runtime is not None:
            metrics["runtime_train_sec"] = train_runtime
        metrics["force_limit"] = eff_force_limit
        results[name] = {
            "metrics": metrics,
            "metadata": getattr(controller, "metadata", {}),
            "final_state": result.y[:, -1].tolist(),
        }
        rows.append({"controller": name, **metrics})
        rollouts[name] = result
        settle_t = metrics.get("settle_and_remain_time")
        print(
            f"{name}: success={metrics.get('success')} settle_time={settle_t} "
            f"max_theta={metrics.get('max_abs_theta'):.4f} force_limit={eff_force_limit} "
            f"cost={metrics.get('cost'):.4g}"
        )

    out_path = Path(args.out)
    write_json(out_path, results)
    write_summary_csv(out_path.with_name("summary.csv"), rows)
    plot_state_rollouts(out_path.with_name("state_rollouts.png"), rollouts)
    plot_control_rollouts(out_path.with_name("control_rollouts.png"), rollouts)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

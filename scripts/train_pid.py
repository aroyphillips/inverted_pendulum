#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import artifact_path, write_json
from controllers.base import TrainingTimer
from controllers.pid import tune_pid
from eval.metrics import rollout_metrics
from eval.rollout import rollout_controller
from models.dataset_views import load_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune and save a PID baseline controller.")
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz")
    parser.add_argument("--out", default=str(artifact_path("pid")))
    parser.add_argument("--mode", choices=["theta_only", "theta_plus_cart"], default="theta_plus_cart")
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--force-limit", type=float, default=5.0)
    args = parser.parse_args()

    dataset = load_control_dataset(args.dataset)
    with TrainingTimer() as timer:
        controller, history = tune_pid(
            dataset.initial_state,
            mode=args.mode,
            n_trials=args.trials,
            seed=args.seed,
            force_limit=args.force_limit,
            t_end=args.t_end,
            dt=args.dt,
        )
    controller.metadata["runtime_train_sec"] = timer.elapsed
    controller.save(args.out)
    result = rollout_controller(controller, dataset.initial_state, t_end=args.t_end, dt=args.dt, force_limit=args.force_limit)
    metrics = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
    report = {"controller": controller.to_payload(), "metrics": metrics, "history_top10": sorted(history, key=lambda r: r["score"])[:10]}
    write_json(str(args.out).replace(".json", ".report.json"), report)
    print(f"saved {args.out}")
    print(f"success={metrics.get('success')} cost={metrics.get('cost'):.6g} max_abs_theta={metrics.get('max_abs_theta'):.6g}")


if __name__ == "__main__":
    main()


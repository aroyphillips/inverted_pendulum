#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import artifact_path, write_json
from controllers.sindy_mpc import train_sindy_mpc
from models.dataset_views import load_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit SINDy dynamics and tune the induced MPC controller."
    )
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz")
    parser.add_argument("--out", default=str(artifact_path("sindy_mpc")))
    parser.add_argument("--max-train-points", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-limit", type=float, default=5.0)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    dataset = load_control_dataset(args.dataset)
    controller = train_sindy_mpc(
        dataset,
        max_train_points=args.max_train_points,
        seed=args.seed,
        force_limit=args.force_limit,
        t_end=args.t_end,
        dt=args.dt,
    )
    controller.save(args.out)
    write_json(str(args.out).replace(".pkl", ".report.json"), controller.metadata)
    print(f"saved {args.out}")
    print(f"best horizon={controller.metadata.get('horizon')}  "
          f"best_score={controller.metadata.get('best_tuning_score'):.4g}")
    for equation in controller.metadata["model"]["equations"]:
        print(equation)


if __name__ == "__main__":
    main()

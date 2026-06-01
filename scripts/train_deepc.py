#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import artifact_path, write_json
from controllers.deepc import DeePCConfig, DeePCController
from controllers.pid import PIDController
from models.dataset_views import load_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save a regularized DeePC controller.")
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz")
    parser.add_argument("--out", default=str(artifact_path("deepc")))
    parser.add_argument("--t-ini", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--max-hankel-columns", type=int, default=2000)
    parser.add_argument("--lambda-g", type=float, default=1e-4)
    parser.add_argument("--lambda-y-ini", type=float, default=1e4)
    parser.add_argument("--lambda-u-rate", type=float, default=1e-3)
    parser.add_argument("--lambda-terminal", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-limit", type=float, default=5.0)
    args = parser.parse_args()

    dataset = load_control_dataset(args.dataset)
    fallback = PIDController.load(artifact_path("pid")) if artifact_path("pid").exists() else PIDController()
    config = DeePCConfig(
        t_ini=args.t_ini,
        horizon=args.horizon,
        lambda_g=args.lambda_g,
        lambda_y_ini=args.lambda_y_ini,
        lambda_u_rate=args.lambda_u_rate,
        lambda_terminal=args.lambda_terminal,
        max_hankel_columns=args.max_hankel_columns,
        seed=args.seed,
    )
    controller = DeePCController(config=config, force_limit=args.force_limit, fallback=fallback).fit(dataset.states, dataset.inputs)
    controller.save(args.out)
    write_json(str(args.out).replace(".pkl", ".report.json"), controller.metadata)
    print(f"saved {args.out}")
    print(f"hankel_columns={controller.metadata['hankel_columns']} Yp_rank={controller.metadata['Yp_rank']['rank']}")


if __name__ == "__main__":
    main()


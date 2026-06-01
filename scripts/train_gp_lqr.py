#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np

from common import artifact_path, write_json
from controllers.gp_lqr import train_gp_lqr
from models.dataset_views import load_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GP dynamics and save the induced LQR controller.")
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz")
    parser.add_argument("--out", default=str(artifact_path("gp_lqr")))
    parser.add_argument("--max-train-points", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-limit", type=float, default=5.0)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    dataset = load_control_dataset(args.dataset)
    controller = train_gp_lqr(
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
    print(f"validation_rmse_mean={controller.metadata['model'].get('validation_rmse_mean')}")
    print(f"near_upright_rmse={controller.metadata.get('near_upright_rmse')}")
    print(f"best alpha={controller.metadata.get('alpha')}  Q_diag={np.diag(controller.q).tolist()}")


if __name__ == "__main__":
    main()


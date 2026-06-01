#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import artifact_path, write_json
from controllers.sindy_lqr import train_sindy_lqr
from models.dataset_views import load_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train controlled SINDy and save the induced LQR controller.")
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz")
    parser.add_argument("--out", default=str(artifact_path("sindy_lqr")))
    parser.add_argument("--threshold", type=float, default=1e-3,
                        help="Starting threshold (grid search uses default grid if not overridden).")
    parser.add_argument("--max-train-points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force-limit", type=float, default=5.0)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    dataset = load_control_dataset(args.dataset)
    controller = train_sindy_lqr(
        dataset,
        threshold=args.threshold,
        max_train_points=args.max_train_points,
        seed=args.seed,
        force_limit=args.force_limit,
        t_end=args.t_end,
        dt=args.dt,
    )
    controller.save(args.out)
    write_json(str(args.out).replace(".pkl", ".report.json"), controller.metadata)
    print(f"saved {args.out}")
    print(f"best alpha={controller.metadata.get('alpha')}  "
          f"best_score={controller.metadata.get('best_tuning_score'):.4g}")
    for equation in controller.metadata["model"]["equations"]:
        print(equation)


if __name__ == "__main__":
    main()


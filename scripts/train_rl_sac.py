#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import artifact_path, write_json
from controllers.rl_sac import train_sac_controller
from envs.inverted_pendulum_env import EnvConfig
from models.dataset_views import load_control_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC on the black-box simulator environment.")
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz")
    parser.add_argument("--out", default=str(artifact_path("rl_sac")))
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--force-limit", type=float, default=5.0)
    args = parser.parse_args()

    dataset = load_control_dataset(args.dataset)
    report = train_sac_controller(
        dataset.initial_state,
        timesteps=args.timesteps,
        output_path=args.out,
        seed=args.seed,
        env_config=EnvConfig(dt=args.dt, t_end=args.t_end, force_limit=args.force_limit),
    )
    write_json(str(args.out).replace(".zip", ".report.json"), report)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()


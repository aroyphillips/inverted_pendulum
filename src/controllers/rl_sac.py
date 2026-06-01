from __future__ import annotations

import json
from pathlib import Path

import time

import numpy as np

from controllers.base import Controller
from envs.inverted_pendulum_env import EnvConfig, InvertedPendulumEnv
from eval.metrics import rollout_metrics
from eval.rollout import rollout_controller


class RLController(Controller):
    def __init__(
        self,
        model,
        observation_scale: tuple[float, float, float, float] = (1.0, 2.0, 1.0, 5.0),
        force_limit: float = 5.0,
        name: str = "rl_sac",
    ) -> None:
        super().__init__(name=name, force_limit=force_limit)
        self.model = model
        self.observation_scale = np.asarray(observation_scale, dtype=float)
        self.metadata["observation_scale"] = self.observation_scale.tolist()
        self.metadata["force_limit"] = self.force_limit

    def act(self, t: float, x: np.ndarray) -> float:
        # Must match InvertedPendulumEnv._obs() exactly: 5D sin/cos observation
        # observation_scale slot 2 (theta) is unused — sin/cos replace raw theta
        s = self.observation_scale
        xi = np.asarray(x, dtype=float)
        obs = np.array([xi[0] / s[0], xi[1] / s[1], np.sin(xi[2]), np.cos(xi[2]), xi[3] / s[3]], dtype=np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return self.clipped(float(np.asarray(action).reshape(-1)[0]))

    @classmethod
    def load_sac(
        cls,
        path: Path,
        observation_scale: tuple[float, float, float, float] = (1.0, 2.0, 1.0, 5.0),
        force_limit: float = 5.0,
    ) -> "RLController":
        try:
            from stable_baselines3 import SAC
        except ImportError as exc:
            raise ImportError(
                "RL SAC controller requires stable-baselines3 and torch. "
                "Install them with `pip install stable-baselines3 torch`."
            ) from exc
        model = SAC.load(str(path))
        # Try to read force_limit from companion metadata file.
        meta_path = Path(path).with_suffix(".metadata.json")
        if meta_path.exists():
            with meta_path.open() as f:
                meta = json.load(f)
            force_limit = float(meta.get("force_limit", force_limit))
            obs_scale_list = meta.get("observation_scale")
            if obs_scale_list is not None:
                observation_scale = tuple(obs_scale_list)  # type: ignore[assignment]
        ctrl = cls(model=model, observation_scale=observation_scale, force_limit=force_limit, name="rl_sac")
        return ctrl


def train_sac_controller(
    initial_state: np.ndarray,
    timesteps: int = 500000,
    output_path: Path = Path("artifacts/rl/sac.zip"),
    seed: int = 0,
    env_config: EnvConfig | None = None,
    checkpoint_freq: int = 20000,
    t_end_eval: float = 20.0,
    dt_eval: float = 0.02,
    n_eval_states: int = 24,
) -> dict:
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise ImportError(
            "SAC training requires stable-baselines3 and torch. "
            "Install them with `pip install stable-baselines3 torch`."
        ) from exc

    env_config = env_config or EnvConfig()

    # Fixed deterministic eval set: canonical state + jittered variants.
    # Uses seed+999 to keep eval states separate from the training RNG.
    _eval_rng = np.random.default_rng(seed + 999)
    _eval_jitter = np.asarray(env_config.jitter_scale)
    eval_states = [np.asarray(initial_state, dtype=float)]
    for _ in range(max(0, n_eval_states - 1)):
        eval_states.append(np.asarray(initial_state, dtype=float) + _eval_rng.normal(scale=_eval_jitter))

    raw_env = InvertedPendulumEnv(initial_state=initial_state, config=env_config, seed=seed)
    env = Monitor(raw_env)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_path.parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Curriculum callback: advance curriculum_fraction with time, but only when
    # a rolling window of recent episode settle-streak outcomes meets success_threshold.
    class CurriculumCallback(BaseCallback):
        def __init__(
            self,
            total_steps: int,
            curriculum_end: int,
            success_threshold: float = 0.85,
            window: int = 200,
        ) -> None:
            super().__init__()
            self.curriculum_end = curriculum_end
            self.success_threshold = success_threshold
            self.window = window
            self._recent_successes: list[bool] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                streak = info.get("settle_streak", 0)
                self._recent_successes.append(streak >= raw_env.config.settle_streak_steps)
                if len(self._recent_successes) > self.window:
                    self._recent_successes.pop(0)
            time_frac = min(1.0, self.num_timesteps / self.curriculum_end)
            if len(self._recent_successes) >= self.window // 2:
                rate = sum(self._recent_successes) / len(self._recent_successes)
                if rate >= self.success_threshold:
                    raw_env.curriculum_fraction = time_frac
                # else: hold curriculum_fraction; don't advance until policy improves
            else:
                # Not enough data yet — use pure time gating
                raw_env.curriculum_fraction = time_frac
            return True

    # Validation callback: every checkpoint_freq steps, evaluate on all eval_states
    # and keep the checkpoint with the best mean score across them.
    class BestCheckpointCallback(BaseCallback):
        def __init__(self, freq: int, eval_states: list, ckpt_dir: Path) -> None:
            super().__init__()
            self.freq = freq
            self.eval_states = eval_states
            self.ckpt_dir = ckpt_dir
            self.best_score = float("inf")
            self.best_step = 0

        def _on_step(self) -> bool:
            if self.num_timesteps % self.freq == 0:
                ckpt_path = self.ckpt_dir / f"sac_{self.num_timesteps}"
                self.model.save(str(ckpt_path))
                ctrl = RLController(
                    model=self.model,
                    observation_scale=tuple(env_config.observation_scale),
                    force_limit=env_config.force_limit,
                )
                state_results = []
                for state in self.eval_states:
                    result = rollout_controller(ctrl, state, t_end=t_end_eval, dt=dt_eval, force_limit=env_config.force_limit)
                    m = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
                    state_results.append((bool(m.get("success", False)), float(m.get("cost", 1e12))))
                scores = [c if ok else c + 2000.0 for ok, c in state_results]
                n_success = sum(ok for ok, _ in state_results)
                score = float(np.mean(scores))
                if score < self.best_score:
                    self.best_score = score
                    self.best_step = self.num_timesteps
                    best_path = self.ckpt_dir / "best"
                    self.model.save(str(best_path))
                    print(f"  [checkpoint] step={self.num_timesteps} NEW BEST score={score:.4g} success={n_success}/{len(self.eval_states)}")
                else:
                    print(f"  [checkpoint] step={self.num_timesteps} score={score:.4g} success={n_success}/{len(self.eval_states)}")
            return True

    callbacks = [
        CurriculumCallback(total_steps=timesteps, curriculum_end=min(100000, timesteps)),
        BestCheckpointCallback(freq=checkpoint_freq, eval_states=eval_states, ckpt_dir=ckpt_dir),
    ]

    tic = time.perf_counter()
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        seed=seed,
        learning_rate=3e-4,
        buffer_size=min(1_000_000, timesteps),
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        learning_starts=10_000,
    )
    model.learn(total_timesteps=int(timesteps), callback=callbacks, progress_bar=True)
    runtime = time.perf_counter() - tic

    # Load best checkpoint if available, else use final model.
    best_ckpt = ckpt_dir / "best.zip"
    if best_ckpt.exists():
        from stable_baselines3 import SAC as SAC2
        model = SAC2.load(str(best_ckpt))
        print(f"  Loaded best checkpoint from {best_ckpt}")

    model.save(str(output_path))

    # Save companion metadata with force_limit and observation_scale.
    meta = {
        "force_limit": env_config.force_limit,
        "observation_scale": list(env_config.observation_scale),
        "algorithm": "SAC",
        "timesteps": int(timesteps),
        "runtime_train_sec": runtime,
        "env_config": env_config.__dict__,
    }
    meta_path = output_path.with_suffix(".metadata.json")
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    return {
        "runtime_train_sec": runtime,
        "timesteps": int(timesteps),
        "algorithm": "SAC",
        "output_path": str(output_path),
        "env_config": env_config.__dict__,
        "force_limit": env_config.force_limit,
    }


from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from config import PendulumParams
from dynamics import inverted_pendulum_dynamics

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


@dataclass
class EnvConfig:
    dt: float = 0.02
    t_end: float = 20.0
    force_limit: float = 5.0
    exact_reset_probability: float = 0.5
    jitter_scale: tuple[float, float, float, float] = (0.02, 0.05, 0.03, 0.1)
    # slots: (x_scale, x_dot_scale, _unused_theta, theta_dot_scale)
    # slot 2 is unused — sin(theta) and cos(theta) replace raw theta and need no scaling.
    # Defaults are neutral (1.0); the notebook computes data-driven values via
    # data_driven_obs_scale(dataset) and passes them here before training.
    observation_scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    settle_bonus: float = 50.0          # one-time reward when settle-and-remain streak is achieved
    settle_streak_steps: int = 50       # consecutive in-tolerance steps needed for bonus
    fall_penalty: float = 100.0         # extra penalty applied when the pendulum falls
    # Reward weights — Bryson's rule: Q_i = 1/tol_i^2, R = 1/u_max^2.
    # Settling tolerances: |x|≤0.25 m, |ẋ|≤0.25 m/s, |θ|≤2°≈0.035 rad, |θ̇|≤0.1 rad/s.
    # Heavy theta weight follows from the strict 2° tolerance; no ODE knowledge required.
    q_x: float = 1.0 / (0.25**2)                   # = 16.0
    q_xdot: float = 1.0 / (0.25**2)                # = 16.0
    q_theta: float = 1.0 / (0.034906585**2)         # ≈ 820.9  (np.deg2rad(2.0))
    q_thetadot: float = 1.0 / (0.1**2)             # = 100.0
    r_u: float = 1.0 / (5.0**2)                    # = 0.04
    # Two-stage shaping: upright bonus when |theta_err| < near_upright_threshold
    upright_bonus: float = 5.0
    recovery_bonus: float = 1.0
    near_upright_threshold: float = 0.35   # radians (~20 degrees)


class InvertedPendulumEnv(gym.Env if gym is not None else object):
    metadata = {"render_modes": []}

    def __init__(
        self,
        initial_state: np.ndarray,
        config: EnvConfig | None = None,
        params: PendulumParams = PendulumParams(),
        seed: int | None = None,
    ) -> None:
        if gym is None or spaces is None:
            raise ImportError(
                "InvertedPendulumEnv requires gymnasium. Install it with `pip install gymnasium`."
            )
        super().__init__()
        self.initial_state = np.asarray(initial_state, dtype=float)
        self.config = config or EnvConfig()
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.state = self.initial_state.copy()
        self.t = 0.0
        self.steps = 0
        self.max_steps = int(round(self.config.t_end / self.config.dt))
        high = np.full(5, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-self.config.force_limit], dtype=np.float32),
            high=np.array([self.config.force_limit], dtype=np.float32),
            dtype=np.float32,
        )
        # Settle-and-remain bonus tracking
        self._settle_streak = 0
        self._settle_bonus_given = False
        # Curriculum: fraction 0→1 controls jitter scaling (updated externally by callback)
        self.curriculum_fraction: float = 1.0

    def _obs(self) -> np.ndarray:
        scale = self.config.observation_scale
        x, x_dot, theta, theta_dot = self.state
        return np.array([
            x / scale[0],
            x_dot / scale[1],
            np.sin(theta),
            np.cos(theta),
            theta_dot / scale[3],
        ], dtype=np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        # Curriculum: scale jitter linearly from 2× (early training) to 1× (full curriculum_fraction)
        jitter = np.asarray(self.config.jitter_scale) * (1.0 + (1.0 - self.curriculum_fraction))
        if self.rng.random() < self.config.exact_reset_probability:
            self.state = self.initial_state.copy()
        else:
            self.state = self.initial_state + self.rng.normal(scale=jitter)
        self.t = 0.0
        self.steps = 0
        self._settle_streak = 0
        self._settle_bonus_given = False
        return self._obs(), {}

    def step(self, action):
        force = float(np.clip(np.asarray(action, dtype=float).reshape(-1)[0], -self.config.force_limit, self.config.force_limit))
        sol = solve_ivp(
            fun=lambda tau, state: inverted_pendulum_dynamics(
                tau, state, self.params, lambda _t, _x: force
            ),
            t_span=(self.t, self.t + self.config.dt),
            y0=self.state,
            method="DOP853",
            rtol=1e-8,
            atol=1e-10,
            t_eval=[self.t + self.config.dt],
        )
        if sol.success and np.all(np.isfinite(sol.y)):
            self.state = sol.y[:, -1]
        else:
            self.state = np.full(4, np.nan)

        x, x_dot, theta, theta_dot = self.state
        theta_err = np.arctan2(np.sin(theta), np.cos(theta))
        cfg = self.config
        base_cost = (
            cfg.q_x * x**2
            + cfg.q_xdot * x_dot**2
            + cfg.q_theta * theta_err**2
            + cfg.q_thetadot * theta_dot**2
            + cfg.r_u * force**2
        )
        if abs(theta_err) < cfg.near_upright_threshold:
            # Dense upright bonus when near vertical
            reward = float(-base_cost + cfg.upright_bonus)
        else:
            # Smooth cos-shaped recovery signal when far from vertical
            recovery = cfg.recovery_bonus * (np.cos(theta_err) + 1.0) / 2.0
            reward = float(-base_cost + recovery)
        self.t += self.config.dt
        self.steps += 1

        terminated = bool(abs(theta) > np.pi / 2.0 or abs(x) > 3.0 or not np.all(np.isfinite(self.state)))

        # Fall penalty
        if terminated:
            reward -= self.config.fall_penalty

        # Settle-and-remain bonus: track streak of in-tolerance steps
        in_tol = bool(
            abs(theta_err) < np.deg2rad(2.0)
            and abs(theta_dot) < 0.1
            and abs(x) < 0.25
            and abs(x_dot) < 0.25
        )
        if in_tol:
            self._settle_streak += 1
        else:
            self._settle_streak = 0
        if (
            not self._settle_bonus_given
            and self._settle_streak >= self.config.settle_streak_steps
        ):
            reward += self.config.settle_bonus
            self._settle_bonus_given = True

        truncated = bool(self.steps >= self.max_steps)
        info = {"t": self.t, "force": force, "state": self.state.copy(), "settle_streak": self._settle_streak}
        return self._obs(), reward, terminated, truncated, info


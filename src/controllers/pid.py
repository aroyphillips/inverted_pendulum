from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import differential_evolution

from controllers.base import Controller, TrainingTimer, load_controller_json, save_controller_json
from eval.metrics import DEFAULT_Q, DEFAULT_R, rollout_metrics
from eval.rollout import rollout_controller


@dataclass
class PIDGains:
    k_theta_p: float = 48.0
    k_theta_d: float = 10.0
    k_x_p: float = -3.8
    k_x_d: float = -5.6
    k_i: float = 0.0




class PIDController(Controller):
    def __init__(
        self,
        gains: PIDGains | None = None,
        mode: str = "theta_plus_cart",
        force_limit: float = 5.0,
        integral_limit: float = 0.1,
    ) -> None:
        super().__init__(name="pid", force_limit=force_limit)
        self.gains = gains or PIDGains()
        self.mode = mode
        self.integral_limit = float(integral_limit)
        self.integral_theta = 0.0
        self.last_t: float | None = None
        self.last_theta: float | None = None
        self.metadata.update(
            {
                "mode": self.mode,
                "gains": asdict(self.gains),
                "force_limit": self.force_limit,
                "integral_limit": self.integral_limit,
            }
        )

    def reset(self, x0: np.ndarray) -> None:
        super().reset(x0)
        self.integral_theta = 0.0
        self.last_t = None
        self.last_theta = float(np.asarray(x0, dtype=float)[2])

    def act(self, t: float, x: np.ndarray) -> float:
        state = np.asarray(x, dtype=float)
        x_pos, x_dot, theta, theta_dot = state

        if self.last_t is not None and t > self.last_t:
            dt = float(t - self.last_t)
            prev_theta = theta if self.last_theta is None else self.last_theta
            self.integral_theta += 0.5 * dt * (float(theta) + float(prev_theta))
            self.integral_theta = float(
                np.clip(self.integral_theta, -self.integral_limit, self.integral_limit)
            )
        self.last_t = float(t)
        self.last_theta = float(theta)

        g = self.gains
        u = -g.k_theta_p * theta - g.k_theta_d * theta_dot - g.k_i * self.integral_theta
        if self.mode == "theta_plus_cart":
            u += -g.k_x_p * x_pos - g.k_x_d * x_dot
        elif self.mode != "theta_only":
            raise ValueError(f"Unknown PID mode: {self.mode}")
        return self.clipped(u)

    def to_payload(self) -> dict:
        return {
            "name": self.name,
            "mode": self.mode,
            "force_limit": self.force_limit,
            "integral_limit": self.integral_limit,
            "gains": asdict(self.gains),
            "metadata": self.metadata,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "PIDController":
        gains = PIDGains(**payload["gains"])
        ctrl = cls(
            gains=gains,
            mode=payload.get("mode", "theta_plus_cart"),
            force_limit=payload.get("force_limit", 5.0),
            integral_limit=payload.get("integral_limit", 0.1),
        )
        ctrl.metadata.update(payload.get("metadata", {}))
        return ctrl

    @classmethod
    def load(cls, path: Path) -> "PIDController":
        return cls.from_payload(load_controller_json(path))

    def save(self, path: Path) -> None:
        save_controller_json(path, self.to_payload())


class TwoPhasePIDController(PIDController):
    """PID controller with separate gains for recovery and regulation phases.

    Uses aggressive gains while |theta| > theta_switch_rad, then switches to
    regulation gains once the pendulum is close to upright.
    """

    def __init__(
        self,
        recovery_gains: PIDGains,
        regulation_gains: PIDGains,
        theta_switch_rad: float = 0.15,
        mode: str = "theta_plus_cart",
        force_limit: float = 5.0,
        integral_limit: float = 0.1,
    ) -> None:
        super().__init__(
            gains=recovery_gains,
            mode=mode,
            force_limit=force_limit,
            integral_limit=integral_limit,
        )
        self.name = "two_phase_pid"
        self.recovery_gains = recovery_gains
        self.regulation_gains = regulation_gains
        self.theta_switch_rad = float(theta_switch_rad)

    def act(self, t: float, x: np.ndarray) -> float:
        theta = float(np.asarray(x, dtype=float)[2])
        self.gains = self.recovery_gains if abs(theta) > self.theta_switch_rad else self.regulation_gains
        return super().act(t, x)

    def to_payload(self) -> dict:
        payload = super().to_payload()
        payload.update({
            "name": self.name,
            "recovery_gains": asdict(self.recovery_gains),
            "regulation_gains": asdict(self.regulation_gains),
            "theta_switch_rad": self.theta_switch_rad,
        })
        return payload

    @classmethod
    def from_payload(cls, payload: dict) -> "TwoPhasePIDController":
        ctrl = cls(
            recovery_gains=PIDGains(**payload["recovery_gains"]),
            regulation_gains=PIDGains(**payload["regulation_gains"]),
            theta_switch_rad=payload.get("theta_switch_rad", 0.15),
            mode=payload.get("mode", "theta_plus_cart"),
            force_limit=payload.get("force_limit", 5.0),
            integral_limit=payload.get("integral_limit", 0.1),
        )
        ctrl.metadata.update(payload.get("metadata", {}))
        return ctrl

    @classmethod
    def load(cls, path: Path) -> "TwoPhasePIDController":
        return cls.from_payload(load_controller_json(path))


def _sample_gains(rng: np.random.Generator, mode: str) -> PIDGains:
    if mode == "theta_only":
        return PIDGains(
            k_theta_p=float(rng.uniform(20.0, 120.0)),
            k_theta_d=float(rng.uniform(2.0, 30.0)),
            k_x_p=0.0,
            k_x_d=0.0,
            k_i=float(rng.choice([0.0, rng.uniform(0.0, 1.0)])),
        )
    return PIDGains(
        k_theta_p=float(rng.uniform(20.0, 120.0)),
        k_theta_d=float(rng.uniform(2.0, 30.0)),
        k_x_p=float(rng.uniform(-10.0, 2.0)),
        k_x_d=float(rng.uniform(-12.0, 2.0)),
        k_i=float(rng.choice([0.0, rng.uniform(0.0, 0.5)])),
    )


def _score_gains(
    gains: PIDGains,
    test_states: List[np.ndarray],
    mode: str,
    force_limit: float,
    t_end: float,
    dt: float,
) -> Tuple[float, int]:
    """Return (score, num_successes) for a set of gains."""
    controller = PIDController(gains=gains, mode=mode, force_limit=force_limit)
    scores = []
    successes = 0
    for x0 in test_states:
        result = rollout_controller(controller, x0, t_end=t_end, dt=dt, force_limit=force_limit)
        metrics = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
        if metrics.get("success"):
            successes += 1
        scores.append(float(metrics.get("cost", 1e12)))
    score = float(np.mean(scores) + 2000.0 * (len(test_states) - successes))
    return score, successes


def tune_pid(
    initial_state: np.ndarray,
    mode: str = "theta_plus_cart",
    n_trials: int = 300,
    seed: int = 0,
    force_limit: float = 5.0,
    t_end: float = 20.0,
    dt: float = 0.02,
    jitter_scale: Iterable[float] = (0.02, 0.05, 0.03, 0.1),
    num_jitter_rollouts: int = 4,
) -> Tuple[PIDController, list[dict]]:
    """Three-stage PID tuning: random seeds → differential evolution → coordinate descent.

    Scoring: mean_cost + 2000 * (num_test_states - num_settle_and_remain_successes).
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(initial_state, dtype=float)
    jitter_scale_arr = np.asarray(list(jitter_scale), dtype=float)

    test_states: List[np.ndarray] = [base]
    for _ in range(num_jitter_rollouts):
        test_states.append(base + rng.normal(scale=jitter_scale_arr))

    history: list[dict] = []
    best_score = float("inf")
    best_gains: PIDGains = PIDGains()  # neutral default; overwritten after first evaluation

    def evaluate(gains: PIDGains, stage: str, trial_idx: int) -> float:
        nonlocal best_score, best_gains
        score, successes = _score_gains(gains, test_states, mode, force_limit, t_end, dt)
        record = {
            "trial": trial_idx,
            "stage": stage,
            "score": score,
            "successes": successes,
            "gains": asdict(gains),
        }
        history.append(record)
        if score < best_score:
            best_score = score
            best_gains = gains
        return score

    # ------------------------------------------------------------------ #
    # Stage 1: fixed seeds + random candidates (~n_trials/3 evaluations)  #
    # ------------------------------------------------------------------ #
    n_stage1 = max(3, n_trials // 3)
    # Empirically-discovered seed gains from prior search runs (not ODE-derived)
    seed_candidates = [
        PIDGains(80.0, 20.0, -4.0, -7.0, 0.0),   # aggressive recovery
        PIDGains(50.0, 12.0, -2.5, -4.5, 0.0),    # moderate
        PIDGains(48.0, 10.0, -3.8, -5.6, 0.0),    # legacy default
        PIDGains(25.0, 5.5, -1.0, -2.0, 0.0),     # legacy default 2
    ]
    for g in seed_candidates:
        evaluate(g, "seed", len(history))
    n_random = max(0, n_stage1 - len(seed_candidates))
    for _ in range(n_random):
        evaluate(_sample_gains(rng, mode), "random", len(history))

    # ------------------------------------------------------------------ #
    # Stage 2: differential evolution (~n_trials/3 evaluations)          #
    # ------------------------------------------------------------------ #
    n_stage2 = n_trials // 3
    trial_counter = [0]

    def de_objective(params: np.ndarray) -> float:
        if mode == "theta_only":
            g = PIDGains(k_theta_p=params[0], k_theta_d=params[1], k_x_p=0.0, k_x_d=0.0, k_i=params[2])
        else:
            g = PIDGains(
                k_theta_p=params[0],
                k_theta_d=params[1],
                k_x_p=params[2],
                k_x_d=params[3],
                k_i=params[4],
            )
        result = evaluate(g, "de", len(history))
        trial_counter[0] += 1
        return result

    if mode == "theta_only":
        bounds = [(20.0, 120.0), (2.0, 30.0), (0.0, 0.5)]
        x0_de = [best_gains.k_theta_p, best_gains.k_theta_d, best_gains.k_i]
    else:
        bounds = [(20.0, 120.0), (2.0, 30.0), (-10.0, 2.0), (-12.0, 2.0), (0.0, 0.5)]
        x0_de = [best_gains.k_theta_p, best_gains.k_theta_d, best_gains.k_x_p, best_gains.k_x_d, best_gains.k_i]

    # Use a small population so maxiter * popsize ≈ n_stage2 total calls.
    popsize = max(3, min(6, n_stage2 // 10))
    maxiter = max(1, n_stage2 // (popsize * len(bounds)))
    differential_evolution(
        de_objective,
        bounds=bounds,
        maxiter=maxiter,
        popsize=popsize,
        seed=int(seed),
        tol=1e-6,
        mutation=(0.5, 1.5),
        recombination=0.7,
        init="latinhypercube",
        polish=False,
    )

    # ------------------------------------------------------------------ #
    # Stage 3: coordinate descent around best-found gains (~n_trials/3)  #
    # ------------------------------------------------------------------ #
    n_stage3 = max(0, n_trials - len(history))
    if n_stage3 > 0 and mode != "theta_only":
        gain_fields = ["k_theta_p", "k_theta_d", "k_x_p", "k_x_d", "k_i"]
        step_fracs = [0.10, 0.10, 0.10, 0.10, 0.20]
        current = best_gains
        n_rounds = max(1, n_stage3 // (2 * len(gain_fields)))
        for _ in range(n_rounds):
            improved = False
            for field, frac in zip(gain_fields, step_fracs):
                val = float(getattr(current, field))
                step = max(0.01, abs(val) * frac) if val != 0.0 else frac
                for delta in [+step, -step]:
                    candidate_dict = asdict(current)
                    candidate_dict[field] = val + delta
                    g_new = PIDGains(**candidate_dict)
                    s = evaluate(g_new, "coord_descent", len(history))
                    if s < best_score:
                        current = g_new
                        improved = True
                        break
            if not improved:
                break

    # ------------------------------------------------------------------ #
    # Assemble result                                                      #
    # ------------------------------------------------------------------ #
    best_controller = PIDController(gains=best_gains, mode=mode, force_limit=force_limit)
    best_controller.metadata.update(
        {
            "runtime_train_sec": 0.0,   # filled in below by TrainingTimer wrapper
            "n_trials": len(history),
            "seed": int(seed),
            "best_score": float(best_score),
            "tuning_history_top10": sorted(history, key=lambda row: row["score"])[:10],
            "Q": DEFAULT_Q.tolist(),
            "R": DEFAULT_R.tolist(),
        }
    )
    return best_controller, history

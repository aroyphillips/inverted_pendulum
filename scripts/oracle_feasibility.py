#!/usr/bin/env python3
"""Oracle feasibility sweep: find the minimum force limit for stabilization.

Sweeps force_limit from 1 N to 15 N using an exact LQR controller derived
from the linearized dynamics at the upright equilibrium.  Reports the minimum
limit that achieves settle-and-remain success and writes the full sweep to
artifacts/oracle/feasibility.json.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cont2discrete

# common.py is on the path because scripts/ adds src/ at import time
import common  # noqa: F401 – side effect: adds src/ to sys.path

from config import PendulumParams
from controllers.base import Controller
from dynamics import inverted_pendulum_dynamics
from eval.metrics import rollout_metrics
from eval.rollout import rollout_controller
from models.dataset_views import load_control_dataset
from models.lqr_utils import discrete_lqr_gain, finite_difference_linearization


# --------------------------------------------------------------------------- #
# Analytic physics helpers                                                     #
# --------------------------------------------------------------------------- #

def theta_ddot_at_state(state: np.ndarray, force: float, params: PendulumParams) -> float:
    """Return the angular acceleration given state and applied force."""
    ydot = inverted_pendulum_dynamics(0.0, state, params, lambda t, x: force)
    return float(ydot[3])


def continuous_linearization(params: PendulumParams) -> tuple[np.ndarray, np.ndarray]:
    """Exact analytic continuous-time A, B at upright equilibrium (theta=0)."""
    M, m, l, g = params.M, params.m, params.l, params.g
    # State: [x, x_dot, theta, theta_dot]
    A_c = np.array([
        [0.0, 1.0,              0.0,              0.0],
        [0.0, 0.0,     m * g / M,                0.0],
        [0.0, 0.0,              0.0,              1.0],
        [0.0, 0.0, (M + m) * g / (M * l),        0.0],
    ], dtype=float)
    B_c = np.array([[0.0], [1.0 / M], [0.0], [1.0 / (M * l)]], dtype=float)
    return A_c, B_c


# --------------------------------------------------------------------------- #
# Discrete-step function for numerical linearization check                    #
# --------------------------------------------------------------------------- #

def discrete_step(x: np.ndarray, u: np.ndarray, params: PendulumParams, dt: float) -> np.ndarray:
    sol = solve_ivp(
        fun=lambda tau, state: inverted_pendulum_dynamics(tau, state, params, lambda _t, _x: float(u[0])),
        t_span=(0.0, dt),
        y0=x,
        method="DOP853",
        rtol=1e-8,
        atol=1e-10,
        t_eval=[dt],
    )
    return sol.y[:, -1]


# --------------------------------------------------------------------------- #
# Simple LQR controller                                                        #
# --------------------------------------------------------------------------- #

class LQRController(Controller):
    def __init__(self, k: np.ndarray, force_limit: float = 5.0) -> None:
        super().__init__(name="lqr_oracle", force_limit=force_limit)
        self.k = np.asarray(k, dtype=float)

    def act(self, t: float, x: np.ndarray) -> float:
        u = float(np.ravel(-self.k @ np.asarray(x, dtype=float))[0])
        return self.clipped(u)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Oracle feasibility sweep over force limits.")
    parser.add_argument("--dataset", default="data/full_example/random_walk_dataset.npz")
    parser.add_argument("--out", default="artifacts/oracle/feasibility.json")
    parser.add_argument("--force-min", type=float, default=1.0)
    parser.add_argument("--force-max", type=float, default=15.0)
    parser.add_argument("--force-step", type=float, default=0.5)
    parser.add_argument("--t-end", type=float, default=20.0)
    parser.add_argument("--dt", type=float, default=0.02)
    args = parser.parse_args()

    params = PendulumParams()
    dataset = load_control_dataset(args.dataset)
    x0 = dataset.initial_state

    # ------------------------------------------------------------------ #
    # 1. Confirm infeasibility at ±1 N                                    #
    # ------------------------------------------------------------------ #
    print("=" * 60)
    print("Initial state:", x0)
    print(f"  theta = {np.degrees(x0[2]):.2f} deg,  theta_dot = {x0[3]:.3f} rad/s")
    for u_probe in [-1.0, 0.0, 1.0]:
        tdd = theta_ddot_at_state(x0, u_probe, params)
        print(f"  theta_ddot(u={u_probe:+.1f}) = {tdd:+.4f} rad/s² "
              f"({'positive—not recovering' if tdd > 0 else 'negative—recovering'})")
    print()

    # ------------------------------------------------------------------ #
    # 2. Build LQR from exact continuous-time linearization               #
    # ------------------------------------------------------------------ #
    A_c, B_c = continuous_linearization(params)
    A_d, B_d, _, _, _ = cont2discrete((A_c, B_c, np.eye(4), np.zeros((4, 1))), args.dt, method="zoh")

    # Also verify with finite-difference of the DOP853 integrator
    A_num, B_num = finite_difference_linearization(
        lambda x, u: discrete_step(x, u, params, args.dt),
        np.zeros(4),
        np.zeros(1),
    )
    print("Continuous-time linearization (analytic A_c):")
    print(A_c)
    print("Discrete-time A_d (ZOH, dt=0.02) vs numerical A_num max diff:",
          float(np.max(np.abs(A_d - A_num))))

    Q = np.diag([1.0, 0.2, 100.0, 10.0])   # heavy theta / theta_dot penalty
    R = np.array([[0.05]])
    K, _ = discrete_lqr_gain(A_d, B_d, Q, R)
    print(f"\nLQR gain K = {K}")
    print()

    # ------------------------------------------------------------------ #
    # 3. Sweep force limits                                               #
    # ------------------------------------------------------------------ #
    force_limits = np.arange(args.force_min, args.force_max + 0.01 * args.force_step, args.force_step)
    sweep_results = []
    first_success = None

    print(f"{'force_limit':>12} | {'success':>7} | {'settle_t':>9} | {'max_theta_deg':>13} | {'cost':>10}")
    print("-" * 62)

    for fl in force_limits:
        fl = float(fl)
        controller = LQRController(K, force_limit=fl)
        result = rollout_controller(controller, x0, t_end=args.t_end, dt=args.dt, force_limit=fl)
        m = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
        success = bool(m.get("success", False))
        settle_t = m.get("settle_and_remain_time")
        max_theta_deg = float(np.degrees(float(m.get("max_abs_theta", np.pi))))
        cost = float(m.get("cost", float("inf")))

        settle_str = f"{settle_t:.2f}" if settle_t is not None else "  None"
        print(f"{fl:>12.1f} | {str(success):>7} | {settle_str:>9} | {max_theta_deg:>13.2f} | {cost:>10.4g}")

        sweep_results.append({
            "force_limit": fl,
            "success": success,
            "settle_and_remain_time": settle_t,
            "max_abs_theta": float(m.get("max_abs_theta", float("nan"))),
            "max_theta_deg": max_theta_deg,
            "cost": cost,
            "no_fall": bool(m.get("no_fall", False)),
        })

        if success and first_success is None:
            first_success = fl

    # ------------------------------------------------------------------ #
    # 4. Summary                                                          #
    # ------------------------------------------------------------------ #
    print()
    if first_success is not None:
        print(f"Minimum force limit for LQR stabilization: {first_success:.1f} N")
    else:
        print(f"LQR did NOT stabilize within [{args.force_min}, {args.force_max}] N — increase sweep range.")

    output = {
        "initial_state": x0.tolist(),
        "lqr_K": K.tolist(),
        "lqr_Q": Q.tolist(),
        "lqr_R": R.tolist(),
        "A_d": A_d.tolist(),
        "B_d": B_d.tolist(),
        "min_feasible_force_limit": first_success,
        "sweep": sweep_results,
    }
    from common import write_json
    write_json(args.out, output)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

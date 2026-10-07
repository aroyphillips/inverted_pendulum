# 07 — SINDy-MPC

**Notebook:** [notebooks/07_sindy_mpc.ipynb](../notebooks/07_sindy_mpc.ipynb)  
**Source:** [src/controllers/sindy_mpc.py](../src/controllers/sindy_mpc.py), [src/models/sindy.py](../src/models/sindy.py)  
**Artifact:** `artifacts/controllers/sindy_mpc.pkl`

---

## Overview

This notebook applies the same **SINDy-C** model identification as method 05 but replaces the fixed infinite-horizon LQR gain with a **Model Predictive Controller** solved online at each control step. The key distinction is the control law:

| Method | Linearisation point | Control law | Input constraints |
|---|---|---|---|
| 05 — SINDy-LQR | Upright equilibrium (once) | Fixed gain $u = -K\mathbf{x}$ | Clip after the fact |
| 07 — SINDy-MPC | Current state (every step) | QP over horizon $N$, receding horizon | Enforced inside the optimiser |

Two advantages motivate this choice over LQR:

1. **Explicit constraint satisfaction.** The force limit $|u_k| \leq u_{\max}$ appears as a hard constraint in the QP, so the planner accounts for actuator saturation when building the receding-horizon trajectory rather than discovering it through clipping.
2. **Improved far-from-equilibrium accuracy.** Successive linearisation at the current operating point provides a better local model for large initial angles where the equilibrium linearisation (used by LQR) is least reliable.

Model identification (SINDy feature library, STLSQ algorithm) is identical to method 05; see [docs/05_sindy_lqr.md](05_sindy_lqr.md) for full details.

---

## Successive Linearisation MPC

At each control step, given current state $\mathbf{x}$:

**Step 1 — Linearise at the current operating point.**

$$A_t = \frac{\partial F}{\partial \mathbf{x}}\bigg|_{(\mathbf{x}, 0)}, \qquad B_t = \frac{\partial F}{\partial u}\bigg|_{(\mathbf{x}, 0)}$$

where $F(\mathbf{x}, u) = \Theta([\mathbf{x}^\top, u]^\top)\Xi$ is the identified sparse model. Jacobians are computed by central finite differences on the SINDy predict step.

**Step 2 — Compute affine offset.**

$$\mathbf{d}_t = F(\mathbf{x}, 0) - A_t\,\mathbf{x}$$

This offset ensures the linearised model $\mathbf{x}_{k+1} \approx A_t\mathbf{x}_k + B_t u_k + \mathbf{d}_t$ is exact at the current state $\mathbf{x}$ when $u = 0$, correcting for the nonlinear residual that the linear approximation ignores.

**Step 3 — Solve a finite-horizon QP.**

$$\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \left[\mathbf{x}_k^\top Q\,\mathbf{x}_k + r\,u_k^2\right] + \mathbf{x}_N^\top P\,\mathbf{x}_N$$

subject to

$$\mathbf{x}_{k+1} = A_t\,\mathbf{x}_k + B_t\,u_k + \mathbf{d}_t, \quad k = 0, \ldots, N-1$$
$$|u_k| \leq u_{\max}, \quad k = 0, \ldots, N-1$$
$$\mathbf{x}_0 = \mathbf{x}_{\text{current}}$$

**Step 4 — Apply first input, discard remainder (receding horizon).**

$$u_{\text{applied}} = u_0^*$$

The optimisation is solved again at the next time step from the new observed state.

---

## Terminal Cost

The terminal cost matrix $P$ is the solution of the **Discrete Algebraic Riccati Equation** (DARE):

$$P = A_{\text{eq}}^\top P\,A_{\text{eq}} - A_{\text{eq}}^\top P\,B_{\text{eq}}\,(B_{\text{eq}}^\top P\,B_{\text{eq}} + R)^{-1} B_{\text{eq}}^\top P\,A_{\text{eq}} + Q$$

where $(A_{\text{eq}}, B_{\text{eq}})$ are the Jacobians evaluated at the upright equilibrium, identical to the matrices used by the SINDy-LQR controller. Using the DARE solution as terminal cost provides two benefits:

- **Stability**: it is a standard Lyapunov-function terminal cost for the linearised system, giving the closed-loop system a certificate of stability near equilibrium.
- **Consistency**: in the limit $N \to \infty$ (or near equilibrium), the MPC law recovers the LQR gain, making the two methods directly comparable.

A **fallback control law** $u = -K\mathbf{x}$ (LQR gain derived from $P$) is applied if OSQP reports the QP as infeasible or times out. This prevents open-loop failure under numerical edge cases.

---

## Implementation Details

The CVXPY problem is built **once** with `cp.Parameter` objects for $(A_t, B_t, \mathbf{d}_t, \mathbf{x}_0)$ and updated in-place at each control step. OSQP is invoked with `warm_start=True` so previous iterates initialise the solver, keeping per-step solve time low.

Because CVXPY problems cannot be pickled, `SINDyMPCController` implements `__getstate__` / `__setstate__` to exclude CVXPY objects from the serialised state and rebuild the problem on load.

---

## Hyperparameter Tuning

`train_sindy_mpc` performs a grid search over STLSQ hyperparameters and prediction horizon.

### $(\xi_{\min}, \lambda)$ Grid

Identical to method 05 (5 candidates ranging from $\xi_{\min} \in [10^{-4}, 5\times10^{-3}]$).

### Horizon Grid

$N \in \{20, 30, 50\}$ steps. At $\Delta t = 0.02\,\text{s}$ these correspond to prediction horizons of 0.4, 0.6, and 1.0 seconds.

### Selection Criterion

$$\text{score}(\xi_{\min}, \lambda, N) = c + 2000 \cdot \mathbb{1}[\text{not successful}]$$

For efficiency, SINDy is fit once per $(\xi_{\min}, \lambda)$ pair and the three horizon values are tested without refitting.

---

## Evaluation

9 rollouts (1 fixed + 8 jittered initial states), same protocol as notebook 01 and method 05. `controller.metadata` exposes:

- `model.equations` — symbolic equations of the discovered model
- $(A, B, P, K)$ — equilibrium linearisation and DARE solution
- `horizon` — selected prediction horizon $N$
- `force_limit` — $u_{\max}$ used during training and inference
- `best_tuning_score` — objective value of the selected configuration
- `Q`, `R` — cost matrices used in the QP

---

## Validation Notebooks

Two additional notebooks extend stress-test and cross-validation coverage for SINDy-MPC, mirroring the structure of notebooks 05a and 05b. The same 100-case ValidationSpec (seed=42) is reused from 05a for direct LQR vs MPC comparison on identical initial conditions.

### 07a — Stress Test (100 Random Initial Conditions)

**Notebook:** [notebooks/07a_sindy_mpc_stress_test.ipynb](../notebooks/07a_sindy_mpc_stress_test.ipynb)

Requires `05a_sindy_lqr_stress_test.ipynb` to have been run first (loads the shared ValidationSpec).

Evaluates the pre-trained SINDy-MPC baseline controller on the same 100 initial states used for SINDy-LQR in notebook 05a.

**Protocol:**
- Loads `artifacts/validation/sindy_lqr_val_spec.npz` (forces ∈ [0.5, 5] N, angles ∈ [5, 90] degrees, seed=42).
- Loads `artifacts/controllers/sindy_mpc.pkl`; no retraining occurs.
- Runs rollouts for all feasible cases at $t_{\text{end}} = 20\,\text{s}$, $\Delta t = 0.02\,\text{s}$.
- Side-by-side bar chart shows MPC vs LQR success rates by angle bucket; scatter plot shows MPC outcome per initial condition.

**Outputs saved to `artifacts/validation/`:**

| File | Contents |
|---|---|
| `sindy_mpc_stress_rollouts.npz` | States/inputs/time arrays for all 100 MPC rollouts |
| `sindy_mpc_stress_metrics.npz` | Per-case metrics (success, cost, settle time, …) |

---

### 07b — Five-Fold Cross-Validation

**Notebook:** [notebooks/07b_sindy_mpc_crossval.ipynb](../notebooks/07b_sindy_mpc_crossval.ipynb)

Requires `05a_sindy_lqr_stress_test.ipynb` and `07a_sindy_mpc_stress_test.ipynb` to have been run first.

Mirrors the 05b protocol, retraining SINDy-MPC five times with different 80/20 splits.

**Protocol:**
- **Fold splits:** same 5-fold stratified-by-angle assignment as notebook 05b (seed=0) so MPC and LQR are evaluated on exactly the same held-out sets.
- **Training data (per fold):** all transition pairs from the original random-walk dataset plus the baseline-MPC-controller rollout trajectories for the 80 train-fold validation cases.
- **Hyperparameters:** threshold, ridge, and horizon are fixed to the baseline values selected during tuning.
- **Evaluation:** each held-out case is evaluated from its original impulse-derived initial state using the fold-specific retrained MPC controller.
- **Comparison:** per-fold metrics are reported alongside the corresponding SINDy-LQR fold metrics from `sindy_lqr_fold_metrics.npz`.

**Outputs saved to `artifacts/validation/`:**

| File | Contents |
|---|---|
| `sindy_mpc_fold_metrics.npz` | Per-case metrics annotated with fold index |

**Result table:** per-fold success rate, median settle time, and median cost; aggregate CV metrics vs SINDy-LQR on the same held-out initial conditions.

---

## References

- Kaiser, E., Kutz, J. N. & Brunton, S. L. (2018). Sparse identification of nonlinear dynamics for model predictive control in the low-data limit. *Proceedings of the Royal Society A*, 474(2219), 20180335.
- Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences*, 113(15), 3932–3937.
- Rawlings, J. B., Mayne, D. Q. & Diehl, M. (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing.
- Borrelli, F., Bemporad, A. & Morari, M. (2017). *Predictive Control for Linear and Hybrid Systems*. Cambridge University Press.
- Mayne, D. Q., Rawlings, J. B., Rao, C. V. & Scokaert, P. O. M. (2000). Constrained model predictive control: Stability and optimality. *Automatica*, 36(6), 789–814.
- Anderson, B. D. O. & Moore, J. B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice-Hall.
- Bryson, A. E. & Ho, Y.-C. (1969). *Applied Optimal Control*. Blaisdell.

# 05 — Controlled SINDy + LQR

**Notebook:** [notebooks/05_sindy_lqr.ipynb](../notebooks/05_sindy_lqr.ipynb)  
**Source:** [src/controllers/sindy_lqr.py](../src/controllers/sindy_lqr.py), [src/models/sindy.py](../src/models/sindy.py)  
**Artifact:** `artifacts/controllers/sindy_lqr.pkl`

---

## Overview

This notebook applies **Sparse Identification of Nonlinear Dynamics (SINDy)** [Brunton et al., 2016] in its controlled variant (SINDy-C) to discover a compact, interpretable discrete-time model of the controlled pendulum. The identified sparse polynomial-trigonometric model is then linearised at the upright equilibrium and used to synthesise a **discrete LQR** gain. A key advantage over GP (notebook 03) is that the discovered model is a human-readable symbolic equation rather than an implicit kernel regression.

---

## SINDy Model Identification

The controlled SINDy problem seeks a sparse coefficient matrix $\Xi$ such that

$$\mathbf{x}_{t+1} \approx \Theta(\mathbf{x}_t, u_t)\,\Xi$$

where $\Theta \in \mathbb{R}^{N \times p}$ is a library matrix of $p$ candidate nonlinear features evaluated at each of the $N$ training transitions, and $\Xi \in \mathbb{R}^{p \times 4}$ contains the sparse model coefficients for each state dimension.

### Feature Library

`sindy_library` constructs $p = 20$ candidate features of the augmented state $[\mathbf{x}^\top, u]^\top = [x, \dot{x}, \theta, \dot{\theta}, u]^\top$:

| Index | Feature | Basis justification |
|---|---|---|
| 0 | $1$ | Constant / bias term |
| 1–4 | $x,\; \dot{x},\; \theta,\; \dot{\theta}$ | Linear state terms |
| 5 | $u$ | Linear control term |
| 6–10 | $x^2,\; \dot{x}^2,\; \theta^2,\; \dot{\theta}^2,\; u^2$ | Degree-2 polynomial terms |
| 11–15 | $x\theta,\; \dot{x}\dot{\theta},\; \theta\dot{\theta},\; xu,\; \theta u$ | Degree-2 cross-product terms |
| 16–17 | $\sin\theta,\; \cos\theta - 1$ | $\theta$ is a periodic angular variable; $\sin$ and $\cos$ form the standard Fourier basis for smooth periodic functions. $\cos\theta - 1$ is the centered even term, equal to 0 at the target state $\theta = 0$ (upright) |
| 18 | $u\cos\theta$ | Interaction between control input and angular orientation; captures how the effect of a horizontal force on a rotating body varies with its angular configuration |
| 19 | $\dot{\theta}^2\sin\theta$ | Product of squared angular rate and angular sine; captures the interaction between rotation speed and angular displacement natural to rotating systems |

The trig terms exploit the fact that $\theta$ is a periodic angular variable: $\sin\theta$ and $\cos\theta$ form the canonical Fourier basis for smooth observables of an angle. STLSQ will discard any term whose coefficient falls below the sparsity threshold, so inclusion in the library does not guarantee selection.

### Sequential Thresholded Least Squares (STLSQ)

`stlsq` solves the sparse regression problem using the **Sequentially Thresholded Least Squares** algorithm [Brunton et al., 2016]:

1. **Initialise** with the ridge-regularised least-squares solution:
   $$\Xi^{(0)} = (\Theta^\top\Theta + \lambda I)^{-1}\Theta^\top X_{\text{next}}$$

2. **Threshold** all coefficients below $\xi_{\min}$:
   $$\Xi^{(k+1)}_{ij} \leftarrow 0 \quad \text{if} \quad |\Xi^{(k)}_{ij}| < \xi_{\min}$$

3. **Re-fit** each state equation on the surviving (non-zero) terms only:
   $$\Xi^{(k+1)}_{\mathcal{A}_j, j} = (\Theta_{\mathcal{A}_j}^\top\Theta_{\mathcal{A}_j} + \lambda I_{\mathcal{A}_j})^{-1}\Theta_{\mathcal{A}_j}^\top (X_{\text{next}})_j$$
   where $\mathcal{A}_j = \{i : |\Xi^{(k)}_{ij}| \geq \xi_{\min}\}$ is the active set for dimension $j$.

4. Repeat steps 2–3 for `max_iter = 12` iterations.

The two hyperparameters are the **threshold** $\xi_{\min}$ (sparsity level) and the **ridge** $\lambda$ (numerical conditioning). A higher $\xi_{\min}$ yields sparser, more interpretable equations at the risk of losing physically important terms; a higher $\lambda$ stabilises the normal equations when $\Theta^\top\Theta$ is ill-conditioned.

### Learned Equations

After fitting, `model.equations()` produces human-readable symbolic equations, e.g.:

```
x_next       = 1.000*x + 0.020*x_dot + ...
x_dot_next   = 1.000*x_dot + 0.010*theta + 0.020*u + ...
theta_next   = 1.000*theta + 0.020*theta_dot + ...
theta_dot_next = 0.290*sin(theta) + 0.040*theta_dot + 0.020*u + ...
```

The notebook prints these equations immediately after fitting to verify qualitative plausibility (e.g., that $\sin\theta$ appears in the angular velocity equation and $u$ has a non-zero coefficient in at least one state equation).

---

## Linearisation and LQR

The sparse identified model $F(\mathbf{x}, u) = \Theta([\mathbf{x}^\top, u]^\top)\Xi$ is linearised at the upright equilibrium $(\mathbf{x}^*, u^*) = (\mathbf{0}, 0)$ by central finite differences (identical to notebook 03):

$$A_{ij} = \frac{F_i(\mathbf{x}^* + \varepsilon\mathbf{e}_j,\, 0) - F_i(\mathbf{x}^* - \varepsilon\mathbf{e}_j,\, 0)}{2\varepsilon}, \qquad B_i = \frac{F_i(\mathbf{0},\, \varepsilon) - F_i(\mathbf{0},\, -\varepsilon)}{2\varepsilon}$$

Because $\Theta$ includes differentiable functions of $(\mathbf{x}, u)$, the analytic Jacobian exists, but finite differences are used for consistency across all model types.

The LQR gain $K$ is computed from the **Discrete Algebraic Riccati Equation**:

$$P = A^\top P A - A^\top P B (B^\top P B + R)^{-1} B^\top P A + Q$$

with weights set by **Bryson's rule** [Bryson & Ho, 1969]: $Q = \mathrm{diag}(1/\text{tol}_x^2, 1/\text{tol}_{\dot{x}}^2, 1/\text{tol}_\theta^2, 1/\text{tol}_{\dot\theta}^2)$, $R = 1/u_{\max}^2$. The heavy $\theta$ penalty ($Q_\theta \approx 820$) follows from the strict 2° settling tolerance relative to the 0.25 m cart tolerance — a task-specification consequence, not a physics equation. The control law is $u = -\alpha(K\mathbf{x})$.

---

## Hyperparameter Tuning

`train_sindy_lqr` performs a **joint grid search** over the STLSQ hyperparameters $(\xi_{\min}, \lambda)$ and the gain multiplier $\alpha$, evaluated by closed-loop rollout.

### $(\xi_{\min}, \lambda)$ Grid (5 candidates)

| $\xi_{\min}$ | $\lambda$ |
|---|---|
| $10^{-4}$ | $10^{-6}$ |
| $10^{-3}$ | $10^{-6}$ |
| $5 \times 10^{-3}$ | $10^{-6}$ |
| $10^{-3}$ | $10^{-4}$ |
| $5 \times 10^{-3}$ | $10^{-4}$ |

### $\alpha$ Range

$\alpha \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$

### Selection Criterion

$$\text{score}(\xi_{\min}, \lambda, \alpha) = c + 2000 \cdot \mathbb{1}[\text{not successful}]$$

### Physics Sanity Check

For each $(\xi_{\min}, \lambda)$ candidate, the predicted angular acceleration at the canonical initial state under maximum force is computed:

$$\ddot{\theta}_{\text{pred}} = \bigl(A\,\mathbf{x}_0 + B\,u_{\max} - \mathbf{x}_0\bigr)_3 / \Delta t$$

This scalar check verifies that the identified model has the correct sign and approximate magnitude for the dominant angular acceleration, catching degenerate sparse solutions that may have discarded the $\sin\theta$ or $u$ terms prematurely.

---

## Evaluation

9 rollouts (1 fixed + 8 jittered), same protocol as notebook 01. `controller.metadata` exposes:

- `model.equations` — symbolic equations of the discovered model
- `model.active_terms_per_state` — number of nonzero terms per state dimension
- $(A, B, K, P)$ — linearisation and LQR solution
- `alpha` — selected gain multiplier
- `tuning_results` — all candidates with scores and $\ddot{\theta}$ sanity values

---

---

## Validation Notebooks

Two additional notebooks extend the stress-test coverage of the SINDy-LQR pipeline.

### 05a — Stress Test (100 Random Initial Conditions)

**Notebook:** [notebooks/05a_sindy_lqr_stress_test.ipynb](../notebooks/05a_sindy_lqr_stress_test.ipynb)

Evaluates the pre-trained baseline controller on 100 randomly generated initial states.

**Protocol:**
- Each case draws an impulse force uniformly from [0.5, 5] N and a target crossing angle uniformly from [5, 90] degrees.  The initial state is the system state at the moment the pendulum first crosses that angle under the impulse.
- All 100 cases use the same seed (42) so the validation set is fully reproducible.
- Cases where the impulse cannot reach the target angle within the simulation horizon are marked infeasible.
- The baseline controller is loaded from `artifacts/controllers/sindy_lqr.pkl`; no retraining occurs.

**Outputs saved to `artifacts/validation/`:**

| File | Contents |
|---|---|
| `sindy_lqr_val_spec.npz` | Forces, angles, initial states, feasibility flags (seed=42) |
| `sindy_lqr_stress_rollouts.npz` | States/inputs/time arrays for all 100 rollouts |
| `sindy_lqr_stress_metrics.npz` | Per-case metrics (success, cost, settle time, …) |

**Plots:** success rate bucketed by target angle and impulse force; scatter of every case coloured by outcome (success / failure / infeasible IC).

---

### 05b — Five-Fold Cross-Validation

**Notebook:** [notebooks/05b_sindy_lqr_crossval.ipynb](../notebooks/05b_sindy_lqr_crossval.ipynb)

Requires `05a_sindy_lqr_stress_test.ipynb` to have been run first.

Retrains SINDy-LQR five times, each time on a different 80/20 partition of the validation cases augmented with the original training dataset, and evaluates each retrained controller on the 20 held-out cases.

**Protocol:**
- **Training data (per fold):** all transition pairs from the original 500-trajectory random-walk dataset plus the baseline-controller rollout trajectories for the 80 train-fold validation cases.
- **Hyperparameters:** threshold, ridge, and alpha are fixed to the values selected during baseline tuning so the comparison measures generalization, not a different configuration.
- **Fold splits:** cases are sorted by target angle and assigned to folds in round-robin order so each held-out fold covers the full angle range rather than clustering in an easy or hard region.
- **Evaluation:** each held-out case is evaluated from its original impulse-derived initial state using the fold-specific retrained controller.

**Outputs saved to `artifacts/validation/`:**

| File | Contents |
|---|---|
| `sindy_lqr_fold_metrics.npz` | Per-case metrics annotated with fold index |

**Result table:** per-fold success rate, median settle time, and median cost; aggregate CV metrics compared against the baseline controller on the same initial conditions.

---

## References

- Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences*, 113(15), 3932–3937.
- Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). Sparse Identification of Nonlinear Dynamics with Control (SINDy-C). *IFAC-PapersOnLine*, 49(18), 710–715.
- Kaiser, E., Kutz, J. N. & Brunton, S. L. (2018). Sparse identification of nonlinear dynamics for model predictive control in the low-data limit. *Proceedings of the Royal Society A*, 474(2219), 20180335.
- Anderson, B. D. O. & Moore, J. B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice-Hall.
- Bryson, A. E. & Ho, Y.-C. (1969). *Applied Optimal Control*. Blaisdell. (Bryson's rule for Q/R selection.)

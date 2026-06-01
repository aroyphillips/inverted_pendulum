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

| Index | Feature | Physical motivation |
|---|---|---|
| 0 | $1$ | constant / bias |
| 1–4 | $x,\; \dot{x},\; \theta,\; \dot{\theta}$ | linear state terms |
| 5 | $u$ | linear control term |
| 6–10 | $x^2,\; \dot{x}^2,\; \theta^2,\; \dot{\theta}^2,\; u^2$ | quadratic terms |
| 11–15 | $x\theta,\; \dot{x}\dot{\theta},\; \theta\dot{\theta},\; xu,\; \theta u$ | bilinear cross-terms |
| 16–17 | $\sin\theta,\; \cos\theta - 1$ | pendulum restoring force, kinematic coupling |
| 18 | $u\cos\theta$ | state-dependent control effectiveness |
| 19 | $\dot{\theta}^2\sin\theta$ | centripetal coupling in pole equation |

The physically motivated trig terms ($\sin\theta$ and $\dot{\theta}^2\sin\theta$) appear in the exact continuous-time equations of motion for the pendulum, so SINDy can in principle exactly recover the governing physics if the data is sufficiently rich.

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

The notebook prints these equations immediately after fitting to verify physical plausibility (e.g., that $\sin\theta$ dominates the $\dot{\theta}$ equation and $u$ has the correct sign).

---

## Linearisation and LQR

The sparse identified model $F(\mathbf{x}, u) = \Theta([\mathbf{x}^\top, u]^\top)\Xi$ is linearised at the upright equilibrium $(\mathbf{x}^*, u^*) = (\mathbf{0}, 0)$ by central finite differences (identical to notebook 03):

$$A_{ij} = \frac{F_i(\mathbf{x}^* + \varepsilon\mathbf{e}_j,\, 0) - F_i(\mathbf{x}^* - \varepsilon\mathbf{e}_j,\, 0)}{2\varepsilon}, \qquad B_i = \frac{F_i(\mathbf{0},\, \varepsilon) - F_i(\mathbf{0},\, -\varepsilon)}{2\varepsilon}$$

Because $\Theta$ includes differentiable functions of $(\mathbf{x}, u)$, the analytic Jacobian exists, but finite differences are used for consistency across all model types.

The LQR gain $K$ is computed from the **Discrete Algebraic Riccati Equation**:

$$P = A^\top P A - A^\top P B (B^\top P B + R)^{-1} B^\top P A + Q$$

with default weights $Q = \mathrm{diag}(1, 0.2, 100, 10)$, $R = 0.05$ (heavy penalty on $\theta$ and $\dot{\theta}$ to prioritise upright stabilisation over cart position). The control law is $u = -\alpha(K\mathbf{x})$.

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

## References

- Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). Discovering governing equations from data by sparse identification of nonlinear dynamical systems. *Proceedings of the National Academy of Sciences*, 113(15), 3932–3937.
- Brunton, S. L., Proctor, J. L. & Kutz, J. N. (2016). Sparse Identification of Nonlinear Dynamics with Control (SINDy-C). *IFAC-PapersOnLine*, 49(18), 710–715.
- Kaiser, E., Kutz, J. N. & Brunton, S. L. (2018). Sparse identification of nonlinear dynamics for model predictive control in the low-data limit. *Proceedings of the Royal Society A*, 474(2219), 20180335.
- Anderson, B. D. O. & Moore, J. B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice-Hall.

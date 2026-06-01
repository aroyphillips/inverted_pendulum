# 02 — Koopman-Lifted DeePC

**Notebook:** [notebooks/02_koopman_willems_deepc.ipynb](../notebooks/02_koopman_willems_deepc.ipynb)  
**Source:** [src/controllers/koopman_deepc.py](../src/controllers/koopman_deepc.py), [src/controllers/deepc.py](../src/controllers/deepc.py), [src/models/koopman.py](../src/models/koopman.py)  
**Artifact:** `artifacts/controllers/koopman_deepc.pkl`

---

## Overview

This notebook implements the **Willems' Fundamental Lemma** [Willems et al., 2005] for data-enabled predictive control in a nonlinear **Koopman feature space**. The core idea is that while the original dynamics $\mathbf{x}_{t+1} = f(\mathbf{x}_t, u_t)$ are nonlinear, the Koopman operator acts linearly on an infinite-dimensional space of observables. A finite nonlinear feature lift $\boldsymbol{\phi}: \mathbb{R}^4 \to \mathbb{R}^{n_z}$ approximates this lifting, after which the DeePC behavioural framework — which relies on linearity — becomes more accurate [Shi & Li, 2023; Pan et al., 2023].

---

## Koopman Operator Theory

For a dynamical system $\mathbf{x}_{t+1} = f(\mathbf{x}_t)$, the **Koopman operator** $\mathcal{K}$ acts on scalar observable functions $\varphi: \mathcal{X} \to \mathbb{R}$ as

$$(\mathcal{K}\varphi)(\mathbf{x}) = \varphi(f(\mathbf{x}))$$

so that $\mathcal{K}$ is a linear (albeit infinite-dimensional) operator even when $f$ is nonlinear. **Extended Dynamic Mode Decomposition (EDMD)** [Williams et al., 2015] approximates the action of $\mathcal{K}$ on a finite dictionary $\{\varphi_1, \ldots, \varphi_{n_z}\}$ by solving the least-squares problem:

$$\min_{A_K, B_K} \sum_t \bigl\|\boldsymbol{\phi}(\mathbf{x}_{t+1}) - A_K\,\boldsymbol{\phi}(\mathbf{x}_t) - B_K\,u_t\bigr\|^2$$

The EDMD solution provides the finite-rank approximation of the controlled Koopman operator, enabling linear analysis of nonlinear systems.

---

## Feature Lift

`KoopmanLift` maps the 4D state $\mathbf{x} = [x, \dot{x}, \theta, \dot{\theta}]^\top$ to $\mathbf{z} = \boldsymbol{\phi}(\mathbf{x}) \in \mathbb{R}^{13}$ via the feature dictionary:

$$\boldsymbol{\phi}(\mathbf{x}) = \bigl[x,\; \dot{x},\; \theta,\; \dot{\theta},\; \sin\theta,\; \cos\theta - 1,\; x^2,\; \dot{x}^2,\; \theta^2,\; \dot{\theta}^2,\; x\theta,\; \dot{x}\dot{\theta},\; \theta\dot{\theta}\bigr]^\top$$

The trigonometric terms $\sin\theta$ and $\cos\theta - 1$ are motivated by the analytic equations of motion; the quadratic terms capture second-order coupling. After lifting, all features are **standardised** to zero mean and unit variance across the dataset:

$$\tilde{\mathbf{z}} = \Sigma^{-1/2}(\boldsymbol{\phi}(\mathbf{x}) - \boldsymbol{\mu}_\phi)$$

where $\boldsymbol{\mu}_\phi$ and $\Sigma$ are fit from the training data via `NumpyStandardScaler`. Standardisation ensures that no lifted dimension dominates the Hankel row space simply by having a larger dynamic range.

`edmdc_diagnostics` fits the ridge-regularised EDMD model $\hat{A}_K, \hat{B}_K$ and reports the one-step RMSE on held-out lifted states as a diagnostic of how well the lifting linearises the dynamics.

---

## Willems' Fundamental Lemma and Hankel Matrices

For a linear time-invariant (LTI) system of order $n$, **Willems' Fundamental Lemma** [Willems et al., 2005] states: if the input sequence $u^d$ used to generate the dataset is **persistently exciting of order** $T_{\text{ini}} + N + n$, then any $L$-step input–output trajectory $(u, y)$ of the system is a linear combination of the columns of the Hankel matrix $\mathcal{H}_L(u^d, y^d)$.

In practice, the multi-trajectory Hankel matrices are constructed by sliding a window of length $T_{\text{ini}} + N$ over each of the 500 trajectories and stacking the resulting columns (randomly subsampled to at most 1500):

$$\mathcal{H} = \begin{bmatrix} U_p \\ U_f \\ Z_p \\ Z_f \end{bmatrix} \in \mathbb{R}^{(T_\text{ini}+N)(m+n_z) \times \text{cols}}$$

where $m=1$ is the input dimension, $n_z = 13$ is the lifted state dimension, $T_{\text{ini}} = 3$, and $N = 50$ is the prediction horizon.

---

## QP Formulation

At each timestep, `DeePCController.act` solves the quadratic programme:

$$\min_{\mathbf{g},\, \boldsymbol{\sigma}} \quad \underbrace{\mathbf{z}_f^\top \bar{Q}\, \mathbf{z}_f}_{\text{stage cost}} + \underbrace{\mathbf{u}_f^\top \bar{R}\, \mathbf{u}_f}_{\text{input cost}} + \underbrace{\lambda_g \|\mathbf{g}\|^2}_{\text{Tikhonov}} + \underbrace{\lambda_{\sigma} \|\boldsymbol{\sigma}\|^2}_{\text{ini. slack}} + \underbrace{\lambda_{\text{rate}} \|D\,\mathbf{u}_f\|^2}_{\text{smoothness}} + \underbrace{\lambda_{\text{term}}\, \mathbf{z}_T^\top Q\, \mathbf{z}_T}_{\text{terminal}}$$

subject to

$$\begin{bmatrix} U_p \\ Z_p \end{bmatrix} \mathbf{g} = \begin{bmatrix} \mathbf{u}_\text{ini} \\ \tilde{\mathbf{z}}_\text{ini} \end{bmatrix} + \boldsymbol{\sigma}, \qquad -u_{\max}\mathbf{1} \leq \mathbf{u}_f \leq u_{\max}\mathbf{1}$$

with $\mathbf{z}_f = Z_f\mathbf{g}$, $\mathbf{u}_f = U_f\mathbf{g}$, and $\mathbf{z}_T$ the last $n_z$ entries of $\mathbf{z}_f$ (terminal lifted state). The block-diagonal matrices $\bar{Q} = I_N \otimes Q$ and $\bar{R} = I_N \otimes R$ expand the per-step cost across the horizon. $D \in \mathbb{R}^{(N-1) \times N}$ is the finite-difference matrix enforcing input-rate smoothness.

The cost weight on lifted dimensions is:

$$Q = \mathrm{diag}\underbrace{[1.0,\; 0.2,\; 100.0,\; 10.0}_{\text{original states}},\; \underbrace{0.1,\; \ldots,\; 0.1}_{9 \text{ lifted dims}}], \qquad R = [0.05]$$

### Regularisation parameters

| Symbol | Role | Value |
|---|---|---|
| $\lambda_g$ | Tikhonov regularisation on $\mathbf{g}$ (prevents overfitting to noise) | $10^{-4}$ |
| $\lambda_\sigma$ | Penalty for past-data inconsistency (soft constraint) | $10^{4}$ |
| $\lambda_{\text{rate}}$ | Input-rate smoothness (anti-chattering) | $10^{-3}$ |
| $\lambda_{\text{term}}$ | Terminal state multiplier | $10.0$ |

The QP is built once with **CVXPY** and solved at each step by **OSQP** [Stellato et al., 2020] with warm-starting, reusing the previous iterate to reduce solve time.

---

## Fallback

When OSQP returns a non-optimal status, the PID controller (artifact `artifacts/controllers/pid.json`) provides the action for that step. The raw 4D state — not the lifted state — is passed to the fallback. Fallback rate and per-step solve times are reported by `controller.diagnostics()`.

---

## Evaluation

9 rollouts (1 fixed + 8 jittered), same protocol as notebook 01. `controller.diagnostics()` additionally reports QP feasibility rate and mean solve time per step.

---

## References

- Willems, J. C., Rapisarda, P., Markovsky, I. & De Moor, B. L. M. (2005). A note on persistency of excitation. *Systems & Control Letters*, 54(4), 325–329.
- Coulson, J., Lygeros, J. & Dörfler, F. (2019). Data-Enabled Predictive Control: In the Shallows of the DeePC. *Proceedings of the European Control Conference (ECC)*, 307–312.
- Koopman, B. O. (1931). Hamiltonian systems and transformation in Hilbert space. *Proceedings of the National Academy of Sciences*, 17(5), 315–318.
- Williams, M. O., Kevrekidis, I. G. & Rowley, C. W. (2015). A Data-Driven Approximation of the Koopman Operator: Extending Dynamic Mode Decomposition. *Journal of Nonlinear Science*, 25(6), 1307–1346.
- Shi, H. & Li, P. (2023). A Koopman-based DeePC framework for data-driven predictive control of nonlinear systems. *arXiv:2302.14187*.
- Stellato, B., Banjac, G., Goulart, P., Bemporad, A. & Boyd, S. (2020). OSQP: An operator splitting solver for quadratic programs. *Mathematical Programming Computation*, 12, 637–672.

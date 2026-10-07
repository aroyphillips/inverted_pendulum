# 03 — Gaussian Process Dynamics + LQR

**Notebook:** [notebooks/03_gp_lqr.ipynb](../notebooks/03_gp_lqr.ipynb)  
**Source:** [src/controllers/gp_lqr.py](../src/controllers/gp_lqr.py), [src/models/gp_dynamics.py](../src/models/gp_dynamics.py)  
**Artifact:** `artifacts/controllers/gp_lqr.pkl`

---

## Overview

This notebook implements an **indirect model-based control** pipeline: (1) fit a nonparametric **Gaussian Process** model of the one-step state transitions from data, (2) linearise the learned mean dynamics at the upright equilibrium by finite differences, and (3) compute a **discrete Linear Quadratic Regulator** from the linearised model. The GP provides principled uncertainty quantification over the dynamics; the near-upright validation RMSE and predictive uncertainty are used as data-quality diagnostics before the LQR is trusted.

**Method character:** The pipeline is *model-based hybrid*, not fully black-box. The GP is learned from data with no physics equations, but the LQR gain is derived from a local linearisation of that learned model. Cost matrices $(Q, R)$ are selected by a heuristic-guided rollout search (Bryson anchor + alternatives), not by a pure model-free optimizer. See the [Q/R Tuning](#qr-tuning) section for a precise description of what is and is not data-driven.

---

## GP Dynamics Model

The system transition is modelled as

$$\mathbf{x}_{t+1} = \mathbf{x}_t + \boldsymbol{\Delta}(\mathbf{x}_t, u_t) + \boldsymbol{\epsilon}_t, \qquad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \sigma_n^2 I)$$

where $\boldsymbol{\Delta}: \mathbb{R}^5 \to \mathbb{R}^4$ is the unknown one-step state delta. `GPDynamicsModel` fits one independent GP $f_j \sim \mathcal{GP}(0, k(\cdot, \cdot))$ per output dimension $j \in \{1,\ldots,4\}$.

### Kernel

The covariance function combines a **squared-exponential** (RBF) kernel with automatic relevance determination (ARD) and a white noise term:

$$k(\mathbf{z}, \mathbf{z}') = \sigma_f^2 \exp\!\left(-\tfrac{1}{2}(\mathbf{z}-\mathbf{z}')^\top \Lambda^{-2} (\mathbf{z}-\mathbf{z}')\right) + \sigma_n^2\,\delta(\mathbf{z}, \mathbf{z}')$$

where $\sigma_f^2$ is the signal variance, $\Lambda = \mathrm{diag}(\ell_1, \ldots, \ell_5)$ are per-dimension length scales (ARD), and $\sigma_n^2$ is the noise variance. All hyperparameters $\{\sigma_f, \ell_1, \ldots, \ell_5, \sigma_n\}$ are optimised by maximising the log marginal likelihood [Rasmussen & Williams, 2006]:

$$\log p(\mathbf{y} | X) = -\tfrac{1}{2}\mathbf{y}^\top K^{-1}\mathbf{y} - \tfrac{1}{2}\log |K| - \tfrac{n}{2}\log 2\pi$$

where $K_{ij} = k(\mathbf{z}_i, \mathbf{z}_j)$ is the Gram matrix of training inputs.

Initialisation bounds are: $\sigma_f^2 \in [10^{-2}, 10^2]$, $\sigma_n^2 \in [10^{-8}, 1]$.

### Input Preprocessing

Both inputs $\tilde{\mathbf{z}} = (\mathbf{z} - \boldsymbol{\mu}_z)/\boldsymbol{\sigma}_z$ and targets $\tilde{\boldsymbol{\Delta}} = (\boldsymbol{\Delta} - \boldsymbol{\mu}_\Delta)/\boldsymbol{\sigma}_\Delta$ are standardised before fitting (separate `StandardScaler` for inputs and outputs). This prevents numerical ill-conditioning in the Gram matrix and avoids any one output dimension dominating length-scale optimisation.

### Training Data

`stratified_transition_sample` draws up to $N_{\text{train}} = 5000$ transitions, stratified by binned pole angle $\theta$, so that near-upright samples (where the GP must be accurate for control) are not systematically underrepresented despite being rare in random-walk data.

A minimum of 10 near-upright transitions ($|\theta| < 0.1$ rad) in the training split is required for the linearisation to be considered reliable. A warning is issued if this threshold is not met.

### Posterior Prediction

The posterior predictive mean and variance for a new input $\mathbf{z}^*$ are:

$$\mu_j(\mathbf{z}^*) = \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}_j$$
$$\sigma_j^2(\mathbf{z}^*) = k(\mathbf{z}^*, \mathbf{z}^*) - \mathbf{k}_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{k}_*$$

where $\mathbf{k}_* = [k(\mathbf{z}^*, \mathbf{z}_1), \ldots, k(\mathbf{z}^*, \mathbf{z}_n)]^\top$. Only the posterior mean is used in the controller; $\sigma_j^2$ is reported as a model-confidence diagnostic.

---

## Linearisation

The learned mean transition map $F: (\mathbf{x}, u) \mapsto \mathbf{x} + \hat{\boldsymbol{\Delta}}(\mathbf{x}, u)$ is linearised at the upright equilibrium $(\mathbf{x}^*, u^*) = (\mathbf{0}, 0)$ by **central finite differences**:

$$A_{ij} = \frac{F_i(\mathbf{x}^* + \varepsilon\mathbf{e}_j,\, u^*) - F_i(\mathbf{x}^* - \varepsilon\mathbf{e}_j,\, u^*)}{2\varepsilon}, \qquad B_i = \frac{F_i(\mathbf{x}^*,\, u^* + \varepsilon) - F_i(\mathbf{x}^*,\, u^* - \varepsilon)}{2\varepsilon}$$

with $\varepsilon = 10^{-4}$. This yields the discrete-time Jacobians $A \in \mathbb{R}^{4\times4}$, $B \in \mathbb{R}^{4\times1}$ of the learned dynamics at the operating point.

### Linearisation Diagnostics

`linearize_with_diagnostics` returns a diagnostics dictionary alongside $(A, B)$:

| Field | Description |
|---|---|
| `controllability_rank` | Rank of the controllability matrix $[B\ AB\ A^2B\ A^3B]$ |
| `controllable` | True iff rank equals state dimension (4) |
| `eigenvalues_A` | Eigenvalues of $A$ (stability indicator) |
| `norm_B` | Frobenius norm of $B$ (near-zero indicates a degenerate input Jacobian) |
| `gp_std_at_eq` | GP predictive standard deviation at the equilibrium (data-coverage indicator) |

If the pair is uncontrollable or $B$ is near-zero, the DARE will be skipped and a warning is issued.

---

## Discrete LQR

Given the linearised model $(A, B)$, the LQR seeks the static gain $K$ that minimises the infinite-horizon quadratic cost

$$V(\mathbf{x}_0) = \sum_{t=0}^{\infty} \bigl[\mathbf{x}_t^\top Q\,\mathbf{x}_t + u_t^\top R\,u_t\bigr]$$

subject to $\mathbf{x}_{t+1} = A\mathbf{x}_t + B u_t$. The solution is obtained via the **Discrete Algebraic Riccati Equation (DARE)**:

$$P = A^\top P A - A^\top P B\,(B^\top P B + R)^{-1} B^\top P A + Q$$

with optimal gain

$$K = (B^\top P B + R)^{-1} B^\top P A$$

and control law $u = -\alpha (K\mathbf{x})$, where $\alpha$ is a scalar multiplier tuned by closed-loop rollout validation.

### DARE Robustness

`safe_discrete_lqr_gain` wraps the solver with the following guardrails:

1. **Controllability check** — if rank$([B\ AB\ A^2B\ A^3B]) < 4$, the solve is skipped.
2. **Finiteness check** — non-finite $A$ or $B$ entries skip the solve.
3. **Damped fallback** — if the standard DARE fails (e.g., near-unit-circle eigenvalues), a second attempt uses $A \leftarrow 0.99\,A$ to regularise the problem.
4. **Structured diagnostics** — all outcomes (success, failure reason, solver status) are recorded in `controller.metadata['lqr_diagnostics']`.

If every Q/R candidate fails the DARE, `train_gp_lqr` raises a `RuntimeError` with diagnostic information rather than silently returning a broken controller.

---

## Q/R Tuning {#qr-tuning}

`train_gp_lqr` fits the GP **once**, then jointly searches over cost matrices $(Q, R)$ and the gain multiplier $\alpha$ using closed-loop rollouts on the canonical initial state.

### What is and is not data-driven

The GP model is learned entirely from data (no physics equations). However, the $(Q, R)$ search space is defined by two human-specified anchor types:

- **Bryson's rule** [Bryson & Ho, 1969]: $Q_i = 1/\text{tol}_i^2$ from per-state settling tolerances, $R = 1/u_{\max}^2$. This encodes the task specification, not a black-box optimum.
- **Non-Bryson alternatives**: additional candidates with different weight orderings that are not derived from Bryson's rule.

**The selection step is data-driven** in the sense that the winning $(Q, R, \alpha)$ triple is chosen by rollout performance on the learned model. But the search space is constructed from human heuristics. This should not be described as a fully black-box Q/R optimizer.

### Q/R Grid (8 candidates)

| Candidate | $Q$ | $R$ | Description |
|---|---|---|---|
| 1 | $Q_{\text{Bryson}} \times 0.25$ | $R_{\text{Bryson}}$ | Softer state cost |
| 2 | $Q_{\text{Bryson}}$ | $R_{\text{Bryson}}$ | Bryson anchor |
| 3 | $Q_{\text{Bryson}} \times 4.0$ | $R_{\text{Bryson}}$ | Stricter state cost |
| 4 | $Q_{\text{Bryson}}$ | $R_{\text{Bryson}} \times 0.25$ | Softer input cost |
| 5 | $Q_{\text{Bryson}}$ | $R_{\text{Bryson}} \times 4.0$ | Stricter input cost |
| 6 | $\text{diag}(4,1,400,4)$ | $0.04$ | Pole-priority, low $R$ |
| 7 | $\text{diag}(10,10,200,20)$ | $0.1$ | Balanced penalties |
| 8 | $\text{diag}(1,1,4000,50)$ | $0.5$ | Pole-angle dominated |

Candidates that fail the DARE (uncontrollable pair, ill-conditioned $A$) are skipped and logged.

### $\alpha$ Range

$\alpha \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$

This multiplier compensates for modelling error: if the GP linearisation underestimates the true system gain, $\alpha > 1$ recovers sufficient control authority.

### Selection Criterion

$$\text{score}(Q, R, \alpha) = c + 2000 \cdot \mathbb{1}[\text{not successful}]$$

where $c$ is the rollout LQR cost. The best $(Q, R, \alpha)$ minimises this score; all candidates (including DARE failures) are logged in `controller.metadata["tuning_results"]`.

---

## Evaluation

9 rollouts (1 fixed + 8 jittered), same protocol as notebook 01. `controller.metadata` exposes learned $(A, B, K, P)$ matrices, the selected $\alpha$, near-upright RMSE, near-upright train/val point counts, linearisation diagnostics, DARE diagnostics, and the full tuning grid results.

---

## References

- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Anderson, B. D. O. & Moore, J. B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice-Hall.
- Bryson, A. E. & Ho, Y.-C. (1969). *Applied Optimal Control*. Blaisdell. (Bryson's rule for Q/R selection.)
- Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A Model-Based and Data-Efficient Approach to Policy Search. *Proceedings of the 28th ICML*, 465–472.
- Hewing, L., Kabzan, J. & Zeilinger, M. N. (2020). Cautious Model Predictive Control Using Gaussian Process Regression. *IEEE Transactions on Control Systems Technology*, 28(6), 2736–2743.

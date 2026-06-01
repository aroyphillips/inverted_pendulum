# 03 — Gaussian Process Dynamics + LQR

**Notebook:** [notebooks/03_gp_lqr.ipynb](../notebooks/03_gp_lqr.ipynb)  
**Source:** [src/controllers/gp_lqr.py](../src/controllers/gp_lqr.py), [src/models/gp_dynamics.py](../src/models/gp_dynamics.py)  
**Artifact:** `artifacts/controllers/gp_lqr.pkl`

---

## Overview

This notebook implements an **indirect model-based control** pipeline: (1) fit a nonparametric **Gaussian Process** model of the one-step state transitions from data, (2) linearise the learned mean dynamics at the upright equilibrium, and (3) compute a **discrete Linear Quadratic Regulator** from the linearised model. The GP provides principled uncertainty quantification over the dynamics, and the near-upright validation RMSE is used as a data-quality diagnostic before the LQR is trusted.

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

---

## Discrete LQR

Given the linearised model $(A, B)$, the LQR seeks the static gain $K$ that minimises the infinite-horizon quadratic cost

$$V(\mathbf{x}_0) = \sum_{t=0}^{\infty} \bigl[\mathbf{x}_t^\top Q\,\mathbf{x}_t + u_t^\top R\,u_t\bigr]$$

subject to $\mathbf{x}_{t+1} = A\mathbf{x}_t + B u_t$. The solution is obtained via the **Discrete Algebraic Riccati Equation (DARE)**:

$$P = A^\top P A - A^\top P B\,(B^\top P B + R)^{-1} B^\top P A + Q$$

with optimal gain

$$K = (B^\top P B + R)^{-1} B^\top P A$$

and control law $u = -\alpha (K\mathbf{x})$, where $\alpha$ is a scalar multiplier tuned by closed-loop rollout validation.

---

## Hyperparameter Tuning

`train_gp_lqr` fits the GP **once**, then jointly searches over cost matrices $(Q, R)$ and the gain multiplier $\alpha$ using closed-loop rollouts on the canonical initial state.

### Q/R Grid (5 candidates)

| $Q$ diagonal | $R$ |
|---|---|
| $[1, 0.2, 10, 1]$ | $0.10$ |
| $[1, 0.2, 20, 1]$ | $0.05$ |
| $[1, 0.2, 50, 5]$ | $0.05$ |
| $[1, 0.2, 100, 10]$ | $0.05$ |
| $[1, 0.2, 100, 10]$ | $0.01$ |

The increasing weight on $\theta$ (from 10 to 100) and $\dot{\theta}$ reflects the physical priority of keeping the pole upright over minimising cart displacement.

### $\alpha$ Range

$\alpha \in \{0.5, 0.75, 1.0, 1.25, 1.5, 2.0\}$

This multiplier compensates for modelling error: if the GP linearisation underestimates the true system gain, $\alpha > 1$ recovers sufficient control authority.

### Selection Criterion

$$\text{score}(Q, R, \alpha) = c + 2000 \cdot \mathbb{1}[\text{not successful}]$$

where $c$ is the rollout LQR cost. The best $(Q, R, \alpha)$ minimises this score; all candidates are logged in `controller.metadata["tuning_results"]`.

---

## Evaluation

9 rollouts (1 fixed + 8 jittered), same protocol as notebook 01. `controller.metadata` exposes learned $(A, B, K, P)$ matrices, the selected $\alpha$, near-upright RMSE, and the full tuning grid results.

---

## References

- Rasmussen, C. E. & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press.
- Anderson, B. D. O. & Moore, J. B. (1990). *Optimal Control: Linear Quadratic Methods*. Prentice-Hall.
- Deisenroth, M. P. & Rasmussen, C. E. (2011). PILCO: A Model-Based and Data-Efficient Approach to Policy Search. *Proceedings of the 28th ICML*, 465–472.
- Hewing, L., Kabzan, J. & Zeilinger, M. N. (2020). Cautious Model Predictive Control Using Gaussian Process Regression. *IEEE Transactions on Control Systems Technology*, 28(6), 2736–2743.

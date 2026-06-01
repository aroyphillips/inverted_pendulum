# 06 — Regularized DeePC

**Notebook:** [notebooks/06_regularized_deepc.ipynb](../notebooks/06_regularized_deepc.ipynb)  
**Source:** [src/controllers/deepc.py](../src/controllers/deepc.py), [src/models/hankel.py](../src/models/hankel.py)  
**Artifact:** `artifacts/controllers/deepc.pkl`

---

## Overview

This notebook implements **Data-Enabled Predictive Control (DeePC)** [Coulson et al., 2019] directly in the measured 4D state space. DeePC is a non-parametric, data-driven model predictive control framework that replaces the system model with a behavioural representation derived directly from historical input–output trajectories, avoiding any explicit system identification step. The theoretical underpinning is **Willems' Fundamental Lemma** [Willems et al., 2005], which establishes that a sufficiently rich finite dataset is sufficient to represent all future system behaviours.

This notebook is the non-parametric counterpart to the Koopman-lifted DeePC in notebook 02; both use the same QP structure but differ in state representation (4D measured states vs. 13D lifted states).

---

## Willems' Fundamental Lemma

For a controllable LTI system of order $n$ with input dimension $m$ and output dimension $p$, Willems' Fundamental Lemma [Willems et al., 2005] states:

> Let $(u^d, y^d)$ be a trajectory of length $T$ with $u^d$ **persistently exciting of order** $L + n$. Then the set of all length-$L$ trajectories of the system equals the column span of the block Hankel matrix $\mathcal{H}_L(u^d, y^d)$.

Persistent excitation of order $L + n$ requires $T \geq (m+1)(L+n) - 1$. In practice, the 500 random-walk trajectories of length 1000 provide far more data than the minimum requirement, and the rank diagnostics (printed via `controller.diagnostics()`) confirm that $\text{rank}(U_f)$ matches the expected value.

The multi-trajectory extension [van Waarde et al., 2020] used here concatenates Hankel columns from all 500 trajectories, with columns randomly subsampled to a cap of 1500 to keep the QP tractable.

---

## Hankel Partitioning

For each sliding window of length $T_{\text{ini}} + N$ within a trajectory, the data is partitioned into four block matrices:

$$\mathcal{H} = \begin{bmatrix} U_p \\ U_f \\ Y_p \\ Y_f \end{bmatrix}, \qquad \text{where} \quad U_p,U_f \in \mathbb{R}^{T_\text{ini} \cdot m \times \text{cols}}, \quad Y_p,Y_f \in \mathbb{R}^{T_\text{ini} \cdot p \times \text{cols}}$$

with $m = 1$ (force input), $p = 4$ (state output), $T_{\text{ini}} = 3$, $N = 50$ (horizon), and `cols` $\leq 1500$.

The subscripts $p$ and $f$ denote *past* ($T_\text{ini}$ samples) and *future* ($N$ samples) sub-blocks. The **past** sub-blocks $U_p$, $Y_p$ collectively form the *initial condition* constraint: the coefficients $\mathbf{g}$ must reproduce the most recent $T_{\text{ini}}$ steps of observed data. The **future** sub-blocks $U_f$, $Y_f$ define the predicted trajectory over the horizon.

---

## QP Formulation

At each control step, DeePC selects a coefficient vector $\mathbf{g} \in \mathbb{R}^{\text{cols}}$ that weights the library of past trajectories, with the following objective:

$$\min_{\mathbf{g},\, \boldsymbol{\sigma}} \quad \underbrace{\mathbf{y}_f^\top \bar{Q}\, \mathbf{y}_f + \mathbf{u}_f^\top \bar{R}\, \mathbf{u}_f}_{\text{predicted stage cost}} + \underbrace{\lambda_g \|\mathbf{g}\|^2}_{\text{Tikhonov regularisation}} + \underbrace{\lambda_\sigma \|\boldsymbol{\sigma}\|^2}_{\text{ini. consistency slack}} + \underbrace{\lambda_{\text{rate}} \|D\,\mathbf{u}_f\|^2}_{\text{input smoothness}} + \underbrace{\lambda_{\text{term}}\,\mathbf{y}_T^\top Q\,\mathbf{y}_T}_{\text{terminal cost}}$$

subject to

$$\underbrace{\begin{bmatrix} U_p \\ Y_p \end{bmatrix}}_{\text{ini. matrix}} \mathbf{g} = \underbrace{\begin{bmatrix} \mathbf{u}_\text{ini} \\ \mathbf{y}_\text{ini} \end{bmatrix}}_{\text{current ini.}} + \boldsymbol{\sigma}, \qquad -u_{\max}\mathbf{1} \leq \mathbf{u}_f \leq u_{\max}\mathbf{1}$$

where

- $\mathbf{y}_f = Y_f\mathbf{g} \in \mathbb{R}^{N \cdot p}$ is the predicted future state sequence
- $\mathbf{u}_f = U_f\mathbf{g} \in \mathbb{R}^{N \cdot m}$ is the predicted future input sequence
- $\boldsymbol{\sigma} \in \mathbb{R}^{T_\text{ini}(m+p)}$ is a slack variable softening the initialisation constraint
- $\mathbf{y}_T$ is the last $p$ entries of $\mathbf{y}_f$ (terminal predicted state)
- $\bar{Q} = I_N \otimes Q$ and $\bar{R} = I_N \otimes R$ are block-diagonal cost matrices
- $D \in \mathbb{R}^{(N-1) \times N}$ is the finite-difference operator: $D_{i,i} = -1$, $D_{i,i+1} = 1$

The first element of $U_f\mathbf{g}$ is applied as the control action; the remainder of the predicted trajectory is discarded (receding horizon).

### Initialisation Buffer

At runtime, $\mathbf{u}_\text{ini}$ is maintained as a rolling buffer of the last $T_\text{ini} = 3$ applied forces. $\mathbf{y}_\text{ini}$ is constructed by repeating the current state $T_\text{ini}$ times (Markovian initialisation), which is valid for the inverted pendulum since the full state is observable.

### Regularisation Roles

| Parameter | Symbol | Role | Value |
|---|---|---|---|
| Tikhonov penalty | $\lambda_g$ | Prevents $\mathbf{g}$ from concentrating on a single noisy trajectory; acts as an implicit model regulariser | $10^{-4}$ |
| Initialisation slack | $\lambda_\sigma$ | Relaxes exact past-data consistency (necessary for noisy data) while strongly penalising deviation | $10^4$ |
| Input rate | $\lambda_{\text{rate}}$ | Penalises $\|\Delta u_t\|^2$; reduces actuation chattering and mechanical wear | $10^{-3}$ |
| Terminal cost | $\lambda_{\text{term}}$ | Additional weight on $\mathbf{y}_T$ beyond the stage cost; improves stability by approximating the infinite-horizon value function at the horizon boundary | $10.0$ |

The effect of $\lambda_g$ on robustness is studied in [Coulson et al., 2019; Berberich et al., 2020]: too small allows over-fitting to noise; too large biases the solution toward the origin of $\mathbf{g}$-space.

### Cost Matrices

$$Q = \mathrm{diag}(1.0,\; 0.2,\; 100.0,\; 10.0), \qquad R = [0.05]$$

The large weights on $\theta$ (100) and $\dot{\theta}$ (10) reflect the physical priority of keeping the pole upright. Cart position is weakly penalised.

---

## Solver

The QP is compiled **once** using CVXPY [Diamond & Boyd, 2016] and solved at every control step by **OSQP** [Stellato et al., 2020] with **warm starting**, which reuses the previous optimal $\mathbf{g}$ as the initial point and typically reduces solve time by 3–10× versus cold starts. OSQP exploits the sparse structure of the KKT system, making it suitable for the large Hankel matrices arising here.

If OSQP returns a non-optimal status (`infeasible`, `unbounded`, or solver error), the PID fallback controller handles that timestep. The fallback rate is monitored via `controller.diagnostics()`.

---

## Hankel Rank Diagnostics

`rank_report` computes the numerical rank of $Y_p$ and $U_f$ using a tolerance of $10^{-6} \cdot \sigma_{\max}$. The resulting rank, stable rank ($\|M\|_F^2 / \|M\|_2^2$), and effective rank are stored in `controller.metadata`. A rank deficiency in $U_f$ signals that the data lacks sufficient excitation for the prediction horizon, which would violate the conditions of the Fundamental Lemma.

---

## Evaluation

9 rollouts (1 fixed + 8 jittered), same protocol as notebooks 01–05. `controller.diagnostics()` reports:

- QP status counts (`optimal`, `optimal_inaccurate`, `infeasible`, etc.)
- Feasibility rate (fraction of steps solved to optimality)
- Mean per-step solve time (ms)

---

## References

- Willems, J. C., Rapisarda, P., Markovsky, I. & De Moor, B. L. M. (2005). A note on persistency of excitation. *Systems & Control Letters*, 54(4), 325–329.
- Coulson, J., Lygeros, J. & Dörfler, F. (2019). Data-Enabled Predictive Control: In the Shallows of the DeePC. *Proceedings of the European Control Conference (ECC)*, 307–312.
- Berberich, J., Köhler, J., Müller, M. A. & Allgöwer, F. (2020). Data-Driven Model Predictive Control with Stability and Robustness Guarantees. *IEEE Transactions on Automatic Control*, 66(4), 1702–1717.
- van Waarde, H. J., De Persis, C., Camlibel, M. K. & Tesi, P. (2020). Willems' Fundamental Lemma for State-Space Systems and Its Extension to Multiple Datasets. *IEEE Control Systems Letters*, 4(3), 602–607.
- Diamond, S. & Boyd, S. (2016). CVXPY: A Python-Embedded Modeling Language for Convex Optimization. *Journal of Machine Learning Research*, 17(83), 1–5.
- Stellato, B., Banjac, G., Goulart, P., Bemporad, A. & Boyd, S. (2020). OSQP: An operator splitting solver for quadratic programs. *Mathematical Programming Computation*, 12, 637–672.

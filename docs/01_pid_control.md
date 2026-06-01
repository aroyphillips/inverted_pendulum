# 01 — PID Baseline Controller

**Notebook:** [notebooks/01_pid_control.ipynb](../notebooks/01_pid_control.ipynb)  
**Source:** [src/controllers/pid.py](../src/controllers/pid.py)  
**Artifact:** `artifacts/controllers/pid.json`

---

## System Description

The inverted pendulum on a cart is described by the state vector

$$\mathbf{x} = [x,\; \dot{x},\; \theta,\; \dot{\theta}]^\top \in \mathbb{R}^4$$

where $x$ (m) is the cart position, $\dot{x}$ (m/s) the cart velocity, $\theta$ (rad) the pole angle from vertical, and $\dot{\theta}$ (rad/s) the pole angular velocity. The scalar control input $u$ (N) is a horizontal force applied to the cart, saturated to $|u| \leq u_{\max} = 5$ N.

---

## Control Law

The `PIDController` implements a linear combination of proportional, derivative, and integral terms on both the pole angle and the cart position:

$$u(t) = -k_{\theta p}\,\theta(t) - k_{\theta d}\,\dot{\theta}(t) - k_i\!\int_0^t\!\theta(\tau)\,d\tau - k_{xp}\,x(t) - k_{xd}\,\dot{x}(t)$$

The integral is approximated by the trapezoidal rule and clamped:

$$I_\theta(t) \leftarrow \mathrm{clip}\!\left(I_\theta(t-\Delta t) + \tfrac{\Delta t}{2}\bigl[\theta(t) + \theta(t-\Delta t)\bigr],\; -I_{\max},\; I_{\max}\right)$$

with $I_{\max} = 0.1$ rad·s. The final command is hard-saturated:

$$u_{\text{applied}} = \mathrm{clip}(u,\; -u_{\max},\; u_{\max})$$

This structure is equivalent to the static gain form $u = -\mathbf{K}\mathbf{x}$ from a discrete LQR (dropping the integrator), which motivates the LQR-informed warm-start used during tuning (see §Tuning, Stage 1).

The `TwoPhasePIDController` variant extends this by switching between separate *recovery* gains (when $|\theta| > \theta_{\text{switch}}$) and *regulation* gains (when near upright), allowing more aggressive initial corrections while preserving stability near equilibrium.

---

## Training / Hyperparameter Tuning

Gain tuning solves the following black-box optimisation over the gain vector $\mathbf{k} = [k_{\theta p}, k_{\theta d}, k_{xp}, k_{xd}, k_i]^\top$:

$$\min_{\mathbf{k}} \quad J(\mathbf{k}) = \frac{1}{N}\sum_{i=1}^{N} c_i(\mathbf{k}) + 2000\cdot(N - S(\mathbf{k}))$$

where $N$ is the number of test rollouts, $S(\mathbf{k})$ is the number of successful rollouts (see §Success Criterion), and $c_i(\mathbf{k})$ is the LQR-style trajectory cost for rollout $i$:

$$c_i(\mathbf{k}) = \int_0^{T} \bigl[\mathbf{x}(t)^\top Q\,\mathbf{x}(t) + u(t)^\top R\,u(t)\bigr]\,dt$$

with $Q = \mathrm{diag}(1, 0.2, 20, 1)$ and $R = 0.05$. The penalty term $2000(N-S)$ strongly prioritises feasibility (all four settling tolerances satisfied) over raw cost minimisation.

**Test states:** the canonical `initial_state` plus $N_{\text{jitter}} = 4$ perturbations drawn from $\mathcal{N}(\mathbf{0}, \mathrm{diag}(0.02, 0.05, 0.03, 0.10)^2)$.

`tune_pid` uses a three-stage budget of $n_{\text{trials}} = 300$ total function evaluations:

### Stage 1 — Seed Initialisation and Random Sampling (~100 evaluations)

Five deterministic seeds are evaluated first. The primary seed is the LQR gain computed from the exact linearisation of the continuous-time pendulum equations at the upright equilibrium:

$$K_{\text{LQR}} \approx [-3.2,\; -6.0,\; 64.2,\; 15.8]$$

which corresponds to $\mathbf{k}^{(0)} = (k_{\theta p}=64.2, k_{\theta d}=15.8, k_{xp}=-3.2, k_{xd}=-6.0, k_i=0)$. The remaining budget for this stage is filled with uniform random samples within the search bounds (Table 1).

### Stage 2 — Differential Evolution (~100 evaluations)

`scipy.optimize.differential_evolution` [Storn & Price, 1997] is run with the following settings:

**Table 1 — Search bounds and DE settings**

| Parameter | Lower | Upper |
|---|---|---|
| $k_{\theta p}$ | 20 | 120 |
| $k_{\theta d}$ | 2 | 30 |
| $k_{xp}$ | −10 | 2 |
| $k_{xd}$ | −12 | 2 |
| $k_i$ | 0 | 0.5 |
| Population size | \multicolumn{2}{c|}{$\max(3, \min(6, n_{\text{stage2}}/10))$} |
| Mutation $F$ | 0.5 | 1.5 |
| Crossover $CR$ | \multicolumn{2}{c|}{0.7} |
| Initialisation | \multicolumn{2}{c|}{Latin hypercube} |

The population is initialised by a Latin hypercube design [McKay et al., 1979] rather than random sampling to achieve better initial coverage of the parameter space.

### Stage 3 — Coordinate Descent (~100 evaluations)

Starting from the best solution found so far, each gain is independently perturbed by $\pm\delta_i$ where $\delta_i = \max(0.01, 0.10\cdot|k_i|)$ (0.20 for $k_i$). If either perturbation lowers the objective, the improvement is accepted greedily. The process repeats until no gain in a full sweep yields improvement or the evaluation budget is exhausted.

---

## Success Criterion

A rollout is declared a **success** if there exists a time $t^* \leq T/2$ such that all four state tolerances are simultaneously satisfied and remain satisfied for all $t \geq t^*$:

$$|x(t)| \leq 0.25\;\text{m}, \quad |\dot{x}(t)| \leq 0.25\;\text{m/s}, \quad |\theta(t)| \leq 2^\circ, \quad |\dot{\theta}(t)| \leq 0.1\;\text{rad/s}$$

The settle-and-remain time $t^*$ must occur before the halfway point $T/2 = 10$ s, ensuring the system spends at least as long settled as it took to settle.

---

## Evaluation

After tuning, 9 rollouts are executed (1 fixed + 8 jittered, $T = 20$ s, $\Delta t = 0.02$ s). Results are ranked by: (1) success flag (failures last), then (2) LQR cost $c_i$. The notebook visualises the best and worst cases as state/force time-series and a cart–pendulum animation.

---

## References

- Åström, K. J. & Hägglund, T. (1995). *PID Controllers: Theory, Design, and Tuning* (2nd ed.). ISA.
- Storn, R. & Price, K. (1997). Differential Evolution — A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces. *Journal of Global Optimization*, 11(4), 341–359.
- McKay, M. D., Beckman, R. J. & Conover, W. J. (1979). A Comparison of Three Methods for Selecting Values of Input Variables in the Analysis of Output from a Computer Code. *Technometrics*, 21(2), 239–245.

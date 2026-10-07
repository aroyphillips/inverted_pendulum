# 04 — Reinforcement Learning with Soft Actor–Critic (SAC)

**Notebook:** [notebooks/04_rl_sac.ipynb](../notebooks/04_rl_sac.ipynb)  
**Source:** [src/controllers/rl_sac.py](../src/controllers/rl_sac.py), [src/envs/inverted_pendulum_env.py](../src/envs/inverted_pendulum_env.py)  
**Artifact:** artifacts/rl/sac.zip (+ sac.metadata.json)

---

## Methods (Reproducible description)

This document gives a precise, reproducible description of the SAC controller used in experiments. It contains the mathematical objective, algorithmic updates, network architectures, hyperparameter choices, training curriculum, checkpointing/evaluation protocol, and instructions needed to replicate results.

**Problem formulation.** The control task is an infinite-horizon discounted Markov decision process $(\mathcal{S},\mathcal{A},P,r,\gamma)$. The simulator state is
$$\mathbf{x} = [x,\;\dot{x},\;\theta,\;\dot{\theta}]^\top\in\mathbb{R}^4,$$
with continuous scalar control input $u\in[-u_{\max},u_{\max}]$. The agent outputs a normalized action $a\in(-1,1)$ which is scaled to torque/force: $u = \operatorname{clip}(a\cdot u_{\max}, -u_{\max}, u_{\max})$.

Transitions are produced by integrating the system ODE at fixed step $\Delta t=0.02\,$s. The per-step base reward is the negative quadratic cost:
$$r(\mathbf{x}_t,u_t) = -\bigl(\mathbf{x}_t^\top Q\,\mathbf{x}_t + u_t^\top R\,u_t\bigr),$$
where $Q$ and $R$ are set by **Bryson's rule** [Bryson & Ho, 1969]: $Q_i = 1/\text{tol}_i^2$ from the task settling tolerances and $R = 1/u_{\max}^2$ — no knowledge of system equations required. The reward also includes a two-stage shaping term (upright bonus near $\theta = 0$, cosine-shaped recovery signal elsewhere), a one-time settle-and-remain streak bonus, and a fall penalty. All parameter values are recorded in `sac.metadata.json`. The discount factor used is $\gamma=0.99$.

Observations: the policy receives a 5-dimensional observation

$$\mathbf{o}_t = \bigl[x/s_0,\;\dot{x}/s_1,\;\sin\theta,\;\cos\theta,\;\dot{\theta}/s_3\bigr]^\top$$

where $\mathbf{s} = [s_0, s_1, s_3]$ is the per-state standard deviation vector computed from the training dataset via `data_driven_obs_scale(dataset)`. This normalises each channel to approximately unit variance without requiring prior knowledge of the system's physical operating range. The $\sin\theta / \cos\theta$ channels replace raw $\theta$; they require no scaling since they are already bounded in $[-1, 1]$.

**SAC objective and updates.** We use the maximum-entropy RL objective: for a stochastic policy $\pi_\phi(a|\mathbf{o})$ and temperature $\alpha>0$,
$$J(\pi) = \sum_{t=0}^\infty \mathbb{E}_{(\mathbf{o}_t,a_t)\sim\rho_\pi}\Big[\gamma^t\big(r(\mathbf{o}_t,a_t) + \alpha\,\mathcal{H}(\pi(\cdot|\mathbf{o}_t))\big)\Big],$$
where $\mathcal{H}(\pi(\cdot|\mathbf{o})) = -\mathbb{E}_{a\sim\pi}[\log\pi(a|\mathbf{o})]$. The implementation follows the soft actor–critic variant with double Q-networks, target networks, and automatic temperature tuning.

At each gradient update we minimise the following critic loss for each Q-network $Q_{\theta_i}$ (using bootstrap from target networks):
$$\mathcal{L}_Q(\theta_i) = \mathbb{E}_{(\mathbf{o},a,r,\mathbf{o}')\sim\mathcal{D}}\Big[\tfrac{1}{2}\Big(Q_{\theta_i}(\mathbf{o},a) - y\Big)^2\Big],$$
with target
$$y = r + \gamma\big(\min_{j=1,2}\bar{Q}_{\theta_j}(\mathbf{o}',a') - \alpha\log\pi_\phi(a'|\mathbf{o}')\big),\quad a'\sim\pi_\phi(\cdot|\mathbf{o}').$$

The policy is updated by minimising the expected KL-style actor objective (equivalently maximising entropy-augmented Q-value):
$$\mathcal{L}_\pi(\phi) = \mathbb{E}_{\mathbf{o}\sim\mathcal{D},\epsilon\sim\mathcal{N}}\big[\alpha\log\pi_\phi(f_\phi(\mathbf{o},\epsilon)|\mathbf{o}) - \min_{j=1,2} Q_{\theta_j}(\mathbf{o},f_\phi(\mathbf{o},\epsilon))\big],$$
where $f_\phi(\mathbf{o},\epsilon)$ denotes the reparameterised Gaussian action (sampled then squashed via $\tanh$) used to compute an action and its log-density.

Temperature (entropy) tuning follows automatic gradient-based adjustment towards a target entropy $\bar{\mathcal{H}}$ by minimising
$$\mathcal{L}(\alpha) = -\mathbb{E}_{a\sim\pi_\phi}\big[\alpha\big(\log\pi_\phi(a|\mathbf{o}) + \bar{\mathcal{H}}\big)\big],$$
whose gradient update (after reparameterisation to ensure positivity, e.g. optimise $\log\alpha$) enforces $\mathbb{E}[-\log\pi_\phi] \approx -\bar{\mathcal{H}}$.

Target networks are updated with soft updates (Polyak averaging):
$$\bar{\theta} \leftarrow (1-\tau)\bar{\theta} + \tau\theta,\qquad \tau\in(0,1).$$

**Network architectures and optimisers.** The standard configuration used in reported experiments is:
- Policy (actor): MLP with two hidden layers of 256 units each, ReLU activations; outputs Gaussian mean and log-std per action dimension; actions sampled and squashed via $\tanh$.
- Q-networks (critics): two independent MLPs, each with two hidden layers of 256 units and ReLU activations; input concatenates observation and action.

All networks are initialised using PyTorch/NumPy default seeds (see Reproducibility). Optimisation uses Adam with learning rates and hyperparameters listed below.

**Hyperparameters (final chosen values).** These are the values used for the main experiments and recorded in `sac.metadata.json`:

| Parameter | Value |
|---|---:|
| Policy architecture | MLP, 2×256 (ReLU) |
| Q architecture | 2×256 (ReLU) |
| Optimiser | Adam (beta1=0.9, beta2=0.999) |
| Actor LR | 3e-4 |
| Critic LR | 3e-4 |
| Temperature LR | 3e-4 |
| Batch size | 256 |
| Replay buffer size | min(1,000,000, total_steps) |
| Discount ($\gamma$) | 0.99 |
| Soft update ($\tau$) | 0.005 |
| Learning starts | 10,000 |
| Initial temperature (\alpha) | auto-initialised to match target entropy |
| Target entropy (\bar{\mathcal{H}}) | -1 (automatic target) |
| Gradient steps per env step | 1 (configurable) |
| Total environment steps | 500,000 |
| Replay sampling | uniform |

Notes on selection: the 3e-4 learning rate and 256 hidden units are standard settings for continuous-control SAC and were selected following a small grid search over {1e-4, 3e-4, 1e-3} (LR) and {128, 256} (hidden size). Reported results use the above values; all tested combinations are retained in experiment metadata.

**Training schedule, curriculum and evaluation.**
- Curriculum fraction $f(t)$: linearly ramp from 0 to 1 over $t_{\text{curr}}=\min(100{,}000,\;N_{\text{steps}})$ training steps. When $f=0$ episodes always start from the canonical upright initial state; when $f=1$ the full curriculum initial-state distribution is used. The curriculum is applied only to episode initialisation and does not modify dynamics or reward.
- Checkpointing: every $\Delta t_{\text{ckpt}}=20{,}000$ environment steps the current model is saved and a deterministic evaluation rollout (policy mean, no action noise) is executed from the canonical initial state. A scalar score is computed (successful rollouts prefer lower cost and successes are promoted); the best-scoring checkpoint is retained as `best.zip`.
- Evaluation protocol: final reported performance consists of 9 rollouts — 1 fixed initial condition and 8 jittered initial conditions (Gaussian perturbations around the canonical initial state). Metrics computed per rollout include total cost (integrated stage cost), settle-and-remain time (time until state remains within predefined tolerances), and binary success (remain upright within the tolerance for the episode duration). The same evaluation code is used at checkpoint time and for final assessment.

**Implementation details and numerical stability.**
- Action squashing: Gaussian samples are squashed with $\tanh$; log-density correction for the squashing transform is applied as in Haarnoja et al. to obtain correct policy entropy.
- Reward scaling: raw simulator reward follows the quadratic form above; no additional multiplicative rescaling is applied in the default configuration. If reward scaling is used in an experiment, the scaling factor is recorded in metadata.
- Gradient clipping: optional (configurable) gradient clipping to a global norm is available in the trainer; the experiments reported here did not rely on aggressive clipping.

**Hyperparameter tuning methodology.**
Hyperparameters were tuned with a limited grid search and pragmatic constraints to keep compute reasonable. Search details recorded in experiment metadata include the parameter grid, random seeds, number of repeats per configuration (typically 3), and the selection criterion (mean deterministic evaluation cost across repeats). Primary tuning focused on actor/critic learning rates and hidden widths; secondary sweeps verified stability to target entropy choices and replay buffer sizes. The final chosen hyperparameters balance performance and robustness across seeds.

**Reproducibility checklist (exact steps to reproduce experiments).**
1. Create and activate virtual environment; install dependencies from `requirements.txt` (exact versions recorded in `sac.metadata.json`).
2. Ensure working directory is the repository root. Run training (example):

```bash
python scripts/train_rl_sac.py --timesteps 500000 --force-limit 5.0 --seed 42 --output artifacts/rl/sac.zip
```

3. Training saves checkpoints to `artifacts/rl/checkpoints/`. The script logs `sac.metadata.json`, which includes `force_limit`, `observation_scale`, final hyperparameters, total training steps, wall-clock time and the random seeds used for each repeat.
4. Evaluate the best checkpoint with the notebook `notebooks/04_rl_sac.ipynb` or with the evaluation helper:

```bash
python scripts/evaluate_controller.py --checkpoint artifacts/rl/checkpoints/best.zip --n-jitter 8 --seed 0
```

**Determinism and seeds.** To maximise reproducibility the training script accepts an explicit random seed which is applied to NumPy, Python `random`, and the deep learning framework (PyTorch) where supported. Exact hardware (GPU/CPU) and non-deterministic CUDA operations may produce small numerical differences; the `sac.metadata.json` records hardware/driver information used for a run.

**Checkpointing and experimental artifacts.** Saved artifacts include:
- `sac_{t}.zip` — full model snapshot at checkpoint step $t$.
- `best.zip` — selected best deterministic-checkpoint according to the evaluation score.
- `sac.metadata.json` — JSON file containing: hyperparameters, `EnvConfig`, `force_limit`, `observation_scale`, seed(s), wall-clock runtime, package versions, and notes on selection.

**Evaluation metrics and definitions.**
- Cost: integrated stage cost over the episode, identical to the negative sum of per-step rewards but reported as a positive cost for clarity.
- Settle-and-remain time: the earliest time $t_s$ after which state components remain within predefined tolerances (e.g., $\pm2^\circ$ for pendulum angle) for the remainder of the episode; if not achieved, value is `None`.
- Success: Boolean flag indicating the episode satisfied the settle-and-remain criterion before episode end.

**Notes and limitations.** The SAC controller is model-free and therefore does not produce interpretable gain matrices; behaviour and performance depend on neural network weights and training randomness. Results reported in the manuscript are averaged across random seeds (see `sac.metadata.json` for per-seed results). When comparing controllers, we recommend using identical evaluation seeds and the deterministic policy (policy mean) for rollouts used to select best checkpoints.

---

## Notebook usage and quick commands

By default the notebook will load the packaged artifact if present (`TRAIN_IF_MISSING = False`). To retrain inside the notebook set `TRAIN_IF_MISSING = True` and the cell will call `train_sac_controller` with the same default hyperparameters as above. For fully reproducible offline runs prefer invoking `scripts/train_rl_sac.py` with explicit `--seed` and `--output` arguments.

Example commands:

```bash
# Train (single-run):
python scripts/train_rl_sac.py --timesteps 500000 --force-limit 5.0 --seed 42 --output artifacts/rl/sac.zip

# Deterministic evaluation of a checkpoint:
python scripts/evaluate_controller.py --checkpoint artifacts/rl/checkpoints/best.zip --n-jitter 8 --seed 0
```

---

## References

- Haarnoja, T., Zhou, A., Abbeel, P. & Levine, S. (2018). Soft Actor–Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor. Proceedings of ICML.
- Haarnoja, T., Zhou, A., Hartikainen, K., Tucker, G., Ha, S., Tan, J., … Levine, S. (2018b). Soft Actor–Critic Algorithms and Applications. arXiv:1812.05905.
- Fujimoto, S., van Hoof, H. & Meger, D. (2018). Addressing Function Approximation Error in Actor–Critic Methods. Proceedings of ICML.
- Bengio, Y., Louradour, J., Collobert, R. & Weston, J. (2009). Curriculum Learning. Proceedings of ICML.
- Bryson, A. E. & Ho, Y.-C. (1969). *Applied Optimal Control*. Blaisdell. (Bryson's rule for Q/R selection.)
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D. & Meger, D. (2018). Deep Reinforcement Learning That Matters. AAAI.

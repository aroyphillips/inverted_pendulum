Repository to generate and evaluate data for inverted pendulum problem for data-driven control.

# To-dos:
  - [x] 1.1 Define system of equations relating $\theta_{t}$ to $\theta_{t+1}$ e.g. via $\ddot{\theta} = C \sin(\theta))$  and implement via code 
  - [x] 1.2 Define system of equations relating input force $F$ to pendulum $\theta$  and implement via code
  - [ ] 2.1 Generate dataset of $F(t)$ and $\theta(t)$, e.g. via random walk of $F(t)$
  - [ ] 2.1.1 Modify code to sample from random initial conditions of $\{x, \dot{x}, \theta < \pi/6, \dot{\theta}\}$
  - [ ] 2.1.2 Optionally introduce measurement noise into the system
  - [ ] 2.2 Verify dataset consists of physically valid systems, e.g. via conservation of energy
  - [ ] 2.3 Plot the timeseries for  $\{x, \dot{x}, \theta < \pi/6, \dot{\theta}\}$
  - [ ] 3.1 Distribute data to students allowing
  - [ ] 3.2 Evaluate data and determine winners

# Deadlines:

4/10: Tasks 1 [x]

4/13: Tasks 2 [ ]

4/20: Evaluate responses [ ]


# Division of responsibility:
Oscar to implement Tasks 2 and let Dani and Roy know if there are any questions


# Math background

Constants: $M, m, l, g$

State variables: $x(t), \dot{x}(t), \theta(t), \ddot(\theta(t)$

Equations:

$$(M-m)\ddot{x} - ml\ddot{\theta}\cos(\theta) + ml\dot{\theta}^2\sin(\theta) = F$$

$$l\ddot{\theta}-g\sin(\theta)-\ddot{x}\cos(\theta)$$


## Numerical integration methods
- Euler's method 

$$y_{n+1} \ \equiv \ y_n + h \cdot f(t_n,y_n)$$ 

$$h<<1$$

- Runge-Kutta

# Some relevant papers:

Section IV: NMPC Applied to an Inverted Pendulum  for  [Nonlinear model predictive control of an inverted pendulum][https://ieeexplore.ieee.org/document/5160391]

[Data-Based Control of Feedback Linearizable Systems][https://ieeexplore.ieee.org/document/10054127] includes application to the double inverted pendulum

[Willems’ Fundamental Lemma for Nonlinear Systems With Koopman Linear Embedding][https://arxiv.org/pdf/2409.16389]

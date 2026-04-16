from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


def plot_states_and_input(
    t: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    title: str = "Trajectory",
    theta_in_degrees: bool = False,
) -> Tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
    y_plot = np.array(y, copy=True)
    labels = ["x", "x_dot", "theta (rad)", "theta_dot (rad/s)"]

    if theta_in_degrees:
        y_plot[2, :] = np.rad2deg(y_plot[2, :])
        y_plot[3, :] = np.rad2deg(y_plot[3, :])
        labels[2] = "theta (deg)"
        labels[3] = "theta_dot (deg/s)"

    for i in range(4):
        axes[i].plot(t, y_plot[i, :], linewidth=1.5)
        axes[i].set_ylabel(labels[i])
        axes[i].grid(True, alpha=0.3)

    axes[4].plot(t, u, color="tab:red", linewidth=1.5)
    axes[4].set_ylabel("force")
    axes[4].set_xlabel("time (s)")
    axes[4].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def animate_cart_pendulum(
    t: np.ndarray,
    y: np.ndarray,
    pendulum_length: float,
    stride: int = 5,
    figsize: Tuple[float, float] = (8, 4),
) -> FuncAnimation:
    x = y[0, :]
    theta = y[2, :]
    cart_width = 0.3
    cart_height = 0.15

    fig, ax = plt.subplots(figsize=figsize)
    x_margin = max(1.0, 0.2 * np.max(np.abs(x)) + 0.5)
    ax.set_xlim(np.min(x) - x_margin, np.max(x) + x_margin)
    ax.set_ylim(-pendulum_length - 0.5, pendulum_length + 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    (rod_line,) = ax.plot([], [], "-", lw=2)
    cart_patch = plt.Rectangle((0, 0), cart_width, cart_height, fill=False, lw=2)
    ax.add_patch(cart_patch)
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

    def _init():
        rod_line.set_data([], [])
        time_text.set_text("")
        return rod_line, cart_patch, time_text

    def _update(frame_idx: int):
        i = frame_idx * stride
        i = min(i, t.size - 1)

        xc = x[i]
        th = theta[i]
        xp = xc + pendulum_length * np.sin(th)
        yp = pendulum_length * np.cos(th)

        cart_patch.set_xy((xc - cart_width / 2.0, -cart_height / 2.0))
        rod_line.set_data([xc, xp], [0.0, yp])
        time_text.set_text(f"t = {t[i]:.2f} s")
        return rod_line, cart_patch, time_text

    total_frames = int(np.ceil(t.size / stride))
    anim = FuncAnimation(fig, _update, init_func=_init, frames=total_frames, interval=30, blit=True)
    return anim

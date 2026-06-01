#!/usr/bin/env python3
"""Patch notebooks 01-06: update cell-3 (helpers), cell-4 (run+summary), cell-5 (show best/worst).

Run from the project root:
    python scripts/patch_notebooks.py
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

# ---------------------------------------------------------------------------
# Shared helpers cell (cell index 3) — identical for all 6 notebooks
# ---------------------------------------------------------------------------
CELL3_SOURCE = """\
T_END = 20.0
DT    = 0.02


def run_fixed_and_jittered(controller, n_jitter=8, seed=0):
    rng = np.random.default_rng(seed)
    jitter_scale = np.array([0.02, 0.05, 0.03, 0.1])
    force_limit = float(getattr(controller, "force_limit", 5.0))
    cases = [("fixed", ds.initial_state)]
    for i in range(n_jitter):
        cases.append((f"jitter_{i:02d}", ds.initial_state + rng.normal(scale=jitter_scale)))
    rows, rollouts = [], {}
    for label, x0 in cases:
        result = rollout_controller(controller, x0, t_end=T_END, dt=DT, force_limit=force_limit)
        metrics = rollout_metrics(result.t, result.y, result.u, result.control_times_sec)
        rows.append({"case": label, **metrics})
        rollouts[label] = result
    # Successes first, then by cost
    rows.sort(key=lambda r: (not bool(r.get("success")), float(r.get("cost", np.inf))))
    return rows, rollouts


def show_case(label, result, title):
    fig, axes = plot_states_and_input(result.t, result.y, result.u, title=title, theta_in_degrees=True)
    plt.show()

    metrics = rollout_metrics(result.t, result.y, result.u)
    t_settle = metrics.get("settle_and_remain_time")
    t = result.t
    theta_deg = np.degrees(result.y[2, :])
    success = bool(metrics.get("success", False))

    fig2, axes2 = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax = axes2[0]
    ax.plot(t, theta_deg, color="steelblue", linewidth=1.5, label="\\u03b8 (deg)")
    ax.axhline( 2.0, color="green", linestyle="--", linewidth=1, label="\\u00b12\\u00b0 tolerance")
    ax.axhline(-2.0, color="green", linestyle="--", linewidth=1)
    if t_settle is not None:
        ax.axvspan(t_settle, float(t[-1]), alpha=0.13, color="green",
                   label=f"Settled  t_s={t_settle:.2f} s")
    ax.axvline(float(t[-1]) / 2, color="gray", linestyle=":", linewidth=1, label="t_end/2")
    ax.set_ylabel("\\u03b8 (degrees)")
    ax.set_title(f"{title}  \\u2014  {'\\u2713 SUCCESS' if success else '\\u2717 FAIL'}")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax2 = axes2[1]
    ax2.plot(t, result.u, color="darkorange", linewidth=1.4, label="Force (N)")
    ax2.axhline(0, color="k", linewidth=0.5)
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("force (N)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    plt.show()

    if t_settle is not None:
        post = float(t[-1]) - t_settle
        flag = "PASS" if t_settle <= float(t[-1]) / 2 else "FAIL"
        print(f"  settle_t={t_settle:.2f} s  post_settle={post:.2f} s  "
              f"[{flag}: t_s \\u2264 {float(t[-1])/2:.0f} s]")
    else:
        print("  Did not settle and remain within all tolerances.")
    try:
        anim = animate_cart_pendulum(result.t, result.y, pendulum_length=0.5, stride=5)
        display(HTML(anim.to_jshtml()))
    except Exception as exc:
        print(f"  Animation skipped ({exc})")
"""

# ---------------------------------------------------------------------------
# Per-notebook cell-4 and cell-5 sources
# ---------------------------------------------------------------------------

def cell4_standard(seed: int) -> str:
    return f"""\
rows, rollouts = run_fixed_and_jittered(controller, n_jitter=8, seed={seed})
successes = sum(1 for r in rows if r.get("success"))
print(f"Successes: {{successes}}/{{len(rows)}}  (force_limit={{getattr(controller,'force_limit',5.0)}} N, t_end={{T_END}} s)")
for r in rows:
    t_s = r.get("settle_and_remain_time")
    t_s_str = f"{{t_s:.2f}} s" if t_s is not None else "  None  "
    print(f"  {{r['case']:15s}}  success={{str(r.get('success')):5s}}  "
          f"settle_t={{t_s_str:8s}}  cost={{r.get('cost', float('inf')):.4g}}")
rows[:3]
"""

def cell4_deepc(seed: int) -> str:
    return f"""\
rows, rollouts = run_fixed_and_jittered(controller, n_jitter=8, seed={seed})
successes = sum(1 for r in rows if r.get("success"))
print(f"Successes: {{successes}}/{{len(rows)}}  (force_limit={{getattr(controller,'force_limit',5.0)}} N, t_end={{T_END}} s)")
for r in rows:
    t_s = r.get("settle_and_remain_time")
    t_s_str = f"{{t_s:.2f}} s" if t_s is not None else "  None  "
    print(f"  {{r['case']:15s}}  success={{str(r.get('success')):5s}}  "
          f"settle_t={{t_s_str:8s}}  cost={{r.get('cost', float('inf')):.4g}}")
print()
controller.diagnostics()
"""

def cell5_source(controller_label: str) -> str:
    return f"""\
best  = rows[0]["case"]
worst = rows[-1]["case"]
show_case(best,  rollouts[best],  f"{controller_label} best case: {{best}}")
show_case(worst, rollouts[worst], f"{controller_label} worst case: {{worst}}")
"""

# ---------------------------------------------------------------------------
# Notebook specifications
# ---------------------------------------------------------------------------

NOTEBOOKS = [
    {
        "file": "01_pid_control.ipynb",
        "cell4": cell4_standard(seed=1),
        "cell5": cell5_source("PID"),
    },
    {
        "file": "02_koopman_willems_deepc.ipynb",
        "cell4": cell4_deepc(seed=2),
        "cell5": cell5_source("Koopman DeePC"),
    },
    {
        "file": "03_gp_lqr.ipynb",
        "cell4": cell4_standard(seed=3),
        "cell5": cell5_source("GP-LQR"),
    },
    {
        "file": "04_rl_sac.ipynb",
        "cell4": cell4_standard(seed=4),
        "cell5": cell5_source("SAC"),
    },
    {
        "file": "05_sindy_lqr.ipynb",
        "cell4": cell4_standard(seed=5),
        "cell5": cell5_source("SINDy-LQR"),
    },
    {
        "file": "06_regularized_deepc.ipynb",
        "cell4": cell4_deepc(seed=6),
        "cell5": cell5_source("DeePC"),
    },
]


def _to_lines(src: str) -> list[str]:
    """Convert a multi-line string to a list of source lines as Jupyter stores them."""
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        pass  # last line intentionally no newline
    return lines


def patch_notebook(spec: dict) -> None:
    path = NB_DIR / spec["file"]
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = nb["cells"]
    assert len(cells) >= 6, f"{path.name}: expected ≥6 cells, got {len(cells)}"

    cells[3]["source"] = _to_lines(CELL3_SOURCE)
    cells[4]["source"] = _to_lines(spec["cell4"])
    cells[5]["source"] = _to_lines(spec["cell5"])

    # Clear stale outputs from cells 3-5
    for idx in (3, 4, 5):
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None

    path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  patched {path.name}")


if __name__ == "__main__":
    for spec in NOTEBOOKS:
        patch_notebook(spec)
    print("Done.")

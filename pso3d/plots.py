"""Matplotlib figures (PNG) and a convergence GIF."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RUST, DARK, GREY = "#20808D", "#A84B2F", "#1B474D", "#7A7974"


def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def plot_convergence(best_history, path, title="Best fitness per iteration (baseline run)"):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogy(range(1, len(best_history) + 1), best_history, color=TEAL, lw=2)
    ax.axvline(30, color=GREY, ls="--", lw=1)
    ax.text(30.6, max(best_history) * 0.6, "plateau region\n(early stopping\ncandidate)", color=GREY, fontsize=8)
    ax.set_xlabel("iteration"); ax.set_ylabel("f(x_best)  [m$^2$]  (log)"); ax.set_title(title, loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_scenario_3d(scenario, estimate, positions_history, path):
    _style()
    fig = plt.figure(figsize=(7, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    a = scenario.anchors
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], marker="s", s=60, color=DARK, label="anchors (M=4)")
    for j, p in enumerate(a):
        ax.text(p[0], p[1], p[2] + 1, f"a{j + 1}", color=DARK, fontsize=8)
    x0 = positions_history[0]
    xT = positions_history[-1]
    ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=12, color=GREY, alpha=0.6, label="swarm start (t=0)")
    ax.scatter(xT[:, 0], xT[:, 1], xT[:, 2], s=12, color=TEAL, alpha=0.9, label="swarm end (t=60)")
    tp = scenario.true_position
    ax.scatter(*tp, marker="*", s=160, color=RUST, label="true node p")
    ax.scatter(*estimate, marker="x", s=90, color="black", label="PSO estimate p̂")
    ax.set_xlim(0, scenario.field_size[0]); ax.set_ylim(0, scenario.field_size[1]); ax.set_zlim(0, scenario.field_size[2])
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("60 x 60 x 20 m field: swarm start vs. converged swarm", loc="left")
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.95))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_sweep(rows, label, xlabel, path, logx=False):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for method, color, name in (("pso", TEAL, "Simplified PSO (best-of-iteration)"), ("std", DARK, "Standard PSO (pbest/gbest)"),
                                ("lsq", RUST, "Least-squares trilateration"), ("lsq_gn", "#FFC553", "LSQ + Gauss-Newton")):
        rs = [r for r in rows if r.method == method and r.label == label]
        if not rs:
            continue
        xs = [r.value for r in rs]
        axes[0].plot(xs, [r.rmse for r in rs], "o-", color=color, label=name)
        axes[1].plot(xs, [r.evaluations for r in rs], "o-", color=color, label=name)
    axes[0].set_ylabel("RMSE [m]"); axes[1].set_ylabel("fitness evaluations per node")
    for ax in axes:
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Accuracy vs {xlabel}", loc="left"); axes[1].set_title(f"Cost vs {xlabel}", loc="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_error_cdf(errors_by_method: dict, path):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for (name, e), color in zip(errors_by_method.items(), (TEAL, DARK, RUST, "#FFC553")):
        e = np.sort(e); ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=color, lw=2, label=name)
    ax.set_xlabel("localization error [m]"); ax.set_ylabel("CDF"); ax.set_xlim(left=0)
    ax.set_title("Error CDF, Monte-Carlo (sigma = 0.5 m, 4 anchors)", loc="left"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def make_convergence_gif(scenario, positions_history, estimate_history, path, every=2, fps=6):
    """Animated 3D swarm using Pillow (no imageio needed)."""
    from PIL import Image
    _style()
    frames = []
    a = scenario.anchors; tp = scenario.true_position
    T = len(positions_history)
    for t in list(range(0, T, every)) + [T - 1]:
        fig = plt.figure(figsize=(5.6, 4.6))
        ax = fig.add_subplot(111, projection="3d")
        x = positions_history[t]
        ax.scatter(a[:, 0], a[:, 1], a[:, 2], marker="s", s=50, color=DARK)
        ax.scatter(x[:, 0], x[:, 1], x[:, 2], s=16, color=TEAL)
        ax.scatter(*tp, marker="*", s=140, color=RUST)
        if t > 0:
            e = estimate_history[min(t, len(estimate_history)) - 1]
            ax.scatter(*e, marker="x", s=70, color="black")
        ax.set_xlim(0, 60); ax.set_ylim(0, 60); ax.set_zlim(0, 20)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"Simplified PSO, iteration {t}/{T - 1}", loc="left", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes())
        frames.append(img.convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)

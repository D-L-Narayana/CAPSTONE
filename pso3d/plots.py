"""Matplotlib figures (PNG) and a convergence GIF.

matplotlib (Agg backend) and Pillow are imported only here; the rest of ``pso3d`` stays
numpy-only.  Every ``plot_*`` function renders to ``path``; the ``_*_figure`` helpers build
and return the figure without saving (handy for tests and for callers that customise it).
Scenario-dependent values (field size, number of anchors, last iteration, plateau marker)
are derived from the inputs instead of being hard-coded.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RUST, DARK, GREY = "#20808D", "#A84B2F", "#1B474D", "#7A7974"
GOLD = "#FFC553"
METHOD_COLOURS = {"pso": TEAL, "std": DARK, "lsq": RUST, "lsq_gn": GOLD,
                  "amcmpso": "#5C9EAD", "std_tol": "#8E6C8A", "std_ws": "#C98B2B", "amcmpso_ws": "#4B6A5B"}
PALETTE = (TEAL, DARK, RUST, GOLD, "#5C9EAD", "#8E6C8A", "#C98B2B", "#4B6A5B")
CRLB_LABEL = "CRLB bound (Gaussian noise)"


def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def _method_label(key: str) -> str:
    """Human-readable method name from the experiments registry (falls back to the key)."""
    try:
        from .experiments import METHODS
    except ImportError:
        return key
    spec = METHODS.get(key)
    return spec.label if spec is not None else key


def _method_colour(key: str, index: int = 0) -> str:
    return METHOD_COLOURS.get(key, PALETTE[index % len(PALETTE)])


def _save(fig, path) -> None:
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _field_text(scenario) -> str:
    return " x ".join(f"{float(s):g}" for s in scenario.field_size) + " m"


def _set_field_limits(ax, scenario) -> None:
    """Axis limits [0, field_size] on a 3D axes."""
    fx, fy, fz = (float(s) for s in scenario.field_size)
    ax.set_xlim(0, fx); ax.set_ylim(0, fy); ax.set_zlim(0, fz)


# ----------------------------------------------------------------------------- convergence
def _convergence_figure(best_history, title, plateau_iteration):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.semilogy(range(1, len(best_history) + 1), best_history, color=TEAL, lw=2)
    if plateau_iteration is not None:
        ax.axvline(plateau_iteration, color=GREY, ls="--", lw=1)
        ax.text(plateau_iteration + 0.6, max(best_history) * 0.6, "plateau region\n(early stopping\ncandidate)",
                color=GREY, fontsize=8)
    ax.set_xlabel("iteration"); ax.set_ylabel("f(x_best)  [m$^2$]  (log)"); ax.set_title(title, loc="left")
    return fig


def plot_convergence(best_history, path, title="Best fitness per iteration (baseline run)", plateau_iteration=None):
    """Best fitness per iteration (log scale).  ``plateau_iteration`` (1-based) adds a dashed marker
    with the "plateau region" annotation; ``None`` draws no marker."""
    _save(_convergence_figure(best_history, title, plateau_iteration), path)


# ----------------------------------------------------------------------------- 3D scenario
def _scenario_figure(scenario, estimate, positions_history):
    _style()
    fig = plt.figure(figsize=(7, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    a = np.asarray(scenario.anchors, dtype=float)
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], marker="s", s=60, color=DARK, label=f"anchors (M={len(a)})")
    for j, p in enumerate(a):
        ax.text(p[0], p[1], p[2] + 1, f"a{j + 1}", color=DARK, fontsize=8)
    hist = np.asarray(positions_history, dtype=float)          # (T+1, N, 3)
    last = len(hist) - 1
    x0, xT = hist[0], hist[-1]
    ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2], s=12, color=GREY, alpha=0.6, label="swarm start (t=0)")
    ax.scatter(xT[:, 0], xT[:, 1], xT[:, 2], s=12, color=TEAL, alpha=0.9, label=f"swarm end (t={last})")
    tp = np.asarray(scenario.true_position, dtype=float)
    ax.scatter(*tp, marker="*", s=160, color=RUST, label="true node p")
    ax.scatter(*np.asarray(estimate, dtype=float), marker="x", s=90, color="black", label="PSO estimate p̂")
    _set_field_limits(ax, scenario)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(f"{_field_text(scenario)} field: swarm start vs. converged swarm", loc="left")
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.95))
    return fig


def plot_scenario_3d(scenario, estimate, positions_history, path):
    _save(_scenario_figure(scenario, estimate, positions_history), path)


# ----------------------------------------------------------------------------- parameter sweeps
def _crlb_by_value(rows, label) -> dict:
    """Mean finite CRLB bound per sweep value (the bound is independent of the method)."""
    per_value: dict = {}
    for r in rows:
        c = getattr(r, "crlb_rmse", float("nan"))
        if r.label == label and np.isfinite(c):
            per_value.setdefault(r.value, []).append(float(c))
    return {x: float(np.mean(v)) for x, v in per_value.items()}


def _sweep_figure(rows, label, xlabel, logx):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    keys: list[str] = []
    for r in rows:
        if r.label == label and r.method not in keys:
            keys.append(r.method)
    for i, key in enumerate(keys):
        rs = [r for r in rows if r.method == key and r.label == label]
        xs = [r.value for r in rs]
        colour, name = _method_colour(key, i), _method_label(key)
        axes[0].plot(xs, [r.rmse for r in rs], "o-", color=colour, label=name)
        axes[1].plot(xs, [r.evaluations for r in rs], "o-", color=colour, label=name)
    crlb = _crlb_by_value(rows, label)
    if crlb:
        xs = sorted(crlb)
        axes[0].plot(xs, [crlb[x] for x in xs], "--", color=GREY, lw=1.2, label=CRLB_LABEL)
    axes[0].set_ylabel("RMSE [m]"); axes[1].set_ylabel("fitness evaluations per node")
    for ax in axes:
        ax.set_xlabel(xlabel)
        if logx:
            ax.set_xscale("log")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Accuracy vs {xlabel}", loc="left"); axes[1].set_title(f"Cost vs {xlabel}", loc="left")
    return fig


def plot_sweep(rows, label, xlabel, path, logx=False):
    """RMSE and fitness evaluations vs one swept parameter, one curve per method present in ``rows``
    (colours/labels from the experiments registry) plus the CRLB reference when the rows carry it."""
    _save(_sweep_figure(rows, label, xlabel, logx), path)


# ----------------------------------------------------------------------------- error CDF
def _error_cdf_figure(errors_by_method: dict, title):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for i, (name, e) in enumerate(errors_by_method.items()):
        e = np.sort(np.asarray(e, dtype=float))
        ax.plot(e, np.arange(1, len(e) + 1) / len(e), color=PALETTE[i % len(PALETTE)], lw=2, label=name)
    ax.set_xlabel("localization error [m]"); ax.set_ylabel("CDF"); ax.set_xlim(left=0)
    ax.set_title(title, loc="left"); ax.legend(fontsize=8)
    return fig


def plot_error_cdf(errors_by_method: dict, path, title="Error CDF, Monte-Carlo (sigma = 0.5 m, 4 anchors)"):
    _save(_error_cdf_figure(errors_by_method, title), path)


# ----------------------------------------------------------------------------- method comparison bars
def _method_bars_figure(rows, label, value):
    _style()
    sel = [r for r in rows if r.label == label and np.isclose(r.value, value)]
    if not sel:
        raise ValueError(f"no rows for {label} = {value}")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    xs = np.arange(len(sel))
    ax.bar(xs, [r.rmse for r in sel], color=[_method_colour(r.method, i) for i, r in enumerate(sel)])
    ax.set_xticks(xs)
    ax.set_xticklabels([_method_label(r.method) for r in sel], rotation=20, ha="right", fontsize=8)
    bounds = [r.crlb_rmse for r in sel if np.isfinite(getattr(r, "crlb_rmse", float("nan")))]
    if bounds:
        ax.axhline(float(np.mean(bounds)), color=GREY, ls="--", lw=1.2, label=CRLB_LABEL)
        ax.legend(fontsize=8)
    ax.set_ylabel("RMSE [m]")
    ax.set_title(f"RMSE per method at {label} = {value:g} ({sel[0].trials} trials)", loc="left")
    return fig


def plot_method_bars(rows, path, label, value):
    """RMSE bars for every method at one sweep point (``label == value``) with the CRLB bound as a
    dashed reference line when the rows carry a finite ``crlb_rmse``."""
    _save(_method_bars_figure(rows, label, value), path)


# ----------------------------------------------------------------------------- CRLB heat map
def _crlb_heatmap_figure(xs, ys, grid2d, title):
    _style()
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    g = np.ma.masked_invalid(np.asarray(grid2d, dtype=float))     # inf/nan cells (singular geometry) in grey
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    cmap = matplotlib.colormaps["viridis"].with_extremes(bad=GREY)
    im = ax.imshow(g, origin="lower", extent=(xs[0], xs[-1], ys[0], ys[-1]), aspect="equal", cmap=cmap)
    fig.colorbar(im, ax=ax, label="CRLB RMSE bound [m]")
    ax.grid(False)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title, loc="left")
    return fig


def plot_crlb_heatmap(xs, ys, grid2d, path, title):
    """Image of one z-slice of a CRLB coverage grid: ``grid2d[iy, ix]`` is the RMS bound at (xs[ix], ys[iy])."""
    _save(_crlb_heatmap_figure(xs, ys, grid2d, title), path)


# ----------------------------------------------------------------------------- GIF
def make_convergence_gif(scenario, positions_history, estimate_history, path, every=2, fps=6,
                         title="Simplified PSO"):
    """Animated 3D swarm using Pillow (no imageio needed)."""
    from PIL import Image
    _style()
    frames = []
    a = np.asarray(scenario.anchors, dtype=float); tp = np.asarray(scenario.true_position, dtype=float)
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
        _set_field_limits(ax, scenario)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"{title}, iteration {t}/{T - 1}", loc="left", fontsize=10)
        fig.tight_layout()
        fig.canvas.draw()
        img = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba().tobytes())
        frames.append(img.convert("P", palette=Image.ADAPTIVE))
        plt.close(fig)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)

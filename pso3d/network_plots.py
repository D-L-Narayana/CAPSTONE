"""Figures for the multi-node iterative localization study (matplotlib, Agg backend only).

Palette shared with :mod:`pso3d.plots`.  Kept apart from the numpy-only core so that
``pso3d.network`` can be used without matplotlib.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEAL, RUST, DARK, GREY = "#20808D", "#A84B2F", "#1B474D", "#7A7974"


def _style() -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130})


def _positions(scenario_or_arrays) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(scenario_or_arrays, "generate"):
        beacons, nodes = scenario_or_arrays.generate()
    else:
        beacons, nodes = scenario_or_arrays
    return (np.asarray(beacons, dtype=float).reshape(-1, 3), np.asarray(nodes, dtype=float).reshape(-1, 3))


def _fmt(value) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.2f}"


def plot_rounds(result, path, title: str | None = None) -> None:
    """Settled count and newly settled nodes per round (left), fitness evaluations per round (right)."""
    _style()
    rounds = [r.round for r in result.rounds]
    newly = [r.newly_settled for r in result.rounds]
    total = [r.settled_total for r in result.rounds]
    evals = [r.evaluations for r in result.rounds]
    n = int(np.asarray(result.settled).shape[0])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    ax.bar(rounds, newly, color=RUST, alpha=0.85, label="newly settled")
    ax.plot(rounds, total, "o-", color=TEAL, lw=2, label="settled total")
    ax.axhline(n, color=GREY, ls="--", lw=1)
    if rounds:
        ax.text(rounds[0], n * 1.01, f"N = {n} nodes", color=GREY, fontsize=8, va="bottom")
    ax.set_xlabel("round"); ax.set_ylabel("nodes"); ax.set_xticks(rounds)
    ax.set_ylim(0, max(n * 1.12, 1.0))
    ax.set_title("Nodes settled per round", loc="left"); ax.legend(fontsize=8, loc="center right")
    ax2.bar(rounds, evals, color=DARK, alpha=0.85)
    ax2.set_xlabel("round"); ax2.set_ylabel("fitness evaluations"); ax2.set_xticks(rounds)
    ax2.set_ylim(0, max(max(evals) if evals else 0, 1) * 1.12)     # closed-form runs report 0 evaluations
    ax2.set_title("Cost per round", loc="left")
    fig.suptitle(title or f"coverage {100 * result.coverage_fraction:.0f} %  ·  N_NL = {result.n_not_localized}  ·  "
                 f"E_l = {_fmt(result.e_l)} m²  ·  RMSE = {_fmt(result.rmse)} m", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def plot_network_map(scenario_or_arrays, result, path, title: str | None = None) -> None:
    """3-D map: beacons, true node positions (hollow when not localized), estimates and error segments."""
    _style()
    beacons, nodes = _positions(scenario_or_arrays)
    settled = np.asarray(result.settled, dtype=bool)
    est = np.asarray(result.estimates, dtype=float)
    n = nodes.shape[0]
    fig = plt.figure(figsize=(7.6, 5.8))
    ax = fig.add_subplot(111, projection="3d")
    if beacons.shape[0]:
        ax.scatter(beacons[:, 0], beacons[:, 1], beacons[:, 2], marker="s", s=55, color=DARK,
                   depthshade=False, label=f"beacons (M = {beacons.shape[0]})")
    if settled.any():
        ax.scatter(nodes[settled, 0], nodes[settled, 1], nodes[settled, 2], marker="o", s=26, color=RUST,
                   depthshade=False, label=f"true position, localized ({int(settled.sum())})")
        ax.scatter(est[settled, 0], est[settled, 1], est[settled, 2], marker="x", s=34, color=TEAL,
                   depthshade=False, label="estimate")
        for p, q in zip(nodes[settled], est[settled]):
            ax.plot([p[0], q[0]], [p[1], q[1]], [p[2], q[2]], color=GREY, lw=1)
    if (~settled).any():
        ax.scatter(nodes[~settled, 0], nodes[~settled, 1], nodes[~settled, 2], marker="o", s=34,
                   facecolors="none", edgecolors=RUST, linewidths=1.2, depthshade=False,
                   label=f"not localized ({int((~settled).sum())})")
    field = getattr(result, "field_size", None) or getattr(scenario_or_arrays, "field_size", None)
    if field is None:
        allp = np.vstack([beacons, nodes]) if n or beacons.shape[0] else np.ones((1, 3))
        field = tuple(float(v) for v in np.maximum(allp.max(axis=0), 1.0))
    fx, fy, fz = (float(v) for v in field)
    ax.set_xlim(0, fx); ax.set_ylim(0, fy); ax.set_zlim(0, fz)
    try:
        ax.set_box_aspect((fx, fy, max(fz, 0.25 * max(fx, fy))))
    except Exception:  # very old matplotlib without 3-D box aspect
        pass
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title or f"Iterative localization: {int(settled.sum())}/{n} nodes settled in "
                 f"{len(result.rounds)} round(s), E_l = {_fmt(result.e_l)} m², RMSE = {_fmt(result.rmse)} m", loc="left",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8, bbox_to_anchor=(0.0, 0.95))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)

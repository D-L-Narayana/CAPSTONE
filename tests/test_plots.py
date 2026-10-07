"""Tests for pso3d.plots: every plot function writes a PNG (> 5 kB) to tmp_path and the formerly
hard-coded scenario values (field size, number of anchors, final iteration, plateau marker) are
derived from the inputs. No GIF is rendered here."""
from __future__ import annotations

import os

import numpy as np
import pytest

from pso3d import plots                      # selects the Agg backend before pyplot is imported here
from pso3d.config import Scenario
from pso3d.experiments import StudyRow
import matplotlib.pyplot as plt

MIN_PNG = 5 * 1024
METHODS8 = ("pso", "std", "lsq", "lsq_gn", "amcmpso", "std_tol", "std_ws", "amcmpso_ws")


def _png_ok(path) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > MIN_PNG


def _rows():
    rows = []
    for i, m in enumerate(METHODS8):
        base = 8.0 - 0.8 * i
        for v in (0.1, 0.5, 1.0):
            rows.append(StudyRow("noise_sigma", v, m, base * v, 0.7 * base * v, 1.6 * base * v,
                                 0.0 if m == "lsq" else 1200.0, 1.0, 120, 3,
                                 crlb_rmse=0.9 * v, iterations_mean=60.0))
    return rows


def test_convergence_default_has_no_plateau_marker(tmp_path):
    hist = list(np.geomspace(100.0, 1.0, 12))
    path = tmp_path / "conv.png"
    plots.plot_convergence(hist, str(path))
    assert _png_ok(path)
    fig = plots._convergence_figure(hist, "t", None)
    ax = fig.axes[0]
    assert len(ax.lines) == 1 and len(ax.texts) == 0
    plt.close(fig)


def test_convergence_plateau_argument_draws_marker_at_given_iteration():
    hist = list(np.geomspace(100.0, 1.0, 12))
    fig = plots._convergence_figure(hist, "custom", 4)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert np.allclose(ax.lines[1].get_xdata(), [4, 4])
    assert len(ax.texts) == 1 and "plateau" in ax.texts[0].get_text()
    assert ax.get_title(loc="left") == "custom"
    plt.close(fig)


def test_scenario_3d_labels_and_limits_come_from_inputs(tmp_path):
    sc = Scenario(field_size=(30.0, 40.0, 10.0),
                  anchors=np.array([[0, 0, 0], [30, 0, 2], [0, 40, 3], [30, 40, 10], [15, 20, 5]], float),
                  true_position=np.array([10.0, 12.0, 4.0]), noise=np.zeros(5))
    rng = np.random.default_rng(0)
    hist = rng.uniform(0, [30, 40, 10], (7, 6, 3))        # 7 frames (t = 0..6) of 6 particles
    est = np.array([10.5, 11.5, 4.5])
    path = tmp_path / "scenario.png"
    plots.plot_scenario_3d(sc, est, hist, str(path))
    assert _png_ok(path)
    fig = plots._scenario_figure(sc, est, hist)
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "swarm end (t=6)" in labels
    assert "anchors (M=5)" in labels
    assert np.allclose(ax.get_xlim(), (0, 30)) and np.allclose(ax.get_ylim(), (0, 40)) and np.allclose(ax.get_zlim(), (0, 10))
    assert "30 x 40 x 10 m" in ax.get_title(loc="left")
    plt.close(fig)


def test_field_limits_helper_used_by_gif_frames():
    sc = Scenario(field_size=(10.0, 20.0, 5.0))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plots._set_field_limits(ax, sc)
    assert np.allclose(ax.get_xlim(), (0, 10)) and np.allclose(ax.get_ylim(), (0, 20)) and np.allclose(ax.get_zlim(), (0, 5))
    plt.close(fig)


def test_sweep_png_with_all_methods_and_crlb_line(tmp_path):
    path = tmp_path / "sweep.png"
    plots.plot_sweep(_rows(), "noise_sigma", "sigma [m]", str(path), logx=True)
    assert _png_ok(path)
    fig = plots._sweep_figure(_rows(), "noise_sigma", "sigma [m]", True)
    acc, cost = fig.axes[0], fig.axes[1]
    labels = [t.get_text() for t in acc.get_legend().get_texts()]
    assert len(acc.lines) == len(METHODS8) + 1          # one curve per method + the CRLB reference
    assert len(cost.lines) == len(METHODS8)
    assert "Simplified PSO" in labels and any("CRLB" in lab for lab in labels)
    plt.close(fig)


def test_error_cdf_png_plots_every_method(tmp_path):
    rng = np.random.default_rng(1)
    errs = {f"method {i}": np.abs(rng.normal(0, 1 + i, 50)) for i in range(6)}
    path = tmp_path / "cdf.png"
    plots.plot_error_cdf(errs, str(path))
    assert _png_ok(path)
    fig = plots._error_cdf_figure(errs, "cdf title")
    ax = fig.axes[0]
    assert len(ax.lines) == 6
    assert ax.get_title(loc="left") == "cdf title"
    plt.close(fig)


def test_method_bars_png_with_crlb_reference(tmp_path):
    rows = _rows()
    path = tmp_path / "bars.png"
    plots.plot_method_bars(rows, str(path), "noise_sigma", 0.5)
    assert _png_ok(path)
    fig = plots._method_bars_figure(rows, "noise_sigma", 0.5)
    ax = fig.axes[0]
    assert len(ax.patches) == len(METHODS8)
    assert [p.get_height() for p in ax.patches] == pytest.approx([(8.0 - 0.8 * i) * 0.5 for i in range(len(METHODS8))])
    assert any(np.allclose(line.get_ydata(), 0.45) for line in ax.lines), "CRLB reference line expected"
    plt.close(fig)


def test_method_bars_without_crlb_has_no_reference_line_and_validates_point():
    rows = [StudyRow("n_anchors", 4.0, m, 5.0 - 0.5 * i, 4.0, 8.0, 1200.0, 1.0, 120, 3)
            for i, m in enumerate(("pso", "std", "lsq"))]
    fig = plots._method_bars_figure(rows, "n_anchors", 4.0)
    ax = fig.axes[0]
    assert len(ax.patches) == 3 and len(ax.lines) == 0
    plt.close(fig)
    with pytest.raises(ValueError):
        plots._method_bars_figure(rows, "n_anchors", 99.0)


def test_crlb_heatmap_png(tmp_path):
    xs = np.linspace(0, 60, 13)
    ys = np.linspace(0, 60, 13)
    X, Y = np.meshgrid(xs, ys)
    grid = 0.5 + 0.01 * np.hypot(X - 30, Y - 30)
    grid[0, 0] = np.inf                                   # singular geometry cells are tolerated
    path = tmp_path / "heat.png"
    plots.plot_crlb_heatmap(xs, ys, grid, str(path), "CRLB bound, z = 8.5 m")
    assert _png_ok(path)

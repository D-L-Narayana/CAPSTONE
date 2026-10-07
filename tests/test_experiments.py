"""Tests for the Monte-Carlo experiments registry (pso3d.experiments) and the study CLI
(scripts/run_experiments.py)."""
from __future__ import annotations

import csv
import importlib.util
import os

import numpy as np
import pytest

from pso3d import experiments
from pso3d.config import Scenario
from pso3d.experiments import METHODS, MethodSpec, StudyRow, monte_carlo, sweep

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEGACY = ("pso", "std", "lsq", "lsq_gn")
NEW = ("amcmpso", "std_tol", "std_ws", "amcmpso_ws")
LEGACY_LABELS = {"pso": "Simplified PSO", "std": "Standard PSO",
                 "lsq": "Least-squares trilateration", "lsq_gn": "LSQ + Gauss-Newton"}
LEGACY_HEADER = ["parameter", "value", "method", "rmse_m", "mean_error_m", "p90_error_m",
                 "fitness_evaluations", "runtime_ms", "memory_floats", "trials"]
EXTRA_COLUMNS = ["crlb_rmse_m", "iterations_mean"]
# noise and anchors sweeps: 5 values x 4 methods; particles and iterations sweeps: 5 values x 2 swarm methods
LEGACY_ROW_COUNT = 5 * 4 + 5 * 4 + 5 * 2 + 5 * 2

# captured from the unmodified pso3d.experiments before the registry refactor
# (monte_carlo(method, trials=5, noise_sigma=0.5, n_anchors=4, n_particles=20, n_iterations=60, seed=123))
LEGACY_PINNED = {
    "pso": dict(rmse=9.360750487431718, mean_error=7.766320477074943, p90_error=13.751901464339651,
                evaluations=1200.0, memory_floats=120),
    "std": dict(rmse=1.7423535486759325, mean_error=1.2938536124832674, p90_error=2.59416205824404,
                evaluations=1220.0, memory_floats=183),
    "lsq": dict(rmse=4.014155706357515, mean_error=3.6829342319508447, p90_error=5.541805679613662,
                evaluations=0.0, memory_floats=15),
    "lsq_gn": dict(rmse=2.1649482934555473, mean_error=1.91405307013627, p90_error=3.0770531528238716,
                   evaluations=8.0, memory_floats=15),
}


def _load_script():
    path = os.path.join(ROOT, "scripts", "run_experiments.py")
    spec = importlib.util.spec_from_file_location("run_experiments_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


# ----------------------------------------------------------------------------- legacy protection (S2)
@pytest.mark.parametrize("method", sorted(LEGACY_PINNED))
def test_legacy_monte_carlo_numbers_pinned(method):
    """The per-trial RNG consumption order (anchors -> true position -> noise -> PSO seed) is frozen."""
    r = monte_carlo(method, 5, 0.5, 4, 20, 60, seed=123)
    expected = LEGACY_PINNED[method]
    for key in ("rmse", "mean_error", "p90_error", "evaluations"):
        assert abs(r[key] - expected[key]) <= 1e-9, (method, key, r[key], expected[key])
    assert r["memory_floats"] == expected["memory_floats"]
    assert r["trials"] == 5
    assert len(r["errors"]) == 5


# ----------------------------------------------------------------------------- registry
def test_registry_has_all_methods_with_labels_and_factories():
    assert set(METHODS) == set(LEGACY) | set(NEW)
    assert tuple(METHODS)[:4] == LEGACY, "legacy keys first, in the legacy order"
    for key, spec in METHODS.items():
        assert isinstance(spec, MethodSpec)
        assert spec.key == key
        assert isinstance(spec.label, str) and spec.label
        assert callable(spec.factory)
        assert spec.closed_form == (key in ("lsq", "lsq_gn"))
    for key, label in LEGACY_LABELS.items():
        assert METHODS[key].label == label
    assert "AMCMPSO" in METHODS["amcmpso"].label
    assert experiments.LEGACY_METHODS == LEGACY
    assert experiments.EXTENDED_METHODS == NEW


def test_study_row_keeps_legacy_positional_layout_and_adds_defaults():
    row = StudyRow("noise_sigma", 0.5, "pso", 8.4, 6.3, 14.7, 1200.0, 1.3, 120, 200)
    assert (row.label, row.value, row.method, row.trials) == ("noise_sigma", 0.5, "pso", 200)
    assert np.isnan(row.crlb_rmse) and np.isnan(row.iterations_mean)
    row2 = StudyRow("noise_sigma", 0.5, "pso", 8.4, 6.3, 14.7, 1200.0, 1.3, 120, 200, 0.9, 60.0)
    assert row2.crlb_rmse == 0.9 and row2.iterations_mean == 60.0


# ----------------------------------------------------------------------------- new methods
@pytest.mark.parametrize("method", NEW)
def test_new_methods_run_and_return_finite_metrics(method):
    r = monte_carlo(method, 3, 0.5, 4, 20, 10, seed=7)
    for key in ("rmse", "mean_error", "p90_error", "crlb_rmse", "iterations_mean", "evaluations"):
        assert np.isfinite(r[key]), (method, key, r[key])
    assert r["rmse"] >= r["mean_error"] >= 0.0
    assert r["evaluations"] > 0
    assert 0 < r["iterations_mean"] <= 10
    assert r["memory_floats"] > 0
    assert r["trials"] == 3 and len(r["errors"]) == 3


def test_tolerance_stopping_uses_fewer_iterations_than_fixed_budget():
    fixed = monte_carlo("std", 3, 0.5, 4, 20, 60, seed=11)
    tol = monte_carlo("std_tol", 3, 0.5, 4, 20, 60, seed=11)
    assert fixed["iterations_mean"] == 60.0
    assert tol["iterations_mean"] < 60.0
    assert tol["evaluations"] < fixed["evaluations"]


# ----------------------------------------------------------------------------- CRLB reference
def test_crlb_rmse_formula_matches_direct_computation_and_scales_with_sigma():
    sc = Scenario()
    p = sc.true_position
    diff = p - sc.anchors
    J = diff / np.linalg.norm(diff, axis=1)[:, None]
    expected = 0.5 * np.sqrt(np.trace(np.linalg.inv(J.T @ J)))
    assert abs(experiments._crlb_rmse(sc.anchors, p, 0.5) - expected) < 1e-12
    assert abs(experiments._crlb_rmse(sc.anchors, p, 1.0) - 2.0 * expected) < 1e-12
    # all anchors in the plane z = 0 and the node in that plane -> singular information matrix
    flat = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [60.0, 60.0, 0.0]])
    assert experiments._crlb_rmse(flat, np.array([20.0, 10.0, 0.0]), 0.5) == np.inf


def test_crlb_reference_column_matches_positions_and_bounds_lsq_gn():
    r_gn = monte_carlo("lsq_gn", 20, 0.1, 4, 20, 60, seed=123)
    r_lsq = monte_carlo("lsq", 20, 0.1, 4, 20, 60, seed=123)
    assert np.isfinite(r_gn["crlb_rmse"]) and r_gn["crlb_rmse"] > 0
    # the bound depends only on the anchors and the true positions, which are drawn identically
    # for every method under the same seed
    assert r_gn["crlb_rmse"] == r_lsq["crlb_rmse"]
    # recompute from the frozen RNG order: (fixed 4 anchors) -> true position -> noise, per trial
    base = Scenario()
    rng = np.random.default_rng(123)
    bounds = []
    for _ in range(20):
        true_p = rng.uniform(np.zeros(3), base.upper)
        rng.normal(0.0, 0.1, size=4)
        bounds.append(experiments._crlb_rmse(base.anchors, true_p, 0.1))
    assert abs(r_gn["crlb_rmse"] - float(np.mean(bounds))) < 1e-12
    # the Monte-Carlo RMSE of the refined closed-form solution is not below the bound by more than
    # the 20-trial sampling tolerance used throughout this project (bound x 0.8)
    assert 0.8 * r_gn["crlb_rmse"] <= r_gn["rmse"]
    assert r_gn["rmse"] <= r_lsq["rmse"]
    # with sigma = 0.5 m the non-linearity of the estimator dominates and the bound is met outright
    r_gn5 = monte_carlo("lsq_gn", 20, 0.5, 4, 20, 60, seed=123)
    assert r_gn5["crlb_rmse"] <= r_gn5["rmse"]
    assert abs(r_gn5["crlb_rmse"] - 5.0 * r_gn["crlb_rmse"]) < 1e-9     # bound is linear in sigma


def test_iterations_mean_for_legacy_methods():
    assert monte_carlo("pso", 2, 0.5, 4, 10, 15, seed=3)["iterations_mean"] == 15.0
    assert monte_carlo("std", 2, 0.5, 4, 10, 15, seed=3)["iterations_mean"] == 15.0
    assert monte_carlo("lsq", 2, 0.5, 4, 10, 15, seed=3)["iterations_mean"] == 0.0
    gn = monte_carlo("lsq_gn", 2, 0.5, 4, 10, 15, seed=3)
    assert gn["iterations_mean"] == gn["evaluations"] > 0


# ----------------------------------------------------------------------------- sweep / noise model / validation
def test_sweep_filters_methods_and_fills_new_columns():
    kw = dict(noise_sigma=0.5, n_anchors=4, n_particles=5, n_iterations=5)
    rows = sweep("noise_sigma", [0.5, 1.0], trials=2, methods=["lsq"], **kw)
    assert [r.method for r in rows] == ["lsq", "lsq"]
    assert [r.value for r in rows] == [0.5, 1.0]
    assert all(isinstance(r, StudyRow) for r in rows)
    assert all(np.isfinite(r.crlb_rmse) and r.crlb_rmse > 0 for r in rows)
    assert all(r.iterations_mean == 0.0 for r in rows)
    assert all(r.trials == 2 and r.label == "noise_sigma" for r in rows)
    # default method set is the legacy four; closed-form methods are skipped for swarm-only labels
    rows = sweep("n_particles", [5], trials=1, **kw)
    assert [r.method for r in rows] == ["pso", "std"]
    rows = sweep("n_anchors", [4], trials=1, **kw)
    assert [r.method for r in rows] == list(LEGACY)


def test_noise_model_object_is_used_for_the_noise_draw():
    class ZeroNoise:
        name = "zero"

        def __init__(self):
            self.calls = []

        def sample(self, true_ranges, rng):
            self.calls.append(np.array(true_ranges, float))
            assert isinstance(rng, np.random.Generator)
            return np.zeros_like(true_ranges)

    model = ZeroNoise()
    r = monte_carlo("lsq", 4, 0.5, 5, 5, 5, seed=5, noise_model=model)
    assert len(model.calls) == 4
    assert all(c.shape == (5,) for c in model.calls)
    assert r["rmse"] < 1e-6                      # noise-free ranges -> exact closed-form solution
    gaussian = monte_carlo("lsq", 4, 0.5, 5, 5, 5, seed=5)
    assert gaussian["rmse"] > 1e-3
    assert np.isfinite(r["crlb_rmse"]) and r["crlb_rmse"] > 0


def test_validation_errors():
    with pytest.raises(ValueError):
        monte_carlo("lsq", 1, 0.5, 3, 5, 5)
    with pytest.raises(ValueError):
        monte_carlo("no_such_method", 1, 0.5, 4, 5, 5)
    with pytest.raises(ValueError):
        sweep("noise_sigma", [0.5], trials=1, methods=["no_such_method"],
              noise_sigma=0.5, n_anchors=4, n_particles=5, n_iterations=5)


# ----------------------------------------------------------------------------- CLI
def test_cli_default_run_reproduces_legacy_layout(tmp_path):
    mod = _load_script()
    out = tmp_path / "res"
    fig = tmp_path / "fig"
    rc = mod.main(["--trials", "2", "--out", str(out), "--figures", str(fig)])
    assert rc == 0
    assert (out / "sweeps.csv").exists() and (out / "results.md").exists()
    with open(out / "sweeps.csv", "rb") as f:
        raw = f.read()
    assert b"\r" not in raw, "sweeps.csv must use LF line endings (csv.writer defaults to CRLF)"
    header, rows = _read_csv(out / "sweeps.csv")
    assert header == LEGACY_HEADER
    assert len(rows) == LEGACY_ROW_COUNT
    assert {r[2] for r in rows} == set(LEGACY)
    assert all(len(r) == len(LEGACY_HEADER) for r in rows)
    assert all(r[9] == "2" for r in rows)
    md = (out / "results.md").read_text()
    assert md.startswith("# Monte-Carlo parameter study")
    assert "Extended methods" not in md and "CRLB" not in md
    for name in ("sweep_noise.png", "sweep_anchors.png", "sweep_particles.png", "sweep_iterations.png", "error_cdf.png"):
        assert (fig / name).stat().st_size > 5 * 1024, name
    assert not (fig / "method_bars.png").exists()


def test_cli_extended_run_adds_methods_columns_tables_and_bars(tmp_path):
    mod = _load_script()
    out = tmp_path / "res"
    fig = tmp_path / "fig"
    rc = mod.main(["--trials", "2", "--out", str(out), "--figures", str(fig), "--methods", "lsq_gn", "--extended"])
    assert rc == 0
    assert (out / "sweeps.csv").exists() and (out / "results.md").exists()
    header, rows = _read_csv(out / "sweeps.csv")
    assert header == LEGACY_HEADER + EXTRA_COLUMNS
    assert {r[2] for r in rows} == {"lsq_gn", *NEW}
    assert all(len(r) == len(header) and float(r[10]) > 0 for r in rows)      # CRLB column filled
    for r in rows:
        if r[0] == "n_iterations" and r[2] in ("amcmpso", "std_ws", "amcmpso_ws"):
            assert float(r[11]) == float(r[1])                                 # fixed budget runs all T iterations
        if r[0] == "n_iterations" and r[2] == "std_tol":
            assert 0 < float(r[11]) <= float(r[1])                            # tolerance stop never exceeds T
    md = (out / "results.md").read_text()
    assert "Extended methods" in md and "CRLB" in md
    assert "interpretation" in md                 # AMCMPSO honesty note travels with its numbers
    for name in ("sweep_noise.png", "error_cdf.png", "method_bars.png"):
        assert (fig / name).stat().st_size > 5 * 1024, name


def test_cli_non_legacy_method_without_extended_flag_and_no_plots(tmp_path):
    mod = _load_script()
    out = tmp_path / "res"
    fig = tmp_path / "fig"
    rc = mod.main(["--trials", "1", "--out", str(out), "--figures", str(fig), "--methods", "std_tol,lsq", "--no-plots"])
    assert rc == 0
    header, rows = _read_csv(out / "sweeps.csv")
    assert header == LEGACY_HEADER + EXTRA_COLUMNS       # a non-legacy method alone switches the layout
    assert [r[2] for r in rows].count("lsq") == 10        # noise + anchors sweeps only
    assert [r[2] for r in rows].count("std_tol") == 20    # all four sweeps
    assert len(rows) == 30
    assert "Extended methods" in (out / "results.md").read_text()
    assert not fig.exists()                               # no figure is written or directory created


def test_cli_closed_form_only_keeps_legacy_layout_and_marks_empty_swarm_sweeps(tmp_path):
    mod = _load_script()
    out = tmp_path / "res"
    rc = mod.main(["--trials", "1", "--out", str(out), "--figures", str(tmp_path / "fig"), "--methods", "lsq,lsq_gn", "--no-plots"])
    assert rc == 0
    header, rows = _read_csv(out / "sweeps.csv")
    assert header == LEGACY_HEADER                        # legacy methods only -> legacy columns
    assert [r[2] for r in rows] == ["lsq", "lsq_gn"] * 10  # noise + anchors sweeps, no swarm sweeps
    md = (out / "results.md").read_text()
    assert md.count("swarm methods only") == 2            # Swarm size N and Iterations T sections
    assert "## Error CDF" in md


def test_cli_rejects_unknown_method(tmp_path):
    mod = _load_script()
    rc = mod.main(["--trials", "1", "--out", str(tmp_path), "--figures", str(tmp_path), "--methods", "nope"])
    assert rc == 2
    assert not (tmp_path / "sweeps.csv").exists()

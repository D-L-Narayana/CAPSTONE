"""Tests for the multi-node iterative auto-localization simulator.

Covers ``pso3d.network`` (scenario generation, localizer factories, the iterative scheme and its
metrics), ``pso3d.network_plots`` (PNG output) and ``scripts/run_network.py`` (CLI).  Everything is
kept small: <= 20 nodes, <= 3 rounds, StandardPSO with 10 particles x 20 iterations.
"""
from __future__ import annotations

import csv
import importlib.util
import os
from types import SimpleNamespace

import numpy as np
import pytest

from pso3d.network import (
    LocalizeOutcome,
    NetworkResult,
    NetworkScenario,
    RoundStats,
    iterative_localize,
    lsq_gn_localizer,
    pso_localizer,
    reference_rank,
    settled_only_error,
    summary_table,
    warm_started_localizer,
)
from pso3d.network_plots import plot_network_map, plot_rounds
from pso3d.standard_pso import StandardPSO
from pso3d.trilateration import gauss_newton_refine, least_squares_trilateration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Hand-made geometry (field 100 x 100 x 30 m, radio range 25 m):
#   * five non-coplanar beacons in the corner region,
#   * four "bridge" nodes ~20 m away that see all five beacons in round 1,
#   * one target node that sees no beacon but all four bridge nodes (so it can only settle in round 2),
#   * one far-away node that sees nothing at all.
BEACONS = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 5.0], [0.0, 10.0, 10.0], [10.0, 10.0, 0.0], [5.0, 5.0, 15.0]])
BRIDGE = np.array([[20.0, 2.0, 3.0], [20.0, 8.0, 12.0], [22.0, 5.0, 6.0], [18.0, 9.0, 1.0]])
TARGET = np.array([[40.0, 5.0, 6.0]])
FAR = np.array([[95.0, 95.0, 25.0]])
LOWER = np.zeros(3)
UPPER = np.array([100.0, 100.0, 30.0])


def _load_script():
    path = os.path.join(ROOT, "scripts", "run_network.py")
    spec = importlib.util.spec_from_file_location("run_network_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingBiasedLocalizer:
    """Closed-form localizer that records every call and adds a known bias to its answer."""

    def __init__(self, bias):
        self.bias = np.asarray(bias, float)
        self.calls: list[tuple[int, np.ndarray, np.ndarray]] = []

    def __call__(self, anchors, measured, lower, upper, seed):
        self.calls.append((int(seed), np.array(anchors, float), np.array(measured, float)))
        p0 = least_squares_trilateration(anchors, measured)
        p, used = gauss_newton_refine(anchors, measured, p0)
        return LocalizeOutcome(p + self.bias, used, used)


class RecordingNoise:
    """Noise model per the shared contract: additive noise with the shape of ``true_ranges``."""

    name = "recording"

    def __init__(self, offset=0.5):
        self.offset = offset
        self.shapes: list[tuple[int, ...]] = []

    def sample(self, true_ranges, rng):
        self.shapes.append(np.shape(true_ranges))
        return np.full(np.shape(true_ranges), self.offset)


class FakeWarmPSO:
    """Optimiser stand-in that accepts ``x0`` and simply returns it."""

    def __init__(self, seed=None, x0=None, n_particles=0):
        self.seed, self.x0, self.n_particles = seed, x0, n_particles

    def run(self, fitness, lower, upper):
        return SimpleNamespace(estimate=np.asarray(self.x0, float), fitness_evaluations=1, iterations_run=0)


class FakeColdPSO:
    """Optimiser stand-in without ``x0`` support."""

    def __init__(self, seed=None, n_particles=0):
        self.seed, self.n_particles = seed, n_particles

    def run(self, fitness, lower, upper):
        return SimpleNamespace(estimate=np.ones(3), fitness_evaluations=1, iterations_run=0)


# ----------------------------------------------------------------------------------------------- scenario


def test_generate_shapes_bounds_and_determinism():
    sc = NetworkScenario(n_nodes=20, n_beacons=6, seed=3)
    beacons, nodes = sc.generate()
    assert beacons.shape == (6, 3) and nodes.shape == (20, 3)
    upper = np.asarray(sc.field_size, float)
    assert (beacons >= 0).all() and (beacons <= upper).all()
    assert (nodes >= 0).all() and (nodes <= upper).all()
    beacons2, nodes2 = NetworkScenario(n_nodes=20, n_beacons=6, seed=3).generate()
    assert np.array_equal(beacons, beacons2) and np.array_equal(nodes, nodes2)
    # beacons are drawn first, so they do not depend on the number of nodes
    beacons3, _ = NetworkScenario(n_nodes=7, n_beacons=6, seed=3).generate()
    assert np.array_equal(beacons, beacons3)
    _, nodes4 = NetworkScenario(n_nodes=20, n_beacons=6, seed=4).generate()
    assert not np.array_equal(nodes, nodes4)
    assert NetworkScenario().field_size == (100.0, 100.0, 30.0)
    assert (NetworkScenario().n_nodes, NetworkScenario().n_beacons, NetworkScenario().radio_range) == (50, 10, 25.0)


# ----------------------------------------------------------------------------------------------- localizers


def test_pso_localizer_runs_the_class_with_the_given_seed_and_counts_evaluations():
    p = np.array([6.0, 4.0, 7.0])
    d = np.linalg.norm(BEACONS - p, axis=1)
    loc = pso_localizer(StandardPSO, n_particles=10, n_iterations=20)
    out = loc(BEACONS, d, LOWER, UPPER, seed=5)
    assert isinstance(out, LocalizeOutcome)
    assert out.position.shape == (3,)
    assert out.evaluations == 10 * 21 and out.iterations == 20
    again = loc(BEACONS, d, LOWER, UPPER, seed=5)
    assert np.array_equal(out.position, again.position)
    other = loc(BEACONS, d, LOWER, UPPER, seed=6)
    assert not np.array_equal(out.position, other.position)
    assert np.linalg.norm(out.position - p) < 1.0


def test_lsq_gn_localizer_is_exact_on_noise_free_ranges():
    p = np.array([6.0, 4.0, 7.0])
    d = np.linalg.norm(BEACONS - p, axis=1)
    out = lsq_gn_localizer()(BEACONS, d, LOWER, UPPER, seed=0)
    assert np.linalg.norm(out.position - p) < 1e-6
    assert out.evaluations == out.iterations >= 1


def test_warm_started_localizer_passes_closed_form_start_as_x0_when_supported():
    p = np.array([6.0, 4.0, 7.0])
    d = np.linalg.norm(BEACONS - p, axis=1)
    warm = warm_started_localizer(FakeWarmPSO, n_particles=3)
    assert warm.warm_start_supported is True
    out = warm(BEACONS, d, LOWER, UPPER, seed=1)
    assert np.allclose(out.position, p, atol=1e-6)          # x0 = LSQ -> GN = the exact solution here
    cold = warm_started_localizer(FakeColdPSO, n_particles=3)
    assert cold.warm_start_supported is False
    out2 = cold(BEACONS, d, LOWER, UPPER, seed=1)
    assert np.allclose(out2.position, np.ones(3))
    # the real optimiser works either way (with or without x0 support in its constructor)
    real = warm_started_localizer(StandardPSO, n_particles=10, n_iterations=20)
    assert isinstance(real.warm_start_supported, bool)
    out3 = real(BEACONS, d, LOWER, UPPER, seed=2)
    assert (out3.position >= LOWER).all() and (out3.position <= UPPER).all()
    assert out3.evaluations == 10 * 21


# ----------------------------------------------------------------------------------------------- iterative scheme


def test_noise_free_round_one_settles_exactly_the_nodes_with_enough_beacon_references():
    sc = NetworkScenario(n_nodes=20, n_beacons=8, radio_range=45.0, seed=5)
    beacons, nodes = sc.generate()
    res = iterative_localize(sc, lsq_gn_localizer(), noise=0.0, max_rounds=1)
    expected = np.zeros(20, bool)
    for i, p in enumerate(nodes):
        d = np.linalg.norm(beacons - p, axis=1)
        near = np.argsort(d)[:6]
        near = near[d[near] <= sc.radio_range]
        if len(near) >= 4:
            centred = beacons[near] - beacons[near].mean(axis=0)
            expected[i] = np.linalg.matrix_rank(centred) == 3
    assert expected.any() and not expected.all()            # the scenario is informative
    assert np.array_equal(res.settled, expected)
    err = np.linalg.norm(res.estimates[expected] - nodes[expected], axis=1)
    assert (err < 1e-3).all()
    assert len(res.rounds) == 1
    assert isinstance(res.rounds[0], RoundStats)
    assert res.rounds[0].round == 1 and res.rounds[0].newly_settled == int(expected.sum())
    assert res.rounds[0].settled_total == int(expected.sum())
    assert res.n_not_localized == int((~expected).sum())


def test_rounds_are_monotonic_and_consistent():
    sc = NetworkScenario(n_nodes=20, n_beacons=6, radio_range=35.0, seed=2)
    res = iterative_localize(sc, lsq_gn_localizer(), noise=0.0, max_rounds=3)
    assert isinstance(res, NetworkResult)
    assert 1 <= len(res.rounds) <= 3
    totals = [r.settled_total for r in res.rounds]
    assert totals == sorted(totals)
    for k, r in enumerate(res.rounds):
        assert r.round == k + 1
        assert r.settled_total == sum(x.newly_settled for x in res.rounds[: k + 1])
        assert r.evaluations >= 0 and r.refs_available_mean >= 0.0
    assert res.rounds[-1].settled_total == int(res.settled.sum())
    assert res.total_evaluations == sum(r.evaluations for r in res.rounds)
    assert res.estimates.shape == (20, 3) and res.settled.shape == (20,) and res.settled.dtype == bool
    assert res.runtime_s >= 0.0


def test_nodes_with_too_few_references_stay_unsettled():
    sc = NetworkScenario(n_nodes=5, n_beacons=5, radio_range=25.0, seed=1)
    nodes = np.vstack([BRIDGE, FAR])
    res = iterative_localize(sc, lsq_gn_localizer(), noise=0.0, max_rounds=3, positions=(BEACONS, nodes))
    assert res.settled.tolist() == [True, True, True, True, False]
    assert res.n_not_localized == 1
    assert res.coverage_fraction == pytest.approx(0.8)
    # round 1 settles the four bridge nodes, round 2 finds nothing new and the scheme stops
    assert [r.newly_settled for r in res.rounds] == [4, 0]
    assert res.rounds[0].refs_available_mean == pytest.approx(4.0)   # (5 + 5 + 5 + 5 + 0) / 5
    assert res.rounds[1].refs_available_mean == pytest.approx(0.0)
    assert np.allclose(res.estimates[:4], BRIDGE, atol=1e-6)
    assert np.isnan(res.estimates[4]).all()


def test_settled_nodes_act_as_references_through_their_estimates_not_truths():
    bias = np.array([2.0, 0.0, 0.0])
    sc = NetworkScenario(n_nodes=5, n_beacons=5, radio_range=25.0, seed=1)
    nodes = np.vstack([BRIDGE, TARGET])
    loc = RecordingBiasedLocalizer(bias)
    res = iterative_localize(sc, loc, noise=0.0, max_rounds=3, positions=(BEACONS, nodes))
    assert [r.newly_settled for r in res.rounds] == [4, 1]
    assert res.settled.all()
    # round 1: the bridge nodes only see beacons (true positions) and come out biased by exactly +2 m in x
    assert np.allclose(res.estimates[:4], BRIDGE + bias, atol=1e-6)
    # round 2: the target sees no beacon, so its anchors must be the bridge ESTIMATES, not the truths
    assert len(loc.calls) == 5
    _, anchors, measured = loc.calls[4]
    assert anchors.shape == (4, 3)
    for row in anchors:
        assert np.min(np.linalg.norm(BRIDGE + bias - row, axis=1)) < 1e-6
        assert np.min(np.linalg.norm(BRIDGE - row, axis=1)) > 1.0
    # ... while the ranges are measured between the TRUE positions
    for row, d in zip(anchors, measured):
        j = int(np.argmin(np.linalg.norm(BRIDGE + bias - row, axis=1)))
        assert d == pytest.approx(np.linalg.norm(BRIDGE[j] - TARGET[0]))
    # error accumulation: the inherited +2 m plus the localizer's own +2 m
    assert np.allclose(res.estimates[4], TARGET[0] + 2 * bias, atol=1e-6)
    seeds = [c[0] for c in loc.calls]
    assert len(set(seeds)) == 5 and all(0 <= s < 2**31 for s in seeds)


def test_e_l_rmse_and_coverage_match_independent_computation():
    sc = NetworkScenario(n_nodes=16, n_beacons=8, radio_range=50.0, seed=11)
    _, nodes = sc.generate()
    loc = pso_localizer(StandardPSO, n_particles=10, n_iterations=20)
    res = iterative_localize(sc, loc, noise=0.5, max_rounds=2)
    assert res.settled.any()
    sq = ((res.estimates[res.settled] - nodes[res.settled]) ** 2).sum(axis=1)
    assert res.e_l == pytest.approx(float(sq.mean()))
    assert res.rmse == pytest.approx(float(np.sqrt(sq.mean())))
    assert res.coverage_fraction == pytest.approx(float(res.settled.mean()))
    assert res.n_not_localized == int((~res.settled).sum())
    errs = settled_only_error(res, nodes)
    assert errs.shape == (int(res.settled.sum()),)
    assert np.allclose(errs, np.sqrt(sq))
    assert res.total_evaluations == 10 * 21 * int(res.settled.sum())
    table = summary_table(res)
    for key in ("N_NL", "E_l", "RMSE", "coverage", "evaluations"):
        assert key in table
    assert table.count("|") > 10


def test_deterministic_for_a_seed_and_default_seed_is_scenario_seed_plus_one():
    sc = NetworkScenario(n_nodes=12, n_beacons=8, radio_range=50.0, seed=3)
    loc = pso_localizer(StandardPSO, n_particles=10, n_iterations=20)
    a = iterative_localize(sc, loc, noise=0.5, max_rounds=2, seed=7)
    b = iterative_localize(sc, loc, noise=0.5, max_rounds=2, seed=7)
    assert a.settled.any()
    assert np.array_equal(a.estimates, b.estimates, equal_nan=True)
    assert np.array_equal(a.settled, b.settled)
    key = lambda r: [(x.round, x.newly_settled, x.settled_total, x.evaluations, x.refs_available_mean) for x in r.rounds]
    assert key(a) == key(b)
    assert a.e_l == b.e_l and a.total_evaluations == b.total_evaluations
    c = iterative_localize(sc, loc, noise=0.5, max_rounds=2)
    d = iterative_localize(sc, loc, noise=0.5, max_rounds=2, seed=sc.seed + 1)
    assert np.array_equal(c.estimates, d.estimates, equal_nan=True)
    e = iterative_localize(sc, loc, noise=0.5, max_rounds=2, seed=8)
    both = a.settled & e.settled
    assert both.any() and not np.allclose(a.estimates[both], e.estimates[both])


def test_noise_model_object_is_used_once_per_localized_node():
    sc = NetworkScenario(n_nodes=10, n_beacons=8, radio_range=50.0, seed=9)
    nm = RecordingNoise(offset=0.5)
    res = iterative_localize(sc, lsq_gn_localizer(), noise=nm, max_rounds=2, max_refs=5)
    assert res.settled.any()
    assert len(nm.shapes) == int(res.settled.sum())
    assert all(len(s) == 1 and 4 <= s[0] <= 5 for s in nm.shapes)
    _, nodes = sc.generate()
    assert settled_only_error(res, nodes).mean() > 1e-2       # a constant +0.5 m bias is not filtered out
    exact = iterative_localize(sc, lsq_gn_localizer(), noise=0.0, max_rounds=2, max_refs=5)
    assert np.array_equal(exact.settled, res.settled)
    assert settled_only_error(exact, nodes).max() < 1e-3


def test_coplanar_reference_sets_are_rejected_unless_allowed():
    flat = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [0.0, 20.0, 0.0], [20.0, 20.0, 0.0]])
    node = np.array([[10.0, 10.0, 5.0]])
    sc = NetworkScenario(n_nodes=1, n_beacons=4, radio_range=25.0)
    strict = iterative_localize(sc, lsq_gn_localizer(), noise=0.0, max_rounds=2, positions=(flat, node))
    assert not strict.settled.any() and strict.n_not_localized == 1
    assert strict.rounds[0].refs_available_mean == pytest.approx(4.0)
    assert strict.coverage_fraction == 0.0 and np.isnan(strict.e_l) and np.isnan(strict.rmse)
    loose = iterative_localize(sc, lsq_gn_localizer(), noise=0.0, max_rounds=2, positions=(flat, node),
                               require_non_coplanar=False)
    assert loose.settled.all()
    assert reference_rank(flat) == 2
    assert reference_rank(BEACONS) == 3
    assert reference_rank(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])) == 1


# ----------------------------------------------------------------------------------------------- plots


def test_plots_write_png_files(tmp_path):
    sc = NetworkScenario(n_nodes=12, n_beacons=6, radio_range=40.0, seed=2)
    res = iterative_localize(sc, lsq_gn_localizer(), noise=0.3, max_rounds=2)
    p1 = tmp_path / "rounds.png"
    p2 = tmp_path / "map.png"
    plot_rounds(res, p1)
    plot_network_map(sc, res, p2)
    assert p1.exists() and p1.stat().st_size > 5000
    assert p2.exists() and p2.stat().st_size > 5000
    beacons, nodes = sc.generate()
    p3 = tmp_path / "map_from_arrays.png"
    plot_network_map((beacons, nodes), res, p3)
    assert p3.exists() and p3.stat().st_size > 5000


# ----------------------------------------------------------------------------------------------- script


def test_script_main_writes_csv_and_markdown(tmp_path):
    mod = _load_script()
    rc = mod.main(["--nodes", "12", "--beacons", "6", "--rounds", "2", "--localizer", "lsq_gn",
                   "--out", str(tmp_path), "--figures", str(tmp_path), "--no-plots"])
    assert rc == 0
    csv_path, md_path = tmp_path / "network.csv", tmp_path / "network.md"
    assert csv_path.exists() and md_path.exists()
    # the CSV artefacts are committed to the repository: LF line endings only, no CRLF
    for name in ("network.csv", "network_nodes.csv"):
        raw = (tmp_path / name).read_bytes()
        assert raw.count(b"\n") >= 2 and b"\r" not in raw, f"{name} must use LF line endings"
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert 1 <= len(rows) <= 2
    assert {"round", "newly_settled", "settled_total", "refs_available_mean", "evaluations"} <= set(rows[0])
    assert rows[0]["round"] == "1"
    text = md_path.read_text()
    for key in ("N_NL", "E_l", "RMSE", "coverage", "lsq_gn", "gaussian:0.5"):
        assert key in text
    assert not (tmp_path / "network_rounds.png").exists()
    assert not (tmp_path / "network_map.png").exists()


def test_script_noise_spec_and_pso_localizer_choice(tmp_path):
    mod = _load_script()
    noise = mod.parse_noise_spec("gaussian:0.25")
    assert hasattr(noise, "sample") or float(noise) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        mod.parse_noise_spec("banana:1")
    rc = mod.main(["--nodes", "8", "--beacons", "6", "--range", "50", "--rounds", "1", "--localizer", "std",
                   "--particles", "10", "--iterations", "20", "--noise", "gaussian:0.1", "--seed", "4",
                   "--out", str(tmp_path), "--figures", str(tmp_path), "--no-plots"])
    assert rc == 0
    text = (tmp_path / "network.md").read_text()
    assert "std" in text and "gaussian:0.1" in text


def test_script_writes_both_figures_unless_disabled(tmp_path):
    mod = _load_script()
    rc = mod.main(["--nodes", "10", "--beacons", "6", "--range", "40", "--rounds", "2", "--localizer", "lsq_gn",
                   "--out", str(tmp_path), "--figures", str(tmp_path)])
    assert rc == 0
    for name in ("network_rounds.png", "network_map.png"):
        path = tmp_path / name
        assert path.exists() and path.stat().st_size > 5000

"""Tests for ``pso3d.amcmpso`` (AMCMPSO *interpretation*: adaptive coefficients plus
swarm-mean and centre-of-mass guidance).

Every number pinned below is the output of this repository's implementation on the
slide-16 scenario; none is a value from Alhasan et al. (2023), whose exact equations are
not transcribed here.  ``tests/fixtures/amcmpso_cases.json`` is created by
``_ensure_fixture`` when it is absent (it is never overwritten) and re-read by
``test_fixture_matches_fresh_computation``.
"""
from __future__ import annotations

import dataclasses
import json
import os

import numpy as np
import pytest

import pso3d.amcmpso as amcmpso_module
from pso3d import RangeErrorFitness, Scenario
from pso3d.amcmpso import AMCMPSO, AMCMPSOResult, centre_of_mass, coefficient_schedule
from pso3d.pso import PSOResult

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "amcmpso_cases.json")
N, T = 20, 60
W_MAX, W_MIN = 0.9, 0.4
C1_START, C1_END = 2.0, 0.5
C2_START, C2_END = 0.5, 2.0
GN_ESTIMATE = np.array([37.42, 11.88, 9.36])   # closed-form LSQ + Gauss-Newton estimate of the baseline scenario


class FireAt:
    """Local stopping rule following the duck-typed protocol: stop after iteration ``at``."""

    def __init__(self, at: int):
        self.at = at
        self.name = f"fire-at-{at}"
        self.resets = 0
        self.calls: list[tuple[int, float, int]] = []

    def reset(self) -> None:
        self.resets += 1
        self.calls = []

    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool:
        self.calls.append((iteration, float(best_fitness), int(evaluations)))
        return iteration >= self.at


def _run(n_particles: int = N, n_iterations: int = T, **kwargs):
    kwargs.setdefault("seed", 1)
    sc = Scenario()
    fit = RangeErrorFitness(sc.anchors, sc.measured())
    res = AMCMPSO(n_particles, n_iterations, **kwargs).run(fit, sc.lower, sc.upper)
    return sc, fit, res


# --------------------------------------------------------------------------- constructor / schedules

def test_constructor_defaults_are_unchanged():
    a = AMCMPSO()
    assert (a.n_particles, a.n_iterations) == (20, 60)
    assert (a.w_max, a.w_min, a.c1_start, a.c1_end, a.c2_start, a.c2_end) == (0.9, 0.4, 2.0, 0.5, 0.5, 2.0)
    assert (a.c_mean, a.c_com, a.seed, a.clip, a.record_positions) == (0.5, 0.5, 1, True, False)
    assert (a.schedule, a.stopping, a.x0, a.diagnostics) == ("linear", None, None, False)


def test_linear_schedule_matches_legacy_formula_and_is_monotone():
    s = coefficient_schedule(T, "linear")
    assert set(s) >= {"w", "c1", "c2"}
    assert len(s["w"]) == len(s["c1"]) == len(s["c2"]) == T
    assert s["w"][0] == W_MAX and s["c1"][0] == C1_START and s["c2"][0] == C2_START
    assert s["w"][-1] == pytest.approx(0.9 - 0.5 * (T - 1) / T, abs=1e-12)
    for t in range(T):                       # exactly the expression used by the original loop (bit-exact)
        frac = t / T
        assert s["w"][t] == W_MAX - (W_MAX - W_MIN) * frac
        assert s["c1"][t] == C1_START + (C1_END - C1_START) * frac
        assert s["c2"][t] == C2_START + (C2_END - C2_START) * frac
    assert np.all(np.diff(s["w"]) < 0) and np.all(np.diff(s["c1"]) < 0) and np.all(np.diff(s["c2"]) > 0)


def test_cosine_schedule_endpoints_midpoint_symmetry_bounds():
    s = coefficient_schedule(T, "cosine")          # T is even
    assert len(s["w"]) == len(s["c1"]) == len(s["c2"]) == T
    assert s["w"][0] == W_MAX and s["c1"][0] == C1_START and s["c2"][0] == C2_START
    s_last = (1.0 - np.cos(np.pi * (T - 1) / T)) / 2.0
    assert s["w"][-1] == pytest.approx(W_MAX - (W_MAX - W_MIN) * s_last, abs=1e-12)
    assert s["c1"][-1] == pytest.approx(C1_START + (C1_END - C1_START) * s_last, abs=1e-12)
    assert s["c2"][-1] == pytest.approx(C2_START + (C2_END - C2_START) * s_last, abs=1e-12)
    half = T // 2
    assert s["w"][half] == pytest.approx((W_MAX + W_MIN) / 2, abs=1e-12)
    assert s["c1"][half] == pytest.approx((C1_START + C1_END) / 2, abs=1e-12)
    assert s["c2"][half] == pytest.approx((C2_START + C2_END) / 2, abs=1e-12)
    for t in range(1, T):                           # s(t) + s(T - t) == 1 -> symmetric about T/2
        assert s["w"][t] + s["w"][T - t] == pytest.approx(W_MAX + W_MIN, abs=1e-12)
    assert np.all((s["w"] <= W_MAX) & (s["w"] >= W_MIN))
    assert np.all((s["c1"] <= C1_START) & (s["c1"] >= C1_END))
    assert np.all((s["c2"] >= C2_START) & (s["c2"] <= C2_END))
    assert np.all(np.diff(s["w"]) < 0) and np.all(np.diff(s["c1"]) < 0) and np.all(np.diff(s["c2"]) > 0)


def test_cosine_keeps_exploration_longer_than_linear():
    lin, cos = coefficient_schedule(T, "linear"), coefficient_schedule(T, "cosine")
    half = T // 2
    assert np.all(cos["w"][1:half] > lin["w"][1:half])          # inertia stays higher in the first half
    assert np.all(cos["c2"][1:half] < lin["c2"][1:half])        # social pull grows more slowly
    assert np.all(cos["w"][half + 1:] < lin["w"][half + 1:])    # and drops faster in the second half


def test_schedule_rejects_unknown_name():
    with pytest.raises(ValueError):
        coefficient_schedule(T, "bogus")
    with pytest.raises(ValueError):
        AMCMPSO(schedule="bogus")


def test_schedule_custom_parameters_and_length():
    s = coefficient_schedule(10, "linear", w_max=1.0, w_min=0.2,
                             c1_start=2.5, c1_end=0.5, c2_start=0.5, c2_end=2.5)
    assert len(s["w"]) == 10
    assert s["w"][0] == 1.0 and s["c1"][0] == 2.5 and s["c2"][0] == 0.5
    assert s["w"][-1] == pytest.approx(1.0 - 0.8 * 0.9, abs=1e-12)
    assert s["c2"][-1] == pytest.approx(0.5 + 2.0 * 0.9, abs=1e-12)
    assert coefficient_schedule(0)["w"].shape == (0,)


# --------------------------------------------------------------------------- centre of mass

def test_centre_of_mass_weights():
    x = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
    f = np.array([1e6, 0.0, 1e6, 1e6])             # particle 1 is (numerically) perfect
    com = centre_of_mass(x, f)
    assert com.shape == (3,)
    assert np.allclose(com, x[1], atol=1e-6)        # ~zero-fitness particle dominates
    assert np.allclose(centre_of_mass(x, np.full(4, 3.0)), x.mean(axis=0), atol=1e-12)   # equal f -> plain mean
    wgt = 1.0 / (f + 1e-9)                          # explicit formula with epsilon 1e-9
    assert np.allclose(com, (wgt[:, None] * x).sum(axis=0) / wgt.sum(), rtol=0, atol=1e-15)
    two = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    c2 = centre_of_mass(two, np.array([1.0, 3.0]))  # better particles are heavier
    assert 0.0 < c2[0] < 5.0


# --------------------------------------------------------------------------- baseline behaviour

def test_baseline_seed1_pin_two_decimals():
    sc, _, res = _run()
    assert np.allclose(np.round(res.estimate, 2), [37.42, 11.88, 9.36])
    assert round(res.error(sc.true_position), 2) == 0.96
    assert res.error(sc.true_position) < 1.0
    assert res.best_fitness == pytest.approx(res.best_history[-1])
    assert np.array_equal(res.estimate, res.estimate_history[-1])


def test_evaluations_iterations_and_histories_without_stopping():
    _, fit, res = _run()
    assert res.iterations_run == T
    assert res.fitness_evaluations == fit.evaluations == N * (T + 1) == 1220
    assert res.distance_computations == N * (T + 1) * 4
    assert len(res.best_history) == len(res.estimate_history) == T + 1
    assert all(b <= a + 1e-12 for a, b in zip(res.best_history, res.best_history[1:]))   # gbest never worsens
    assert res.positions_history is None
    assert res.diversity_history == []


def test_swarm_memory_floats():
    assert AMCMPSO(N, T).swarm_state_floats() == 9 * N + 6 == 186
    assert AMCMPSO(7, 3).swarm_state_floats() == 9 * 7 + 6
    _, _, res = _run()
    assert res.swarm_state_floats == 186 and res.swarm_state_bytes == 186 * 8


def test_result_type_and_fields():
    _, _, res = _run()
    assert isinstance(res, AMCMPSOResult) and isinstance(res, PSOResult)
    names = {f.name for f in dataclasses.fields(res)}
    assert {"improvement_rate", "diversity_history", "method", "stopped_by", "warm_start"} <= names
    assert res.method == "amcmpso"
    assert res.stopped_by is None
    assert res.warm_start is False
    assert dataclasses.fields(res)[0].name == "estimate"      # PSOResult positional order untouched


# --------------------------------------------------------------------------- stopping rules

def test_local_stopping_rule_fires_at_iteration_5():
    rule = FireAt(5)
    _, fit, res = _run(stopping=rule)
    assert rule.resets == 1
    assert res.iterations_run == 5
    assert res.fitness_evaluations == fit.evaluations == N * 6
    assert res.stopped_by == "fire-at-5"
    assert [c[0] for c in rule.calls] == [1, 2, 3, 4, 5]                      # 1-based iteration counter
    assert [c[2] for c in rule.calls] == [N * (k + 1) for k in range(1, 6)]   # evaluations after each iteration
    assert [c[1] for c in rule.calls] == pytest.approx(res.best_history[1:])  # best-ever fitness after each iteration
    assert len(res.best_history) == len(res.estimate_history) == 6


def test_stopping_rule_that_never_fires_runs_full_length():
    rule = FireAt(10 ** 6)
    _, _, res = _run(stopping=rule)
    assert res.iterations_run == T and res.fitness_evaluations == N * (T + 1)
    assert res.stopped_by is None and len(rule.calls) == T
    _, _, first = _run(stopping=FireAt(1))
    assert first.iterations_run == 1 and first.fitness_evaluations == 2 * N and first.stopped_by == "fire-at-1"


# --------------------------------------------------------------------------- warm start

def test_x0_warm_start_replaces_particle_zero_without_touching_rng():
    _, _, cold = _run(record_positions=True)
    x0 = np.array([70.0, -5.0, 10.0])               # outside the field in x and y -> clipped
    sc, _, warm = _run(record_positions=True, x0=x0)
    assert warm.warm_start is True and cold.warm_start is False
    assert np.array_equal(warm.positions_history[0][0], np.clip(x0, sc.lower, sc.upper))
    assert np.array_equal(warm.positions_history[0][0], [60.0, 0.0, 10.0])
    assert np.array_equal(warm.positions_history[0][1:], cold.positions_history[0][1:])
    assert warm.positions_history.shape == cold.positions_history.shape == (T + 1, N, 3)


def test_x0_from_closed_form_estimate_is_at_least_as_good_initially():
    _, _, cold = _run()
    sc, _, warm = _run(x0=GN_ESTIMATE)
    f_x0 = float(RangeErrorFitness(sc.anchors, sc.measured())(GN_ESTIMATE[None, :])[0])
    assert warm.best_history[0] <= f_x0 + 1e-12
    assert warm.best_history[0] <= cold.best_history[0]
    assert warm.error(sc.true_position) < 1.0
    assert warm.fitness_evaluations == N * (T + 1)


# --------------------------------------------------------------------------- diagnostics

def test_improvement_rate_definition():
    _, _, res = _run()
    f0, fT = res.best_history[0], res.best_history[-1]
    assert 0.0 < res.improvement_rate <= 1.0
    assert res.improvement_rate == pytest.approx((f0 - fT) / f0, rel=1e-12)
    assert res.improvement_rate > 0.9           # the swarm improves on the best random start by far


def test_diversity_history_definition_and_contraction():
    _, _, res = _run(diagnostics=True, record_positions=True)
    assert len(res.diversity_history) == res.iterations_run == T
    assert all(np.isfinite(d) and d >= 0.0 for d in res.diversity_history)
    assert res.diversity_history[-1] < res.diversity_history[0]
    for t in range(1, T + 1):                   # mean Euclidean distance to the swarm mean after iteration t
        pos = res.positions_history[t]
        expected = float(np.linalg.norm(pos - pos.mean(axis=0), axis=1).mean())
        assert res.diversity_history[t - 1] == pytest.approx(expected, rel=1e-12)
    _, _, plain = _run(diagnostics=True)        # diagnostics do not alter the optimisation
    assert np.array_equal(plain.estimate, res.estimate) and plain.diversity_history == res.diversity_history
    assert _run()[2].diversity_history == []    # off by default


def test_diversity_history_follows_iterations_when_stopped():
    _, _, res = _run(diagnostics=True, stopping=FireAt(5))
    assert res.iterations_run == 5 and len(res.diversity_history) == 5


# --------------------------------------------------------------------------- structure of the loop

def test_run_calls_module_centre_of_mass(monkeypatch):
    _, _, ref = _run()
    monkeypatch.setattr(amcmpso_module, "centre_of_mass", lambda x, f: np.asarray(x).mean(axis=0))
    _, _, patched = _run()
    assert patched.best_history != ref.best_history


def test_run_calls_module_coefficient_schedule(monkeypatch):
    _, _, ref = _run()
    const = {"w": np.full(T, 0.7), "c1": np.full(T, 1.4), "c2": np.full(T, 1.4)}
    monkeypatch.setattr(amcmpso_module, "coefficient_schedule", lambda *a, **k: const)
    _, _, patched = _run()
    assert patched.best_history != ref.best_history


def test_cosine_schedule_run_differs_but_converges():
    sc, _, lin = _run()
    _, _, cos = _run(schedule="cosine")
    assert cos.iterations_run == T and cos.fitness_evaluations == N * (T + 1)
    assert cos.best_history != lin.best_history
    assert cos.error(sc.true_position) < 1.5
    assert (cos.estimate >= sc.lower).all() and (cos.estimate <= sc.upper).all()


def test_clipping_keeps_particles_inside_field():
    sc, _, res = _run(seed=3, record_positions=True)
    ph = res.positions_history
    assert (ph >= sc.lower).all() and (ph <= sc.upper).all()


# --------------------------------------------------------------------------- fixture

def _computed_cases() -> dict:
    schedules = {name: {k: np.asarray(v, dtype=float).tolist() for k, v in coefficient_schedule(10, name).items()}
                 for name in ("linear", "cosine")}
    sc, _, res = _run()
    return {
        "description": "Values produced by this repository's AMCMPSO interpretation (slide-16 scenario, "
                       "N=20, T=60, seed=1) and its coefficient schedules for T=10; "
                       "not figures from the base paper.",
        "schedule_T10": schedules,
        "baseline_seed1": {
            "estimate": res.estimate.tolist(),
            "best_fitness": float(res.best_fitness),
            "error_m": float(res.error(sc.true_position)),
            "improvement_rate": float(res.improvement_rate),
            "fitness_evaluations": int(res.fitness_evaluations),
            "iterations_run": int(res.iterations_run),
        },
    }


def _ensure_fixture() -> dict:
    cases = _computed_cases()
    # refuse to persist neutral/placeholder values: the schedule and the run must be real
    assert cases["schedule_T10"]["linear"]["w"][0] == W_MAX
    assert cases["schedule_T10"]["cosine"]["w"][5] == pytest.approx((W_MAX + W_MIN) / 2, abs=1e-12)
    assert cases["baseline_seed1"]["improvement_rate"] > 0.0
    if not os.path.exists(FIXTURE):
        os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
        with open(FIXTURE, "w", encoding="utf-8") as fh:
            json.dump(cases, fh, indent=2)
            fh.write("\n")
    return cases


def test_fixture_written_by_helper():
    _ensure_fixture()
    assert os.path.exists(FIXTURE)
    with open(FIXTURE, encoding="utf-8") as fh:
        data = json.load(fh)
    assert {"schedule_T10", "baseline_seed1"} <= set(data)
    assert set(data["schedule_T10"]) == {"linear", "cosine"}
    for name in ("linear", "cosine"):
        assert set(data["schedule_T10"][name]) == {"w", "c1", "c2"}
        assert all(len(data["schedule_T10"][name][k]) == 10 for k in ("w", "c1", "c2"))
    assert len(data["baseline_seed1"]["estimate"]) == 3


def test_fixture_matches_fresh_computation():
    fresh = _ensure_fixture()
    with open(FIXTURE, encoding="utf-8") as fh:
        data = json.load(fh)
    for name in ("linear", "cosine"):
        for k in ("w", "c1", "c2"):
            assert np.allclose(data["schedule_T10"][name][k], fresh["schedule_T10"][name][k], rtol=0, atol=1e-9)
    stored, now = data["baseline_seed1"], fresh["baseline_seed1"]
    assert np.allclose(stored["estimate"], now["estimate"], rtol=0, atol=1e-9)
    for key in ("best_fitness", "error_m", "improvement_rate"):
        assert abs(stored[key] - now[key]) <= 1e-9
    assert stored["fitness_evaluations"] == now["fitness_evaluations"] == N * (T + 1)
    assert stored["iterations_run"] == now["iterations_run"] == T

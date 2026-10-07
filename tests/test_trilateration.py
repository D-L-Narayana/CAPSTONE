"""Tests for the closed-form estimators in ``pso3d.trilateration``.

Frozen behaviour (also pinned by tests/test_pso.py and results/baseline.json):
  * ``least_squares_trilateration`` on the slide-16 baseline -> (37.13, 11.44, 11.65), error 3.20 m
  * ``gauss_newton_refine`` from that start -> (37.42, 11.88, 9.36), error 0.96 m after 10 iterations
New behaviour: weighted rows, ``GNResult`` diagnostics, damping, Levenberg-Marquardt, warm start,
plus the fixture ``tests/fixtures/closed_form_cases.json`` (written once, then compared at 1e-9).
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from pso3d.centroid import centroid, weighted_centroid
from pso3d.config import Scenario
from pso3d.trilateration import (GNResult, gauss_newton, gauss_newton_refine, least_squares_trilateration,
                                 levenberg_marquardt, warm_start_estimate)

FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "closed_form_cases.json")

# "10 codes" document, code 2: six anchors with fixed per-anchor noise
P02_ANCHORS = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 4.0], [0.0, 60.0, 6.0], [60.0, 60.0, 9.0],
                        [30.0, 30.0, 20.0], [10.0, 45.0, 12.0]])
P02_TRUE = np.array([31.0, 17.5, 7.2])
P02_NOISE = np.array([0.5, -0.4, 0.6, -0.3, 0.4, -0.5])
# code 5: weighted centroid example (perfect ranges)
P05_ANCHORS = np.array([[0.0, 0.0, 2.0], [60.0, 0.0, 4.0], [0.0, 60.0, 3.0], [60.0, 60.0, 6.0], [30.0, 30.0, 18.0]])
P05_TRUE = np.array([16.0, 38.0, 5.0])

LOWER = np.zeros(3)
UPPER = np.array([60.0, 60.0, 20.0])


def ranges(anchors, true_p, noise=None) -> np.ndarray:
    d = np.linalg.norm(np.asarray(anchors, float) - np.asarray(true_p, float), axis=1)
    return d if noise is None else d + np.asarray(noise, float)


def residual_norm_at(anchors, measured, p) -> float:
    return float(np.linalg.norm(np.linalg.norm(np.asarray(p, float) - np.asarray(anchors, float), axis=1) - measured))


def inside_box(p) -> bool:
    return bool(np.isfinite(p).all() and (p >= LOWER).all() and (p <= UPPER).all())


# --------------------------------------------------------------------------- frozen behaviour
class TestFrozenBehaviour:
    def test_lsq_baseline_pin(self):
        sc = Scenario()
        p = least_squares_trilateration(sc.anchors, sc.measured())
        assert p.shape == (3,)
        assert np.allclose(np.round(p, 2), [37.13, 11.44, 11.65])
        assert round(float(np.linalg.norm(p - sc.true_position)), 2) == 3.20

    def test_gauss_newton_refine_baseline_pin(self):
        sc = Scenario()
        meas = sc.measured()
        p0 = least_squares_trilateration(sc.anchors, meas)
        p, used = gauss_newton_refine(sc.anchors, meas, p0)
        assert np.allclose(np.round(p, 2), [37.42, 11.88, 9.36])
        assert round(float(np.linalg.norm(p - sc.true_position)), 2) == 0.96
        assert used == 10

    def test_noise_free_ranges_are_solved_exactly_from_lists(self):
        sc = Scenario(noise=np.zeros(4))
        meas = sc.measured()
        p = least_squares_trilateration(sc.anchors.tolist(), meas.tolist())
        assert np.allclose(p, sc.true_position, atol=1e-6)
        p2, used = gauss_newton_refine(sc.anchors.tolist(), meas.tolist(), (p + 1.0).tolist())
        assert np.allclose(p2, sc.true_position, atol=1e-6)
        assert 1 <= used <= 10


# --------------------------------------------------------------------------- weighted least squares
class TestWeightedLeastSquares:
    def test_weights_none_is_identical_to_the_unweighted_call(self):
        sc = Scenario()
        meas = sc.measured()
        assert np.allclose(least_squares_trilateration(sc.anchors, meas),
                           least_squares_trilateration(sc.anchors, meas, weights=None), atol=1e-12, rtol=0)

    def test_unit_weights_equal_the_unweighted_solution(self):
        sc = Scenario()
        meas = sc.measured()
        p_plain = least_squares_trilateration(sc.anchors, meas)
        for w in (np.ones(4), [1, 1, 1, 1], np.full(4, 1.0)):
            assert np.allclose(least_squares_trilateration(sc.anchors, meas, weights=w), p_plain, atol=1e-9, rtol=0)

    def test_weighting_a_corrupt_anchor_near_zero_recovers_the_true_position(self):
        d = ranges(P02_ANCHORS, P02_TRUE)
        d[3] += 8.0                                       # one grossly biased range (not the reference anchor)
        err_plain = float(np.linalg.norm(least_squares_trilateration(P02_ANCHORS, d) - P02_TRUE))
        w = np.ones(6)
        w[3] = 1e-6
        err_weighted = float(np.linalg.norm(least_squares_trilateration(P02_ANCHORS, d, weights=w) - P02_TRUE))
        assert err_plain > 0.5
        assert err_weighted < 1e-3
        assert err_weighted < 1e-2 * err_plain

    def test_weights_are_scale_invariant(self):
        d = ranges(P02_ANCHORS, P02_TRUE, P02_NOISE)
        w = np.array([1.0, 0.5, 2.0, 1e-3, 1.0, 3.0])
        assert np.allclose(least_squares_trilateration(P02_ANCHORS, d, weights=w),
                           least_squares_trilateration(P02_ANCHORS, d, weights=7.0 * w), atol=1e-9, rtol=0)

    def test_zero_weight_removes_that_equation(self):
        d = ranges(P02_ANCHORS, P02_TRUE)
        d[4] += 5.0
        w = np.ones(6)
        w[4] = 0.0
        p = least_squares_trilateration(P02_ANCHORS, d, weights=w)
        assert np.allclose(p, P02_TRUE, atol=1e-6)

    def test_invalid_weights_are_rejected(self):
        d = ranges(P02_ANCHORS, P02_TRUE)
        with pytest.raises(ValueError):
            least_squares_trilateration(P02_ANCHORS, d, weights=np.ones(5))      # one weight per anchor
        with pytest.raises(ValueError):
            least_squares_trilateration(P02_ANCHORS, d, weights=[1, 1, -1, 1, 1, 1])


# --------------------------------------------------------------------------- Gauss-Newton with diagnostics
class TestGaussNewton:
    def test_matches_refine_on_the_baseline(self):
        sc = Scenario()
        meas = sc.measured()
        p0 = least_squares_trilateration(sc.anchors, meas)
        p_ref, used = gauss_newton_refine(sc.anchors, meas, p0)
        res = gauss_newton(sc.anchors, meas, p0)
        assert isinstance(res, GNResult)
        assert np.allclose(res.position, p_ref, atol=1e-9, rtol=0)
        assert res.iterations == used == 10
        assert res.converged is False                       # the 10-step budget ran out before the 1e-9 step size
        assert res.residual_norm == pytest.approx(residual_norm_at(sc.anchors, meas, res.position), abs=1e-12)
        assert res.evaluations == res.iterations + 1        # one residual/Jacobian per step + the final residual

    def test_converges_on_noise_free_data_within_six_iterations(self):
        sc = Scenario(noise=np.zeros(4))
        meas = sc.measured()
        res = gauss_newton(sc.anchors, meas, sc.true_position + 1.0)
        assert res.converged is True
        assert 1 <= res.iterations <= 6
        assert np.allclose(res.position, sc.true_position, atol=1e-6)
        assert res.residual_norm < 1e-6

    def test_zero_iterations_reports_the_start_point(self):
        sc = Scenario()
        meas = sc.measured()
        p0 = np.array([30.0, 30.0, 10.0])
        res = gauss_newton(sc.anchors, meas, p0, iterations=0)
        assert np.array_equal(res.position, p0)
        assert res.iterations == 0 and res.converged is False
        assert res.residual_norm == pytest.approx(residual_norm_at(sc.anchors, meas, p0), abs=1e-12)
        assert res.evaluations == 1

    def test_does_not_modify_the_start_array(self):
        sc = Scenario()
        p0 = np.array([30.0, 30.0, 10.0])
        gauss_newton(sc.anchors, sc.measured(), p0, iterations=3)
        assert np.array_equal(p0, [30.0, 30.0, 10.0])

    def test_damping_shrinks_the_step(self):
        sc = Scenario()
        meas = sc.measured()
        p0 = least_squares_trilateration(sc.anchors, meas)
        plain = gauss_newton(sc.anchors, meas, p0, iterations=1).position
        damped = gauss_newton(sc.anchors, meas, p0, iterations=1, damping=1.0).position
        assert np.linalg.norm(damped - p0) < np.linalg.norm(plain - p0)
        assert np.linalg.norm(damped - p0) > 0.0

    def test_damped_iteration_still_converges_on_noise_free_data(self):
        sc = Scenario(noise=np.zeros(4))
        meas = sc.measured()
        res = gauss_newton(sc.anchors, meas, sc.true_position + 1.0, iterations=30, damping=1e-3)
        assert res.converged is True
        assert np.allclose(res.position, sc.true_position, atol=1e-6)

    def test_negative_damping_is_rejected(self):
        sc = Scenario()
        with pytest.raises(ValueError):
            gauss_newton(sc.anchors, sc.measured(), np.zeros(3), damping=-1.0)


# --------------------------------------------------------------------------- Levenberg-Marquardt
class TestLevenbergMarquardt:
    def test_baseline_residual_not_above_start_and_not_above_gauss_newton(self):
        sc = Scenario()
        meas = sc.measured()
        p0 = least_squares_trilateration(sc.anchors, meas)
        start_residual = residual_norm_at(sc.anchors, meas, p0)
        lm = levenberg_marquardt(sc.anchors, meas, p0)
        gn = gauss_newton(sc.anchors, meas, p0)
        assert isinstance(lm, GNResult)
        assert lm.residual_norm <= start_residual
        assert lm.residual_norm <= gn.residual_norm + 1e-9
        assert lm.residual_norm == pytest.approx(residual_norm_at(sc.anchors, meas, lm.position), abs=1e-12)
        assert np.allclose(np.round(lm.position, 2), [37.42, 11.88, 9.36])
        assert lm.evaluations >= lm.iterations + 1

    def test_residual_norm_never_increases_across_iterations(self):
        sc = Scenario()
        meas = sc.measured()
        p0 = least_squares_trilateration(sc.anchors, meas)
        norms = [levenberg_marquardt(sc.anchors, meas, p0, iterations=k).residual_norm for k in range(0, 21)]
        assert norms[0] == pytest.approx(residual_norm_at(sc.anchors, meas, p0), abs=1e-12)
        assert all(later <= earlier for earlier, later in zip(norms, norms[1:]))
        assert norms[-1] < norms[0]

    def test_bad_starts_never_end_worse_than_the_start(self):
        sc = Scenario()
        meas = sc.measured()
        for start in ([0.0, 0.0, 0.0], [60.0, 60.0, 20.0], [60.0, 0.0, 20.0], [0.0, 60.0, 0.0],
                      [200.0, -100.0, 50.0], [-50.0, -50.0, -50.0]):
            p0 = np.array(start)
            res = levenberg_marquardt(sc.anchors, meas, p0)
            assert np.isfinite(res.position).all()
            assert res.residual_norm <= residual_norm_at(sc.anchors, meas, p0)

    def test_start_at_the_minimiser_is_returned_unchanged(self):
        sc = Scenario(noise=np.zeros(4))
        meas = sc.measured()
        res = levenberg_marquardt(sc.anchors, meas, sc.true_position)
        assert np.allclose(res.position, sc.true_position, atol=1e-9)
        assert res.residual_norm <= 1e-9
        assert res.iterations == 0
        assert res.converged is True

    def test_converges_on_noise_free_data(self):
        sc = Scenario(noise=np.zeros(4))
        meas = sc.measured()
        res = levenberg_marquardt(sc.anchors, meas, sc.true_position + 1.0)
        assert res.converged is True
        assert np.allclose(res.position, sc.true_position, atol=1e-6)
        assert res.residual_norm < 1e-6

    def test_invalid_initial_damping_is_rejected(self):
        sc = Scenario()
        with pytest.raises(ValueError):
            levenberg_marquardt(sc.anchors, sc.measured(), np.zeros(3), lam0=0.0)


# --------------------------------------------------------------------------- warm start
class TestWarmStart:
    def test_baseline_equals_lsq_plus_gauss_newton_inside_the_field(self):
        sc = Scenario()
        meas = sc.measured()
        expected, _ = gauss_newton_refine(sc.anchors, meas, least_squares_trilateration(sc.anchors, meas))
        ws = warm_start_estimate(sc.anchors, meas, sc.lower, sc.upper)
        assert ws.shape == (3,)
        assert np.allclose(ws, expected, atol=1e-9, rtol=0)
        assert inside_box(ws)

    def test_collinear_anchors_fall_back_without_exception(self):
        anchors = np.array([[0.0, 0.0, 0.0], [20.0, 0.0, 0.0], [40.0, 0.0, 0.0], [60.0, 0.0, 0.0]])
        d = ranges(anchors, [30.0, 10.0, 5.0])
        ws = warm_start_estimate(anchors, d, LOWER, UPPER)
        assert inside_box(ws)

    def test_coplanar_anchors_fall_back_without_exception(self):
        anchors = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [60.0, 60.0, 0.0]])
        d = ranges(anchors, [20.0, 40.0, 12.0], [0.2, -0.1, 0.3, -0.2])
        ws = warm_start_estimate(anchors, d, LOWER, UPPER)
        assert inside_box(ws)

    def test_near_coplanar_result_is_clipped_into_the_field(self):
        anchors = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [60.0, 60.0, 0.01]])
        d = ranges(anchors, [20.0, 40.0, 12.0], [0.2, -0.1, 0.3, -0.2])
        ws = warm_start_estimate(anchors, d, LOWER, UPPER)
        assert inside_box(ws)

    def test_non_finite_range_still_gives_a_finite_point_inside_the_field(self):
        sc = Scenario()
        meas = sc.measured()
        meas[2] = np.nan
        ws = warm_start_estimate(sc.anchors, meas, sc.lower, sc.upper)
        assert inside_box(ws)

    def test_non_finite_anchor_row_is_ignored(self):
        sc = Scenario()
        meas = sc.measured()
        anchors = np.vstack([sc.anchors, [[np.nan, 30.0, 10.0]]])    # a fifth anchor with an unknown position
        ws = warm_start_estimate(anchors, np.append(meas, 25.0), sc.lower, sc.upper)
        assert inside_box(ws)
        expected, _ = gauss_newton_refine(sc.anchors, meas, least_squares_trilateration(sc.anchors, meas))
        assert np.allclose(ws, expected, atol=1e-9, rtol=0)

    def test_accepts_python_lists_and_clips_to_the_given_box(self):
        sc = Scenario()
        meas = sc.measured()
        ws = warm_start_estimate(sc.anchors.tolist(), meas.tolist(), [0, 0, 0], [60, 60, 9])
        assert ws.shape == (3,)
        assert ws[2] == pytest.approx(9.0)                   # the GN estimate (z = 9.36) is clipped to the box


# --------------------------------------------------------------------------- "10 codes" examples
class TestCode2Example:
    def test_lsq_pin(self):
        d = ranges(P02_ANCHORS, P02_TRUE, P02_NOISE)
        p = least_squares_trilateration(P02_ANCHORS, d)
        assert np.allclose(np.round(p, 2), [31.5, 17.48, 7.63])
        assert round(float(np.linalg.norm(p - P02_TRUE)), 2) == 0.66

    def test_gauss_newton_does_not_worsen_the_range_residual(self):
        d = ranges(P02_ANCHORS, P02_TRUE, P02_NOISE)
        p = least_squares_trilateration(P02_ANCHORS, d)
        res = gauss_newton(P02_ANCHORS, d, p)
        assert res.residual_norm <= residual_norm_at(P02_ANCHORS, d, p)
        assert np.linalg.norm(res.position - P02_TRUE) < 1.0


# --------------------------------------------------------------------------- fixture
# Levenberg-Marquardt's last accepted steps are at the 1e-9 level, so its final iterate (unlike its
# residual norm, stable to ~1e-13) is only reproducible to ~1e-7 across linear-algebra builds, and
# its accept/reject counters are not reproducible at all; the fixture therefore stores no LM
# counters and compares the LM position at 1e-6. Everything else is compared at 1e-9.
FIXTURE_TOLERANCES = {"/baseline/lm/position": 1e-6}


def _gn_dict(res: GNResult) -> dict:
    return {"position": [float(v) for v in res.position], "iterations": int(res.iterations),
            "converged": bool(res.converged), "residual_norm": float(res.residual_norm),
            "evaluations": int(res.evaluations)}


def _lm_dict(res: GNResult) -> dict:
    return {"position": [float(v) for v in res.position], "converged": bool(res.converged),
            "residual_norm": float(res.residual_norm)}


def compute_cases() -> dict:
    sc = Scenario()
    meas = sc.measured()
    lsq = least_squares_trilateration(sc.anchors, meas)
    d02 = ranges(P02_ANCHORS, P02_TRUE, P02_NOISE)
    lsq02 = least_squares_trilateration(P02_ANCHORS, d02)
    d05 = ranges(P05_ANCHORS, P05_TRUE)
    return {
        "baseline": {
            "anchors": sc.anchors.tolist(), "true_position": sc.true_position.tolist(), "noise": sc.noise.tolist(),
            "measured": meas.tolist(), "lsq": lsq.tolist(),
            "gn": _gn_dict(gauss_newton(sc.anchors, meas, lsq)),
            "lm": _lm_dict(levenberg_marquardt(sc.anchors, meas, lsq)),
            "warm_start": warm_start_estimate(sc.anchors, meas, sc.lower, sc.upper).tolist(),
        },
        "code2": {
            "anchors": P02_ANCHORS.tolist(), "true_position": P02_TRUE.tolist(), "noise": P02_NOISE.tolist(),
            "measured": d02.tolist(), "lsq": lsq02.tolist(),
            "gn": _gn_dict(gauss_newton(P02_ANCHORS, d02, lsq02)),
            "warm_start": warm_start_estimate(P02_ANCHORS, d02, LOWER, UPPER).tolist(),
        },
        "code5": {
            "anchors": P05_ANCHORS.tolist(), "true_position": P05_TRUE.tolist(), "measured": d05.tolist(),
            "centroid": centroid(P05_ANCHORS).tolist(),
            "weighted_centroid": weighted_centroid(P05_ANCHORS, d05).tolist(),
            "weighted_centroid_power2": weighted_centroid(P05_ANCHORS, d05, power=2.0).tolist(),
        },
    }


def _assert_same(expected, actual, path="") -> None:
    tol = FIXTURE_TOLERANCES.get(path, 1e-9)
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and set(expected) == set(actual), path
        for key in expected:
            _assert_same(expected[key], actual[key], f"{path}/{key}")
    elif isinstance(expected, bool):
        assert actual is expected, path
    elif isinstance(expected, int):
        assert actual == expected, path
    elif isinstance(expected, float):
        assert abs(actual - expected) <= tol, path
    else:
        assert np.allclose(np.asarray(expected, float), np.asarray(actual, float), atol=tol, rtol=0), path


def test_fixture_file_round_trips_the_pinned_cases():
    cases = compute_cases()
    # only a validated computation may (re)generate the fixture
    assert np.allclose(np.round(cases["baseline"]["lsq"], 2), [37.13, 11.44, 11.65])
    assert np.allclose(np.round(cases["baseline"]["gn"]["position"], 2), [37.42, 11.88, 9.36])
    assert np.allclose(np.round(cases["code2"]["lsq"], 2), [31.5, 17.48, 7.63])
    assert np.allclose(np.round(cases["code5"]["weighted_centroid"], 2), [25.23, 33.19, 8.31])
    if not os.path.exists(FIXTURE_PATH):
        os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
        payload = {"version": 1,
                   "description": "Closed-form reference outputs (least squares, Gauss-Newton, Levenberg-Marquardt, "
                                  "warm start, centroids) for the slide-16 baseline and the code-2/code-5 examples of "
                                  "the 10-codes document. tests/test_trilateration.py writes this file when it is "
                                  "absent and otherwise compares against it at 1e-9.",
                   "cases": cases}
        with open(FIXTURE_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
    with open(FIXTURE_PATH, encoding="utf-8") as fh:
        stored = json.load(fh)
    assert stored["version"] == 1
    assert set(stored["cases"]) == {"baseline", "code2", "code5"}
    _assert_same(stored["cases"], cases)

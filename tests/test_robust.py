"""Behavioural tests for ``pso3d.robust``: Huber and weighted range-error fitness functions.

Run with ``python3 -m pytest -q tests/test_robust.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pso3d.config import Scenario
from pso3d.fitness import RangeErrorFitness
from pso3d.pso import SimplifiedPSO
from pso3d.robust import HuberRangeFitness, WeightedRangeFitness, huber_loss

ROBUST_CLASSES = {
    "huber": lambda anchors, measured: HuberRangeFitness(anchors, measured, delta=1.0),
    "weighted": lambda anchors, measured: WeightedRangeFitness(anchors, measured, np.ones(len(measured))),
}


def baseline():
    sc = Scenario()
    return sc, sc.anchors, sc.measured()


def random_points(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform([0.0, 0.0, 0.0], [60.0, 60.0, 20.0], size=(n, 3))


def measured_with_residual(anchors, p, r: float) -> np.ndarray:
    """Ranges that give residual exactly ``r`` on anchor 0 and 0 elsewhere when evaluated at ``p``."""
    d = np.linalg.norm(anchors - p, axis=1)
    out = d.copy()
    out[0] = d[0] - r
    return out


# --------------------------------------------------------------------------- huber_loss


def test_huber_loss_matches_its_definition_elementwise():
    delta = 1.5
    r = np.linspace(-5.0, 5.0, 101)
    expected = np.where(np.abs(r) <= delta, r ** 2, 2.0 * delta * np.abs(r) - delta ** 2)
    assert np.allclose(huber_loss(r, delta), expected, atol=1e-12)
    assert np.allclose(huber_loss(r, delta), huber_loss(-r, delta))
    assert huber_loss(0.0, delta) == 0.0
    assert huber_loss(delta, delta) == pytest.approx(delta ** 2)


def test_huber_loss_is_continuous_with_continuous_gradient_at_delta():
    delta = 0.8
    eps = 1e-9
    assert abs(huber_loss(delta + eps, delta) - huber_loss(delta - eps, delta)) < 1e-6
    h = 1e-6

    def slope(x):
        return (huber_loss(x + h, delta) - huber_loss(x - h, delta)) / (2.0 * h)

    assert abs(slope(delta + 10 * h) - slope(delta - 10 * h)) < 1e-4
    assert slope(delta + 10 * h) == pytest.approx(2.0 * delta, abs=1e-6)
    assert slope(-delta - 10 * h) == pytest.approx(-2.0 * delta, abs=1e-6)


# --------------------------------------------------------------------------- HuberRangeFitness


def test_huber_equals_squared_loss_when_all_residuals_are_within_delta():
    sc, anchors, measured = baseline()
    squared = RangeErrorFitness(anchors, measured)
    huber = HuberRangeFitness(anchors, measured, delta=1.0)
    # at the true position the residuals are the baseline offsets (|n_j| <= 0.5 < delta)
    assert huber(sc.true_position) == pytest.approx(squared(sc.true_position), abs=1e-12)
    # with a huge delta the two functions coincide everywhere
    wide = HuberRangeFitness(anchors, measured, delta=1e6)
    pts = random_points(200, seed=1)
    assert np.allclose(wide(pts), RangeErrorFitness(anchors, measured)(pts), rtol=1e-12, atol=1e-9)


def test_huber_grows_linearly_beyond_delta():
    delta = 1.0
    sc, anchors, _ = baseline()
    p = sc.true_position

    def loss_at(r):
        return HuberRangeFitness(anchors, measured_with_residual(anchors, p, r), delta=delta)(p)

    assert loss_at(delta) == pytest.approx(delta ** 2, abs=1e-9)
    assert loss_at(2 * delta) - loss_at(delta) == pytest.approx(2 * delta ** 2, abs=1e-9)
    assert loss_at(3 * delta) - loss_at(2 * delta) == pytest.approx(2 * delta ** 2, abs=1e-9)
    assert loss_at(-2 * delta) == pytest.approx(loss_at(2 * delta), abs=1e-9)
    # squared loss would grow quadratically instead
    assert loss_at(10 * delta) == pytest.approx(2 * delta * 10 * delta - delta ** 2, abs=1e-9)


def test_huber_never_exceeds_the_squared_loss():
    _, anchors, measured = baseline()
    pts = random_points(300, seed=2)
    f_sq = RangeErrorFitness(anchors, measured)(pts)
    f_hu = HuberRangeFitness(anchors, measured, delta=1.0)(pts)
    assert np.all(f_hu <= f_sq + 1e-9)
    assert np.any(f_hu < f_sq - 1.0)  # far from the solution the residuals exceed delta


def test_huber_argmin_is_less_affected_by_an_outlier_than_the_squared_loss():
    # Eight anchors on the corners of a cube centred on the node (good geometry), exact ranges and
    # one NLOS-like +10 m outlier.  The squared loss averages the outlier into the estimate
    # (bias ~ 10 / (8/3) m along the outlier direction); the Huber loss caps the outlier's pull at
    # 2*delta per metre, so the bias shrinks to ~0.6*delta = 0.3 m (grid step 0.25 m).
    p = np.array([37.0, 12.0, 8.5])
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    anchors = p + 20.0 * corners
    measured = np.linalg.norm(anchors - p, axis=1)
    measured[7] += 10.0
    offsets = np.arange(-4.0, 4.0 + 1e-9, 0.25)
    gx, gy, gz = np.meshgrid(offsets, offsets, offsets, indexing="ij")
    grid = p + np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    err_sq = np.linalg.norm(grid[np.argmin(RangeErrorFitness(anchors, measured)(grid))] - p)
    err_hu = np.linalg.norm(grid[np.argmin(HuberRangeFitness(anchors, measured, delta=0.5)(grid))] - p)
    assert err_sq > 2.0
    assert err_hu < 0.75
    assert err_hu < err_sq / 3.0


def test_huber_delta_default_and_validation():
    _, anchors, measured = baseline()
    assert HuberRangeFitness(anchors, measured).delta == 1.0
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            HuberRangeFitness(anchors, measured, delta=bad)


# --------------------------------------------------------------------------- WeightedRangeFitness


def test_weighted_with_unit_weights_equals_range_error_fitness():
    sc, anchors, measured = baseline()
    weighted = WeightedRangeFitness(anchors, measured, np.ones(4))
    squared = RangeErrorFitness(anchors, measured)
    pts = random_points(100, seed=3)
    assert np.allclose(weighted(pts), squared(pts), rtol=1e-12, atol=1e-12)
    assert weighted(sc.true_position) == pytest.approx(squared(sc.true_position), abs=1e-12)
    assert weighted([37.0, 12.0, 8.5]) == pytest.approx(squared(np.array([37.0, 12.0, 8.5])), abs=1e-12)


def test_weighted_formula_and_weight_validation():
    sc, anchors, measured = baseline()
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    fit = WeightedRangeFitness(anchors, measured, weights)
    pts = random_points(50, seed=4)
    r = np.linalg.norm(pts[:, None, :] - anchors[None, :, :], axis=2) - measured
    assert np.allclose(fit(pts), (weights * r ** 2).sum(axis=1), rtol=1e-12, atol=1e-9)
    assert fit(sc.true_position) == pytest.approx(float((weights * sc.noise ** 2).sum()), abs=1e-12)
    with pytest.raises(ValueError):
        WeightedRangeFitness(anchors, measured, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        WeightedRangeFitness(anchors, measured, [1.0, -2.0, 3.0, 4.0])


def test_weighted_from_sigmas_uses_inverse_variance_weights():
    _, anchors, measured = baseline()
    sigmas = np.array([0.5, 1.0, 2.0, 0.25])
    fit = WeightedRangeFitness.from_sigmas(anchors, measured, sigmas)
    assert isinstance(fit, WeightedRangeFitness)
    assert np.allclose(fit.weights, 1.0 / sigmas ** 2)
    with pytest.raises(ValueError):
        WeightedRangeFitness.from_sigmas(anchors, measured, [0.5, 0.0, 1.0, 1.0])


# --------------------------------------------------------------------------- shared behaviour


@pytest.mark.parametrize("key", list(ROBUST_CLASSES))
def test_counters_increment_for_single_and_batch_inputs(key):
    _, anchors, measured = baseline()
    fit = ROBUST_CLASSES[key](anchors, measured)
    assert isinstance(fit, RangeErrorFitness)
    assert (fit.evaluations, fit.distance_computations) == (0, 0)
    fit(np.array([37.0, 12.0, 8.5]))
    assert (fit.evaluations, fit.distance_computations) == (1, 4)
    fit(random_points(20, seed=5))
    assert (fit.evaluations, fit.distance_computations) == (21, 84)
    fit([37.0, 12.0, 8.5])
    assert (fit.evaluations, fit.distance_computations) == (22, 88)
    fit.reset()
    assert (fit.evaluations, fit.distance_computations) == (0, 0)


@pytest.mark.parametrize("key", list(ROBUST_CLASSES))
def test_accepts_python_lists_and_returns_float_or_vector(key):
    _, anchors, measured = baseline()
    fit = ROBUST_CLASSES[key](anchors.tolist(), measured.tolist())
    single = fit([37.0, 12.0, 8.5])
    assert isinstance(single, float)
    assert single > 0.0
    batch = fit([[37.0, 12.0, 8.5], [10.0, 10.0, 5.0], [50.0, 50.0, 15.0]])
    assert isinstance(batch, np.ndarray)
    assert batch.shape == (3,) and batch.dtype == np.float64
    assert batch[0] == pytest.approx(single, abs=1e-12)
    assert batch[1] > single and batch[2] > single
    with pytest.raises(ValueError):
        fit(np.ones((5, 2)))


@pytest.mark.parametrize("key", list(ROBUST_CLASSES))
def test_evaluate_one_matches_call_and_counts_one_evaluation(key):
    _, anchors, measured = baseline()
    fit = ROBUST_CLASSES[key](anchors, measured)
    value = fit.evaluate_one([37.0, 12.0, 8.5])
    assert isinstance(value, float)
    assert (fit.evaluations, fit.distance_computations) == (1, 4)
    assert value == pytest.approx(fit(np.array([37.0, 12.0, 8.5])), abs=1e-12)


# --------------------------------------------------------------------------- optimiser drop-in


def _baseline_run(fit, seed=1):
    sc = Scenario()
    return SimplifiedPSO(20, 60, w=0.7, c=1.4, seed=seed, clip=False).run(fit, sc.lower, sc.upper)


def test_robust_classes_reproduce_the_baseline_run_when_they_reduce_to_the_squared_loss():
    # With a huge delta (all residuals in the quadratic branch) or unit weights the per-particle loss
    # values equal RangeErrorFitness, so the frozen Review-1 run (seed 1) must follow the same trajectory.
    sc, anchors, measured = baseline()
    ref = _baseline_run(RangeErrorFitness(anchors, measured))
    assert np.allclose(np.round(ref.estimate, 2), [37.43, 11.90, 9.29])
    for fit in (HuberRangeFitness(anchors, measured, delta=1e6),
                WeightedRangeFitness(anchors, measured, np.ones(4))):
        res = _baseline_run(fit)
        assert np.allclose(res.estimate, ref.estimate, rtol=0.0, atol=1e-9)
        assert res.best_fitness == pytest.approx(ref.best_fitness, abs=1e-9)
        assert res.iterations_run == ref.iterations_run == 60
        assert (res.fitness_evaluations, res.distance_computations) == (1200, 4800)
        assert (fit.evaluations, fit.distance_computations) == (1200, 4800)
    # a realistic delta leaves the well-behaved baseline ranges (|n_j| <= 0.5 m) essentially unchanged
    res = _baseline_run(HuberRangeFitness(anchors, measured, delta=1.0))
    assert res.error(sc.true_position) < 1.0
    assert res.fitness_evaluations == 1200


def test_huber_reduces_the_pso_error_under_a_single_nlos_outlier():
    # Six anchors with exact ranges except one +8 m NLOS range. For every seed the Huber-driven run
    # ends closer to the true node than the squared-loss run (observed ~5.5 m versus ~9.3 m).
    sc, anchors4, _ = baseline()
    anchors = np.vstack([anchors4, [[30.0, 30.0, 10.0], [10.0, 50.0, 15.0]]])
    p = sc.true_position
    measured = np.linalg.norm(anchors - p, axis=1)
    measured[5] += 8.0
    pairs = []
    for seed in range(1, 6):
        err_sq = _baseline_run(RangeErrorFitness(anchors, measured), seed).error(p)
        err_hu = _baseline_run(HuberRangeFitness(anchors, measured, delta=1.0), seed).error(p)
        assert err_hu < err_sq
        pairs.append((err_sq, err_hu))
    mean_sq, mean_hu = np.mean(pairs, axis=0)
    assert mean_hu < 0.8 * mean_sq

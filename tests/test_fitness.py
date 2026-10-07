"""RangeErrorFitness: input handling (lists, tuples, arrays), return shapes, counters, evaluate_one, reset."""
from __future__ import annotations

import numpy as np
import pytest

from pso3d import RangeErrorFitness, Scenario


def make():
    sc = Scenario()
    return sc, RangeErrorFitness(sc.anchors, sc.measured())


def test_python_list_point_returns_float():
    sc, fit = make()
    f = fit([37.0, 12.0, 8.5])
    assert isinstance(f, float)
    # at the true position the residuals are exactly the noise: f = sum(n_j^2) = 0.16 + 0.09 + 0.25 + 0.16
    assert f == pytest.approx(float((sc.noise ** 2).sum()))
    assert f == pytest.approx(0.66)


def test_tuple_point_returns_float():
    sc, fit = make()
    f = fit((37.0, 12.0, 8.5))
    assert isinstance(f, float) and f == pytest.approx(0.66)


def test_list_of_points_returns_vector():
    sc, fit = make()
    pts = [[37.0, 12.0, 8.5], [0.0, 0.0, 0.0], [60.0, 60.0, 20.0]]
    f = fit(pts)
    assert isinstance(f, np.ndarray) and f.shape == (3,)
    assert f[0] == pytest.approx(0.66)
    assert (f[1:] > f[0]).all()


def test_numpy_shapes():
    sc, fit = make()
    single = fit(np.array([37.0, 12.0, 8.5]))
    assert isinstance(single, float)
    batch = fit(np.zeros((5, 3)))
    assert batch.shape == (5,)
    assert np.array_equal(batch, np.full(5, batch[0]))


def test_batch_equals_individual_evaluations_exactly():
    sc, fit = make()
    rng = np.random.default_rng(0)
    pts = rng.uniform(sc.lower, sc.upper, (7, 3))
    batch = fit(pts)
    for i in range(7):
        assert fit(pts[i]) == batch[i]
        assert fit(pts[i].tolist()) == batch[i]


def test_counters_count_points_and_distances():
    sc, fit = make()
    assert fit.evaluations == 0 and fit.distance_computations == 0
    fit(np.zeros((5, 3)))
    assert fit.evaluations == 5 and fit.distance_computations == 5 * fit.n_anchors
    fit([1.0, 2.0, 3.0])
    assert fit.evaluations == 6 and fit.distance_computations == 6 * fit.n_anchors
    fit([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert fit.evaluations == 8 and fit.distance_computations == 8 * 4


def test_evaluate_one_counts_a_single_evaluation():
    sc, fit = make()
    f = fit.evaluate_one([37.0, 12.0, 8.5])
    assert isinstance(f, float) and f == pytest.approx(0.66)
    assert fit.evaluations == 1 and fit.distance_computations == fit.n_anchors
    assert fit.evaluate_one(np.array([1.0, 2.0, 3.0])) == fit(np.array([1.0, 2.0, 3.0]))
    assert fit.evaluations == 3


def test_reset_and_attributes():
    sc, fit = make()
    assert fit.n_anchors == 4
    assert np.array_equal(fit.anchors, sc.anchors) and np.array_equal(fit.measured, sc.measured())
    assert fit.anchors.dtype == float and fit.measured.dtype == float
    fit(np.zeros((3, 3)))
    fit.reset()
    assert fit.evaluations == 0 and fit.distance_computations == 0


def test_noise_free_true_position_scores_zero():
    sc = Scenario(noise=np.zeros(4))
    fit = RangeErrorFitness(sc.anchors, sc.measured())
    assert fit(sc.true_position.tolist()) == pytest.approx(0.0, abs=1e-20)
    assert fit.evaluate_one(sc.true_position) == pytest.approx(0.0, abs=1e-20)

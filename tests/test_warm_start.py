"""Warm start (``x0``) for StandardPSO and SimplifiedPSO.

Contract: after the uniform initial draw, particle 0 is replaced by clip(x0, lower, upper) with zero velocity;
no random numbers are consumed, so particles 1..N-1 are identical to the cold run at t = 0.
"""
from __future__ import annotations

import numpy as np
import pytest

from pso3d import RangeErrorFitness, Scenario, SimplifiedPSO, StandardPSO

X0_GN = [37.42, 11.88, 9.36]        # least squares + Gauss-Newton estimate of the baseline scenario (README table)


def scenario():
    sc = Scenario()
    return sc, RangeErrorFitness(sc.anchors, sc.measured())


def test_standard_pso_warm_start_flags_and_accuracy():
    sc, fit = scenario()
    res = StandardPSO(20, 20, seed=1, x0=X0_GN).run(fit, sc.lower, sc.upper)
    assert res.warm_start is True
    assert res.method == "standard"
    assert res.stopped_by is None
    assert res.error(sc.true_position) <= 0.97
    assert res.iterations_run == 20 and res.fitness_evaluations == 20 * 21
    # gbest starts at x0 and is monotone: the result can only be as good as x0 or better
    f_x0 = RangeErrorFitness(sc.anchors, sc.measured()).evaluate_one(X0_GN)
    assert res.best_fitness <= f_x0 + 1e-15
    assert res.best_history[0] == pytest.approx(f_x0)


def test_standard_pso_warm_start_replaces_only_particle_zero_at_t0():
    sc, fit_cold = scenario()
    cold = StandardPSO(20, 20, seed=1, record_positions=True).run(fit_cold, sc.lower, sc.upper)
    sc, fit_warm = scenario()
    warm = StandardPSO(20, 20, seed=1, x0=X0_GN, record_positions=True).run(fit_warm, sc.lower, sc.upper)
    assert cold.warm_start is False and warm.warm_start is True
    assert cold.positions_history.shape == warm.positions_history.shape == (21, 20, 3)
    assert np.array_equal(warm.positions_history[0][1:], cold.positions_history[0][1:])
    assert np.allclose(warm.positions_history[0][0], X0_GN)
    assert not np.allclose(cold.positions_history[0][0], X0_GN)
    assert warm.fitness_evaluations == cold.fitness_evaluations


def test_x0_outside_the_field_is_clipped():
    sc, fit = scenario()
    res = StandardPSO(20, 3, seed=1, x0=[-5.0, 70.0, 25.0], record_positions=True).run(fit, sc.lower, sc.upper)
    assert np.array_equal(res.positions_history[0][0], [0.0, 60.0, 20.0])
    sc, fit = scenario()
    res = SimplifiedPSO(20, 3, seed=1, x0=[-5.0, 70.0, 25.0], record_positions=True).run(fit, sc.lower, sc.upper)
    assert np.array_equal(res.positions_history[0][0], [0.0, 60.0, 20.0])
    assert res.warm_start is True


def test_simplified_pso_supports_x0():
    sc, fit_cold = scenario()
    cold = SimplifiedPSO(20, 60, seed=1, clip=False, record_positions=True).run(fit_cold, sc.lower, sc.upper)
    sc, fit_warm = scenario()
    warm = SimplifiedPSO(20, 60, seed=1, clip=False, x0=X0_GN, record_positions=True).run(fit_warm, sc.lower, sc.upper)
    assert warm.warm_start is True and warm.method == "simplified"
    assert np.allclose(warm.positions_history[0][0], X0_GN)
    assert np.array_equal(warm.positions_history[0][1:], cold.positions_history[0][1:])
    assert warm.fitness_evaluations == 1200 and warm.iterations_run == 60
    assert warm.error(sc.true_position) <= 0.97


def test_warm_start_with_the_cold_runs_own_particle_is_bit_identical():
    # proves that x0 is injected after the initial draw and that no random numbers are consumed
    sc, fit_cold = scenario()
    cold = SimplifiedPSO(20, 60, seed=1, clip=False, record_positions=True).run(fit_cold, sc.lower, sc.upper)
    sc, fit_warm = scenario()
    warm = SimplifiedPSO(20, 60, seed=1, clip=False, x0=cold.positions_history[0][0],
                         record_positions=True).run(fit_warm, sc.lower, sc.upper)
    assert warm.warm_start is True
    assert np.array_equal(warm.estimate, cold.estimate)
    assert np.array_equal(warm.positions_history, cold.positions_history)
    assert warm.best_history == cold.best_history


def test_review1_baseline_unchanged_when_x0_is_none():
    sc, fit = scenario()
    res = SimplifiedPSO(20, 60, w=0.7, c=1.4, seed=1, clip=False, x0=None).run(fit, sc.lower, sc.upper)
    assert np.allclose(np.round(res.estimate, 2), [37.43, 11.90, 9.29])
    assert round(res.error(sc.true_position), 2) == 0.90
    assert res.fitness_evaluations == 1200 and res.distance_computations == 4800
    assert res.swarm_state_floats == 120
    assert res.warm_start is False and res.stopped_by is None and res.method == "simplified"


def test_x0_with_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        StandardPSO(20, 5, seed=1, x0=[1.0, 2.0])
    with pytest.raises(ValueError):
        SimplifiedPSO(20, 5, seed=1, x0=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

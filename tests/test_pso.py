import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pso3d import Scenario, RangeErrorFitness, SimplifiedPSO, least_squares_trilateration, gauss_newton_refine


def test_baseline_reproduces_review1_numbers():
    sc = Scenario()
    fit = RangeErrorFitness(sc.anchors, sc.measured())
    res = SimplifiedPSO(20, 60, w=0.7, c=1.4, seed=1, clip=False).run(fit, sc.lower, sc.upper)
    assert np.allclose(np.round(res.estimate, 2), [37.43, 11.90, 9.29])
    assert round(res.error(sc.true_position), 2) == 0.90
    assert res.fitness_evaluations == 1200 and res.distance_computations == 4800
    assert res.swarm_state_floats == 120


def test_noise_free_ranges_give_exact_solution():
    sc = Scenario(noise=np.zeros(4))
    meas = sc.measured()
    p = least_squares_trilateration(sc.anchors, meas)
    assert np.allclose(p, sc.true_position, atol=1e-6)
    p2, _ = gauss_newton_refine(sc.anchors, meas, p + 1.0)
    assert np.allclose(p2, sc.true_position, atol=1e-6)


def test_clipping_keeps_particles_inside_field():
    sc = Scenario()
    fit = RangeErrorFitness(sc.anchors, sc.measured())
    res = SimplifiedPSO(20, 30, seed=3, clip=True, record_positions=True).run(fit, sc.lower, sc.upper)
    ph = res.positions_history
    assert (ph >= 0).all() and (ph <= sc.upper).all()

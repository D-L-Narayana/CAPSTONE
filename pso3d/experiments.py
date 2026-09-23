"""Monte-Carlo / parameter study helpers.

Every study localizes many random nodes (uniform in the field) with Gaussian
ranging noise of a given standard deviation and reports RMSE, mean error,
fitness evaluations, run time and swarm memory.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import Scenario
from .fitness import RangeErrorFitness
from .pso import SimplifiedPSO
from .standard_pso import StandardPSO
from .trilateration import least_squares_trilateration, gauss_newton_refine


@dataclass
class StudyRow:
    label: str
    value: float
    method: str
    rmse: float
    mean_error: float
    p90_error: float
    evaluations: float
    runtime_ms: float
    memory_floats: int
    trials: int


def random_anchors(rng: np.random.Generator, n: int, upper: np.ndarray, base: Scenario) -> np.ndarray:
    """First four anchors are the fixed non-coplanar corners of the default scenario;
    extra anchors are drawn uniformly in the field."""
    if n <= 4:
        return base.anchors[:n]
    extra = rng.uniform(np.zeros(3), upper, (n - 4, 3))
    return np.vstack([base.anchors, extra])


def monte_carlo(method: str, trials: int, noise_sigma: float, n_anchors: int,
                n_particles: int, n_iterations: int, seed: int = 123,
                base: Scenario | None = None) -> dict:
    base = base or Scenario()
    rng = np.random.default_rng(seed)
    upper = base.upper
    errors, evals, times = [], [], []
    mem = 0
    for k in range(trials):
        anchors = random_anchors(rng, n_anchors, upper, base)
        true_p = rng.uniform(np.zeros(3), upper)
        noise = rng.normal(0.0, noise_sigma, size=n_anchors)
        meas = np.linalg.norm(anchors - true_p, axis=1) + noise
        fit = RangeErrorFitness(anchors, meas)
        if method == "pso":
            pso = SimplifiedPSO(n_particles, n_iterations, seed=int(rng.integers(0, 2**31 - 1)), clip=True)
            res = pso.run(fit, np.zeros(3), upper)
            est, ev, rt, mem = res.estimate, res.fitness_evaluations, res.runtime_s, res.swarm_state_floats
        elif method == "std":
            pso = StandardPSO(n_particles, n_iterations, seed=int(rng.integers(0, 2**31 - 1)), clip=True)
            res = pso.run(fit, np.zeros(3), upper)
            est, ev, rt, mem = res.estimate, res.fitness_evaluations, res.runtime_s, res.swarm_state_floats
        elif method in ("lsq", "lsq_gn"):
            import time
            t0 = time.perf_counter()
            est = least_squares_trilateration(anchors, meas)
            ev = 0
            if method == "lsq_gn":
                est, gn_it = gauss_newton_refine(anchors, meas, est)
                ev = gn_it                       # one residual/Jacobian evaluation per GN step
            rt = time.perf_counter() - t0
            mem = 3 * n_anchors + 3              # anchors + estimate
        else:
            raise ValueError(method)
        errors.append(float(np.linalg.norm(est - true_p)))
        evals.append(ev); times.append(rt)
    e = np.asarray(errors)
    return dict(rmse=float(np.sqrt((e ** 2).mean())), mean_error=float(e.mean()),
                p90_error=float(np.percentile(e, 90)), evaluations=float(np.mean(evals)),
                runtime_ms=float(np.mean(times) * 1e3), memory_floats=int(mem), trials=trials,
                errors=e)


def sweep(label: str, values, trials: int = 200, **fixed) -> list[StudyRow]:
    rows: list[StudyRow] = []
    for v in values:
        kw = dict(fixed); kw[label] = v
        for method in ("pso", "std", "lsq", "lsq_gn"):
            if method.startswith("lsq") and label in ("n_particles", "n_iterations"):
                continue
            r = monte_carlo(method, trials, kw["noise_sigma"], kw["n_anchors"],
                            kw["n_particles"], kw["n_iterations"])
            rows.append(StudyRow(label, float(v), method, r["rmse"], r["mean_error"], r["p90_error"],
                                 r["evaluations"], r["runtime_ms"], r["memory_floats"], trials))
    return rows

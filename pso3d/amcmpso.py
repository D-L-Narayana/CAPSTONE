"""AMCMPSO - Adaptive Mean Centre-of-Mass PSO, an *interpretation* of the idea behind
Alhasan et al. (2023), J. King Saud Univ. - Comput. Inf. Sci., 35(9), 101782,
https://doi.org/10.1016/j.jksuci.2023.101782.

Status (please keep this paragraph with the code)
--------------------------------------------------
This module is an interpretation of the AMCMPSO idea -- adaptive coefficients plus
swarm-mean and centre-of-mass guidance -- built from the paper's abstract and the team's
literature summary.  The exact velocity equation, the adaptation rules and the constants
of the paper have not been transcribed (the paper text is not in this repository) and the
implementation has not been validated against the paper's reported results.  Therefore no
number produced by this class may be quoted as a reproduction of the paper.  The abstract
reports an improvement rate of 99.86 %, an error below 1.34 cm and 3D coverage above 87 %
for the paper's own setting; nothing here reproduces, confirms or contradicts those figures,
and the ``improvement_rate`` diagnostic below is *this* implementation's definition, not the
paper's metric.

What is implemented (one iteration t = 0 .. T-1; details in docs/amcmpso.md)
----------------------------------------------------------------------------
    w(t), c1(t), c2(t)  = coefficient_schedule(T, schedule, ...)[...][t]
    mean = x.mean(axis=0)                       swarm mean position
    com  = centre_of_mass(x, f)                 inverse-fitness weighted centre of mass
    v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x) + c_mean r3 (mean - x) + c_com r4 (com - x)
    x <- x + v,  then clip to [lower, upper] when clip=True
    evaluate f(x); update personal bests and the global best (synchronous)

r1..r4 are N x 3 uniform draws taken in this order from ``numpy.random.default_rng(seed)``
after the N x 3 uniform start positions.  The random stream is identical to the Phase-1
version of this file, so seeded runs reproduce the Phase-1 numbers exactly.

Phase-2 options: ``schedule`` ('linear' | 'cosine'), ``stopping`` (duck-typed rule with
``reset()``, ``update(iteration_1_based, best_fitness, evaluations) -> bool`` and ``name``),
``x0`` warm start (particle 0 is replaced by clip(x0); no random numbers are consumed) and
``diagnostics`` (``diversity_history``), all reported through ``AMCMPSOResult``.
"""
from __future__ import annotations

import dataclasses
import time
import numpy as np

from .fitness import RangeErrorFitness
from .pso import PSOResult

SCHEDULES = ("linear", "cosine")
_COM_EPSILON = 1e-9


def coefficient_schedule(T: int, schedule: str = "linear", w_max: float = 0.9, w_min: float = 0.4,
                         c1_start: float = 2.0, c1_end: float = 0.5,
                         c2_start: float = 0.5, c2_end: float = 2.0) -> dict:
    """Per-iteration inertia and acceleration coefficients for t = 0 .. T-1.

    linear:  s(t) = t / T                      (exactly the Phase-1 expression)
    cosine:  s(t) = (1 - cos(pi * t / T)) / 2  (slow start, fast middle, slow end)
    w(t)  = w_max   + (w_min  - w_max)   * s(t)
    c1(t) = c1_start + (c1_end - c1_start) * s(t)
    c2(t) = c2_start + (c2_end - c2_start) * s(t)
    Returns ``{"w": array(T), "c1": array(T), "c2": array(T)}``.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
    T = int(T)
    denom = max(T, 1)
    t = np.arange(T, dtype=float)
    if schedule == "linear":
        s = t / denom
    else:
        s = (1.0 - np.cos(np.pi * t / denom)) / 2.0
    w = w_max - (w_max - w_min) * s
    c1 = c1_start + (c1_end - c1_start) * s
    c2 = c2_start + (c2_end - c2_start) * s
    return {"w": w, "c1": c1, "c2": c2}


def centre_of_mass(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Fitness-weighted centre of mass of the swarm: weight_i = 1 / (f_i + 1e-9).

    Better (lower-fitness) particles are heavier; the epsilon guards f == 0.
    ``x`` is (N, 3), ``f`` is (N,); returns a (3,) array.
    """
    x = np.asarray(x, dtype=float)
    f = np.asarray(f, dtype=float)
    wgt = 1.0 / (f + _COM_EPSILON)
    return (wgt[:, None] * x).sum(axis=0) / wgt.sum()


@dataclasses.dataclass
class AMCMPSOResult(PSOResult):
    """``PSOResult`` plus the AMCMPSO diagnostics.

    improvement_rate   (f_0 - f_T) / f_0 with f_0 the best fitness after the initial
                       evaluation and f_T the final best fitness (0.0 when f_0 == 0).
    diversity_history  mean Euclidean distance of the particles to the swarm mean after
                       each iteration; filled only when ``diagnostics=True``.
    stopped_by / warm_start / method follow the shared ``PSOResult`` contract and are
    declared here as well so that the result carries them regardless of the base class.
    """
    improvement_rate: float = 0.0
    diversity_history: list[float] = dataclasses.field(default_factory=list)
    stopped_by: str | None = None
    warm_start: bool = False
    method: str = "amcmpso"


def _result_kwargs(**kwargs) -> dict:
    """Keep only the keyword arguments that are fields of ``AMCMPSOResult``."""
    names = {fld.name for fld in dataclasses.fields(AMCMPSOResult)}
    return {k: v for k, v in kwargs.items() if k in names}


class AMCMPSO:
    def __init__(self, n_particles: int = 20, n_iterations: int = 60,
                 w_max: float = 0.9, w_min: float = 0.4,
                 c1_start: float = 2.0, c1_end: float = 0.5,
                 c2_start: float = 0.5, c2_end: float = 2.0,
                 c_mean: float = 0.5, c_com: float = 0.5,
                 seed: int | None = 1, clip: bool = True, record_positions: bool = False,
                 schedule: str = "linear", stopping=None, x0=None, diagnostics: bool = False):
        if schedule not in SCHEDULES:
            raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")
        self.n_particles, self.n_iterations = int(n_particles), int(n_iterations)
        self.w_max, self.w_min = w_max, w_min
        self.c1_start, self.c1_end, self.c2_start, self.c2_end = c1_start, c1_end, c2_start, c2_end
        self.c_mean, self.c_com = c_mean, c_com
        self.seed, self.clip, self.record_positions = seed, clip, record_positions
        self.schedule, self.stopping, self.x0, self.diagnostics = schedule, stopping, x0, diagnostics

    def swarm_state_floats(self) -> int:
        # position + velocity + personal best, plus mean and centre of mass (2 x 3)
        return self.n_particles * 3 * 3 + 6

    @staticmethod
    def _centre_of_mass(x: np.ndarray, f: np.ndarray) -> np.ndarray:
        # kept for backward compatibility; the module-level function is the implementation
        return centre_of_mass(x, f)

    def run(self, fitness: RangeErrorFitness, lower, upper) -> AMCMPSOResult:
        rng = np.random.default_rng(self.seed)
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        t0 = time.perf_counter()
        fitness.reset()
        rule = self.stopping
        if rule is not None:
            rule.reset()

        # initial swarm: N x 3 uniform draw (first use of the random stream), zero velocity
        x = rng.uniform(lower, upper, (self.n_particles, 3))
        warm = self.x0 is not None
        if warm:
            x[0] = np.clip(np.asarray(self.x0, dtype=float), lower, upper)   # no random numbers consumed
        v = np.zeros_like(x)
        f = fitness(x)
        pbest, pbest_f = x.copy(), f.copy()
        g = int(f.argmin()); gbest, gbest_f = x[g].copy(), float(f[g])
        f_initial = gbest_f
        best_hist, est_hist = [gbest_f], [gbest.copy()]
        pos_hist = [x.copy()] if self.record_positions else None
        div_hist: list[float] = []

        sched = coefficient_schedule(self.n_iterations, self.schedule,
                                     w_max=self.w_max, w_min=self.w_min,
                                     c1_start=self.c1_start, c1_end=self.c1_end,
                                     c2_start=self.c2_start, c2_end=self.c2_end)
        w_s, c1_s, c2_s = sched["w"], sched["c1"], sched["c2"]

        it_run = 0
        stopped_by = None
        for t in range(self.n_iterations):
            w, c1, c2 = w_s[t], c1_s[t], c2_s[t]
            mean = x.mean(axis=0)
            com = centre_of_mass(x, f)
            r1, r2, r3, r4 = (rng.random(x.shape) for _ in range(4))   # same draw order as Phase 1
            v = (w * v + c1 * r1 * (pbest - x) + c2 * r2 * (gbest - x)
                 + self.c_mean * r3 * (mean - x) + self.c_com * r4 * (com - x))
            x = x + v
            if self.clip:
                x = np.clip(x, lower, upper)
            f = fitness(x)
            better = f < pbest_f
            pbest[better], pbest_f[better] = x[better], f[better]
            g = int(pbest_f.argmin())
            if pbest_f[g] < gbest_f:
                gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
            it_run = t + 1
            best_hist.append(gbest_f); est_hist.append(gbest.copy())
            if pos_hist is not None:
                pos_hist.append(x.copy())
            if self.diagnostics:
                div_hist.append(float(np.linalg.norm(x - x.mean(axis=0), axis=1).mean()))
            if rule is not None and rule.update(it_run, gbest_f, fitness.evaluations):
                stopped_by = getattr(rule, "name", type(rule).__name__)
                break

        improvement = (f_initial - gbest_f) / f_initial if f_initial > 0.0 else 0.0
        return AMCMPSOResult(**_result_kwargs(
            estimate=gbest, best_fitness=gbest_f, iterations_run=it_run,
            fitness_evaluations=fitness.evaluations, distance_computations=fitness.distance_computations,
            runtime_s=time.perf_counter() - t0, swarm_state_floats=self.swarm_state_floats(),
            swarm_state_bytes=self.swarm_state_floats() * 8, best_history=best_hist, estimate_history=est_hist,
            positions_history=np.array(pos_hist) if pos_hist is not None else None,
            improvement_rate=float(improvement), diversity_history=div_hist,
            stopped_by=stopped_by, warm_start=warm, method="amcmpso"))

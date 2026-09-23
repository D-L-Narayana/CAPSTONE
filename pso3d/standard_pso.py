"""Standard (Kennedy & Eberhart 1995 / Shi & Eberhart inertia) PSO with pbest/gbest memory.

    v = w v + c1 r1 (pbest - x) + c2 r2 (gbest - x),   x = x + v

Kept here as the *reference* PSO that AMCMPSO extends; costs one extra
N x 3 array (personal bests) of swarm memory compared with SimplifiedPSO.
"""
from __future__ import annotations

import time
import numpy as np

from .fitness import RangeErrorFitness
from .pso import PSOResult


class StandardPSO:
    def __init__(self, n_particles: int = 20, n_iterations: int = 60, w: float = 0.7,
                 c1: float = 1.4, c2: float = 1.4, seed: int | None = 1, clip: bool = True,
                 record_positions: bool = False):
        self.n_particles, self.n_iterations = int(n_particles), int(n_iterations)
        self.w, self.c1, self.c2, self.seed, self.clip = w, c1, c2, seed, clip
        self.record_positions = record_positions

    def swarm_state_floats(self) -> int:
        # position + velocity + personal best (+ gbest, negligible)
        return self.n_particles * 3 * 3 + 3

    def run(self, fitness: RangeErrorFitness, lower, upper) -> PSOResult:
        rng = np.random.default_rng(self.seed)
        lower, upper = np.asarray(lower, float), np.asarray(upper, float)
        t0 = time.perf_counter()
        fitness.reset()
        x = rng.uniform(lower, upper, (self.n_particles, 3))
        v = np.zeros_like(x)
        f = fitness(x)
        pbest, pbest_f = x.copy(), f.copy()
        g = int(f.argmin()); gbest, gbest_f = x[g].copy(), float(f[g])
        best_hist, est_hist = [gbest_f], [gbest.copy()]
        pos_hist = [x.copy()] if self.record_positions else None
        for _ in range(self.n_iterations):
            r1, r2 = rng.random(x.shape), rng.random(x.shape)
            v = self.w * v + self.c1 * r1 * (pbest - x) + self.c2 * r2 * (gbest - x)
            x = x + v
            if self.clip:
                x = np.clip(x, lower, upper)
            f = fitness(x)
            better = f < pbest_f
            pbest[better], pbest_f[better] = x[better], f[better]
            g = int(pbest_f.argmin())
            if pbest_f[g] < gbest_f:
                gbest, gbest_f = pbest[g].copy(), float(pbest_f[g])
            best_hist.append(gbest_f); est_hist.append(gbest.copy())
            if pos_hist is not None:
                pos_hist.append(x.copy())
        return PSOResult(gbest, gbest_f, self.n_iterations, fitness.evaluations,
                         fitness.distance_computations, time.perf_counter() - t0,
                         self.swarm_state_floats(), self.swarm_state_floats() * 8,
                         best_hist, est_hist,
                         np.array(pos_hist) if pos_hist is not None else None)

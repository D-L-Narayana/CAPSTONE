"""AMCMPSO - Adaptive Mean Center of Mass PSO (base paper: Alhasan et al., 2023,
J. King Saud Univ. - Comput. Inf. Sci., 35(9), 101782, https://doi.org/10.1016/j.jksuci.2023.101782).

STATUS: IN PROGRESS (Phase 1, milestone M2 / Review 2).

What is implemented here is a *working scaffold* of the algorithm's structure as we
understand it from the paper's abstract and our literature summary: a standard PSO
whose velocity update is additionally guided by (i) the swarm mean position and
(ii) the fitness-weighted centre of mass of the swarm, with (iii) adaptive
w, c1, c2 that move from exploration to exploitation over the iterations.

The exact update equations, the adaptation schedules and the constants used in the
paper still have to be transcribed from Section 3 of the paper and verified against
its reported numbers (improvement rate 99.86 %, error < 1.34 cm, 3D coverage > 87 %).
Until that is done the numbers produced by this class must NOT be quoted as a
reproduction of the base paper.

TODO (tracked in README "Roadmap"):
  [ ] transcribe the exact velocity equation and adaptation rules from the paper
  [ ] multi-node auto-localization (localized nodes become anchors) and coverage metric
  [ ] paper's noise model and network sizes; comparison table against the paper
"""
from __future__ import annotations

import time
import numpy as np

from .fitness import RangeErrorFitness
from .pso import PSOResult


class AMCMPSO:
    def __init__(self, n_particles: int = 20, n_iterations: int = 60,
                 w_max: float = 0.9, w_min: float = 0.4,
                 c1_start: float = 2.0, c1_end: float = 0.5,
                 c2_start: float = 0.5, c2_end: float = 2.0,
                 c_mean: float = 0.5, c_com: float = 0.5,
                 seed: int | None = 1, clip: bool = True, record_positions: bool = False):
        self.n_particles, self.n_iterations = int(n_particles), int(n_iterations)
        self.w_max, self.w_min = w_max, w_min
        self.c1_start, self.c1_end, self.c2_start, self.c2_end = c1_start, c1_end, c2_start, c2_end
        self.c_mean, self.c_com = c_mean, c_com
        self.seed, self.clip, self.record_positions = seed, clip, record_positions

    def swarm_state_floats(self) -> int:
        # position + velocity + personal best, plus mean and centre of mass (2 x 3)
        return self.n_particles * 3 * 3 + 6

    @staticmethod
    def _centre_of_mass(x: np.ndarray, f: np.ndarray) -> np.ndarray:
        # weight = inverse fitness (better particles are heavier); guard against f == 0
        wgt = 1.0 / (f + 1e-9)
        return (wgt[:, None] * x).sum(axis=0) / wgt.sum()

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
        T = max(self.n_iterations, 1)
        for t in range(self.n_iterations):
            frac = t / T
            # adaptive coefficients: linear schedules (exploration -> exploitation)
            w = self.w_max - (self.w_max - self.w_min) * frac
            c1 = self.c1_start + (self.c1_end - self.c1_start) * frac
            c2 = self.c2_start + (self.c2_end - self.c2_start) * frac
            mean = x.mean(axis=0)
            com = self._centre_of_mass(x, f)
            r1, r2, r3, r4 = (rng.random(x.shape) for _ in range(4))
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
            best_hist.append(gbest_f); est_hist.append(gbest.copy())
            if pos_hist is not None:
                pos_hist.append(x.copy())
        return PSOResult(gbest, gbest_f, self.n_iterations, fitness.evaluations,
                         fitness.distance_computations, time.perf_counter() - t0,
                         self.swarm_state_floats(), self.swarm_state_floats() * 8,
                         best_hist, est_hist,
                         np.array(pos_hist) if pos_hist is not None else None)

"""Simplified PSO exactly as implemented for Review 1 (slide 16).

    v_i <- w * v_i + c * r (.) (x_best - x_i)
    x_i <- x_i + v_i

* one social term toward the best particle of the *current* iteration
  (no personal-best / global-best memory);
* N particles start uniformly in the field with zero velocity;
* fixed number of iterations (optional early stopping / clipping are OFF by
  default so that the original numbers are reproduced bit-for-bit).

Random numbers come from ``numpy.random.default_rng(seed)`` in the same order
as the original script ``p09_pso_localization_3d.py``: first the N x 3 uniform
start positions, then one N x 3 uniform draw per iteration.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import numpy as np

from .fitness import RangeErrorFitness


@dataclass
class PSOResult:
    estimate: np.ndarray
    best_fitness: float
    iterations_run: int
    fitness_evaluations: int
    distance_computations: int
    runtime_s: float
    swarm_state_floats: int             # numbers stored per node for the swarm
    swarm_state_bytes: int              # float64 bytes
    best_history: list[float] = field(default_factory=list)     # best fitness per iteration
    estimate_history: list[np.ndarray] = field(default_factory=list)
    positions_history: list[np.ndarray] | None = None           # (T+1, N, 3) if recorded

    def error(self, true_position: np.ndarray) -> float:
        return float(np.linalg.norm(self.estimate - np.asarray(true_position)))


class SimplifiedPSO:
    def __init__(self, n_particles: int = 20, n_iterations: int = 60, w: float = 0.7,
                 c: float = 1.4, seed: int | None = 1, clip: bool = False,
                 early_stop_patience: int | None = None, record_positions: bool = False):
        self.n_particles = int(n_particles)
        self.n_iterations = int(n_iterations)
        self.w = float(w)
        self.c = float(c)
        self.seed = seed
        self.clip = clip
        self.early_stop_patience = early_stop_patience
        self.record_positions = record_positions

    # memory model from slide 16: N particles x 3 coordinates x 2 (position, velocity)
    def swarm_state_floats(self) -> int:
        return self.n_particles * 3 * 2

    def run(self, fitness: RangeErrorFitness, lower: np.ndarray, upper: np.ndarray) -> PSOResult:
        rng = np.random.default_rng(self.seed)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        t0 = time.perf_counter()
        fitness.reset()

        x = rng.uniform(lower, upper, (self.n_particles, 3))
        v = np.zeros((self.n_particles, 3))
        best_x = x[0].copy()
        best_f = np.inf
        best_hist: list[float] = []
        est_hist: list[np.ndarray] = []
        pos_hist = [x.copy()] if self.record_positions else None
        stall = 0
        it_run = 0

        for it in range(self.n_iterations):
            f = fitness(x)
            i_best = int(f.argmin())
            best_x = x[i_best].copy()          # best of the *current* iteration only
            it_run = it + 1
            # bookkeeping
            improved = f[i_best] < best_f - 1e-12
            best_f = float(f[i_best]) if improved or it == 0 else best_f
            best_hist.append(float(f[i_best]))
            est_hist.append(best_x.copy())
            # velocity / position update (same random-draw order as the original script)
            v = self.w * v + self.c * rng.random((self.n_particles, 3)) * (best_x - x)
            x = x + v
            if self.clip:
                x = np.clip(x, lower, upper)
            if pos_hist is not None:
                pos_hist.append(x.copy())
            # optional early stopping on a plateau of the per-iteration best fitness
            if self.early_stop_patience is not None:
                stall = 0 if improved else stall + 1
                if stall >= self.early_stop_patience:
                    break

        runtime = time.perf_counter() - t0
        return PSOResult(
            estimate=best_x,
            best_fitness=float(best_hist[-1]),
            iterations_run=it_run,
            fitness_evaluations=fitness.evaluations,
            distance_computations=fitness.distance_computations,
            runtime_s=runtime,
            swarm_state_floats=self.swarm_state_floats(),
            swarm_state_bytes=self.swarm_state_floats() * 8,
            best_history=best_hist,
            estimate_history=est_hist,
            positions_history=np.array(pos_hist) if pos_hist is not None else None,
        )

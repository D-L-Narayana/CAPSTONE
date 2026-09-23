"""Range-error fitness  f(p) = sum_j ( ||p - a_j|| - d_hat_j )^2  with evaluation counting."""
from __future__ import annotations

import numpy as np


class RangeErrorFitness:
    """Callable fitness that counts fitness evaluations and distance computations.

    One *fitness evaluation* = scoring one candidate position p (M distances).
    """

    def __init__(self, anchors: np.ndarray, measured: np.ndarray):
        self.anchors = np.asarray(anchors, dtype=float)
        self.measured = np.asarray(measured, dtype=float)
        self.evaluations = 0
        self.distance_computations = 0

    @property
    def n_anchors(self) -> int:
        return self.anchors.shape[0]

    def __call__(self, positions: np.ndarray) -> np.ndarray:
        """positions: (N, 3) -> fitness (N,). Also accepts a single (3,) point."""
        pts = np.atleast_2d(np.asarray(positions, dtype=float))
        dist = np.linalg.norm(pts[:, None, :] - self.anchors[None, :, :], axis=2)  # (N, M)
        self.evaluations += pts.shape[0]
        self.distance_computations += pts.shape[0] * self.n_anchors
        f = ((dist - self.measured) ** 2).sum(axis=1)
        return f if positions.ndim == 2 else f[0]

    def reset(self) -> None:
        self.evaluations = 0
        self.distance_computations = 0

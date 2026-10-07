"""Range-error fitness  f(p) = sum_j ( ||p - a_j|| - d_hat_j )^2  with evaluation counting."""
from __future__ import annotations

import numpy as np


class RangeErrorFitness:
    """Callable fitness that counts fitness evaluations and distance computations.

    One *fitness evaluation* = scoring one candidate position p (M distances).
    Accepts numpy arrays, lists or tuples: a single point of shape (3,) returns a float,
    a batch of shape (N, 3) returns an array of shape (N,).
    """

    def __init__(self, anchors: np.ndarray, measured: np.ndarray):
        self.anchors = np.asarray(anchors, dtype=float)
        self.measured = np.asarray(measured, dtype=float)
        self.evaluations = 0
        self.distance_computations = 0

    @property
    def n_anchors(self) -> int:
        return self.anchors.shape[0]

    def __call__(self, positions) -> np.ndarray | float:
        """positions: (N, 3) -> fitness (N,). Also accepts a single (3,) point -> float."""
        arr = np.asarray(positions, dtype=float)      # convert once; lists/tuples become arrays here
        single = arr.ndim == 1
        pts = arr[None, :] if single else arr
        dist = np.linalg.norm(pts[:, None, :] - self.anchors[None, :, :], axis=2)  # (N, M)
        self.evaluations += pts.shape[0]
        self.distance_computations += pts.shape[0] * self.n_anchors
        f = ((dist - self.measured) ** 2).sum(axis=1)
        return float(f[0]) if single else f

    def evaluate_one(self, p) -> float:
        """Score a single point (counts as one evaluation and M distance computations)."""
        point = np.asarray(p, dtype=float).reshape(self.anchors.shape[1])
        return float(self(point))

    def reset(self) -> None:
        self.evaluations = 0
        self.distance_computations = 0

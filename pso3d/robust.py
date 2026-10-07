"""Robust range-error fitness functions: Huber loss and per-anchor weights.

With residuals ``r_j(p) = ||p - a_j|| - d_hat_j``::

    RangeErrorFitness      f(p) = sum_j r_j^2                      (Review-1 baseline, unchanged)
    HuberRangeFitness      f(p) = sum_j rho_delta(r_j)
                           rho_delta(r) = r^2              if |r| <= delta
                                        = 2 delta |r| - delta^2   otherwise  (continuous value and slope)
    WeightedRangeFitness   f(p) = sum_j w_j r_j^2                  (w_j >= 0; 1 / sigma_j^2 via from_sigmas)

Both classes subclass :class:`pso3d.fitness.RangeErrorFitness` so they drop into every optimiser of the
package. They compute distances themselves and keep the baseline counter semantics: one *evaluation*
per candidate position and ``M`` *distance computations* per evaluation. A single ``(3,)`` point gives
a ``float``; a batch ``(N, 3)`` gives an ``(N,)`` array; Python lists are accepted.
Rationale and guidance for choosing ``delta``/weights: ``docs/noise_models.md``.
"""
from __future__ import annotations

import numpy as np

from .fitness import RangeErrorFitness

__all__ = ["huber_loss", "HuberRangeFitness", "WeightedRangeFitness"]


def _check_delta(delta) -> float:
    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0.0:
        raise ValueError(f"delta must be a positive finite number, got {delta}")
    return delta


def huber_loss(residuals, delta: float):
    """Element-wise Huber loss ``r^2`` for ``|r| <= delta`` else ``2 delta |r| - delta^2``.

    Returns a float for scalar input, otherwise an array shaped like ``residuals``.
    """
    delta = _check_delta(delta)
    r = np.asarray(residuals, dtype=float)
    a = np.abs(r)
    out = np.where(a <= delta, r * r, 2.0 * delta * a - delta * delta)
    return float(out) if out.ndim == 0 else out


class _RobustRangeFitness(RangeErrorFitness):
    """Shared machinery: validated inputs, own distance computation, counters, ``evaluate_one``."""

    name = "robust"

    def __init__(self, anchors, measured):
        super().__init__(anchors, measured)
        # set explicitly so this class does not rely on the base-class internals
        self.anchors = np.asarray(anchors, dtype=float)
        self.measured = np.asarray(measured, dtype=float)
        if self.anchors.ndim != 2 or self.anchors.shape[1] != 3:
            raise ValueError(f"anchors must have shape (M, 3), got {self.anchors.shape}")
        if self.measured.shape != (self.anchors.shape[0],):
            raise ValueError(f"measured must have shape ({self.anchors.shape[0]},), got {self.measured.shape}")
        self.evaluations = 0
        self.distance_computations = 0

    def _loss(self, residuals: np.ndarray) -> np.ndarray:
        """Map per-anchor residuals ``(N, M)`` to fitness values ``(N,)``."""
        raise NotImplementedError

    def __call__(self, positions):
        pts = np.asarray(positions, dtype=float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"positions must have shape (3,) or (N, 3), got {np.shape(positions)}")
        dist = np.linalg.norm(pts[:, None, :] - self.anchors[None, :, :], axis=2)  # (N, M)
        n_points, n_anchors = dist.shape
        self.evaluations += n_points
        self.distance_computations += n_points * n_anchors
        f = np.asarray(self._loss(dist - self.measured), dtype=float)
        return float(f[0]) if single else f

    def evaluate_one(self, p) -> float:
        """Fitness of a single point ``p`` (counts as one evaluation)."""
        pts = np.asarray(p, dtype=float)
        if pts.shape != (3,):
            raise ValueError(f"evaluate_one expects a single (3,) point, got shape {pts.shape}")
        return float(self(pts))


class HuberRangeFitness(_RobustRangeFitness):
    """Sum of Huber losses of the range residuals; equals the squared loss while ``|r_j| <= delta``.

    Beyond ``delta`` the loss grows linearly with slope ``2 delta``, so a single NLOS range cannot
    dominate the sum. Use ``delta`` of about 1-2 times the line-of-sight noise standard deviation.
    """

    name = "huber"

    def __init__(self, anchors, measured, delta: float = 1.0):
        super().__init__(anchors, measured)
        self.delta = _check_delta(delta)

    def _loss(self, residuals: np.ndarray) -> np.ndarray:
        return huber_loss(residuals, self.delta).sum(axis=1)


class WeightedRangeFitness(_RobustRangeFitness):
    """Weighted squared range error ``sum_j w_j r_j^2`` with non-negative per-anchor weights.

    Unit weights reproduce :class:`RangeErrorFitness` exactly; ``w_j = 1 / sigma_j^2``
    (:meth:`from_sigmas`) is the maximum-likelihood choice for independent Gaussian noise.
    """

    name = "weighted"

    def __init__(self, anchors, measured, weights):
        super().__init__(anchors, measured)
        w = np.asarray(weights, dtype=float)
        if w.shape != (self.n_anchors,):
            raise ValueError(f"weights must have shape ({self.n_anchors},), got {w.shape}")
        if not np.all(np.isfinite(w)) or np.any(w < 0.0):
            raise ValueError("weights must be finite and >= 0")
        self.weights = w

    @classmethod
    def from_sigmas(cls, anchors, measured, sigmas) -> "WeightedRangeFitness":
        """Inverse-variance weights ``w_j = 1 / sigma_j^2`` (all ``sigma_j`` must be > 0)."""
        s = np.asarray(sigmas, dtype=float)
        if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError("sigmas must be finite and > 0")
        return cls(anchors, measured, 1.0 / (s * s))

    def _loss(self, residuals: np.ndarray) -> np.ndarray:
        return (self.weights * (residuals * residuals)).sum(axis=1)

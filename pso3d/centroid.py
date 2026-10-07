"""Centroid estimators: the range-free baseline and the inverse-distance weighted centroid.

Plain centroid (no ranges at all):          p = (1/M) sum_j a_j
Weighted centroid (code 5 of the project's "10 codes" document):

    w_j = 1 / d_j^power,    p = sum_j w_j a_j / sum_j w_j

Anchors with shorter measured ranges pull the estimate harder; ``power`` controls how hard
(0 -> plain centroid, large -> nearest anchor). Measured ranges <= 1e-12 are clamped to 1e-12
so that a node sitting on an anchor still gives a finite answer (that anchor then dominates).
Both estimators always return a point inside the bounding box of the anchors, which makes the
weighted centroid a safe fallback for the warm-start helper in ``pso3d.trilateration`` when the
linearised least-squares system is singular.
"""
from __future__ import annotations

import numpy as np

_MIN_RANGE = 1e-12


def centroid(anchors) -> np.ndarray:
    """Arithmetic mean of the anchor positions (uses no range information)."""
    a = np.asarray(anchors, float)
    return a.mean(axis=0)


def weighted_centroid(anchors, measured, power: float = 1.0) -> np.ndarray:
    """Inverse-distance weighted centroid with weights 1 / d_hat_j ** power (power >= 0).

    The weights are evaluated as (d_min / d_j) ** power, which is the same set of weights up to a
    common factor (the result is identical) but cannot overflow for large ``power``.
    """
    a = np.asarray(anchors, float)
    d = np.asarray(measured, float)
    d = np.where(d <= _MIN_RANGE, _MIN_RANGE, d)
    w = (d.min() / d) ** power
    return (a * w[:, None]).sum(axis=0) / w.sum()

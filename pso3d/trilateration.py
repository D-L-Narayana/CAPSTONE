"""Closed-form comparison methods: linearised least-squares trilateration and Gauss-Newton refinement.

Least squares: subtracting the sphere equation of anchor 1 from anchors j = 2..M gives
the linear system  A p = b  with
    A_j = 2 (a_j - a_1),   b_j = ||a_j||^2 - ||a_1||^2 - d_j^2 + d_1^2 ,
solved with numpy.linalg.lstsq (one small (M-1) x 3 solve - the "one 3x3 solve" of slide 16).
"""
from __future__ import annotations

import numpy as np


def least_squares_trilateration(anchors: np.ndarray, measured: np.ndarray) -> np.ndarray:
    a = np.asarray(anchors, float)
    d = np.asarray(measured, float)
    A = 2.0 * (a[1:] - a[0])
    b = (a[1:] ** 2).sum(axis=1) - (a[0] ** 2).sum() - d[1:] ** 2 + d[0] ** 2
    p, *_ = np.linalg.lstsq(A, b, rcond=None)
    return p


def gauss_newton_refine(anchors: np.ndarray, measured: np.ndarray, p0: np.ndarray,
                        iterations: int = 10, tol: float = 1e-9) -> tuple[np.ndarray, int]:
    """Iteratively minimise sum_j (||p - a_j|| - d_j)^2 starting from p0. Returns (p, iterations used)."""
    a = np.asarray(anchors, float)
    d = np.asarray(measured, float)
    p = np.asarray(p0, float).copy()
    used = 0
    for k in range(iterations):
        diff = p - a
        dist = np.linalg.norm(diff, axis=1)
        dist = np.where(dist < 1e-12, 1e-12, dist)
        r = dist - d                       # residuals
        J = diff / dist[:, None]           # Jacobian of ||p - a_j|| wrt p
        step, *_ = np.linalg.lstsq(J, -r, rcond=None)
        p = p + step
        used = k + 1
        if np.linalg.norm(step) < tol:
            break
    return p, used

"""Closed-form estimators: linearised (weighted) least-squares trilateration, Gauss-Newton and
Levenberg-Marquardt refinement with diagnostics, and a warm-start helper for the swarm optimisers.

Least squares: subtracting the sphere equation of anchor 1 from anchors j = 2..M gives
the linear system  A p = b  with
    A_j = 2 (a_j - a_1),   b_j = ||a_j||^2 - ||a_1||^2 - d_j^2 + d_1^2 ,
solved with numpy.linalg.lstsq (one small (M-1) x 3 solve - the "one 3x3 solve" of slide 16).
With weights w_j the rows j = 2..M of A and b are multiplied by sqrt(w_j) (weights=None leaves
the system untouched, so the default result is unchanged).

Gauss-Newton minimises  f(p) = sum_j (||p - a_j|| - d_j)^2  from a start p0:
    r_j = ||p - a_j|| - d_j,   J_j = (p - a_j) / ||p - a_j||,   step = argmin ||J step + r||,
optionally damped:  (J^T J + lambda I) step = -J^T r.  Levenberg-Marquardt adapts lambda from
step to step and only ever accepts steps that reduce ||r||.

``least_squares_trilateration`` (default call) and ``gauss_newton_refine`` keep their exact
historical outputs; ``gauss_newton`` produces the same iterates and adds diagnostics.
Derivations, failure modes and pinned examples: docs/closed_form.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .centroid import centroid, weighted_centroid

_MIN_DISTANCE = 1e-12      # guards the Jacobian when p coincides with an anchor
_LAMBDA_MAX = 1e12         # Levenberg-Marquardt: beyond this damping a step cannot help any more


@dataclass
class GNResult:
    """Outcome of an iterative refinement (``gauss_newton`` / ``levenberg_marquardt``).

    position       final estimate (3,)
    iterations     accepted steps taken
    converged      True when the stopping test was met (see the functions' docstrings)
    residual_norm  ||r(position)|| = sqrt(f(position)), the root of the range-error fitness
    evaluations    residual/Jacobian evaluations spent (includes the one at the returned position)
    """

    position: np.ndarray
    iterations: int
    converged: bool
    residual_norm: float
    evaluations: int


def _linear_system(a: np.ndarray, d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows j = 2..M of the linearised trilateration system A p = b (anchor 1 is the reference)."""
    A = 2.0 * (a[1:] - a[0])
    b = (a[1:] ** 2).sum(axis=1) - (a[0] ** 2).sum() - d[1:] ** 2 + d[0] ** 2
    return A, b


def _residual_and_jacobian(a: np.ndarray, d: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    diff = p - a
    dist = np.linalg.norm(diff, axis=1)
    dist = np.where(dist < _MIN_DISTANCE, _MIN_DISTANCE, dist)
    return dist - d, diff / dist[:, None]


def _step(J: np.ndarray, r: np.ndarray, damping: float) -> np.ndarray:
    """Gauss-Newton step (damping == 0, min-norm least squares as in ``gauss_newton_refine``) or the
    damped step solving (J^T J + damping I) step = -J^T r through the equivalent augmented system."""
    if damping > 0.0:
        n = J.shape[1]
        A = np.vstack([J, np.sqrt(damping) * np.eye(n)])
        b = np.concatenate([-r, np.zeros(n)])
        step, *_ = np.linalg.lstsq(A, b, rcond=None)
        return step
    step, *_ = np.linalg.lstsq(J, -r, rcond=None)
    return step


def least_squares_trilateration(anchors: np.ndarray, measured: np.ndarray,
                                weights: np.ndarray | None = None) -> np.ndarray:
    """Linearised least-squares position from M >= 4 anchors and measured ranges.

    ``weights`` (optional, one non-negative entry per anchor) scales row j of the linear system by
    sqrt(w_j); a weight of 0 removes that anchor's equation. The equation of the reference anchor
    (index 0) is the one subtracted from the others, so its weight does not enter — reorder the
    anchors if the reference itself is unreliable. ``weights=None`` reproduces the unweighted solve.
    """
    a = np.asarray(anchors, float)
    d = np.asarray(measured, float)
    A, b = _linear_system(a, d)
    if weights is not None:
        w = np.asarray(weights, float)
        if w.shape != (a.shape[0],):
            raise ValueError(f"weights must have one entry per anchor ({a.shape[0]}), got shape {w.shape}")
        if not np.isfinite(w).all() or (w < 0.0).any():
            raise ValueError("weights must be finite and non-negative")
        sw = np.sqrt(w[1:])
        A = A * sw[:, None]
        b = b * sw
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


def gauss_newton(anchors, measured, p0, iterations: int = 10, tol: float = 1e-9,
                 damping: float = 0.0) -> GNResult:
    """Gauss-Newton refinement with diagnostics.

    Produces exactly the iterates of ``gauss_newton_refine`` when ``damping == 0``. A positive
    ``damping`` adds lambda I to J^T J (a fixed Levenberg step), which shortens every step and
    helps from poor starts. ``converged`` is True when the last step was shorter than ``tol``.
    ``evaluations`` counts one residual/Jacobian evaluation per step plus the residual at the
    returned position (reported as ``residual_norm``).
    """
    if damping < 0.0:
        raise ValueError("damping must be >= 0")
    a = np.asarray(anchors, float)
    d = np.asarray(measured, float)
    p = np.asarray(p0, float).copy()
    used, converged, evaluations = 0, False, 0
    for k in range(iterations):
        r, J = _residual_and_jacobian(a, d, p)
        evaluations += 1
        step = _step(J, r, damping)
        p = p + step
        used = k + 1
        if np.linalg.norm(step) < tol:
            converged = True
            break
    r, _ = _residual_and_jacobian(a, d, p)
    evaluations += 1
    return GNResult(position=p, iterations=used, converged=converged,
                    residual_norm=float(np.linalg.norm(r)), evaluations=evaluations)


def levenberg_marquardt(anchors, measured, p0, iterations: int = 20, lam0: float = 1e-2,
                        tol: float = 1e-9) -> GNResult:
    """Levenberg-Marquardt refinement: Gauss-Newton with an adaptive damping lambda.

    Each iteration proposes the damped step for the current lambda. If it lowers ||r|| it is
    accepted and lambda is divided by 10; otherwise lambda is multiplied by 10 and the step is
    recomputed (up to lambda = 1e12). The returned position therefore never has a larger
    residual than the start. ``converged`` is True when an accepted step was shorter than
    ``tol``, when the gradient norm ||J^T r|| fell below ``tol``, or when no admissible damping
    could reduce the residual any further (a minimum in working precision). ``evaluations``
    counts the initial residual/Jacobian plus every candidate step evaluated, accepted or not.
    """
    if not lam0 > 0.0:
        raise ValueError("lam0 must be > 0")
    a = np.asarray(anchors, float)
    d = np.asarray(measured, float)
    p = np.asarray(p0, float).copy()
    lam = float(lam0)
    r, J = _residual_and_jacobian(a, d, p)
    evaluations = 1
    cost = float(np.linalg.norm(r))
    used, converged = 0, False
    for _ in range(iterations):
        if np.linalg.norm(J.T @ r) < tol:
            converged = True
            break
        accepted = False
        while lam <= _LAMBDA_MAX:
            step = _step(J, r, lam)
            candidate = p + step
            r_c, J_c = _residual_and_jacobian(a, d, candidate)
            evaluations += 1
            cost_c = float(np.linalg.norm(r_c))
            if cost_c < cost:
                p, r, J, cost = candidate, r_c, J_c, cost_c
                lam /= 10.0
                accepted = True
                break
            lam *= 10.0
        if not accepted:
            converged = True
            break
        used += 1
        if np.linalg.norm(step) < tol:
            converged = True
            break
    return GNResult(position=p, iterations=used, converged=converged, residual_norm=cost,
                    evaluations=evaluations)


def warm_start_estimate(anchors, measured, lower, upper) -> np.ndarray:
    """Cheap starting point for a swarm: least squares -> 10 Gauss-Newton steps -> clip to the box.

    Anchors whose range or position is not finite are ignored. If the linearised system is
    rank-deficient (fewer than four usable anchors, or anchors that are collinear/coplanar so that
    ``matrix_rank(A) < 3``) or the solve yields non-finite values, Gauss-Newton starts from the
    inverse-distance weighted centroid instead. The result is always finite and inside
    [lower, upper] (for a finite box); with no usable anchor at all it is the centre of the box.
    """
    a = np.asarray(anchors, float)
    d = np.asarray(measured, float)
    lo = np.asarray(lower, float)
    hi = np.asarray(upper, float)
    usable = np.isfinite(d) & np.isfinite(a).all(axis=1)
    a_use, d_use = (a, d) if usable.all() else (a[usable], d[usable])
    if d_use.size == 0:
        estimate = (lo + hi) / 2.0
    else:
        start = None
        if a_use.shape[0] >= 4:
            A, _ = _linear_system(a_use, d_use)
            if np.linalg.matrix_rank(A) >= 3:
                candidate = least_squares_trilateration(a_use, d_use)
                if np.isfinite(candidate).all():
                    start = candidate
        if start is None:
            start = weighted_centroid(a_use, d_use)
        estimate = gauss_newton(a_use, d_use, start, iterations=10).position
        if not np.isfinite(estimate).all():
            estimate = start
    estimate = np.where(np.isfinite(estimate), estimate, (lo + hi) / 2.0)
    return np.clip(estimate, lo, hi)

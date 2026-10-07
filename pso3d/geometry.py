"""Anchor-geometry diagnostics for range-based 3D localization (numpy only).

Why this module exists: the Monte-Carlo study shows that most of the heavy error tail of the
localizers is caused by *where the anchors are*, not by the optimiser.  These functions turn that
intuition into numbers that scripts, tests and the browser simulator can share:

* ``tetrahedron_volume``, ``anchor_rank``, ``is_non_coplanar`` - is the anchor layout genuinely 3D?
* ``range_jacobian``, ``gdop`` - geometric dilution of precision of the range set at a point.
* ``crlb`` - Cramer-Rao lower bound of any unbiased position estimate under i.i.d. Gaussian ranging noise.
* ``anchor_plane``, ``mirror_point``, ``flip_ambiguity_risk``, ``flip_risk_hint`` - the mirror-image
  ("flip") ambiguity that near-coplanar anchors cannot resolve.
* ``coverage_grid``, ``scenario_report`` - a CRLB map over the field and a JSON-friendly summary.

All formulas are written out in ``docs/geometry.md``; ``web/geometry.js`` implements the same
definitions for the browser so that both can be compared on shared fixtures.

Notation: anchors ``a_j`` (M x 3), candidate position ``p``, measured ranges ``d_j``,
range fitness ``f(q) = sum_j (||q - a_j|| - d_j)^2`` (the same objective the optimisers minimise).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CRLB", "tetrahedron_volume", "anchor_rank", "is_non_coplanar", "range_jacobian", "gdop", "crlb",
    "anchor_plane", "mirror_point", "flip_ambiguity_risk", "flip_risk_hint", "coverage_grid", "scenario_report",
]

_COND_LIMIT = 1e12       # J^T J with a 2-norm condition number above this is treated as singular
_ZERO_DISTANCE = 1e-12   # guard for a candidate sitting exactly on an anchor
_FITNESS_FLOOR = 1e-12   # guard for a mirror image that fits the ranges perfectly


@dataclass(frozen=True, eq=False)
class CRLB:
    """Cramer-Rao lower bound at one position.

    ``cov``        (3, 3) inverse Fisher information: lower bound of the estimator covariance (m^2).
    ``rmse_bound`` sqrt(trace(cov)): lower bound of the root-mean-square position error (m).
    ``per_axis``   sqrt(diag(cov)): lower bound of the standard deviation along x, y and z (m).
    For a singular geometry every entry is ``inf``.
    """

    cov: np.ndarray
    rmse_bound: float
    per_axis: np.ndarray

    @property
    def singular(self) -> bool:
        return not math.isfinite(self.rmse_bound)


# ----------------------------------------------------------------------------------------------- helpers

def _as_anchors(anchors) -> np.ndarray:
    a = np.asarray(anchors, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"anchors must have shape (M, 3), got {a.shape}")
    return a


def _as_point(p) -> np.ndarray:
    q = np.asarray(p, dtype=float).reshape(-1)
    if q.shape != (3,):
        raise ValueError(f"a position must have 3 coordinates, got shape {np.shape(p)}")
    return q


def _as_sigma(sigma) -> float:
    s = float(sigma)
    if not s > 0.0:
        raise ValueError(f"sigma must be a positive ranging noise level in metres, got {sigma!r}")
    return s


def _range_fitness(anchors: np.ndarray, measured: np.ndarray, q: np.ndarray) -> float:
    """f(q) = sum_j (||q - a_j|| - d_j)^2."""
    return float(np.sum((np.linalg.norm(q - anchors, axis=1) - measured) ** 2))


def _safe_inverse(m: np.ndarray) -> np.ndarray | None:
    """Inverse of a symmetric 3x3 matrix, or ``None`` when it is numerically singular."""
    if not np.all(np.isfinite(m)):
        return None
    try:
        cond = float(np.linalg.cond(m))
    except np.linalg.LinAlgError:
        return None
    if not math.isfinite(cond) or cond > _COND_LIMIT:
        return None
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return None


# ----------------------------------------------------------------------------------------------- layout

def tetrahedron_volume(a, b, c, d) -> float:
    """Volume of the tetrahedron with vertices a, b, c, d:  |det([b - a, c - a, d - a])| / 6."""
    a, b, c, d = (_as_point(v) for v in (a, b, c, d))
    return float(abs(np.linalg.det(np.stack([b - a, c - a, d - a]))) / 6.0)


def anchor_rank(anchors, tol: float = 1e-9) -> int:
    """Numerical rank of the centred anchor matrix ``anchors - mean(anchors)``.

    3 = the anchors span a volume (non-coplanar), 2 = coplanar, 1 = collinear, 0 = a single point.
    ``tol`` is relative to the largest singular value, so the result does not depend on the unit of length.
    """
    a = _as_anchors(anchors)
    if a.shape[0] == 0:
        return 0
    centred = a - a.mean(axis=0)
    s_max = float(np.linalg.svd(centred, compute_uv=False)[0])
    if not s_max > 0.0:
        return 0
    return int(np.linalg.matrix_rank(centred, tol=tol * s_max))


def is_non_coplanar(anchors, min_volume: float = 1e-6) -> bool:
    """True iff some four anchors form a tetrahedron with volume > ``min_volume`` (m^3).

    Fewer than four anchors can never be non-coplanar.  Four non-coplanar anchors are the minimum for an
    unambiguous 3D range fix: with all anchors in one plane the mirror image across that plane has exactly
    the same ranges (see ``mirror_point``).
    """
    a = _as_anchors(anchors)
    if a.shape[0] < 4:
        return False
    for i, j, k, m in itertools.combinations(range(a.shape[0]), 4):
        if tetrahedron_volume(a[i], a[j], a[k], a[m]) > min_volume:
            return True
    return False


# ----------------------------------------------------------------------------------------------- GDOP / CRLB

def range_jacobian(anchors, p) -> np.ndarray:
    """(M, 3) Jacobian of the range vector ||p - a_j|| with respect to p: unit vectors (p - a_j) / ||p - a_j||.

    A zero distance (p on top of an anchor) is guarded with 1e-12, giving a zero row.
    """
    a = _as_anchors(anchors)
    q = _as_point(p)
    diff = q - a
    dist = np.linalg.norm(diff, axis=1)
    dist = np.where(dist < _ZERO_DISTANCE, _ZERO_DISTANCE, dist)
    return diff / dist[:, None]


def gdop(anchors, p) -> float:
    """Geometric dilution of precision  sqrt(trace((J^T J)^-1))  of the range set at p.

    Position RMSE of an efficient estimator ~ GDOP x ranging sigma.  Returns ``inf`` when J^T J is singular
    (condition number > 1e12), e.g. for coplanar anchors at a point in their plane or fewer than three
    independent range directions.
    """
    j = range_jacobian(anchors, p)
    inv = _safe_inverse(j.T @ j)
    if inv is None:
        return float("inf")
    return float(math.sqrt(max(float(np.trace(inv)), 0.0)))


def crlb(anchors, p, sigma) -> CRLB:
    """Cramer-Rao lower bound at p for i.i.d. Gaussian ranging noise with standard deviation ``sigma`` (m).

    Fisher information  FIM = J^T J / sigma^2,  cov = FIM^-1,  rmse_bound = sqrt(trace(cov)) = sigma x GDOP,
    per_axis = sqrt(diag(cov)).  A singular geometry gives a CRLB filled with ``inf``.
    """
    s = _as_sigma(sigma)
    j = range_jacobian(anchors, p)
    cov = _safe_inverse(j.T @ j / (s * s))
    if cov is None:
        return CRLB(np.full((3, 3), np.inf), float("inf"), np.full(3, np.inf))
    per_axis = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return CRLB(cov, float(math.sqrt(max(float(np.trace(cov)), 0.0))), per_axis)


# ----------------------------------------------------------------------------------------------- flip ambiguity

def anchor_plane(anchors) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares plane through the anchors: ``(centroid, unit normal)``.

    The normal is the right-singular vector of the smallest singular value of the centred anchor matrix
    (total least squares).  Its sign is chosen so that it points upwards (positive z component; for a vertical
    plane positive y, then x), which makes ``dot(p - centroid, normal)`` the signed height of p above the plane.
    """
    a = _as_anchors(anchors)
    if a.shape[0] == 0:
        raise ValueError("at least one anchor is required")
    centroid = a.mean(axis=0)
    _, _, vt = np.linalg.svd(a - centroid)
    normal = vt[-1].copy()
    for axis in (2, 1, 0):
        if abs(normal[axis]) > 1e-9:
            if normal[axis] < 0.0:
                normal = -normal
            break
    return centroid, normal / np.linalg.norm(normal)


def mirror_point(anchors, p) -> np.ndarray | None:
    """Reflection of p across the least-squares anchor plane:  p - 2 ((p - c) . n) n.

    This is the "flipped" solution of the localization problem: for perfectly coplanar anchors it has exactly
    the same ranges as p, for nearly coplanar anchors almost the same.  Returns ``None`` when the anchors do
    not define a plane (``anchor_rank`` < 2).
    """
    a = _as_anchors(anchors)
    q = _as_point(p)
    if anchor_rank(a) < 2:
        return None
    centroid, normal = anchor_plane(a)
    return q - 2.0 * float(np.dot(q - centroid, normal)) * normal


def flip_ambiguity_risk(anchors, p, measured) -> float:
    """How well the measured ranges separate the candidate p from its mirror image across the anchor plane.

    With  f(q) = sum_j (||q - a_j|| - d_j)^2  the risk is

        risk = clip( f(p) / max(f(mirror), 1e-12), 0, 1 )

    i.e. the ratio of the fitness at the candidate to the fitness at its mirror, clipped to [0, 1].  Equal fitness
    on both sides gives 1 (indistinguishable - exactly the case for perfectly coplanar anchors); a mirror that
    fits the ranges far worse than the candidate gives a value near 0.  The ratio exceeds 1 (and is clipped)
    when the candidate itself is the worse side.  Returns 1.0 when no mirror exists (``mirror_point`` is None).

    This is the definition shared with the browser implementation.  Note that the ratio is only meaningful for a
    candidate that fits the ranges reasonably well (an estimate or the true position); a noise-free scenario in
    which both sides fit perfectly is degenerate (0 / 1e-12).
    """
    a = _as_anchors(anchors)
    q = _as_point(p)
    d = np.asarray(measured, dtype=float).reshape(-1)
    if d.shape[0] != a.shape[0]:
        raise ValueError(f"expected {a.shape[0]} measured ranges, got {d.shape[0]}")
    mirror = mirror_point(a, q)
    if mirror is None:
        return 1.0
    f_p = _range_fitness(a, d, q)
    f_m = _range_fitness(a, d, mirror)
    return float(min(1.0, max(0.0, f_p / max(f_m, _FITNESS_FLOOR))))


def flip_risk_hint(anchors, p, sigma) -> float:
    """Measurement-free flip indicator for a node at p under Gaussian ranging noise ``sigma``.

    It is the ratio of the *expected* fitness values of ``flip_ambiguity_risk``:  E f(p) = M sigma^2 and
    E f(mirror) = M sigma^2 + f0(mirror)  with  f0(mirror) = sum_j (||mirror - a_j|| - ||p - a_j||)^2, hence

        hint = M sigma^2 / (M sigma^2 + f0(mirror))          in (0, 1]

    1 = the mirror image produces identical ranges (perfectly coplanar anchors, or no plane at all), values near
    0 = the two sides are far apart compared with the noise.  Used by ``scenario_report`` when no measured ranges
    are available.
    """
    a = _as_anchors(anchors)
    q = _as_point(p)
    s = _as_sigma(sigma)
    mirror = mirror_point(a, q)
    if mirror is None:
        return 1.0
    noise_power = a.shape[0] * s * s
    f0 = _range_fitness(a, np.linalg.norm(q - a, axis=1), mirror)
    return float(noise_power / (noise_power + f0))


# ----------------------------------------------------------------------------------------------- field maps

def _rmse_bounds(anchors: np.ndarray, points: np.ndarray, sigma: float) -> np.ndarray:
    """Vectorised ``crlb(...).rmse_bound`` for (P, 3) points; ``inf`` where the geometry is singular."""
    diff = points[:, None, :] - anchors[None, :, :]                      # (P, M, 3)
    dist = np.linalg.norm(diff, axis=2)
    dist = np.where(dist < _ZERO_DISTANCE, _ZERO_DISTANCE, dist)
    jac = diff / dist[:, :, None]
    fim = np.einsum("pmi,pmj->pij", jac, jac) / (sigma * sigma)          # (P, 3, 3)
    with np.errstate(all="ignore"):
        cond = np.linalg.cond(fim)
    ok = np.isfinite(cond) & (cond <= _COND_LIMIT)
    out = np.full(points.shape[0], np.inf)
    if ok.any():
        try:
            inv = np.linalg.inv(fim[ok])
            out[ok] = np.sqrt(np.maximum(np.einsum("pii->p", inv), 0.0))
        except np.linalg.LinAlgError:                                     # fall back to the scalar path
            out[ok] = [crlb(anchors, pt, sigma).rmse_bound for pt in points[ok]]
    return out


def coverage_grid(anchors, field_size, sigma, step):
    """CRLB position bound on a regular grid over the field ``[0, Lx] x [0, Ly] x [0, Lz]``.

    Returns ``(xs, ys, zs, rmse_bound)`` with axes ``np.arange(0, L + 1e-9, step)`` and
    ``rmse_bound[iz, iy, ix] = crlb(anchors, (xs[ix], ys[iy], zs[iz]), sigma).rmse_bound``.
    ``field_size`` may be a scalar (cube) or three lengths.
    """
    a = _as_anchors(anchors)
    s = _as_sigma(sigma)
    size = np.asarray(field_size, dtype=float).reshape(-1)
    if size.size == 1:
        size = np.repeat(size, 3)
    if size.shape != (3,) or not np.all(size >= 0.0):
        raise ValueError(f"field_size must be a scalar or three non-negative lengths, got {field_size!r}")
    h = float(step)
    if not h > 0.0:
        raise ValueError(f"step must be positive, got {step!r}")
    xs, ys, zs = (np.arange(0.0, length + 1e-9, h) for length in size)
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    grid = _rmse_bounds(a, points, s).reshape(zs.size, ys.size, xs.size)
    return xs, ys, zs, grid


def scenario_report(anchors, p, sigma, measured=None) -> dict:
    """Plain-Python summary of the geometry at p (for the CLI, README tables and JSON dumps).

    Keys: ``rank``, ``non_coplanar``, ``gdop``, ``crlb_rmse``, ``crlb_per_axis``, ``mirror`` (list or None),
    ``flip_risk_hint``; when ``measured`` ranges are given, ``flip_risk`` (``flip_ambiguity_risk``) is added.
    """
    a = _as_anchors(anchors)
    q = _as_point(p)
    bound = crlb(a, q, sigma)
    mirror = mirror_point(a, q)
    report = {
        "rank": anchor_rank(a),
        "non_coplanar": is_non_coplanar(a),
        "gdop": gdop(a, q),
        "crlb_rmse": bound.rmse_bound,
        "crlb_per_axis": [float(v) for v in bound.per_axis],
        "mirror": None if mirror is None else [float(v) for v in mirror],
        "flip_risk_hint": flip_risk_hint(a, q, sigma),
    }
    if measured is not None:
        report["flip_risk"] = flip_ambiguity_risk(a, q, measured)
    return report

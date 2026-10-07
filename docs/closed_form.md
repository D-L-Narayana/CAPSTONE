# Closed-form estimators: least squares, Gauss-Newton, Levenberg-Marquardt, centroids

Modules: `pso3d/trilateration.py`, `pso3d/centroid.py`. Tests: `tests/test_trilateration.py`,
`tests/test_centroid.py`. Reference outputs: `tests/fixtures/closed_form_cases.json`.

These estimators are the *comparison methods* of the project (the swarm optimisers are the subject).
They are also used as a warm start for the swarms (`warm_start_estimate`) and as the reference
`lsq` / `lsq_gn` methods of the Monte-Carlo study. Everything here is plain numpy.

Notation: anchors `a_j` (j = 1..M, rows of an (M, 3) array), unknown node `p`, measured ranges
`d̂_j = ‖p − a_j‖ + n_j`.

## 1. Linearised least squares (`least_squares_trilateration`)

Each range defines a sphere `‖p − a_j‖² = d̂_j²`. Subtracting the sphere of the reference anchor
`a_1` from the sphere of anchor `j` removes the quadratic term `‖p‖²` and leaves one *linear*
equation per remaining anchor:

```
A_j · p = b_j,   A_j = 2 (a_j − a_1),   b_j = ‖a_j‖² − ‖a_1‖² − d̂_j² + d̂_1²      (j = 2..M)
```

`A` is (M−1) × 3, so with M = 4 anchors it is one 3 × 3 solve; with more anchors the system is
over-determined and solved in the least-squares sense with `numpy.linalg.lstsq` (minimum-norm
solution when `A` is rank-deficient). No iteration, no starting point, zero fitness evaluations.

The equations are linear in `p` but the noise enters through `d̂_j²`, so the estimate is **not**
the minimiser of the range-error fitness `f(p) = Σ_j (‖p − a_j‖ − d̂_j)²`; it is a biased but
cheap first guess (3.20 m on the baseline, versus 0.96 m for the fitness minimiser).

### Weighted rows (`weights=`)

`least_squares_trilateration(anchors, measured, weights=w)` multiplies row `j` of `A` and `b` by
`√w_j` (j = 2..M), i.e. it minimises `Σ_j w_j (A_j p − b_j)²`. `w` has one non-negative, finite
entry per anchor; a zero removes that anchor's equation; multiplying all weights by a constant
does not change the result; `weights=None` or unit weights reproduce the unweighted solution
exactly. The reference anchor's equation is the one being subtracted, so `w_1` has no effect —
if the reference range itself is suspect, reorder the anchors. Typical weights are
`1/σ_j²` from a noise model or the robust weights of `pso3d.robust`.

## 2. Gauss-Newton refinement (`gauss_newton_refine`, `gauss_newton`)

Gauss-Newton minimises the actual range-error fitness. With residuals and Jacobian

```
r_j(p) = ‖p − a_j‖ − d̂_j,        J_j(p) = (p − a_j) / ‖p − a_j‖          (unit vectors, J is M × 3)
```

each iteration solves the linearised problem `min ‖J δ + r‖` for the step `δ` (`lstsq`, minimum
norm) and sets `p ← p + δ`. Iteration stops after `iterations` steps or when `‖δ‖ < tol`.
Distances below 1e-12 are clamped so a start exactly on an anchor is still defined.

* `gauss_newton_refine(anchors, measured, p0, iterations=10, tol=1e-9) -> (p, used)` is the
  historical function; its outputs are frozen.
* `gauss_newton(anchors, measured, p0, iterations=10, tol=1e-9, damping=0.0) -> GNResult`
  produces exactly the same iterates and adds diagnostics:

| `GNResult` field | meaning |
|---|---|
| `position` | final estimate |
| `iterations` | steps taken |
| `converged` | last step shorter than `tol` (False when the step budget ran out) |
| `residual_norm` | `‖r(position)‖ = √f(position)` |
| `evaluations` | residual/Jacobian evaluations: one per step plus the final residual (`iterations + 1`) |

`damping=λ > 0` replaces the step by the Levenberg step `(JᵀJ + λI) δ = −Jᵀr` (solved through the
equivalent augmented least-squares system). The step is shorter and biased towards steepest
descent, which helps from poor starts at the price of slower final convergence (on the noise-free
baseline, λ = 1e-3 needs 8 steps instead of 5).

Gauss-Newton is *local*: from a start that is far away or badly placed relative to the anchor
geometry it can diverge (see §6) — hence the warm-start recipe below and the Levenberg-Marquardt
variant.

## 3. Levenberg-Marquardt (`levenberg_marquardt`)

`levenberg_marquardt(anchors, measured, p0, iterations=20, lam0=1e-2) -> GNResult` is the classic
adaptive version:

1. compute `r`, `J` at the current point and propose the damped step for the current `λ`;
2. if the candidate lowers `‖r‖`, accept it and set `λ ← λ/10`;
3. otherwise set `λ ← 10 λ` and re-propose (as `λ` grows the step tends to a short
   steepest-descent step); give up at `λ > 1e12`.

Only improving steps are accepted, so the returned position **never has a larger residual than
the start** (`iterations=0` simply returns the start and its residual). `converged` is True when an
accepted step was shorter than `tol`, when the gradient `‖Jᵀr‖` fell below `tol` (already at a
stationary point — a start at the exact minimiser is returned unchanged with 0 iterations), or when
no admissible `λ` could reduce the residual (a minimum in working precision). `evaluations` counts
the initial evaluation plus every candidate, accepted or rejected; `iterations` counts accepted
steps only.

On the baseline, LM from the least-squares start reaches the same minimiser as Gauss-Newton
(residual norm 0.2380 m, estimate (37.42, 11.88, 9.36)). From deliberately bad starts such as
(200, −100, 50) or (−50, −50, −50), where plain Gauss-Newton diverges or lands in a wrong basin,
LM still returns the minimiser.

## 4. Warm start for the swarms (`warm_start_estimate`)

`warm_start_estimate(anchors, measured, lower, upper) -> np.ndarray`:

1. ignore anchors whose range or position is not finite;
2. if at least four usable anchors remain and the linear system has `matrix_rank(A) = 3`, take the
   least-squares solution as the start (unless it is non-finite);
3. otherwise start from the inverse-distance **weighted centroid** (§5), which always lies inside
   the anchors' bounding box;
4. refine with 10 Gauss-Newton steps (fall back to the start if that produced non-finite values);
5. clip to `[lower, upper]`.

The result is finite and inside the box for any input with a finite box. On the baseline it equals
LSQ + Gauss-Newton, (37.42, 11.88, 9.36). The optimisers take it through their `x0` argument
(particle 0 is placed at `x0` with zero velocity; the random stream is unchanged — see
`docs/stopping_and_warm_start.md`); the experiments registry exposes this as the `*_ws` methods.

## 5. Centroids (`centroid`, `weighted_centroid`)

* `centroid(anchors)` — mean of the anchor positions. Range-free; the error is simply the distance
  from the node to the anchor centroid.
* `weighted_centroid(anchors, measured, power=1.0)` — `p = Σ w_j a_j / Σ w_j` with
  `w_j = 1 / d̂_j^power`. Shorter ranges pull harder; `power = 0` gives the plain centroid, a large
  `power` snaps to the nearest anchor. Ranges ≤ 1e-12 are clamped (a node sitting on an anchor
  returns that anchor). The weights are evaluated as `(d̂_min / d̂_j)^power`, identical up to a common
  factor, so large powers cannot overflow. The estimate always lies inside the convex hull of the
  anchors — good as a fallback, poor as an estimator when the node is near the field boundary.

## 6. When least squares fails: geometry

The linear system has `A_j = 2 (a_j − a_1)`, so its rank is the dimension of the affine span of the
anchors:

| anchor configuration | rank of `A` | effect |
|---|---|---|
| ≥ 4 anchors spanning a volume (non-coplanar) | 3 | unique solution |
| all anchors in one plane | 2 | `lstsq` returns the minimum-norm solution, which lies in the anchor plane; the height is undetermined and the two mirror images across the plane fit the ranges equally well (flip ambiguity) |
| all anchors on one line | 1 | only the coordinate along the line is determined |
| fewer than 4 anchors | ≤ 2 | under-determined |

Near-degenerate geometry is the practical problem: with anchors at heights 0, 0, 0 and 0.01 m the
rank is still 3 numerically, but the least-squares height is amplified by the inverse of the
smallest singular value of `A` — on such a case LSQ returns z ≈ 356 m for a node at 12 m. The
warm-start helper survives this (Gauss-Newton pulls the estimate back and the clip bounds it), but
the quality of *any* range-based estimate is governed by the anchor geometry: the Jacobian `J`
is the same matrix that defines GDOP and the Cramér-Rao bound (`pso3d.geometry`, `docs/geometry.md`).
For the baseline, anchor heights span only 0–20 m against 60 m in x and y, which is why the
vertical error dominates every method (Δz = 0.79 m of the 0.90 m PSO error).

Gauss-Newton inherits the degeneracy: in the anchor plane the z-column of `J` is zero, so the
height never moves; from a start on the wrong side of the plane it converges to the mirror image.
Ranges alone cannot resolve this — only a non-coplanar anchor set (or prior knowledge such as the
field bounds) can.

## 7. Pinned examples

All numbers below are computed by this module (tests pin them to two decimals; the fixture stores
the full-precision values and is compared at 1e-9).

**Baseline (slide 16).** Anchors (0,0,0), (60,0,5), (0,60,6), (60,60,20); node (37, 12, 8.5);
noise (+0.4, −0.3, +0.5, −0.4) m.

| method | estimate (m) | error (m) | notes |
|---|---|---|---|
| least squares | (37.13, 11.44, 11.65) | 3.20 | one (M−1)×3 solve, residual norm 0.7349 |
| + Gauss-Newton | (37.42, 11.88, 9.36) | 0.96 | 10 steps, step budget exhausted (`converged=False`), residual norm 0.2380 |
| + Levenberg-Marquardt | (37.42, 11.88, 9.36) | 0.96 | same minimiser; residual never above the start |
| warm start | (37.42, 11.88, 9.36) | 0.96 | = LSQ + GN, already inside the field |

With noise-free ranges least squares is exact and Gauss-Newton started 1 m off in every axis
converges (step < 1e-9) in 5 iterations.

**Code 2 of the "10 codes" document (trilateration with six anchors).** Anchors (0,0,0), (60,0,4),
(0,60,6), (60,60,9), (30,30,20), (10,45,12); node (31.0, 17.5, 7.2); fixed noise
(+0.5, −0.4, +0.6, −0.3, +0.4, −0.5) m. Least squares gives (31.50, 17.48, 7.63), error 0.66 m;
Gauss-Newton lowers the range residual further (1.04 → 0.88) and ends at 0.65 m error.
With one range corrupted by +8 m, giving that anchor a weight of 1e-6 recovers the node to < 1 mm
(unweighted error 12.4 m).

**Code 5 of the same document (weighted centroid).** Anchors (0,0,2), (60,0,4), (0,60,3),
(60,60,6), (30,30,18); node (16, 38, 5); perfect ranges. Plain centroid (30.00, 30.00, 6.60),
error 16.20 m; weighted centroid (25.23, 33.19, 8.31), error 10.92 m.

The fixture `tests/fixtures/closed_form_cases.json` holds the measured ranges and the outputs of all
estimators for these three cases; `tests/test_trilateration.py` regenerates it only when the file is
absent and otherwise asserts agreement at 1e-9, so a change in any estimator shows up as a test
failure rather than as a silently rewritten fixture. Levenberg-Marquardt is the one exception: its
last accepted steps are at the 1e-9 level, so the exact final iterate — and the number of accepted
and rejected steps — depends on the last bits of the linear-algebra library, while its residual
norm is reproducible to about 1e-13. The fixture therefore stores the LM residual norm, convergence
flag and position only, and compares the position at 1e-6; `GNResult.iterations` and
`GNResult.evaluations` of `levenberg_marquardt` are diagnostics, not pinned quantities.

## 8. API summary

```python
from pso3d.trilateration import (least_squares_trilateration, gauss_newton_refine, gauss_newton,
                                 levenberg_marquardt, warm_start_estimate, GNResult)
from pso3d.centroid import centroid, weighted_centroid

least_squares_trilateration(anchors, measured, weights=None) -> np.ndarray
gauss_newton_refine(anchors, measured, p0, iterations=10, tol=1e-9) -> (np.ndarray, int)
gauss_newton(anchors, measured, p0, iterations=10, tol=1e-9, damping=0.0) -> GNResult
levenberg_marquardt(anchors, measured, p0, iterations=20, lam0=1e-2, tol=1e-9) -> GNResult
warm_start_estimate(anchors, measured, lower, upper) -> np.ndarray
centroid(anchors) -> np.ndarray
weighted_centroid(anchors, measured, power=1.0) -> np.ndarray
```

Inputs may be numpy arrays or nested lists; all functions return float arrays of shape (3,).

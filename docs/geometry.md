# Anchor geometry diagnostics (`pso3d.geometry`)

The Monte-Carlo study in the README ends with an observation that is easy to state and hard to act on:
most of the localization error is vertical, and the heavy error tail "is a geometry problem — nodes far
from the anchors and the small height spread (0–20 m) produce weak vertical geometry and flip ambiguity
along z". `pso3d/geometry.py` turns that remark into numbers. Everything in it is plain numpy; the browser
simulator carries the same definitions in `web/geometry.js` (functions `tetrahedronVolume`, `anchorRank`,
`isNonCoplanar`, `rangeJacobian`, `gdop`, `crlb`, `mirrorPoint`, `flipAmbiguityRisk`) so both can be checked
against the same fixtures.

Notation used below: anchors `a_j` (j = 1 … M, an M × 3 array), node or candidate position `p`, measured
ranges `d̂_j = ‖p − a_j‖ + n_j`, ranging noise standard deviation `σ`, and the range-error fitness that all
optimisers in this repository minimise,

```
f(q) = Σ_j ( ‖q − a_j‖ − d̂_j )²
```

The running example is the slide-16 scenario: field 60 × 60 × 20 m, anchors
`(0,0,0) (60,0,5) (0,60,6) (60,60,20)`, true node `(37, 12, 8.5)`, fixed noise `(0.4, −0.3, 0.5, −0.4)` m.

## 1. Is the anchor layout really three-dimensional?

| function | definition |
|---|---|
| `tetrahedron_volume(a, b, c, d)` | `V = │det[b − a, c − a, d − a]│ / 6` (a regular tetrahedron with edge `s` has `V = s³ / (6√2)`) |
| `anchor_rank(anchors, tol=1e-9)` | numerical rank of the centred matrix `A − mean(A)`; a singular value counts when it exceeds `tol × s_max` |
| `is_non_coplanar(anchors, min_volume=1e-6)` | `True` iff some four anchors form a tetrahedron with `V > min_volume` m³; fewer than four anchors → `False` |

Rank 3 means the anchors span a volume, 2 means they lie in a plane, 1 on a line, 0 on a point. The
tolerance is relative, so the answer does not change with the unit of length. The baseline anchors have
rank 3 and a tetrahedron volume of 5 400 m³; the same corners with every height set to 0 have rank 2 and
volume 0.

Why four *non-coplanar* anchors are the minimum: three spheres intersect in two points that are mirror
images of each other across the plane through the three centres. A fourth sphere separates the two only
if its centre is off that plane — the further off, the more clearly.

## 2. Range Jacobian and GDOP

The derivative of the range vector with respect to the position is the matrix of unit vectors pointing
from each anchor to the node,

```
J_j = (p − a_j) / ‖p − a_j‖            (M × 3; a zero distance is guarded with 1e-12)
```

For small ranging errors `δd` the position error of a least-squares fix is `δp ≈ (JᵀJ)⁻¹ Jᵀ δd`. The
geometric dilution of precision summarises how much the geometry amplifies the ranging noise:

```
GDOP(p) = sqrt( trace( (JᵀJ)⁻¹ ) )      position RMSE ≈ GDOP × σ
```

`gdop()` returns `inf` when `JᵀJ` is singular (2-norm condition number above 1e12): coplanar anchors seen
from a point inside their plane, collinear anchors, or fewer than three independent range directions. At
the baseline true node the GDOP is **5.29**; one metre of ranging noise becomes about five metres of
position uncertainty there.

## 3. Cramér–Rao lower bound

For independent Gaussian ranging noise with standard deviation `σ` the Fisher information of the position is

```
FIM = JᵀJ / σ²,      cov ⪰ FIM⁻¹  for every unbiased estimator,
rmse_bound = sqrt(trace(FIM⁻¹)) = σ × GDOP,      per_axis = sqrt(diag(FIM⁻¹))
```

`crlb(anchors, p, sigma)` returns a `CRLB(cov, rmse_bound, per_axis)` dataclass (all `inf` for a singular
geometry; `sigma` must be positive). The bound is linear in `σ`: doubling the noise doubles every entry.

Baseline true node `(37, 12, 8.5)`, `σ = 0.5` m:

| quantity | value |
|---|---|
| GDOP | 5.293 |
| `rmse_bound` | 2.646 m |
| `per_axis` (x, y, z) | 0.438 m, 0.596 m, **2.541 m** |

The vertical bound is almost six times the horizontal ones. The anchors sit at heights 0, 5, 6 and 20 m
across a 60 m × 60 m footprint, so every unit vector `J_j` is nearly horizontal and the range set carries
little information about `z`. This is the formal version of the README's note that "most of the error is
vertical (Δz = 0.79 m) because anchor heights span only 0–20 m against 60 m in x and y".

Two cautions when reading the bound:

* It bounds the root-mean-square error over *random* noise of level `σ`. The slide-16 run uses one fixed
  noise vector whose entries are 0.3–0.5 m, and its single error of 0.90 m is not in contradiction with a
  2.65 m bound for `σ = 0.5` m.
* With `σ = 0.1` m the bound at the true node is 0.53 m; twenty noisy trials of least squares followed by
  Gauss-Newton give a sample RMSE within the sampling scatter of that value (the test suite asserts
  `0.8 × bound ≤ RMSE ≤ 1.5 × bound`). Non-Gaussian errors (NLOS bias, outliers) break the assumptions
  and the bound no longer applies.

## 4. Flip ambiguity

### The least-squares anchor plane and the mirror image

`anchor_plane(anchors)` returns the centroid `c = mean(a_j)` and the unit normal `n` of the total-least-squares
plane: `n` is the right-singular vector of the smallest singular value of `A − c`, oriented so that it points
upwards (`n_z > 0`). For the baseline the anchors lie within ±2.2 m of a plane that is tilted about 13° from
horizontal and passes through `(30, 30, 7.75)`; the singular values are 61.7, 60.0 and 4.38 — the third one is
the "thickness" of the layout.

`mirror_point(anchors, p)` reflects a candidate across that plane,

```
p' = p − 2 ((p − c) · n) n          (None when anchor_rank < 2, i.e. no plane exists)
```

For perfectly coplanar anchors `‖p' − a_j‖ = ‖p − a_j‖` for every j: the ranges cannot distinguish `p` from
`p'`, and any range-based estimator may land on either side. For nearly coplanar anchors the ranges differ
only slightly. The baseline true node is 2.73 m above the plane; its mirror image `(37.84, 12.93, 3.19)` is
5.45 m away, yet its four ranges differ from the true ranges by only −0.46 … +0.31 m — the same size as the
ranging noise.

### Measuring the risk

```
flip_ambiguity_risk(anchors, p, measured) = clip( f(p) / max(f(p'), 1e-12), 0, 1 )
```

the ratio of the fitness at the candidate to the fitness at its mirror image, clipped to [0, 1]; 1.0 when no
mirror exists. Equal fitness on both sides gives 1 (indistinguishable — exactly the case for perfectly
coplanar anchors); a mirror that fits the ranges far worse than the candidate gives a value near 0. If the
candidate itself is the worse side the ratio exceeds 1 and is clipped to 1, so a flipped estimate is reported
as maximal risk. The ratio is meaningful for a candidate that fits the ranges reasonably well (an estimate or
the true position); in a noise-free scenario where both sides fit perfectly the quotient is 0 / 1e-12 and
carries no information.

Baseline values with the slide-16 measured ranges:

| candidate | f(p) | f(mirror) | risk |
|---|---|---|---|
| true node `(37, 12, 8.5)` | 0.660 (= Σ n_j²) | 0.897 | **0.736** |
| Review-1 estimate `(37.43, 11.90, 9.29)` | 0.057 | 0.939 | 0.061 |
| mirror of the Review-1 estimate | 0.939 | 0.057 | 1.0 (clipped) |
| node `(10, 10, 19)`, 17.5 m from the plane, same noise vector | 0.660 | 14.7 | 0.045 |
| any candidate, anchors flattened to z = 0 | equal | equal | 1.0 |

The 0.736 at the true node is the quantitative form of the README remark: the node sits only 2.7 m from the
anchor plane, so the 0–20 m height spread resolves the flip weakly and a wrong-side solution scores almost as
well as the right one. The optimiser's own estimate is nevertheless clearly on the correct side (0.061),
and a node far from the plane is not ambiguous at all (0.045).

`flip_risk_hint(anchors, p, sigma)` is the measurement-free counterpart used by `scenario_report`: with
`f₀(p') = Σ_j (‖p' − a_j‖ − ‖p − a_j‖)²` the expected fitness values under Gaussian noise are
`E f(p) = Mσ²` and `E f(p') = Mσ² + f₀(p')`, so

```
hint = Mσ² / (Mσ² + f₀(p'))        in (0, 1]
```

Baseline: 0.72 at `σ = 0.5` m, 0.09 at `σ = 0.1` m; exactly 1 for coplanar anchors.

### What this means for placing anchors in the 60 × 60 × 20 m field

The same diagnostics at the baseline true node (`σ = 0.5` m) for a few alternative layouts
(`s_min` is the smallest singular value of the centred anchor matrix, i.e. the "thickness" of the layout):

| layout | GDOP | `rmse_bound` | z-bound | `s_min` | `flip_risk_hint` |
|---|---|---|---|---|---|
| baseline corners, heights 0 / 5 / 6 / 20 m | 5.29 | 2.65 m | 2.54 m | 4.4 m | 0.72 |
| corners with alternating heights 0 / 20 / 20 / 0 m | 2.46 | 1.23 m | 1.10 m | 20.0 m | 0.31 |
| baseline + fifth anchor at mid height `(30, 30, 6)` | 3.67 | 1.84 m | 1.76 m | 4.6 m | 0.66 |
| baseline + fifth anchor on the ceiling `(30, 30, 20)` | 2.83 | 1.42 m | 1.27 m | 11.5 m | 0.90 |
| baseline + fifth anchor on the floor `(30, 30, 0)` | 2.33 | 1.17 m | 1.05 m | 8.0 m | 0.14 |

* Check `anchor_rank == 3` and `is_non_coplanar` first; a flat layout makes every point in the anchor plane
  unobservable (`gdop = inf`) and every point above it a coin toss between `p` and `p'`.
* The vertical bound is driven by the spread of anchor heights relative to the footprint, not by the anchor
  count: alternating floor and ceiling corners halve the z-bound with the same four anchors, whereas a fifth
  anchor at the height of the existing ones barely helps.
* Every added anchor moves the least-squares plane and with it the mirror image of every node, so the flip
  hint does not simply fall with more anchors: at the baseline node a fifth anchor on the ceiling raises it to
  0.90, one on the floor lowers it to 0.14. Evaluate `flip_risk_hint` where the nodes are expected.
* Clipping particles to the field only removes wrong-side solutions that fall outside it; for the baseline
  node the mirror image (z = 3.19 m) lies inside the field, so the ambiguity has to be resolved by the ranges
  themselves (more height spread or an anchor well off the plane).
* GDOP and CRLB are local: compute them where the nodes are expected, not only at the field centre.

## 5. Coverage grid and scenario report

```
xs, ys, zs, bound = coverage_grid(anchors, field_size, sigma, step)
bound[iz, iy, ix] == crlb(anchors, (xs[ix], ys[iy], zs[iz]), sigma).rmse_bound
```

The axes are `np.arange(0, L + 1e-9, step)` per dimension (`field_size` may be a scalar or three lengths), so
the grid includes both field boundaries. For the baseline anchors at `σ = 0.5` m and a 10 m step the bound
ranges from 0.79 m to 7.2 m over the field, and for flattened anchors the whole z = 0 layer is `inf`.

`scenario_report(anchors, p, sigma, measured=None)` collects the diagnostics as plain Python types for the
CLI and JSON output:

```json
{"rank": 3, "non_coplanar": true, "gdop": 5.2927, "crlb_rmse": 2.6464,
 "crlb_per_axis": [0.4380, 0.5957, 2.5410], "mirror": [37.8447, 12.9336, 3.1935],
 "flip_risk_hint": 0.7198}
```

When measured ranges are passed, a `flip_risk` entry (`flip_ambiguity_risk`) is added.

## 6. Fixtures and tests

`tests/test_geometry.py` covers the definitions above (volume, rank, GDOP, CRLB, plane, mirror, flip risk,
coverage grid, report) and the Monte-Carlo sanity check. `tests/fixtures/geometry_cases.json` stores GDOP,
CRLB, mirror and flip-risk values of the baseline anchors at the probe points `(37, 12, 8.5)`, `(30, 30, 10)`
and `(5, 55, 2)`; the test writes it once and compares against it on every later run (tolerance 1e-9). The
shared Python/JavaScript parity inputs in `tests/fixtures/parity_inputs.json` exercise the same functions
in both languages.

## 7. Limitations

* The CRLB assumes an unbiased estimator, independent Gaussian noise and a known `σ`; it is a lower bound,
  not a prediction of any particular optimiser's error.
* The flip risk is a fitness ratio, not a probability, and depends on the measured ranges supplied.
* Risk and hint measure how well the ranges separate the two sides, not how far apart they are. The mirror
  image sits at twice the node's distance from the plane: for a node very close to the plane a high value
  means a small positional error, 10 m away from the plane it would mean a gross one.
* The least-squares plane is a single global fit. With many anchors in a strongly three-dimensional layout
  the "mirror image" has no physical meaning — the risk is then simply low.

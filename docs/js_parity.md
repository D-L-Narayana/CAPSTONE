# Python ↔ JavaScript parity of the browser simulator

The browser simulator (`web/`) re-implements the algorithms of the `pso3d` package in plain JavaScript
(`web/pso.js`, `web/geometry.js`): no bundler, no module syntax, ES2020, loadable both as classic browser
scripts and — through a `module.exports` shim at the end of each file — with Node's `require`. This note
records what matches the Python package exactly, what cannot match, and how the parity is tested.

## What matches Python exactly (up to floating-point rounding)

| JavaScript (`web/`) | Python (`pso3d/`) | Checked by |
|---|---|---|
| `Fitness.f(p)` — range-error fitness Σⱼ (‖p − aⱼ‖ − d̂ⱼ)² with evaluation / distance counters | `fitness.RangeErrorFitness` | `tests/test_parity.py` (rel 1e-6), `tests/js/pso.test.js` |
| `leastSquares(anchors, d)` — linearised trilateration (normal equations, 3×3 solve) | `trilateration.least_squares_trilateration` (`lstsq`) | parity test (abs/rel 1e-6); baseline pin (37.1345, 11.4443, 11.6490) |
| `gaussNewton(anchors, d, p0, iterations=10, tol=1e-9)` → `{p, iterations, converged}` — same residual/Jacobian, same iteration counting (`used = k + 1`, stop when ‖step‖ < tol), zero distances clamped at 1e-12 | `trilateration.gauss_newton_refine` | parity test on `p` and `iterations`; baseline pin (37.4211, 11.8843, 9.3592), 10 iterations |
| `warmStart(anchors, d, lower, upper)` — LSQ → GN → clip; the plain anchor centroid replaces a singular LSQ start (Python falls back to the inverse-distance weighted centroid and skips non-finite anchors — only degenerate inputs differ, none of them in the parity fixture) | `trilateration.warm_start_estimate` | `tests/js/closed_form.test.js` |
| `tetrahedronVolume`, `anchorRank`, `isNonCoplanar`, `rangeJacobian`, `gdop`, `crlb`, `mirrorPoint`, `flipAmbiguityRisk` | `geometry.py` (same formulas, see `docs/geometry.md`) | parity test: `gdop`, `crlb.rmseBound` (rel 1e-5), `mirror` (1e-6), `non_coplanar` |
| `amcmpsoSchedule(T, opts)` — w, c1, c2 per iteration with `frac = t / T`, linear or cosine | `amcmpso.coefficient_schedule` | parity test (abs 1e-12) |
| `centreOfMass(x, f)` — weights 1 / (fᵢ + 1e-9) | `amcmpso.centre_of_mass` | parity test (rel 1e-6) |
| Evaluation counts: simplified N·T, standard and AMCMPSO N·(T+1); memory 6N, 9N+3, 9N+6 floats | `SimplifiedPSO`, `StandardPSO`, `AMCMPSO` | `tests/js/pso.test.js` |

The only numerical difference in the closed-form solvers is the linear-algebra route: JavaScript solves the
3×3 normal equations with partial pivoting (`solve3`), Python uses `numpy.linalg.lstsq`. On the shared
fixtures the results agree to about 1e-12 (far inside the 1e-6 parity tolerance); on rank-deficient anchor
sets (all anchors on a line) `lstsq` returns a minimum-norm solution whereas `solve3` reports NaN, which
`warmStart` turns into the anchor-centroid fallback. `gaussNewton` ends its loop early when the normal
equations become singular (returned `converged = false`).

The geometry functions use a small Jacobi eigen-solver for the symmetric 3×3 matrices (scatter matrix of the
anchors, JᵀJ) instead of NumPy's SVD/`inv`; `gdop`/`crlb` report `Infinity` when the smallest eigenvalue of
JᵀJ is ≤ 1e-12 × the largest (2-norm condition number above 1e12 — the same threshold as the Python module),
and a singular `crlb` carries a 3×3 `cov` of `Infinity` like the Python `CRLB`. `anchorRank` compares the
*singular values* of the centred anchor matrix with `tol × s_max` exactly as `numpy.linalg.matrix_rank` does in
`anchor_rank`; the singular values are recovered from the scatter eigenvectors as root-sum-square projections
of the centred anchors, which keeps a tiny out-of-plane spread accurate instead of losing it to rounding in the
squared eigenvalue (a 1e-8 m height step over a 60 m square is rank 2 at `tol = 1e-9` in both implementations).

## What cannot match: random streams

The browser uses the seeded `mulberry32` generator; Python uses `numpy.random.default_rng(seed)`. The two
streams are unrelated, so **particle trajectories, per-iteration fitness histories and final estimates of the
swarm variants are not bit-identical between the simulator and the Python package** — only their statistics
are comparable (the baseline scenario, seed 1, N = 20, T = 60 gives an error of 0.90 m in Python and
1.03 m in the browser for the simplified variant; both are within the spread seen over seeds). The
`mulberry32` implementation itself and the draw order of the simplified variant are pinned in
`tests/js/pso.test.js` (first draws of `mulberry32(1)`, final best fitness 0.06639094051757627 and estimate of
the seed-1 baseline run captured before this change), so a browser run stays repeatable release after release.

Draw order per iteration (one `rand()` call per random factor, coordinate-major within a particle):

| variant | draws per particle and coordinate | Python counterpart |
|---|---|---|
| simplified | `r` | `rng.random((N, 3))` once per iteration |
| standard | `r1`, then `r2` | `rng.random(x.shape)` twice per iteration |
| amcmpso | `r1`, `r2`, `r3`, `r4` | four `rng.random(x.shape)` arrays per iteration |

Noise sampling (`sampleNoise`) is likewise seeded with `mulberry32`; the models (`fixed`, `gaussian`,
`uniformPercent`, `nlos`) use the same formulas as `pso3d/noise.py` but produce different samples.

## Behaviour change: the standard variant is now synchronous

Before this change the browser's standard (pbest/gbest) variant updated the global best *inside* the particle
loop, so particles later in the loop were already pulled towards a best found in the same iteration
(asynchronous update). `pso3d/standard_pso.py` and `pso3d/amcmpso.py` are synchronous: all particles move,
all are evaluated, then personal bests and finally the global best are updated. `web/pso.js` now follows the
Python order for both the standard and the AMCMPSO variant. Consequences:

* the per-iteration best fitness is still non-increasing and the evaluation count is unchanged (N·(T+1));
* the seed-1 baseline run of the standard variant changed from (37.419, 11.882, 9.364), error 0.968 m
  (asynchronous) to (37.420, 11.883, 9.361), error 0.965 m (synchronous) — different trajectory, same quality;
* the simplified variant is untouched (pinned).

## New swarm options (shared contract with the Python package)

`new Swarm(cfg, fitness)` accepts `variant: 'simplified' | 'standard' | 'amcmpso'`, `stopping`, `x0` and
`amcmpso` options; `swarm.step()` returns `false` when the run is finished, `swarm.done` and `swarm.stoppedBy`
(`'patience' | 'tolerance' | 'budget' | 'target' | null`) report why. Rules are checked after each iteration:

| rule (`stoppedBy` = the `type`) | fires when | Python class (`PSOResult.stopped_by` name) |
|---|---|---|
| `{type:'patience', patience}` | best-ever fitness has not improved by more than 1e-12 for `patience` consecutive iterations (`patience: 0` fires after the first iteration) | `stopping.Patience` (`"patience"`) |
| `{type:'tolerance', tol, window, relative}` | over the last `window` iterations the best-ever fitness improved by ≤ `tol·f_old` (relative) or ≤ `tol` (absolute); needs `window + 1` iterations | `stopping.Tolerance` (`"tolerance"`) |
| `{type:'budget', evaluations}` | `fitness.evals ≥ evaluations` (exact: N = 20, budget 600 → 30 simplified / 29 standard iterations) | `stopping.MaxEvaluations` (`"max_evaluations"`) |
| `{type:'target', fitness}` | `bestF ≤ fitness` | `stopping.TargetFitness` (`"target_fitness"`) |

The browser reports the short `type` names in `swarm.stoppedBy`; the Python result carries the rule's `name`
shown in parentheses. Numeric rule parameters may arrive as numeric strings from form fields; anything else
malformed (unknown `type` or `variant`, non-numeric parameters, an `x0` that is not three finite numbers)
throws a `TypeError`/`RangeError` when the `Swarm` is constructed.

`x0` injects a warm start: after the uniform initial draw, particle 0 is placed at `clip(x0, lower, upper)`
with zero velocity, leaving the random stream untouched (the Python optimisers do the same).

## Running the checks

```
node --check web/pso.js web/geometry.js       # syntax
node --test tests/js/                          # Node ≥ 20 built-in runner: pso, closed_form, geometry (+ share, export)
node tests/js/dump.js tests/fixtures/parity_inputs.json   # JSON consumed by tests/test_parity.py
python3 -m pytest -q tests/test_parity.py      # Python ↔ JS comparison on the shared fixture
```

`dump.js` prints one JSON object keyed by case id (`baseline`, `near_coplanar`, `six_anchors`) with
`measured`, `lsq`, `gn`, `fitness[]`, `gdop[]`, `crlb_rmse[]`, `mirror[]`, `non_coplanar`, `schedule` and
`centre_of_mass`; non-finite numbers are written as the strings `"Infinity"` / `"NaN"` because JSON has no
representation for them.

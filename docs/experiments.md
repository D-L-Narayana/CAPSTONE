# Monte-Carlo experiments: method registry, CRLB reference and the study CLI

`pso3d/experiments.py` runs the parameter study behind `results/sweeps.csv`, `results/results.md`
and `figures/sweep_*.png`.  Phase 2 turns the hard-coded `if method == ...` chain into a registry,
adds four methods, a Cramer-Rao lower bound (CRLB) reference column and an iterations column, and
exposes the script as a function with output directories as arguments.  The Phase-1 numbers are
unchanged (see "How the legacy numbers are protected").

## Method keys and labels

| key | label | kind | what runs per trial |
|---|---|---|---|
| `pso` | Simplified PSO | swarm | `SimplifiedPSO(N, T, seed, clip=True)` (Review-1 update rule) |
| `std` | Standard PSO | swarm | `StandardPSO(N, T, seed, clip=True)` (pbest/gbest) |
| `lsq` | Least-squares trilateration | closed form | `least_squares_trilateration` |
| `lsq_gn` | LSQ + Gauss-Newton | closed form | LSQ, then `gauss_newton_refine` (<= 10 steps) |
| `amcmpso` | AMCMPSO (interpretation) | swarm | `AMCMPSO(N, T, seed, clip=True)` |
| `std_tol` | Standard PSO + tolerance stop | swarm | `StandardPSO(..., stopping=Tolerance(1e-2, 10))` |
| `std_ws` | Standard PSO + LSQ/GN warm start | swarm | `StandardPSO(..., x0=warm_start_estimate(...))` |
| `amcmpso_ws` | AMCMPSO (interpretation) + warm start | swarm | `AMCMPSO(..., x0=warm_start_estimate(...))` |

The first four are the *legacy* methods (`LEGACY_METHODS`); the last four are the *extended*
methods (`EXTENDED_METHODS`).  AMCMPSO is this repository's interpretation of the adaptive
mean / centre-of-mass idea of the base paper, not a reproduction of it; its numbers must not be
quoted as results of that paper (see `docs/amcmpso.md`).

Each entry is a `MethodSpec(key, label, closed_form, factory)` in the `METHODS` dict.  A factory has
the signature

```python
factory(anchors, measured, lower, upper, n_particles, n_iterations, seed)
    -> (estimate, evaluations, iterations, memory_floats)
```

`evaluations` counts fitness evaluations for swarm methods and Gauss-Newton steps for `lsq_gn`
(one residual/Jacobian evaluation each, as in Phase 1); the warm start's LSQ solve and Gauss-Newton
steps are not added to the swarm's count.  `memory_floats` is the swarm state (`swarm_state_floats`)
or `3 * M + 3` for the closed-form methods (anchors + estimate).

The tolerance rule stops a run when the best fitness improved by at most 1 % over the last ten
iterations (`pso3d.stopping.Tolerance(1e-2, 10)`); the warm start replaces particle 0 by the
clipped LSQ + Gauss-Newton estimate (`pso3d.trilateration.warm_start_estimate`), consuming no
random numbers.

A warm start is not automatically an improvement, so the study measures it.  The 200-trial extended
study (seed 123; tables in `results/extended/results.md`) is produced by

```bash
python3 scripts/run_experiments.py --trials 200 --extended --out results/extended --figures figures/extended
```

In it the warm-started Standard PSO is the most accurate swarm method at every anchor count.  RMSE at
sigma = 0.5 m for 4 / 5 / 6 / 8 / 10 anchors: plain Standard PSO 6.533 / 5.683 / 3.051 / 2.698 / 2.372 m,
warm-started 3.192 / 2.696 / 1.765 / 1.043 / 0.794 m, LSQ + Gauss-Newton alone 4.174 / 2.828 / 2.124 /
1.147 / 0.835 m (CRLB 2.51 m with four anchors); with four anchors the warm start also cuts the p90
error from 12.93 m to 4.66 m.  Its advantage over the cold swarm shrinks as the noise grows.  Over the
noise sweep with four anchors (sigma 0.1 / 0.25 / 0.5 / 1.0 / 2.0 m) the warm-started swarm reaches
0.592 / 1.336 / 3.192 / 6.177 / 7.502 m against 5.878 / 6.059 / 6.533 / 6.978 / 7.772 m for the cold
swarm, while the closed-form start itself degrades to 7.366 m and 11.187 m at sigma = 1 and 2 m, so at
those noise levels the warm-started swarm is only marginally better than the cold one.

The mechanism is geometric.  With four noisy anchors the LSQ + Gauss-Newton solution can sit on the
*mirror image* of the node across the anchor plane (flip ambiguity, see `docs/geometry.md`), where the
range-error fitness is as low as -- or lower than -- at the true position; a swarm seeded there settles
on that minimum.  The noisier the ranges, the more often the start lands on the wrong side or far from
either minimum, which is why the benefit fades at high noise -- and why a small sample can show the
opposite ordering: a 20-trial check at sigma = 0.5 m with four anchors contained enough flipped starts
to rank the warm-started swarm behind the cold one.  Use the 200-trial setting before drawing
conclusions for a new geometry, and read the CRLB column to see how much of any remaining gap is
geometry rather than optimiser.

## Python API

```python
from pso3d.experiments import METHODS, monte_carlo, sweep, StudyRow

r = monte_carlo("std_ws", trials=200, noise_sigma=0.5, n_anchors=4, n_particles=20, n_iterations=60, seed=123)
r["rmse"], r["mean_error"], r["p90_error"], r["evaluations"], r["runtime_ms"], r["memory_floats"]   # legacy keys
r["errors"]            # per-trial errors (ndarray)
r["crlb_rmse"]         # mean CRLB RMS bound at the sampled true positions (new)
r["iterations_mean"]   # mean optimiser iterations per trial (new)

rows = sweep("noise_sigma", [0.1, 0.5, 1.0], trials=200, methods=["lsq_gn", "amcmpso"],
             noise_sigma=0.5, n_anchors=4, n_particles=20, n_iterations=60)        # list[StudyRow]
```

* `monte_carlo(method, trials, noise_sigma, n_anchors, n_particles, n_iterations, seed=123, base=None, noise_model=None)`
  localizes `trials` random nodes, uniform in the field of `base` (default `Scenario()`), and
  summarises the errors.  `n_anchors < 4` raises `ValueError` (a 3D fix needs four anchors);
  unknown keys raise `ValueError`.
* `noise_model` is any object with `sample(true_ranges, rng) -> additive noise` (the
  `pso3d.noise` models qualify).  When given, it replaces the Gaussian draw; the `crlb_rmse`
  reference still assumes Gaussian noise of `noise_sigma`.
* `sweep(label, values, trials=200, methods=None, **fixed)` runs `monte_carlo` for every value of
  `label` and every selected method; `methods=None` means the legacy four, `"all"` every method.
  Closed-form methods are skipped for the swarm-only labels `n_particles` and `n_iterations`.
  `fixed` may also carry `seed`, `base` and `noise_model`.
* `StudyRow` keeps the Phase-1 positional fields and appends `crlb_rmse` and `iterations_mean`
  (defaults `nan`) at the end.

## How the legacy numbers are protected

The per-trial random-number consumption order inside `monte_carlo` is frozen:

```
anchors (random_anchors) -> true position -> Gaussian noise -> [swarm methods only] PSO seed = rng.integers(0, 2**31 - 1)
```

Closed-form methods never draw a seed; the CRLB reference uses no random numbers; `n_anchors`
validation happens before the generator is created.  `tests/test_experiments.py::test_legacy_monte_carlo_numbers_pinned`
pins `rmse`, `mean_error`, `p90_error`, `evaluations` and `memory_floats` of the four legacy methods
at `trials=5, seed=123` to 1e-9 with values captured from the code *before* the registry existed,
and `scripts/run_experiments.py` with its default arguments writes a `sweeps.csv` with the Phase-1
header and row layout.  To verify a checkout end to end:

```bash
python3 scripts/run_experiments.py --trials 200 --out build/exp --no-plots
diff <(cut -d, -f1-7,9,10 build/exp/sweeps.csv) <(cut -d, -f1-7,9,10 results/sweeps.csv)   # runtime column excluded
```

## The CRLB column

For ranging noise that is zero-mean Gaussian with standard deviation sigma, independent across
anchors, the Fisher information matrix at the true position p is `F = J^T J / sigma^2`, where the
rows of `J` are the unit vectors `(p - a_j) / ||p - a_j||`.  The Cramer-Rao lower bound on the RMS
position error of any unbiased estimator is `sqrt(trace(F^-1))`.  `crlb_rmse` is the mean of this
bound over the trials, evaluated at each sampled true position with the anchors of that trial
(`pso3d.experiments._crlb_rmse`; `pso3d.geometry.crlb` evaluates the same matrix with per-axis
detail).  The bound is `inf` when `F` is singular (for example all anchors and the node in one plane).

Reading it: the bound depends only on the geometry and sigma, not on the method, and scales
linearly with sigma.  At a given sweep point it is identical for all swarm methods and identical
for both closed-form methods; the two groups differ slightly because the swarm methods draw one
extra random number per trial (the PSO seed), so from the second trial on they localize different
nodes than the closed-form methods -- this is the frozen Phase-1 draw order, not a property of the
bound.  It is a reference, not a method: a method whose RMSE sits close to the bound is limited by
geometry and noise, not by the optimiser; a large gap (the Phase-1 swarm methods with four anchors)
points at the optimiser or at its flip/plateau failures.  Because it is a bound for *unbiased*
estimators and the Monte-Carlo RMSE is itself a finite-sample estimate, a sample RMSE can fall
slightly below the mean bound; with 20 trials differences of about 20 % are sampling noise.

`iterations_mean` is the mean number of optimiser iterations actually run per trial (`T` for the
fixed-budget swarm methods, fewer with the tolerance stop, Gauss-Newton steps for `lsq_gn`, 0 for `lsq`).

## Command line

```bash
python3 scripts/run_experiments.py                       # legacy study: 200 trials, results/ and figures/
python3 scripts/run_experiments.py --trials 50 --out build/exp --figures build/exp
python3 scripts/run_experiments.py --methods lsq_gn,std_ws,amcmpso --trials 100 --out build/exp --figures build/exp
python3 scripts/run_experiments.py --extended --trials 200 --out build/ext --figures build/ext
```

| option | default | meaning |
|---|---|---|
| `--trials N` | 200 | random nodes per setting |
| `--out DIR` | `results/` | where `sweeps.csv` and `results.md` are written |
| `--figures DIR` | `figures/` | where the PNG figures are written |
| `--methods a,b` | `pso,std,lsq,lsq_gn` | method keys to run (`all` for every registered method) |
| `--extended` | off | add `amcmpso,std_tol,std_ws,amcmpso_ws` to the selected methods |
| `--no-plots` | off | write `sweeps.csv` and `results.md` only (no figures directory is created) |

Outputs:

* `sweeps.csv`: header `parameter,value,method,rmse_m,mean_error_m,p90_error_m,fitness_evaluations,runtime_ms,memory_floats,trials`.
  When `--extended` is set or a non-legacy method is requested the header gains `crlb_rmse_m,iterations_mean`;
  the default run reproduces the committed layout exactly.  CSV files use LF line endings.
* `results.md`: the four sweep tables and the error-CDF table as before; in extended mode the sweep
  tables gain CRLB and iterations columns and an "Extended methods" table is appended.
* `sweep_noise.png`, `sweep_anchors.png`, `sweep_particles.png`, `sweep_iterations.png` (one curve
  per method, CRLB reference dashed), `error_cdf.png`, and in extended mode `method_bars.png`
  (RMSE per method at sigma = 0.5 m with the CRLB bound as a horizontal reference).

The script exits with status 0 on success and 2 for an unknown method key; `main(argv)` can be
called from Python (the package CLI forwards the same arguments).

## Plot helpers (`pso3d/plots.py`)

All functions render PNGs with the Agg backend; matplotlib and Pillow are imported only in this
module.  Values that used to be hard-coded are now derived from the inputs:

| function | note |
|---|---|
| `plot_convergence(best_history, path, title=..., plateau_iteration=None)` | the dashed "plateau region" marker is drawn only when `plateau_iteration` (1-based) is given |
| `plot_scenario_3d(scenario, estimate, positions_history, path)` | axis limits and the title from `scenario.field_size`, anchor count from `scenario.anchors`, "swarm end (t=T)" from the history length |
| `plot_sweep(rows, label, xlabel, path, logx=False)` | one curve per method present in `rows`, colours and labels from `METHODS`, CRLB reference when the rows carry it |
| `plot_error_cdf(errors_by_method, path, title=...)` | any number of methods |
| `plot_method_bars(rows, path, label, value)` | RMSE bars for one sweep point with the CRLB bound as a reference line when available |
| `plot_crlb_heatmap(xs, ys, grid2d, path, title)` | image of one z-slice of a CRLB coverage grid (`grid2d[iy, ix]`); singular cells (inf) are shown in grey |
| `make_convergence_gif(scenario, positions_history, estimate_history, path, every=2, fps=6, title=...)` | axis limits from `scenario.field_size` |

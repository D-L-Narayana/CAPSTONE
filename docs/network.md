# Multi-node iterative auto-localization in 3D

`pso3d/network.py` simulates a whole sensor network instead of a single node. It transposes the
distributed iterative scheme of Kulkarni, Venayagamoorthy and Cheng (2009, *Bio-inspired node
localization in wireless sensor networks*, Section IV — the background paper kept in the repository
root) from the paper's 2-D square field to the 3-D box fields used throughout this project. The
single-node estimators of the package (PSO variants, least squares + Gauss-Newton) are plugged in
unchanged.

## Scheme

1. **Deployment.** `N` unknown nodes and `M` beacons are placed uniformly at random in the field
   (`NetworkScenario.generate()`, one `numpy.random.default_rng(seed)` stream, beacons first).
2. **Ranging.** Every unknown node measures ranges to the *references* within its radio range `r`:
   the beacons plus the nodes that were settled in earlier rounds. A reference is "in range" when
   the true distance is at most `r`. Ranges are simulated from the true positions and perturbed
   by the noise model (Gaussian sigma by default; any object with `sample(true_ranges, rng)` from
   `pso3d/noise.py` works, e.g. the paper's uniform `d ± d·Pn/100` model or an NLOS mixture).
3. **Localizable nodes.** A node is localizable in a round when it has at least `min_refs`
   references (default 4 — a unique 3-D fix needs four non-coplanar references, where the paper's
   2-D setting needs three) and, when `require_non_coplanar=True` (default), when the chosen
   reference set spans all three dimensions (rank of the centred reference matrix equals 3). At
   most the `max_refs` nearest references are used (default 6; the paper calls that limit
   "arbitrarily chosen" and uses it to bound the time per node).
4. **Estimation.** Each localizable node runs the supplied `Localizer` on its own ranges and the
   positions of its references *as the node knows them*: beacons are exact, settled nodes advertise
   their **estimates**, not their true positions. Estimates are clipped to the field.
5. **Iteration.** Settled nodes serve as references in the next round. Rounds repeat until no new
   node settles, every node is settled, or `max_rounds` is reached. A settled node stays settled;
   it is never re-estimated.

Reference selection and the ranges use the true geometry (who can hear whom), whereas the
localizer only sees advertised positions — exactly the information a real node would have.

## API

```python
from pso3d.network import NetworkScenario, iterative_localize, pso_localizer, lsq_gn_localizer, warm_started_localizer
from pso3d.standard_pso import StandardPSO

scenario = NetworkScenario(field_size=(100.0, 100.0, 30.0), n_nodes=50, n_beacons=10, radio_range=25.0, seed=1)
beacons, nodes = scenario.generate()                     # (M, 3), (N, 3)

localizer = pso_localizer(StandardPSO, n_particles=20, n_iterations=60, clip=True)   # or lsq_gn_localizer()
result = iterative_localize(scenario, localizer, noise=0.5, max_rounds=10, min_refs=4, max_refs=6)
print(result.n_not_localized, result.e_l, result.rmse, result.coverage_fraction, result.total_evaluations)
for r in result.rounds:
    print(r.round, r.newly_settled, r.settled_total, r.refs_available_mean, r.evaluations)
```

* `Localizer` is a callable `(anchors, measured, lower, upper, seed) -> LocalizeOutcome(position,
  evaluations, iterations)`; every node gets its own integer seed drawn from the run's generator.
* `pso_localizer(cls, **kwargs)` constructs `cls(seed=seed, **kwargs)` per node and runs it on a
  fresh `RangeErrorFitness`; `lsq_gn_localizer()` is the closed-form least squares + Gauss-Newton
  estimator (evaluations = Gauss-Newton iterations); `warm_started_localizer(cls, **kwargs)` seeds
  the swarm with the closed-form estimate through the optimiser's `x0` argument. If the optimiser
  does not accept `x0`, the swarm runs plainly and the localizer reports
  `warm_start_supported = False`.
* `noise` is a float sigma (zero-mean Gaussian) or a noise-model object. `seed` (default
  `scenario.seed + 1`) drives the noise and the per-node localizer seeds; the deployment itself is
  fixed by the scenario. `positions=(beacons, nodes)` replaces the random deployment with given
  arrays, which is how the tests build hand-made geometries.
* Helpers: `settled_only_error(result, nodes)` (per-node Euclidean errors of the settled nodes),
  `summary_table(result)` (Markdown), `reference_rank(points)` (3 = non-coplanar, 2 = coplanar,
  1 = collinear).

## Metrics (`NetworkResult`)

| field | meaning |
|---|---|
| `n_not_localized` | **N_NL** — nodes that could not be localized in any round (paper: the first element of the performance doublet) |
| `e_l` | **E_l** — mean squared distance between true and estimated positions over the localized nodes (eq. 4 of the paper with the `z` term added); `nan` when no node settled |
| `rmse` | `sqrt(E_l)`, in metres |
| `coverage_fraction` | settled nodes / N |
| `total_evaluations` | fitness evaluations spent by all localized nodes (0 for the closed-form localizer, which reports Gauss-Newton iterations instead) |
| `rounds` | per round: `newly_settled`, `settled_total`, `refs_available_mean` (mean number of references in range over the nodes still unsettled at the start of the round), `evaluations` |
| `estimates`, `settled` | `(N, 3)` estimates (`nan` rows for unsettled nodes) and the `(N,)` boolean settled mask |
| extras | `settled_round`, `n_refs_used`, `errors` per node; `runtime_s` |

Lower `N_NL` and lower `E_l` mean better performance, as in the paper. Because `E_l` is averaged
over the localized nodes only, a method that settles fewer, easier nodes can show a lower `E_l`
than a method with higher coverage — always read `N_NL`/coverage and `E_l` together.

## How to run

```bash
python scripts/run_network.py                                   # 50 nodes, 10 beacons, r = 25 m, 100 x 100 x 30 m,
                                                                # gaussian:0.5, standard PSO 20 x 60, 10 rounds, seed 1
python scripts/run_network.py --localizer lsq_gn --range 40 --rounds 5
python scripts/run_network.py --localizer std_ws --noise percent:2 --nodes 80 --beacons 12
python scripts/run_network.py --nodes 20 --beacons 6 --rounds 3 --out /tmp/net --figures /tmp/net --no-plots
```

| option | default | meaning |
|---|---|---|
| `--nodes`, `--beacons`, `--range`, `--field X Y Z` | 50, 10, 25, 100 100 30 | deployment |
| `--noise SPEC` | `gaussian:0.5` | `gaussian:SIGMA` · `percent:P` · `nlos:p[:bias[:sigma]]` · `shadowing:dB` (the last three need `pso3d/noise.py`) |
| `--localizer` | `std` | `std` (standard PSO) · `lsq_gn` (closed form) · `amcmpso` (interpretation of the base paper, see `docs/amcmpso.md`) · `std_ws` (standard PSO warm-started from LSQ+GN) · `simplified` (Review-1 PSO) |
| `--particles`, `--iterations` | 20, 60 | swarm budget for the PSO localizers |
| `--rounds`, `--min-refs`, `--max-refs`, `--allow-coplanar` | 10, 4, 6, off | iteration limits |
| `--seed` | 1 | deployment seed (noise and per-node PSO seeds derive from `seed + 1`) |
| `--out DIR`, `--figures DIR`, `--no-plots`, `--quiet` | `results/`, `figures/` | outputs |

Outputs: `network.csv` (one row per round), `network_nodes.csv` (one row per node: true position,
settled flag, round, references used, estimate, error), `network.md` (settings and the summary
table with N_NL, E_l, RMSE, coverage, evaluations) and the figures `network_rounds.png` (settled
count and newly settled nodes per round, cost per round) and `network_map.png` (3-D map with
beacons, true positions — hollow when not localized — estimates and error segments). The CSV files
use LF line endings. The same run is available as `python -m pso3d network ...` when the package
CLI is installed.

## Limits and honest notes

* **Ranges are simulated from the true positions** with additive noise; there is no propagation,
  connectivity or packet model. "In range" means true distance ≤ `r`, so the simulation knows the
  truth that a real network would only observe through its radios.
* **Errors accumulate through settled references.** A node localized from biased references
  inherits their bias and adds its own, so later rounds are typically less accurate than the
  first; the paper reports the same effect and the correction of flip-ambiguity errors when more
  references become available. Nothing here corrects a settled node later.
* **Coverage depends on geometry.** With the paper's `r = 25` in a 100 × 100 field and 10 beacons,
  a 3-D node rarely hears four beacons at once (the paper needs three in 2-D), so the first round
  may settle few or no nodes; larger `--range`, more beacons or a flatter field change this a lot.
  Results are reported per deployment and seed — they are not statistics over many deployments.
* The scheme uses the given single-node estimator as a black box; its accuracy, including the
  heavy error tail for weak reference geometry documented in the Monte-Carlo study, carries over.
* The 3-D transposition (four references, non-coplanarity test, `z` term in `E_l`) is this
  project's generalisation of the paper's 2-D description; the paper's own numbers (2-D, PSO with
  30 particles × 150 iterations, uniform `±d·Pn/100` noise) are not reproduced here.

# Stopping rules and warm start

`pso3d/stopping.py` provides composable stopping rules for the swarm optimisers; `SimplifiedPSO`,
`StandardPSO` (and the other optimisers that follow the same interface) accept them through the
`stopping=` argument and a warm-start point through `x0=`. Neither option touches the random stream,
so a run without them is unchanged (the Review-1 baseline stays bit-exact).

## Interface

```python
class StoppingRule(Protocol):
    name: str
    def reset(self) -> None: ...
    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool: ...
```

The optimiser calls `reset()` once at the start of `run()` and `update()` after the bookkeeping of
every iteration with the 1-based iteration counter, the **best fitness seen so far** (best-ever) and the
number of fitness evaluations spent. `True` means "stop after this iteration"; the optimiser then records
the rule's `name` in `PSOResult.stopped_by` (otherwise `None`). `iterations_run`, `best_history`,
`estimate_history` and `positions_history` reflect the iterations actually run. Rules are consulted
after iterations 1..T only; the initial swarm evaluation of `StandardPSO` is not a stopping point.

| Rule | Fires when | `name` |
|---|---|---|
| `Patience(patience, min_improvement=1e-12)` | the best-ever fitness has not improved by more than `min_improvement` for `patience` consecutive iterations (stall counter ≥ `patience`) | `patience` |
| `Tolerance(tol=1e-2, window=10, relative=True)` | `f_old − f_new ≤ tol · f_old` (relative) or `≤ tol` (absolute), where `f_new` is the current best-ever fitness and `f_old` the best-ever fitness `window` iterations earlier; needs `window + 1` values, so it cannot fire before iteration `window + 1` | `tolerance` |
| `MaxEvaluations(budget)` | `evaluations ≥ budget` | `max_evaluations` |
| `TargetFitness(target)` | `best_fitness ≤ target` (Kulkarni & Venayagamoorthy, Algorithm 1: f(gbest) ≤ f_T) | `target_fitness` |
| `AnyOf(*rules)` | any sub-rule fires; every sub-rule is updated on every call and `reset()` resets all of them; `fired` lists the sub-rules that fired in the last update | `any(name1,name2,…)` |

`Patience` reproduces the former `early_stop_patience` option of `SimplifiedPSO` exactly (same
`1e-12` improvement threshold, same stall counter). `early_stop_patience=p` is still accepted as a
**deprecated alias** for `stopping=Patience(p)`; an explicit `stopping=` argument takes precedence.

Combining rules:

```python
from pso3d import RangeErrorFitness, Scenario, StandardPSO
from pso3d.stopping import AnyOf, MaxEvaluations, Tolerance

sc = Scenario()
fit = RangeErrorFitness(sc.anchors, sc.measured())
rule = AnyOf(Tolerance(1e-2, 10), MaxEvaluations(1000))
res = StandardPSO(20, 60, seed=1, stopping=rule).run(fit, sc.lower, sc.upper)
print(res.iterations_run, res.fitness_evaluations, res.stopped_by)   # 46 940 any(tolerance,max_evaluations)
```

## Baseline numbers (slide-16 scenario, seed 1)

Simplified PSO, `SimplifiedPSO(20, 60, w=0.7, c=1.4, seed=1, clip=False)` — the Review-1 configuration.
The estimate is the best particle of the *last* iteration (Review-1 semantics), so stopping earlier
changes the reported estimate.

| Stopping rule | Iterations | Evaluations | Estimate (m) | Error (m) | `stopped_by` |
|---|---|---|---|---|---|
| none | 60 | 1 200 | (37.43, 11.90, 9.29) | 0.90 | `None` |
| `Patience(10)` (= `early_stop_patience=10`) | 60 | 1 200 | (37.43, 11.90, 9.29) | 0.90 | `None` — never triggers |
| `Patience(5)` | 43 | 860 | (37.41, 11.96, 9.37) | 0.97 | `patience` |
| `Tolerance(1e-2, 10)` (relative, default) | 60 | 1 200 | (37.43, 11.90, 9.29) | 0.90 | `tolerance` — fires on the last iteration |
| `Tolerance(1e-2, 10, relative=False)` | 39 | 780 | (37.41, 11.92, 9.13) | 0.76 | `tolerance` |
| `Tolerance(5e-2, 10)` | 48 | 960 | (37.44, 11.90, 9.31) | 0.93 | `tolerance` |
| `MaxEvaluations(600)` | 30 | 600 | (37.46, 11.86, 8.96) | 0.67 | `max_evaluations` |
| `TargetFitness(0.1)` | 28 | 560 | (37.37, 11.78, 9.32) | 0.92 | `target_fitness` |

The relative 1 %/10-iteration criterion saves nothing on this run: the memory-less swarm keeps lowering
its best-ever fitness by slightly more than 1 % per 10 iterations until the very end (iterations 50 → 60:
0.05702 → 0.05672 m², −0.52 %, which is where it finally fires). The absolute criterion (0.01 m² over 10
iterations) stops at iteration 39 and saves 35 % of the evaluations, which is the "tolerance-based
criterion" the baseline discussion asked for. The lower errors of the shorter runs (0.67–0.76 m) are
not a general property: the swarm had not converged yet, and the accuracy floor of this scenario is set
by the ranging noise (true minimiser of f at (37.42, 11.88, 9.36), error 0.96 m).

Standard PSO, `StandardPSO(20, 60, seed=1)` (pbest/gbest memory, clipping on; 20 initial + 20 evaluations per iteration):

| Stopping rule | Iterations | Evaluations | Estimate (m) | Error (m) | `stopped_by` |
|---|---|---|---|---|---|
| none | 60 | 1 220 | (37.42, 11.89, 9.35) | 0.96 | `None` |
| `Tolerance(1e-2, 10)` | 46 | 940 | (37.42, 11.89, 9.35) | 0.96 | `tolerance` |
| `Tolerance(1e-2, 10, relative=False)` | 28 | 580 | (37.39, 11.90, 9.35) | 0.94 | `tolerance` |
| `MaxEvaluations(600)` | 29 | 600 | (37.44, 11.88, 9.39) | 1.01 | `max_evaluations` |
| `TargetFitness(0.1)` | 16 | 340 | (37.43, 11.81, 9.25) | 0.88 | `target_fitness` |

## Warm start (`x0`)

`x0` is a single 3-D point (list, tuple or array; anything else raises `ValueError`). After the uniform
initial draw of the swarm, particle 0 is replaced by `clip(x0, lower, upper)` with zero velocity. No random
numbers are consumed, so particles 1..N−1 and all later random factors are identical to the cold run with
the same seed, and `PSOResult.warm_start` is `True`. The natural source of `x0` is the closed-form
estimate (`least_squares_trilateration` followed by `gauss_newton_refine`, or
`pso3d.trilateration.warm_start_estimate(anchors, measured, lower, upper)`).

With `x0 = (37.42, 11.88, 9.36)` (the least-squares + Gauss-Newton estimate of the baseline scenario,
f(x0) = 0.0567 m²):

| Run | Iterations | Evaluations | Estimate (m) | Error (m) | `stopped_by` |
|---|---|---|---|---|---|
| `StandardPSO(20, 20, seed=1)` cold | 20 | 420 | (37.41, 11.85, 9.16) | 0.79 | `None` |
| `StandardPSO(20, 20, seed=1, x0=x0)` | 20 | 420 | (37.42, 11.88, 9.36) | 0.96 | `None` |
| `StandardPSO(20, 60, seed=1, x0=x0, stopping=Tolerance(1e-2, 10))` | 11 | 240 | (37.42, 11.88, 9.36) | 0.96 | `tolerance` |
| `SimplifiedPSO(20, 60, seed=1, clip=False, x0=x0)` | 60 | 1 200 | (37.42, 11.88, 9.36) | 0.96 | `None` |

Because gbest starts at the closed-form point, the warm-started standard PSO can only stay at or improve
on it; here it converges to the true minimiser of f (error 0.96 m) and, combined with the tolerance rule,
needs 240 instead of 1 220 evaluations. The warm start therefore buys evaluations, not accuracy, on this
scenario — the remaining 0.96 m is the ranging-noise floor discussed in the README.

## `PSOResult` additions

Three fields were appended with defaults (the positional order of the existing fields is unchanged):
`stopped_by: str | None = None`, `warm_start: bool = False`, `method: str = ""`
(`"simplified"` for `SimplifiedPSO`, `"standard"` for `StandardPSO`).

## Related fix: `RangeErrorFitness`

`RangeErrorFitness.__call__` now converts its argument once with `np.asarray(..., dtype=float)`, so Python
lists and tuples work: a single point of shape `(3,)` returns a `float`, a batch of shape `(N, 3)` returns an
array of shape `(N,)`. The counters are unchanged (`evaluations += N`, `distance_computations += N·M`).
`evaluate_one(p) -> float` scores one point and counts as one evaluation.

Tests: `tests/test_stopping.py`, `tests/test_warm_start.py`, `tests/test_fitness.py`.

# AMCMPSO — adaptive mean / centre-of-mass PSO (interpretation)

Module: `pso3d/amcmpso.py` · tests: `tests/test_amcmpso.py` · fixture: `tests/fixtures/amcmpso_cases.json`

## Status and honesty statement

`AMCMPSO` is an **interpretation** of the idea behind the base paper
(Alhasan et al., 2023, *J. King Saud Univ. – Comput. Inf. Sci.*, 35(9), 101782,
https://doi.org/10.1016/j.jksuci.2023.101782): a standard PSO whose velocity update is
additionally guided by the swarm mean and by a fitness-weighted centre of mass, with inertia
and acceleration coefficients that move from exploration to exploitation over the run.

What is and is not from the paper:

* **From the paper (abstract and the team's literature summary):** the three ingredients above
  — adaptive `w`, `c1`, `c2`; a swarm-mean term; a centre-of-mass term — and the paper's own
  headline figures (improvement rate 99.86 %, error < 1.34 cm, 3D coverage > 87 %), which are
  quoted here only to say that *nothing in this repository reproduces, confirms or contradicts
  them*.
* **Not from the paper (chosen by this project):** the exact velocity equation, the way the
  mean and centre-of-mass terms enter it, the inverse-fitness weights of the centre of mass,
  the coefficient schedules and their end values, the default constants (`c_mean = c_com = 0.5`),
  the stopping rules, the warm start and the diagnostics defined below.
* The paper text is not in the repository; its Section 3 equations and constants have **not**
  been transcribed. No number produced by this class may be quoted as a reproduction of the
  paper. The `improvement_rate` field is *this implementation's* convergence diagnostic, not
  the paper's metric.

The class is a working, tested and configurable interpretation; it is not, and does not claim
to be, a faithful re-implementation of the published algorithm. Transcribing the paper's
equations and constants, and comparing against its reported results, remains open work.

## Algorithm as implemented

Notation: `N` particles, `T` iterations, field bounds `lower`, `upper`, fitness
`f(p) = sum_j (||p − a_j|| − d_j)^2` (`pso3d.fitness.RangeErrorFitness`).

Initialisation (random stream `numpy.random.default_rng(seed)`):

```
x  = uniform(lower, upper, size (N, 3))        # first use of the random stream
x[0] = clip(x0, lower, upper)                  # only when x0 is given; consumes no random numbers
v  = 0
f  = fitness(x)                                # N evaluations
pbest = x, pbest_f = f
gbest = argmin particle, gbest_f = min f       # f_0 := gbest_f
```

One iteration, `t = 0 … T−1` (`w(t)`, `c1(t)`, `c2(t)` from the schedule, see below):

```
mean = x.mean(axis=0)                                  # swarm mean
com  = centre_of_mass(x, f)                            # see below
r1, r2, r3, r4 = four independent uniform (N, 3) draws, in this order
v = w(t)·v + c1(t)·r1·(pbest − x) + c2(t)·r2·(gbest − x)
            + c_mean·r3·(mean − x) + c_com·r4·(com − x)
x = x + v
x = clip(x, lower, upper)                              # when clip=True (default)
f = fitness(x)                                         # N evaluations
pbest/pbest_f updated where f < pbest_f; gbest/gbest_f updated from the best pbest
record gbest_f and gbest; (positions when record_positions; diversity when diagnostics)
stop if the stopping rule says so (see below)
```

All products are element-wise. The update is *synchronous*: every particle is evaluated,
then the personal bests and the global best are refreshed.

### Centre of mass

```
weight_i = 1 / (f_i + 1e-9)
com      = sum_i weight_i · x_i / sum_i weight_i
```

Better (lower-fitness) particles are heavier; the `1e-9` guards a particle with exactly zero
fitness (it then dominates the centre). With equal fitness values the centre of mass equals
the plain mean. Exposed as `pso3d.amcmpso.centre_of_mass(x, f)`.

### Coefficient schedules

`coefficient_schedule(T, schedule="linear", w_max=0.9, w_min=0.4, c1_start=2.0, c1_end=0.5,
c2_start=0.5, c2_end=2.0)` returns arrays `w`, `c1`, `c2` of length `T`, for `t = 0 … T−1`:

```
linear:  s(t) = t / T
cosine:  s(t) = (1 − cos(π · t / T)) / 2
w(t)  = w_max   − (w_max − w_min)     · s(t)        # 0.9 → towards 0.4  (exploration → exploitation)
c1(t) = c1_start + (c1_end − c1_start) · s(t)        # 2.0 → towards 0.5  (cognitive pull fades)
c2(t) = c2_start + (c2_end − c2_start) · s(t)        # 0.5 → towards 2.0  (social pull grows)
```

Both schedules start exactly at the `*_start` / `w_max` values and stop one step short of the
end values (`s(T−1) < 1`), because the Phase-1 code used `t / T` and the linear schedule is kept
bit-exact with it. The cosine schedule has the same endpoints, passes through the midpoint
`(start + end) / 2` at `t = T/2` (even `T`) and satisfies `s(t) + s(T − t) = 1`, i.e. it is
symmetric about the midpoint: it keeps the coefficients near their exploration values longer
at the start and moves faster through the middle of the run. With the default end values `w`
and `c1` decrease strictly, `c2` increases strictly, and all stay within their start/end values.

## Configuration

```python
from pso3d import AMCMPSO, RangeErrorFitness, Scenario

sc = Scenario()
fit = RangeErrorFitness(sc.anchors, sc.measured())
res = AMCMPSO(20, 60, seed=1).run(fit, sc.lower, sc.upper)            # Phase-1 behaviour
res = AMCMPSO(20, 60, seed=1, schedule="cosine").run(fit, sc.lower, sc.upper)
```

Constructor (defaults unchanged from Phase 1): `n_particles=20, n_iterations=60, w_max=0.9,
w_min=0.4, c1_start=2.0, c1_end=0.5, c2_start=0.5, c2_end=2.0, c_mean=0.5, c_com=0.5, seed=1,
clip=True, record_positions=False`, plus the Phase-2 options `schedule="linear"`,
`stopping=None`, `x0=None`, `diagnostics=False`. An unknown schedule name raises `ValueError`.

### Stopping rules

`stopping` accepts any object following the shared stopping protocol:

```
rule.reset()                                          # called once at the start of run()
rule.update(iteration, best_fitness, evaluations)     # called after every iteration; True → stop
rule.name                                             # copied into result.stopped_by
```

`iteration` is 1-based, `best_fitness` is the best-ever fitness after that iteration and
`evaluations` the fitness-evaluation counter at that moment (`N · (iteration + 1)`). When a rule
fires after iteration `k`, `iterations_run == k`, `fitness_evaluations == N · (k + 1)` and
`stopped_by == rule.name`; otherwise `stopped_by is None`. The rules in `pso3d.stopping`
(`Patience`, `Tolerance`, `MaxEvaluations`, `TargetFitness`, `AnyOf`) follow this protocol.

### Warm start

`x0=(x, y, z)` replaces particle 0 of the uniform initial swarm by `clip(x0, lower, upper)`
with zero velocity. No random numbers are consumed, so particles `1 … N−1` are identical to the
cold run with the same seed, and `result.warm_start` is `True`. A natural choice for `x0` is the
closed-form least-squares + Gauss–Newton estimate (`pso3d.trilateration`).

### Diagnostics

`run()` returns an `AMCMPSOResult`, a `PSOResult` with `method == "amcmpso"` and:

* `improvement_rate = (f_0 − f_T) / f_0`, where `f_0` is the best fitness after the initial
  evaluation and `f_T` the final best fitness; `0.0` when `f_0 == 0`. Lies in `[0, 1]` because the
  best-ever fitness never increases. This is a relative-convergence diagnostic of this
  implementation.
* `diversity_history`: one value per iteration run, the mean Euclidean distance of the particles
  to the swarm mean after the position update (and clipping):
  `mean_i ||x_i − mean(x)||`. Filled only when `diagnostics=True` (empty list otherwise); it does
  not influence the optimisation. A shrinking value shows the swarm contracting around the
  best position.

Cost accounting is unchanged from Phase 1: `fitness_evaluations = N · (iterations_run + 1)`
(initial swarm plus one evaluation per particle per iteration), `distance_computations =
fitness_evaluations · M` for `M` anchors, and `swarm_state_floats = 9 N + 6` (position,
velocity and personal best per particle, plus the mean and the centre of mass).

## Reference values of this implementation (not of the paper)

On the slide-16 scenario (`Scenario()` defaults: field 60 × 60 × 20 m, four anchors, node at
(37, 12, 8.5), fixed ranging offsets) with `AMCMPSO(20, 60, seed=1)` the estimate is
(37.42, 11.88, 9.36), the error 0.96 m, 1 220 fitness evaluations, 186 swarm floats — the same
numbers as the Phase-1 version of the class, because the random-draw order was preserved. They
describe this interpretation on this scenario only and must not be presented as results of
Alhasan et al. (2023).

`tests/fixtures/amcmpso_cases.json` stores the linear and cosine schedules for `T = 10` and the
seed-1 result above (estimate, best fitness, error, improvement rate, counters). The test suite
recreates the file when it is missing and otherwise checks the stored values against a fresh
computation (absolute tolerance `1e-9`).

"""Monte-Carlo / parameter study helpers and the method registry.

Every study localizes many random nodes (uniform in the field) with ranging noise
(Gaussian with standard deviation ``noise_sigma`` unless a ``noise_model`` is given) and
reports RMSE, mean error, p90 error, fitness evaluations, iterations, run time, swarm
memory and the Cramer-Rao lower bound (CRLB) on the RMS error for the sampled geometry.

Methods are looked up in ``METHODS`` (key -> ``MethodSpec``).  A method *factory* has the
signature

    factory(anchors, measured, lower, upper, n_particles, n_iterations, seed)
        -> (estimate, evaluations, iterations, memory_floats)

Legacy guarantee (Phase-1 numbers)
----------------------------------
The four Phase-1 methods ``pso``, ``std``, ``lsq`` and ``lsq_gn`` reproduce the committed
``results/sweeps.csv`` exactly.  The per-trial random-number consumption order is frozen:

    anchors (random_anchors) -> true position -> Gaussian noise -> [swarm methods only] PSO seed

Closed-form methods (``closed_form=True``) never draw a PSO seed, exactly as before.
``tests/test_experiments.py`` pins the outputs of the four methods at ``trials=5, seed=123``
to 1e-9 with values captured from the code before the registry was introduced.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

import numpy as np

from .config import Scenario
from .fitness import RangeErrorFitness
from .pso import SimplifiedPSO
from .standard_pso import StandardPSO
from .trilateration import least_squares_trilateration, gauss_newton_refine

Factory = Callable[..., tuple]

# sweep labels for which closed-form methods are skipped (they have no swarm parameters)
SWARM_ONLY_LABELS = ("n_particles", "n_iterations")
TOLERANCE_TOL, TOLERANCE_WINDOW = 1e-2, 10


@dataclass
class StudyRow:
    label: str
    value: float
    method: str
    rmse: float
    mean_error: float
    p90_error: float
    evaluations: float
    runtime_ms: float
    memory_floats: int
    trials: int
    crlb_rmse: float = float("nan")          # mean CRLB RMS bound at the sampled true positions
    iterations_mean: float = float("nan")    # mean optimiser iterations per trial


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    closed_form: bool
    factory: Factory


# ----------------------------------------------------------------------------- CRLB reference
def _crlb_rmse(anchors, p, sigma) -> float:
    """CRLB on the RMS position error for Gaussian ranging noise: sqrt(trace(inv(J^T J / sigma^2))).

    J stacks the unit vectors (p - a_j) / ||p - a_j|| (the range Jacobian), so the Fisher
    information matrix is J^T J / sigma^2 -- the same quantity ``pso3d.geometry.crlb`` evaluates.
    Kept local so the study has no dependency beyond numpy.  Returns inf when the information
    matrix is singular (coplanar anchors with p in their plane, fewer than three anchors) or
    when p coincides with an anchor.
    """
    a = np.asarray(anchors, dtype=float)
    diff = np.asarray(p, dtype=float) - a
    dist = np.linalg.norm(diff, axis=1)
    if a.shape[0] < 3 or np.any(dist < 1e-12):
        return float("inf")
    J = diff / dist[:, None]
    fim = J.T @ J
    if np.linalg.matrix_rank(fim) < 3:
        return float("inf")
    return float(sigma) * float(np.sqrt(np.trace(np.linalg.inv(fim))))


# ----------------------------------------------------------------------------- building blocks
class _LocalTolerance:
    """Fallback tolerance rule (used only when ``pso3d.stopping`` is unavailable):
    stop when the best-ever fitness improved by <= tol * f_old over the last ``window`` iterations."""
    name = "tolerance"

    def __init__(self, tol: float = TOLERANCE_TOL, window: int = TOLERANCE_WINDOW, relative: bool = True):
        self.tol, self.window, self.relative = float(tol), int(window), bool(relative)
        self._history: list[float] = []

    def reset(self) -> None:
        self._history = []

    def update(self, iteration: int, best_fitness: float, evaluations: int) -> bool:
        self._history.append(float(best_fitness))
        if len(self._history) <= self.window:
            return False
        f_old = self._history[-1 - self.window]
        threshold = self.tol * abs(f_old) if self.relative else self.tol
        return (f_old - best_fitness) <= threshold


def _tolerance_rule():
    try:
        from .stopping import Tolerance
    except ImportError:
        return _LocalTolerance(TOLERANCE_TOL, TOLERANCE_WINDOW)
    return Tolerance(TOLERANCE_TOL, TOLERANCE_WINDOW)


def _warm_start(anchors, measured, lower, upper) -> np.ndarray:
    """LSQ -> Gauss-Newton -> clip (via ``trilateration.warm_start_estimate`` when available)."""
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    try:
        from .trilateration import warm_start_estimate
    except ImportError:
        warm_start_estimate = None
    if warm_start_estimate is not None:
        p = np.asarray(warm_start_estimate(anchors, measured, lower, upper), dtype=float)
    else:
        p = least_squares_trilateration(anchors, measured)
        p, _ = gauss_newton_refine(anchors, measured, p)
    p = np.where(np.isfinite(p), p, 0.5 * (lower + upper))
    return np.clip(p, lower, upper)


def _swarm(optimiser, anchors, measured, lower, upper):
    fit = RangeErrorFitness(anchors, measured)
    res = optimiser.run(fit, lower, upper)
    return res.estimate, res.fitness_evaluations, res.iterations_run, res.swarm_state_floats


def _closed_form_memory(anchors) -> int:
    return 3 * int(np.asarray(anchors).shape[0]) + 3     # anchors + estimate


# ----------------------------------------------------------------------------- factories
def _factory_pso(anchors, measured, lower, upper, n_particles, n_iterations, seed):
    return _swarm(SimplifiedPSO(n_particles, n_iterations, seed=seed, clip=True), anchors, measured, lower, upper)


def _factory_std(anchors, measured, lower, upper, n_particles, n_iterations, seed):
    return _swarm(StandardPSO(n_particles, n_iterations, seed=seed, clip=True), anchors, measured, lower, upper)


def _factory_lsq(anchors, measured, lower, upper, n_particles, n_iterations, seed):
    est = least_squares_trilateration(anchors, measured)
    return est, 0, 0, _closed_form_memory(anchors)


def _factory_lsq_gn(anchors, measured, lower, upper, n_particles, n_iterations, seed):
    est = least_squares_trilateration(anchors, measured)
    est, gn_it = gauss_newton_refine(anchors, measured, est)
    return est, gn_it, gn_it, _closed_form_memory(anchors)     # one residual/Jacobian evaluation per GN step


def _factory_amcmpso(anchors, measured, lower, upper, n_particles, n_iterations, seed):
    from .amcmpso import AMCMPSO
    return _swarm(AMCMPSO(n_particles, n_iterations, seed=seed, clip=True), anchors, measured, lower, upper)


def _factory_std_tol(anchors, measured, lower, upper, n_particles, n_iterations, seed):
    pso = StandardPSO(n_particles, n_iterations, seed=seed, clip=True, stopping=_tolerance_rule())
    return _swarm(pso, anchors, measured, lower, upper)


def _factory_std_ws(anchors, measured, lower, upper, n_particles, n_iterations, seed):
    x0 = _warm_start(anchors, measured, lower, upper)
    pso = StandardPSO(n_particles, n_iterations, seed=seed, clip=True, x0=x0)
    return _swarm(pso, anchors, measured, lower, upper)


def _factory_amcmpso_ws(anchors, measured, lower, upper, n_particles, n_iterations, seed):
    from .amcmpso import AMCMPSO
    x0 = _warm_start(anchors, measured, lower, upper)
    return _swarm(AMCMPSO(n_particles, n_iterations, seed=seed, clip=True, x0=x0), anchors, measured, lower, upper)


LEGACY_METHODS: tuple[str, ...] = ("pso", "std", "lsq", "lsq_gn")
EXTENDED_METHODS: tuple[str, ...] = ("amcmpso", "std_tol", "std_ws", "amcmpso_ws")

METHODS: dict[str, MethodSpec] = {
    spec.key: spec for spec in (
        MethodSpec("pso", "Simplified PSO", False, _factory_pso),
        MethodSpec("std", "Standard PSO", False, _factory_std),
        MethodSpec("lsq", "Least-squares trilateration", True, _factory_lsq),
        MethodSpec("lsq_gn", "LSQ + Gauss-Newton", True, _factory_lsq_gn),
        MethodSpec("amcmpso", "AMCMPSO (interpretation)", False, _factory_amcmpso),
        MethodSpec("std_tol", "Standard PSO + tolerance stop", False, _factory_std_tol),
        MethodSpec("std_ws", "Standard PSO + LSQ/GN warm start", False, _factory_std_ws),
        MethodSpec("amcmpso_ws", "AMCMPSO (interpretation) + warm start", False, _factory_amcmpso_ws),
    )
}


def _spec(method: str) -> MethodSpec:
    try:
        return METHODS[method]
    except KeyError:
        raise ValueError(f"unknown method {method!r}; known methods: {list(METHODS)}") from None


def resolve_methods(methods=None) -> tuple[str, ...]:
    """Normalise a method selection to registry order without duplicates.

    ``None`` -> the legacy four; a comma-separated string or an iterable of keys otherwise;
    the word ``all`` selects every registered method.  Unknown keys raise ``ValueError``.
    """
    if methods is None:
        return LEGACY_METHODS
    if isinstance(methods, str):
        methods = [m.strip() for m in methods.split(",") if m.strip()]
    wanted = set(methods)
    if "all" in wanted:
        return tuple(METHODS)
    unknown = sorted(wanted - set(METHODS))
    if unknown:
        raise ValueError(f"unknown method(s) {unknown}; known methods: {list(METHODS)}")
    return tuple(key for key in METHODS if key in wanted)


def random_anchors(rng: np.random.Generator, n: int, upper: np.ndarray, base: Scenario) -> np.ndarray:
    """First four anchors are the fixed non-coplanar corners of the default scenario;
    extra anchors are drawn uniformly in the field."""
    if n <= 4:
        return base.anchors[:n]
    extra = rng.uniform(np.zeros(3), upper, (n - 4, 3))
    return np.vstack([base.anchors, extra])


def monte_carlo(method: str, trials: int, noise_sigma: float, n_anchors: int,
                n_particles: int, n_iterations: int, seed: int = 123,
                base: Scenario | None = None, noise_model=None) -> dict:
    """Localize ``trials`` random nodes with ``method`` and summarise the errors.

    ``noise_model`` (any object with ``sample(true_ranges, rng) -> additive noise``) replaces the
    Gaussian draw; the ``crlb_rmse`` reference always uses Gaussian noise of ``noise_sigma``.
    Returns the legacy keys (rmse, mean_error, p90_error, evaluations, runtime_ms, memory_floats,
    trials, errors) plus ``crlb_rmse`` and ``iterations_mean``.
    """
    spec = _spec(method)
    if int(n_anchors) < 4:
        raise ValueError(f"n_anchors must be >= 4 for a 3D fix, got {n_anchors}")
    if int(trials) < 1:
        raise ValueError(f"trials must be >= 1, got {trials}")
    base = base or Scenario()
    rng = np.random.default_rng(seed)
    lower, upper = np.zeros(3), base.upper
    errors, evals, iters, times, bounds = [], [], [], [], []
    mem = 0
    for _ in range(trials):
        # --- frozen random-draw order: anchors -> true position -> noise -> (swarm methods) seed
        anchors = random_anchors(rng, n_anchors, upper, base)
        true_p = rng.uniform(np.zeros(3), upper)
        true_ranges = np.linalg.norm(anchors - true_p, axis=1)
        if noise_model is None:
            noise = rng.normal(0.0, noise_sigma, size=n_anchors)
        else:
            noise = np.asarray(noise_model.sample(true_ranges, rng), dtype=float)
        meas = true_ranges + noise
        pso_seed = None if spec.closed_form else int(rng.integers(0, 2**31 - 1))
        # --- localize
        t0 = time.perf_counter()
        est, ev, it, mem = spec.factory(anchors, meas, lower, upper, n_particles, n_iterations, pso_seed)
        times.append(time.perf_counter() - t0)
        errors.append(float(np.linalg.norm(np.asarray(est, dtype=float) - true_p)))
        evals.append(ev)
        iters.append(it)
        bounds.append(_crlb_rmse(anchors, true_p, noise_sigma))
    e = np.asarray(errors)
    return dict(rmse=float(np.sqrt((e ** 2).mean())), mean_error=float(e.mean()),
                p90_error=float(np.percentile(e, 90)), evaluations=float(np.mean(evals)),
                runtime_ms=float(np.mean(times) * 1e3), memory_floats=int(mem), trials=trials,
                errors=e, crlb_rmse=float(np.mean(bounds)), iterations_mean=float(np.mean(iters)))


def sweep(label: str, values, trials: int = 200, methods=None, **fixed) -> list[StudyRow]:
    """Run ``monte_carlo`` for every value of ``label`` and every selected method.

    ``fixed`` holds the other settings (``noise_sigma, n_anchors, n_particles, n_iterations`` and
    optionally ``seed``, ``base``, ``noise_model``).  Closed-form methods are skipped for the
    swarm-only labels ``n_particles`` and ``n_iterations``.  ``methods=None`` selects the legacy four.
    """
    keys = resolve_methods(methods)
    rows: list[StudyRow] = []
    for v in values:
        kw = dict(fixed); kw[label] = v
        for key in keys:
            if METHODS[key].closed_form and label in SWARM_ONLY_LABELS:
                continue
            r = monte_carlo(key, trials, kw["noise_sigma"], kw["n_anchors"],
                            kw["n_particles"], kw["n_iterations"],
                            seed=kw.get("seed", 123), base=kw.get("base"), noise_model=kw.get("noise_model"))
            rows.append(StudyRow(label, float(v), key, r["rmse"], r["mean_error"], r["p90_error"],
                                 r["evaluations"], r["runtime_ms"], r["memory_floats"], trials,
                                 r["crlb_rmse"], r["iterations_mean"]))
    return rows

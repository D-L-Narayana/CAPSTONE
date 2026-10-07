"""Multi-node iterative auto-localization in 3D.

Transposes the distributed iterative scheme of Kulkarni, Venayagamoorthy and Cheng (2009,
Section IV) from the 2-D field used in that paper to the 3-D fields of this project:

1. ``N`` unknown nodes and ``M`` beacons are deployed uniformly at random in the field.
2. Every unknown node measures noisy ranges to the *references* within its radio range ``r``:
   the beacons plus the nodes that were settled in earlier rounds.  Settled nodes advertise
   their *estimated* positions, so estimation errors propagate to later rounds exactly as they
   would in a real deployment.
3. A node is *localizable* when it has at least ``min_refs`` references (four in 3-D, three in
   the paper's 2-D setting).  It uses at most the ``max_refs`` nearest ones (six, a limit the
   paper calls "arbitrarily chosen") and, when required, only a non-coplanar reference set.
4. Every localizable node runs the supplied single-node ``Localizer`` on its own ranges and
   becomes a settled node - a reference for the others - in the next round.
5. Rounds repeat until no new node settles, every node is settled, or ``max_rounds`` is hit.

Metrics follow the paper: ``N_NL`` (nodes that could not be localized) and ``E_l`` (mean squared
distance between true and estimated positions over the localized nodes, eq. 4 with the z term
added), plus RMSE = sqrt(E_l), coverage = settled / N, total fitness evaluations and per-round
statistics.  Only numpy is used here; figures live in :mod:`pso3d.network_plots`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
import time
from typing import Protocol

import numpy as np

from .fitness import RangeErrorFitness
from .trilateration import gauss_newton_refine, least_squares_trilateration


# ------------------------------------------------------------------------------------------ scenario


@dataclass
class NetworkScenario:
    """Random deployment of ``n_beacons`` beacons and ``n_nodes`` unknown nodes in a box field.

    ``generate`` draws from one ``numpy.random.default_rng(seed)`` stream - beacons first, then
    nodes - so the beacon layout does not depend on the number of nodes.
    """

    field_size: tuple[float, float, float] = (100.0, 100.0, 30.0)
    n_nodes: int = 50
    n_beacons: int = 10
    radio_range: float = 25.0
    seed: int = 1

    def __post_init__(self) -> None:
        fs = tuple(float(v) for v in self.field_size)
        if len(fs) != 3 or any(v <= 0 for v in fs):
            raise ValueError("field_size must be three positive lengths")
        self.field_size = fs
        self.n_nodes, self.n_beacons = int(self.n_nodes), int(self.n_beacons)
        if self.n_nodes < 0 or self.n_beacons < 0:
            raise ValueError("n_nodes and n_beacons must be non-negative")
        self.radio_range = float(self.radio_range)
        if self.radio_range <= 0:
            raise ValueError("radio_range must be positive")

    @property
    def lower(self) -> np.ndarray:
        return np.zeros(3)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.field_size, dtype=float)

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(beacons (M, 3), nodes (N, 3))`` - uniform in the field, deterministic per seed."""
        rng = np.random.default_rng(self.seed)
        beacons = rng.uniform(self.lower, self.upper, (self.n_beacons, 3))
        nodes = rng.uniform(self.lower, self.upper, (self.n_nodes, 3))
        return beacons, nodes


# ------------------------------------------------------------------------------------------ localizers


@dataclass
class LocalizeOutcome:
    """What a single-node localizer reports back to the network simulator."""

    position: np.ndarray
    evaluations: int
    iterations: int


class Localizer(Protocol):
    """Single-node estimator used by :func:`iterative_localize`.

    ``anchors`` (K, 3) are the reference positions *as known to the node*, ``measured`` (K,) the
    noisy ranges, ``lower``/``upper`` the field bounds and ``seed`` a per-node integer seed.
    """

    def __call__(self, anchors, measured, lower, upper, seed: int) -> LocalizeOutcome: ...


def _outcome_from_result(res) -> LocalizeOutcome:
    position = np.asarray(getattr(res, "estimate"), dtype=float).reshape(3)
    evaluations = int(getattr(res, "fitness_evaluations", 0))
    iterations = int(getattr(res, "iterations_run", 0))
    return LocalizeOutcome(position, evaluations, iterations)


def _accepts_x0(cls) -> bool:
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False
    return "x0" in params


def _range_cost(anchors: np.ndarray, measured: np.ndarray, p: np.ndarray) -> float:
    """f(p) = sum_j (||p - a_j|| - d_j)^2 without touching any evaluation counter."""
    r = np.linalg.norm(anchors - p, axis=1) - measured
    return float((r * r).sum())


def warm_start(anchors, measured, lower, upper) -> np.ndarray:
    """Closed-form starting point: least squares -> Gauss-Newton -> clip to the field.

    The candidate from the two closed-form functions of this package is always computed.  When
    ``pso3d.trilateration.warm_start_estimate`` (which also handles rank-deficient reference sets)
    is available its candidate is computed as well, and the one with the lower range cost wins.
    The result is finite and inside the field.
    """
    anchors = np.asarray(anchors, dtype=float)
    measured = np.asarray(measured, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    candidates: list[np.ndarray] = []
    try:
        p0 = least_squares_trilateration(anchors, measured)
        if np.all(np.isfinite(p0)):
            p, _ = gauss_newton_refine(anchors, measured, p0)
            if np.all(np.isfinite(p)):
                candidates.append(np.clip(p, lower, upper))
    except np.linalg.LinAlgError:
        pass
    try:
        from .trilateration import warm_start_estimate  # optional helper
    except ImportError:
        warm_start_estimate = None
    if warm_start_estimate is not None:
        try:
            q = np.asarray(warm_start_estimate(anchors, measured, lower, upper), dtype=float).reshape(3)
        except (ValueError, TypeError, np.linalg.LinAlgError):
            q = None
        if q is not None and np.all(np.isfinite(q)):
            candidates.append(np.clip(q, lower, upper))
    if not candidates:
        candidates.append(np.clip(anchors.mean(axis=0), lower, upper))
    return min(candidates, key=lambda c: _range_cost(anchors, measured, c))


class _PSOLocalizer:
    """Runs ``cls(seed=seed, **kwargs)`` on a fresh :class:`RangeErrorFitness` for every node."""

    def __init__(self, cls, kwargs: dict):
        self.cls = cls
        self.kwargs = dict(kwargs)
        self.name = getattr(cls, "__name__", str(cls))
        self.warm_start_supported = False

    def _make(self, seed: int, **extra):
        return self.cls(seed=int(seed), **extra, **self.kwargs)

    def __call__(self, anchors, measured, lower, upper, seed: int) -> LocalizeOutcome:
        anchors = np.asarray(anchors, dtype=float)
        measured = np.asarray(measured, dtype=float)
        fitness = RangeErrorFitness(anchors, measured)
        res = self._make(seed).run(fitness, np.asarray(lower, float), np.asarray(upper, float))
        return _outcome_from_result(res)


class _WarmStartedLocalizer(_PSOLocalizer):
    """PSO localizer started from the closed-form estimate (``x0``) when the class supports it."""

    def __init__(self, cls, kwargs: dict):
        super().__init__(cls, kwargs)
        self.warm_start_supported = bool(_accepts_x0(cls))
        suffix = "+warm_start" if self.warm_start_supported else "+warm_start(unsupported, plain run)"
        self.name = f"{self.name}{suffix}"
        self.last_x0: np.ndarray | None = None

    def __call__(self, anchors, measured, lower, upper, seed: int) -> LocalizeOutcome:
        anchors = np.asarray(anchors, dtype=float)
        measured = np.asarray(measured, dtype=float)
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        x0 = warm_start(anchors, measured, lower, upper)
        self.last_x0 = x0
        fitness = RangeErrorFitness(anchors, measured)
        opt = self._make(seed, x0=x0) if self.warm_start_supported else self._make(seed)
        return _outcome_from_result(opt.run(fitness, lower, upper))


class _ClosedFormLocalizer:
    """Least-squares trilateration refined by Gauss-Newton; evaluations = GN iterations."""

    name = "lsq_gn"
    warm_start_supported = False

    def __call__(self, anchors, measured, lower, upper, seed: int) -> LocalizeOutcome:
        anchors = np.asarray(anchors, dtype=float)
        measured = np.asarray(measured, dtype=float)
        p0 = least_squares_trilateration(anchors, measured)
        if not np.all(np.isfinite(p0)):
            p0 = anchors.mean(axis=0)
        p, used = gauss_newton_refine(anchors, measured, p0)
        if not np.all(np.isfinite(p)):
            p = p0
        return LocalizeOutcome(np.asarray(p, dtype=float).reshape(3), int(used), int(used))


def pso_localizer(cls, **kwargs) -> Localizer:
    """Localizer that runs the optimiser ``cls(seed=seed, **kwargs)`` for every node."""
    return _PSOLocalizer(cls, kwargs)


def lsq_gn_localizer() -> Localizer:
    """Closed-form localizer: least squares -> Gauss-Newton (no swarm)."""
    return _ClosedFormLocalizer()


def warm_started_localizer(cls, **kwargs) -> Localizer:
    """Like :func:`pso_localizer` but passes the closed-form estimate as ``x0``.

    If ``cls`` does not accept ``x0`` the optimiser runs plainly and the returned localizer has
    ``warm_start_supported = False`` so callers can report it honestly.
    """
    return _WarmStartedLocalizer(cls, kwargs)


# ------------------------------------------------------------------------------------------ noise


class _GaussianNoise:
    """Zero-mean Gaussian ranging noise, the default when ``noise`` is a float sigma."""

    def __init__(self, sigma: float):
        self.sigma = float(sigma)
        if self.sigma < 0:
            raise ValueError("noise sigma must be non-negative")
        self.name = f"gaussian({self.sigma:g})"

    def sample(self, true_ranges, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(0.0, self.sigma, size=np.shape(true_ranges))


def _as_noise_model(noise):
    if hasattr(noise, "sample"):
        return noise
    return _GaussianNoise(float(noise))


# ------------------------------------------------------------------------------------------ geometry


def reference_rank(points, rel_tol: float = 1e-9) -> int:
    """Rank of the centred reference matrix (3 = non-coplanar, 2 = coplanar, 1 = collinear).

    Singular values below ``rel_tol`` times the largest one are treated as zero.
    """
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    if pts.shape[0] < 2:
        return 0
    centred = pts - pts.mean(axis=0)
    s = np.linalg.svd(centred, compute_uv=False)
    if s.size == 0 or not np.isfinite(s[0]) or s[0] <= 0.0:
        return 0
    return int(np.count_nonzero(s > rel_tol * s[0]))


def is_non_coplanar_set(points, rel_tol: float = 1e-9) -> bool:
    """True when the points span all three dimensions (needed for a unique 3-D fix)."""
    return reference_rank(points, rel_tol) == 3


# ------------------------------------------------------------------------------------------ results


@dataclass
class RoundStats:
    round: int
    newly_settled: int
    settled_total: int
    refs_available_mean: float     # mean number of references in radio range over the nodes unsettled at round start
    evaluations: int               # fitness evaluations spent by the nodes localized in this round


@dataclass
class NetworkResult:
    rounds: list[RoundStats]
    estimates: np.ndarray          # (N, 3); NaN rows for nodes that were never localized
    settled: np.ndarray            # (N,) bool
    n_not_localized: int           # N_NL
    e_l: float                     # mean squared error over settled nodes (NaN when none settled)
    rmse: float                    # sqrt(e_l)
    coverage_fraction: float       # settled / N
    total_evaluations: int
    runtime_s: float
    # extras (defaults keep the positional contract above intact)
    settled_round: np.ndarray | None = None    # (N,) round in which each node settled, 0 = never
    n_refs_used: np.ndarray | None = None      # (N,) references used by each settled node
    errors: np.ndarray | None = None           # (N,) Euclidean error, NaN for unsettled nodes
    n_beacons: int | None = None
    field_size: tuple[float, float, float] | None = None
    radio_range: float | None = None
    localizer_name: str = ""
    noise_name: str = ""
    settings: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.settled.shape[0])

    @property
    def n_localized(self) -> int:
        return int(self.settled.sum())


# ------------------------------------------------------------------------------------------ scheme


def iterative_localize(scenario: NetworkScenario, localizer: Localizer, noise=0.5, max_rounds: int = 10,
                       min_refs: int = 4, max_refs: int = 6, require_non_coplanar: bool = True,
                       seed: int | None = None, positions=None) -> NetworkResult:
    """Run the iterative multi-node scheme and return the metrics.

    ``noise`` is an object with ``.sample(true_ranges, rng)`` (additive noise, same shape) or a
    float Gaussian sigma.  ``seed`` drives the noise and the per-node localizer seeds (default
    ``scenario.seed + 1``); the deployment itself comes from ``scenario.generate()`` unless
    ``positions=(beacons, nodes)`` is given.
    """
    min_refs, max_refs, max_rounds = int(min_refs), int(max_refs), int(max_rounds)
    if min_refs < 1 or max_refs < min_refs:
        raise ValueError("need 1 <= min_refs <= max_refs")
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    t0 = time.perf_counter()
    if positions is None:
        beacons, nodes = scenario.generate()
    else:
        beacons, nodes = positions
    beacons = np.atleast_2d(np.asarray(beacons, dtype=float)).reshape(-1, 3)
    nodes = np.atleast_2d(np.asarray(nodes, dtype=float)).reshape(-1, 3)
    n = nodes.shape[0]
    lower = np.zeros(3)
    upper = np.asarray(scenario.field_size, dtype=float)
    radio_range = float(scenario.radio_range)
    rng = np.random.default_rng(int(scenario.seed) + 1 if seed is None else int(seed))
    noise_model = _as_noise_model(noise)

    estimates = np.full((n, 3), np.nan)
    settled = np.zeros(n, dtype=bool)
    settled_round = np.zeros(n, dtype=int)
    n_refs_used = np.zeros(n, dtype=int)
    rounds: list[RoundStats] = []

    for k in range(1, max_rounds + 1):
        pending = np.flatnonzero(~settled)
        if pending.size == 0:
            break
        # references at the start of the round: beacons (exact) + settled nodes (their ESTIMATES);
        # radio range and the measured ranges are physical, i.e. computed from the TRUE positions
        ref_known = np.vstack([beacons, estimates[settled]])
        ref_true = np.vstack([beacons, nodes[settled]])
        counts: list[int] = []
        newly: dict[int, tuple[np.ndarray, int, int]] = {}
        evaluations = 0
        for i in pending:
            d_true = np.linalg.norm(ref_true - nodes[i], axis=1) if ref_true.shape[0] else np.zeros(0)
            in_range = np.flatnonzero(d_true <= radio_range)
            counts.append(int(in_range.size))
            if in_range.size < min_refs:
                continue
            order = in_range[np.argsort(d_true[in_range], kind="stable")]
            chosen = order[:max_refs]
            anchors = ref_known[chosen]
            if require_non_coplanar and not is_non_coplanar_set(anchors):
                continue
            true_ranges = d_true[chosen]
            measured = true_ranges + np.asarray(noise_model.sample(true_ranges, rng), dtype=float)
            node_seed = int(rng.integers(0, 2**31 - 1))
            outcome = localizer(anchors, measured, lower, upper, node_seed)
            p = np.asarray(outcome.position, dtype=float).reshape(3)
            if not np.all(np.isfinite(p)):
                p = anchors.mean(axis=0)
            p = np.clip(p, lower, upper)
            evaluations += int(outcome.evaluations)
            newly[int(i)] = (p, int(chosen.size), int(outcome.evaluations))
        for i, (p, k_refs, _) in newly.items():
            estimates[i] = p
            settled[i] = True
            settled_round[i] = k
            n_refs_used[i] = k_refs
        rounds.append(RoundStats(k, len(newly), int(settled.sum()),
                                 float(np.mean(counts)) if counts else 0.0, int(evaluations)))
        if not newly:
            break

    errors = np.full(n, np.nan)
    if settled.any():
        errors[settled] = np.linalg.norm(estimates[settled] - nodes[settled], axis=1)
        e_l = float(np.mean(errors[settled] ** 2))
        rmse = float(np.sqrt(e_l))
    else:
        e_l = rmse = float("nan")
    coverage = float(settled.mean()) if n else 0.0
    return NetworkResult(
        rounds=rounds, estimates=estimates, settled=settled, n_not_localized=int((~settled).sum()),
        e_l=e_l, rmse=rmse, coverage_fraction=coverage,
        total_evaluations=int(sum(r.evaluations for r in rounds)), runtime_s=time.perf_counter() - t0,
        settled_round=settled_round, n_refs_used=n_refs_used, errors=errors, n_beacons=int(beacons.shape[0]),
        field_size=tuple(float(v) for v in upper), radio_range=radio_range,
        localizer_name=str(getattr(localizer, "name", type(localizer).__name__)),
        noise_name=str(getattr(noise_model, "name", type(noise_model).__name__)),
        settings=dict(max_rounds=max_rounds, min_refs=min_refs, max_refs=max_refs,
                      require_non_coplanar=bool(require_non_coplanar)),
    )


# ------------------------------------------------------------------------------------------ reporting


def settled_only_error(result: NetworkResult, nodes) -> np.ndarray:
    """Euclidean errors of the settled nodes only (the quantity averaged by ``E_l``)."""
    nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
    mask = np.asarray(result.settled, dtype=bool)
    return np.linalg.norm(np.asarray(result.estimates, dtype=float)[mask] - nodes[mask], axis=1)


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def summary_table(result: NetworkResult) -> str:
    """Markdown summary (overall metrics followed by a per-round table)."""
    n = result.n_nodes
    lines = ["| metric | value |", "|---|---|",
             f"| nodes N | {n} |"]
    if result.n_beacons is not None:
        lines.append(f"| beacons M | {result.n_beacons} |")
    if result.radio_range is not None:
        lines.append(f"| radio range r [m] | {result.radio_range:g} |")
    if result.field_size is not None:
        lines.append("| field [m] | " + " x ".join(f"{v:g}" for v in result.field_size) + " |")
    if result.localizer_name:
        lines.append(f"| localizer | {result.localizer_name} |")
    if result.noise_name:
        lines.append(f"| noise | {result.noise_name} |")
    lines += [f"| rounds run | {len(result.rounds)} |",
              f"| localized nodes N_L | {result.n_localized} |",
              f"| not localized N_NL | {result.n_not_localized} |",
              f"| coverage (settled / N) | {100.0 * result.coverage_fraction:.1f} % |",
              f"| E_l = mean squared error over localized nodes [m^2] | {_fmt(result.e_l)} |",
              f"| RMSE over localized nodes [m] | {_fmt(result.rmse)} |",
              f"| total fitness evaluations | {result.total_evaluations} |",
              f"| runtime [s] | {result.runtime_s:.3f} |",
              "",
              "| round | newly settled | settled total | mean references in range | evaluations |",
              "|---|---|---|---|---|"]
    for r in result.rounds:
        lines.append(f"| {r.round} | {r.newly_settled} | {r.settled_total} | {r.refs_available_mean:.2f} | {r.evaluations} |")
    return "\n".join(lines)

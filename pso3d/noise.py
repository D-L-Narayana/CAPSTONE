"""Ranging-noise models for range-based 3D localization.

Every model is *additive on the distance domain*::

    measured_j = ||p - a_j|| + n_j,        n = model.sample(true_ranges, rng)

and implements the :class:`NoiseModel` protocol: a ``name`` plus ``sample(true_ranges, rng)`` returning
an array shaped like ``true_ranges`` (the last axis indexes anchors; ``(M,)`` or ``(K, M)`` inputs work).
Models are frozen dataclasses - value equality, hashable, no internal state - and all randomness comes
from the ``numpy.random.Generator`` passed to ``sample``, so results are reproducible per seed.

======================================  =============================================================
model                                   noise on anchor j
======================================  =============================================================
``FixedOffsets(offsets)``               ``n_j = offsets[j]``  (the Review-1 baseline numbers)
``Gaussian(sigma)``                     ``n_j ~ N(0, sigma^2)``  (metres)
``UniformPercent(pn_percent)``          ``n_j = d_j * U(-1, 1) * Pn / 100``  (Kulkarni et al. 2009)
``LogNormalShadowing(sigma_db, ...)``   RSSI round trip through a log-distance path-loss model
``NLOS(p, bias_mean, bias_sigma, base)`` base noise plus a positive bias on a random subset of anchors
======================================  =============================================================

Spec strings understood by :func:`from_spec` (the form used by the command-line tools)::

    gaussian:0.5 | percent:2 | nlos:0.2 | nlos:0.2:3.0:1.0 | nlos:0.2:3.0:1.0:0.7 | nlos:0.2@percent:2
    shadowing:4 | shadowing:4:3.0:-40 | fixed:0.4,-0.3,0.5,-0.4 | 0.5 (= gaussian:0.5) | none

See ``docs/noise_models.md`` for formulas, rationale and limitations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "NoiseModel",
    "FixedOffsets",
    "Gaussian",
    "UniformPercent",
    "LogNormalShadowing",
    "NLOS",
    "rssi_from_distance",
    "distance_from_rssi",
    "rssi_round_trip",
    "measure",
    "from_spec",
    "as_noise_model",
    "SPEC_HELP",
]

SPEC_HELP = (
    "gaussian:SIGMA | percent:PN | nlos:P[:BIAS_MEAN[:BIAS_SIGMA[:BASE_SIGMA]]] | "
    "nlos:P[:BIAS_MEAN[:BIAS_SIGMA]]@BASE_SPEC | shadowing:SIGMA_DB[:EXPONENT[:P_REF_DBM]] | "
    "fixed:N1,N2,... | SIGMA (bare number = gaussian:SIGMA) | none (= gaussian:0)"
)

DEFAULT_PATH_LOSS_EXPONENT = 2.8
DEFAULT_REFERENCE_POWER_DBM = -45.0
DEFAULT_NLOS_BASE_SIGMA = 0.5


@runtime_checkable
class NoiseModel(Protocol):
    """Additive ranging noise: ``sample`` returns an array shaped like ``true_ranges``."""

    name: str

    def sample(self, true_ranges: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...


# --------------------------------------------------------------------------- helpers


def _ranges(true_ranges) -> np.ndarray:
    return np.asarray(true_ranges, dtype=float)


def _finite(value, what: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    return value


def _fmt(value: float) -> str:
    """Shortest decimal text that parses back to the same float."""
    return repr(float(value))


def _as_generator(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


# --------------------------------------------------------------------------- models


@dataclass(frozen=True)
class FixedOffsets:
    """Deterministic per-anchor offsets ``n_j = offsets[j]`` in metres.

    ``FixedOffsets((0.4, -0.3, 0.5, -0.4))`` reproduces the Review-1 baseline measurement exactly.
    The offsets are stored as a tuple of floats so that models compare by value.
    """

    offsets: tuple[float, ...]
    name: ClassVar[str] = "fixed"

    def __post_init__(self) -> None:
        arr = np.asarray(self.offsets, dtype=float)
        if arr.ndim != 1 or arr.size == 0:
            raise ValueError(f"offsets must be a non-empty 1-D sequence, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("offsets must be finite numbers")
        object.__setattr__(self, "offsets", tuple(float(v) for v in arr))

    def sample(self, true_ranges, rng: np.random.Generator | None = None) -> np.ndarray:
        d = _ranges(true_ranges)
        if d.ndim == 0 or d.shape[-1] != len(self.offsets):
            raise ValueError(
                f"FixedOffsets has {len(self.offsets)} offsets but the last axis of true_ranges "
                f"has shape {d.shape}")
        return np.broadcast_to(np.asarray(self.offsets, dtype=float), d.shape).copy()

    def describe(self) -> str:
        return "fixed:" + ",".join(_fmt(v) for v in self.offsets)


@dataclass(frozen=True)
class Gaussian:
    """Zero-mean Gaussian ranging noise ``n_j ~ N(0, sigma^2)`` (metres, independent per anchor).

    One normal draw per range is consumed even when ``sigma == 0`` so that random streams stay
    aligned across noise levels.
    """

    sigma: float
    name: ClassVar[str] = "gaussian"

    def __post_init__(self) -> None:
        sigma = _finite(self.sigma, "sigma")
        if sigma < 0.0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        object.__setattr__(self, "sigma", sigma)

    def sample(self, true_ranges, rng: np.random.Generator) -> np.ndarray:
        d = _ranges(true_ranges)
        return np.asarray(rng.normal(0.0, self.sigma, size=d.shape), dtype=float)

    def describe(self) -> str:
        return f"gaussian:{_fmt(self.sigma)}"


@dataclass(frozen=True)
class UniformPercent:
    """Range-proportional uniform noise ``n_j = d_j * U(-1, 1) * Pn / 100``.

    This is the ``d +- d * Pn / 100`` ranging error of Kulkarni et al. (2009): bounded by
    ``d_j * Pn / 100``, zero mean, standard deviation ``d_j * Pn / (100 * sqrt(3))``.
    """

    pn_percent: float
    name: ClassVar[str] = "percent"

    def __post_init__(self) -> None:
        pn = _finite(self.pn_percent, "pn_percent")
        if pn < 0.0:
            raise ValueError(f"pn_percent must be >= 0, got {pn}")
        object.__setattr__(self, "pn_percent", pn)

    def sample(self, true_ranges, rng: np.random.Generator) -> np.ndarray:
        d = _ranges(true_ranges)
        u = rng.uniform(-1.0, 1.0, size=d.shape)
        return np.asarray(d * u * (self.pn_percent / 100.0), dtype=float)

    def describe(self) -> str:
        return f"percent:{_fmt(self.pn_percent)}"


@dataclass(frozen=True)
class LogNormalShadowing:
    """RSSI-derived ranging error from a log-distance path-loss model with log-normal shadowing.

    ``P(d) = P_ref - 10 n log10(d)``; ``P_noisy = P(d) + X`` with ``X ~ N(0, sigma_db^2)`` in dB;
    ``d_hat = 10^((P_ref - P_noisy) / (10 n)) = d * 10^(-X / (10 n))``; the noise is ``d_hat - d``.
    The error is multiplicative (it scales with the distance) and skewed (positive mean); it is
    exactly zero when ``sigma_db == 0``. One normal draw per range is consumed.
    """

    sigma_db: float
    path_loss_exponent: float = DEFAULT_PATH_LOSS_EXPONENT
    reference_power_dbm: float = DEFAULT_REFERENCE_POWER_DBM
    name: ClassVar[str] = "shadowing"

    def __post_init__(self) -> None:
        sigma_db = _finite(self.sigma_db, "sigma_db")
        if sigma_db < 0.0:
            raise ValueError(f"sigma_db must be >= 0, got {sigma_db}")
        object.__setattr__(self, "sigma_db", sigma_db)
        object.__setattr__(self, "path_loss_exponent", _check_exponent(self.path_loss_exponent))
        object.__setattr__(self, "reference_power_dbm", _finite(self.reference_power_dbm, "reference_power_dbm"))

    def sample(self, true_ranges, rng: np.random.Generator) -> np.ndarray:
        d = _ranges(true_ranges)
        noise_db = rng.normal(0.0, self.sigma_db, size=d.shape)
        return rssi_round_trip(d, noise_db, self.path_loss_exponent, self.reference_power_dbm) - d

    def describe(self) -> str:
        return (f"shadowing:{_fmt(self.sigma_db)}:{_fmt(self.path_loss_exponent)}:"
                f"{_fmt(self.reference_power_dbm)}")


@dataclass(frozen=True)
class NLOS:
    """Non-line-of-sight mixture: base noise plus a positive bias on a random subset of anchors.

    For each anchor independently, with probability ``p_nlos`` the range receives an extra bias
    ``|N(bias_mean, bias_sigma^2)|`` (obstructed paths only lengthen a range). Draw order per call:
    the base model's sample, one uniform per anchor (the NLOS mask), one normal per anchor (the bias),
    so ``p_nlos = 0`` reproduces the base model exactly for the same seed. ``base`` defaults to
    ``Gaussian(0.5)``. The default bias parameters are illustrative, not taken from any paper.
    """

    p_nlos: float
    bias_mean: float = 2.0
    bias_sigma: float = 1.0
    base: NoiseModel | None = None
    name: ClassVar[str] = "nlos"

    def __post_init__(self) -> None:
        p = _finite(self.p_nlos, "p_nlos")
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p_nlos must be within [0, 1], got {p}")
        mean = _finite(self.bias_mean, "bias_mean")
        if mean < 0.0:
            raise ValueError(f"bias_mean must be >= 0, got {mean}")
        sigma = _finite(self.bias_sigma, "bias_sigma")
        if sigma < 0.0:
            raise ValueError(f"bias_sigma must be >= 0, got {sigma}")
        base = Gaussian(DEFAULT_NLOS_BASE_SIGMA) if self.base is None else self.base
        if not callable(getattr(base, "sample", None)):
            raise TypeError("base must be a noise model with a sample(true_ranges, rng) method")
        object.__setattr__(self, "p_nlos", p)
        object.__setattr__(self, "bias_mean", mean)
        object.__setattr__(self, "bias_sigma", sigma)
        object.__setattr__(self, "base", base)

    def sample(self, true_ranges, rng: np.random.Generator) -> np.ndarray:
        d = _ranges(true_ranges)
        base = np.asarray(self.base.sample(d, rng), dtype=float)
        if base.shape != d.shape:
            raise ValueError(f"base model returned shape {base.shape}, expected {d.shape}")
        affected = rng.random(d.shape) < self.p_nlos
        bias = np.abs(rng.normal(self.bias_mean, self.bias_sigma, size=d.shape))
        return base + np.where(affected, bias, 0.0)

    def describe(self) -> str:
        head = f"nlos:{_fmt(self.p_nlos)}:{_fmt(self.bias_mean)}:{_fmt(self.bias_sigma)}"
        if isinstance(self.base, Gaussian):
            if self.base.sigma == DEFAULT_NLOS_BASE_SIGMA:
                return head
            return f"{head}:{_fmt(self.base.sigma)}"
        describe = getattr(self.base, "describe", None)
        if describe is None:
            raise ValueError(f"base model {self.base!r} has no spec form")
        return f"{head}@{describe()}"


# --------------------------------------------------------------------------- RSSI helpers


def _check_exponent(path_loss_exponent) -> float:
    n = _finite(path_loss_exponent, "path_loss_exponent")
    if n <= 0.0:
        raise ValueError(f"path_loss_exponent must be > 0, got {n}")
    return n


def rssi_from_distance(distances, path_loss_exponent: float = DEFAULT_PATH_LOSS_EXPONENT,
                       reference_power_dbm: float = DEFAULT_REFERENCE_POWER_DBM) -> np.ndarray:
    """Received power ``P(d) = P_ref - 10 n log10(d)`` in dBm (``P_ref`` is the power at 1 m)."""
    n = _check_exponent(path_loss_exponent)
    p_ref = _finite(reference_power_dbm, "reference_power_dbm")
    d = _ranges(distances)
    with np.errstate(divide="ignore"):
        return np.asarray(p_ref - 10.0 * n * np.log10(d), dtype=float)


def distance_from_rssi(rssi_dbm, path_loss_exponent: float = DEFAULT_PATH_LOSS_EXPONENT,
                       reference_power_dbm: float = DEFAULT_REFERENCE_POWER_DBM) -> np.ndarray:
    """Inverse of :func:`rssi_from_distance`: ``d = 10^((P_ref - P) / (10 n))``."""
    n = _check_exponent(path_loss_exponent)
    p_ref = _finite(reference_power_dbm, "reference_power_dbm")
    p = _ranges(rssi_dbm)
    return np.asarray(np.power(10.0, (p_ref - p) / (10.0 * n)), dtype=float)


def rssi_round_trip(true_ranges, noise_db, path_loss_exponent: float = DEFAULT_PATH_LOSS_EXPONENT,
                    reference_power_dbm: float = DEFAULT_REFERENCE_POWER_DBM) -> np.ndarray:
    """Distances estimated from RSSI after adding the *given* dB offsets ``noise_db``.

    Algebraically ``10^((P_ref - (P(d) + X)) / (10 n)) = d * 10^(-X / (10 n))``: the reference power
    cancels, so the closed form is evaluated, which returns exactly ``d`` when ``X == 0``.
    Example (exponent 2.8): true ``[10, 25, 40]`` m with ``[2, -3, 1.5]`` dB give ``[8.48, 32.00, 35.36]`` m.
    """
    n = _check_exponent(path_loss_exponent)
    _finite(reference_power_dbm, "reference_power_dbm")
    d = _ranges(true_ranges)
    x = _ranges(noise_db)
    return np.asarray(d * np.power(10.0, -x / (10.0 * n)), dtype=float)


# --------------------------------------------------------------------------- measure / coercion / specs


def as_noise_model(obj) -> NoiseModel:
    """Coerce ``obj`` to a noise model.

    Accepts a noise model (anything with a ``sample`` method), a spec string (see :func:`from_spec`),
    a number (Gaussian sigma in metres) or a sequence of per-anchor fixed offsets.
    """
    if callable(getattr(obj, "sample", None)):
        return obj
    if isinstance(obj, str):
        return from_spec(obj)
    if isinstance(obj, (bool, np.bool_)):
        raise TypeError("a boolean is not a noise model")
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return Gaussian(float(obj))
    if isinstance(obj, (int, float, np.integer, np.floating)):
        return Gaussian(float(obj))
    if isinstance(obj, (list, tuple, np.ndarray)):
        return FixedOffsets(obj)
    raise TypeError(f"cannot interpret {type(obj).__name__} as a noise model "
                    "(expected a NoiseModel, a spec string, a sigma or a sequence of offsets)")


def measure(anchors, true_position, model, rng=None) -> np.ndarray:
    """Measured ranges ``||p - a_j|| + n_j`` with ``n = model.sample(true_ranges, rng)``.

    ``anchors`` is ``(M, 3)``, ``true_position`` ``(3,)``; ``model`` is anything accepted by
    :func:`as_noise_model`; ``rng`` is a ``numpy.random.Generator``, an integer seed or ``None``
    (fresh entropy - pass a seed for reproducible runs).
    """
    a = np.asarray(anchors, dtype=float)
    p = np.asarray(true_position, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"anchors must have shape (M, 3), got {a.shape}")
    if p.shape != (3,):
        raise ValueError(f"true_position must have shape (3,), got {p.shape}")
    d = np.linalg.norm(a - p, axis=1)
    noise = np.asarray(as_noise_model(model).sample(d, _as_generator(rng)), dtype=float)
    if noise.shape != d.shape:
        raise ValueError(f"noise model returned shape {noise.shape}, expected {d.shape}")
    return d + noise


_ALIASES = {
    "gaussian": "gaussian", "normal": "gaussian",
    "percent": "percent", "uniform": "percent",
    "nlos": "nlos",
    "shadowing": "shadowing", "rssi": "shadowing", "lognormal": "shadowing",
    "fixed": "fixed",
}


def _num(token: str, what: str) -> float:
    token = token.strip()
    if not token:
        raise ValueError(f"{what}: missing number")
    try:
        value = float(token)
    except ValueError:
        raise ValueError(f"{what}: {token!r} is not a number") from None
    return _finite(value, what)


def _numbers(rest: str, kind: str, low: int, high: int) -> list[float]:
    parts = [p.strip() for p in rest.split(":")] if rest.strip() else []
    if not low <= len(parts) <= high:
        expected = str(low) if low == high else f"{low}-{high}"
        raise ValueError(f"noise spec {kind!r} takes {expected} parameter(s), got {len(parts)}")
    return [_num(part, f"{kind} parameter {i + 1}") for i, part in enumerate(parts)]


def from_spec(spec: str) -> NoiseModel:
    """Parse a noise spec string (case-insensitive, surrounding whitespace ignored).

    Grammar (``SPEC_HELP``)::

        gaussian:SIGMA
        percent:PN
        nlos:P[:BIAS_MEAN[:BIAS_SIGMA[:BASE_SIGMA]]]        base = Gaussian(BASE_SIGMA), default 0.5
        nlos:P[:BIAS_MEAN[:BIAS_SIGMA]]@BASE_SPEC           any other base model
        shadowing:SIGMA_DB[:EXPONENT[:P_REF_DBM]]
        fixed:N1,N2,...
        SIGMA                                               bare number = gaussian:SIGMA
        none                                                = gaussian:0

    Raises ``ValueError`` for unknown kinds, wrong arity, non-numeric fields or invalid values.
    """
    if not isinstance(spec, str):
        raise TypeError(f"noise spec must be a string, got {type(spec).__name__}")
    text = spec.strip().lower()
    if not text:
        raise ValueError("empty noise spec")
    kind, sep, rest = text.partition(":")
    kind, rest = kind.strip(), rest.strip()
    if not kind:
        raise ValueError(f"noise spec {spec!r} has no kind before ':'")
    if kind == "none" and not sep:
        return Gaussian(0.0)
    canonical = _ALIASES.get(kind)
    if canonical is None:
        if sep:
            raise ValueError(f"unknown noise model {kind!r} in {spec!r}; expected one of "
                             f"{sorted(set(_ALIASES.values()))} (see SPEC_HELP)")
        try:
            sigma = float(kind)
        except ValueError:
            raise ValueError(f"unknown noise spec {spec!r} (see SPEC_HELP)") from None
        return Gaussian(_finite(sigma, "sigma"))
    if not sep:
        raise ValueError(f"noise spec {kind!r} needs parameters, e.g. '{kind}:...' (see SPEC_HELP)")
    if canonical == "gaussian":
        (sigma,) = _numbers(rest, kind, 1, 1)
        return Gaussian(sigma)
    if canonical == "percent":
        (pn,) = _numbers(rest, kind, 1, 1)
        return UniformPercent(pn)
    if canonical == "shadowing":
        return LogNormalShadowing(*_numbers(rest, kind, 1, 3))
    if canonical == "fixed":
        return FixedOffsets([_num(token, "fixed offset") for token in rest.split(",")])
    head, at, base_spec = rest.partition("@")
    numbers = _numbers(head, kind, 1, 4)
    if at:
        if len(numbers) == 4:
            raise ValueError("nlos spec: give either a base sigma or an '@BASE_SPEC', not both")
        if not base_spec.strip():
            raise ValueError("nlos spec: empty base spec after '@'")
        base = from_spec(base_spec)
    else:
        base = Gaussian(numbers[3]) if len(numbers) == 4 else None
    return NLOS(*numbers[:3], base=base)

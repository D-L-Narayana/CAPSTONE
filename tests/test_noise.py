"""Behavioural tests for ``pso3d.noise``: ranging-noise models, ``measure``, ``from_spec`` and the shared fixture.

Run with ``python3 -m pytest -q tests/test_noise.py``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pso3d.config import Scenario
from pso3d.noise import (
    NLOS,
    FixedOffsets,
    Gaussian,
    LogNormalShadowing,
    NoiseModel,
    UniformPercent,
    as_noise_model,
    distance_from_rssi,
    from_spec,
    measure,
    rssi_from_distance,
    rssi_round_trip,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "noise_cases.json"
BASELINE_OFFSETS = (0.4, -0.3, 0.5, -0.4)
FIXTURE_SEED = 2026

# One representative instance per model; the dict key is also the expected ``name``.
MODEL_FACTORIES = {
    "fixed": lambda: FixedOffsets(BASELINE_OFFSETS),
    "gaussian": lambda: Gaussian(0.5),
    "percent": lambda: UniformPercent(2.0),
    "shadowing": lambda: LogNormalShadowing(4.0),
    "nlos": lambda: NLOS(0.3, 2.0, 1.0),
}
STOCHASTIC = ("gaussian", "percent", "shadowing", "nlos")

# p03 of the "10 codes" examples: RSSI round trip with fixed dB offsets.
P03_TRUE = [10.0, 25.0, 40.0]
P03_NOISE_DB = [2.0, -3.0, 1.5]
P03_ESTIMATED = [8.48, 32.00, 35.36]
P03_ERRORS = [-1.52, 7.00, -4.64]


def true_ranges() -> np.ndarray:
    return Scenario().true_ranges()


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- protocol / shapes


@pytest.mark.parametrize("key", list(MODEL_FACTORIES))
def test_models_have_names_and_satisfy_the_protocol(key):
    model = MODEL_FACTORIES[key]()
    assert model.name == key
    assert isinstance(model, NoiseModel)
    assert callable(model.sample)


@pytest.mark.parametrize("key", list(MODEL_FACTORIES))
def test_sample_shape_and_dtype_follow_the_input(key):
    model = MODEL_FACTORIES[key]()
    one_d = model.sample(true_ranges(), rng(1))
    assert isinstance(one_d, np.ndarray)
    assert one_d.shape == (4,)
    assert one_d.dtype == np.float64
    two_d = model.sample(np.tile(true_ranges(), (5, 1)), rng(1))
    assert two_d.shape == (5, 4)
    as_list = model.sample(list(true_ranges()), rng(1))
    assert as_list.shape == (4,)


@pytest.mark.parametrize("key", list(MODEL_FACTORIES))
def test_same_seed_gives_identical_samples(key):
    model = MODEL_FACTORIES[key]()
    a = model.sample(true_ranges(), rng(42))
    b = model.sample(true_ranges(), rng(42))
    assert np.array_equal(a, b)


@pytest.mark.parametrize("key", STOCHASTIC)
def test_different_seeds_give_different_non_trivial_samples(key):
    model = MODEL_FACTORIES[key]()
    a = model.sample(true_ranges(), rng(1))
    b = model.sample(true_ranges(), rng(2))
    assert np.any(a != 0.0) and np.any(b != 0.0)
    assert not np.allclose(a, b)


def test_sample_does_not_mutate_inputs_or_model_state():
    d = true_ranges()
    d_copy = d.copy()
    model = FixedOffsets(BASELINE_OFFSETS)
    out = model.sample(d, rng(0))
    out += 100.0
    assert np.array_equal(d, d_copy)
    assert np.allclose(model.sample(d, rng(0)), BASELINE_OFFSETS)


# --------------------------------------------------------------------------- FixedOffsets


def test_fixed_offsets_returns_the_offsets_exactly_and_checks_length():
    model = FixedOffsets(BASELINE_OFFSETS)
    assert np.array_equal(model.sample(true_ranges(), rng(0)), np.asarray(BASELINE_OFFSETS))
    assert np.array_equal(model.sample(np.ones((3, 4)), rng(0)), np.tile(BASELINE_OFFSETS, (3, 1)))
    with pytest.raises(ValueError):
        model.sample(np.ones(3), rng(0))
    with pytest.raises(ValueError):
        model.sample(np.ones(5), rng(0))


# --------------------------------------------------------------------------- Gaussian


def test_gaussian_std_within_10_percent_over_20000_samples():
    model = Gaussian(0.5)
    s = model.sample(np.full(20000, 10.0), rng(7))
    assert abs(s.std() - 0.5) <= 0.05
    assert abs(s.mean()) <= 0.02


def test_gaussian_zero_sigma_is_exactly_zero_and_negative_sigma_rejected():
    assert np.all(Gaussian(0.0).sample(true_ranges(), rng(3)) == 0.0)
    with pytest.raises(ValueError):
        Gaussian(-0.1)


# --------------------------------------------------------------------------- UniformPercent


def test_uniform_percent_bounds_and_zero_mean():
    d = np.linspace(5.0, 50.0, 20000)
    model = UniformPercent(2.0)
    s = model.sample(d, rng(5))
    bound = d * 2.0 / 100.0
    assert np.all(np.abs(s) <= bound + 1e-12)
    relative = s / d
    assert abs(relative.mean()) <= 0.005
    # U(-0.02, 0.02) has standard deviation 0.02 / sqrt(3)
    assert abs(relative.std() - 0.02 / np.sqrt(3.0)) <= 0.1 * 0.02 / np.sqrt(3.0)
    assert np.abs(s).max() > 0.5 * bound.max()


def test_uniform_percent_is_proportional_to_distance_and_validates_pn():
    model = UniformPercent(5.0)
    d = true_ranges()
    a = model.sample(d, rng(9))
    b = model.sample(2.0 * d, rng(9))
    assert np.allclose(b, 2.0 * a, atol=1e-12)
    with pytest.raises(ValueError):
        UniformPercent(-1.0)


# --------------------------------------------------------------------------- LogNormalShadowing / RSSI


def test_shadowing_with_zero_db_noise_returns_exact_zeros():
    model = LogNormalShadowing(0.0)
    s = model.sample(true_ranges(), rng(11))
    assert np.all(s == 0.0)
    assert np.array_equal(rssi_round_trip(true_ranges(), np.zeros(4)), true_ranges())


def test_rssi_round_trip_matches_the_p03_example():
    est = rssi_round_trip(P03_TRUE, P03_NOISE_DB, path_loss_exponent=2.8, reference_power_dbm=-45.0)
    assert est.shape == (3,)
    assert np.array_equal(np.round(est, 2), P03_ESTIMATED)
    assert np.array_equal(np.round(est - np.asarray(P03_TRUE), 2), P03_ERRORS)


def test_rssi_helpers_are_a_consistent_log_distance_model():
    d = np.asarray(P03_TRUE)
    rssi = rssi_from_distance(d, path_loss_exponent=2.8, reference_power_dbm=-45.0)
    # P = P_ref - 10 n log10(d): at 10 m the path loss is exactly 10 n dB
    assert np.isclose(rssi[0], -45.0 - 28.0)
    assert np.allclose(distance_from_rssi(rssi, 2.8, -45.0), d, atol=1e-9)
    literal = distance_from_rssi(rssi + np.asarray(P03_NOISE_DB), 2.8, -45.0)
    assert np.allclose(literal, rssi_round_trip(d, P03_NOISE_DB), atol=1e-9)


def test_shadowing_sample_is_the_round_trip_error_of_a_seeded_db_draw():
    model = LogNormalShadowing(4.0, path_loss_exponent=3.0, reference_power_dbm=-40.0)
    d = true_ranges()
    s = model.sample(d, rng(21))
    noise_db = rng(21).normal(0.0, 4.0, size=d.shape)
    expected = rssi_round_trip(d, noise_db, path_loss_exponent=3.0, reference_power_dbm=-40.0) - d
    assert np.allclose(s, expected, atol=1e-12)
    # multiplicative model: the same dB offset produces an error proportional to the distance
    err = rssi_round_trip([10.0, 20.0], [1.0, 1.0]) - np.array([10.0, 20.0])
    assert np.isclose(err[1], 2.0 * err[0])
    assert err[0] < 0.0  # stronger-than-expected signal -> distance under-estimated


def test_shadowing_validates_parameters():
    with pytest.raises(ValueError):
        LogNormalShadowing(-1.0)
    with pytest.raises(ValueError):
        LogNormalShadowing(2.0, path_loss_exponent=0.0)


# --------------------------------------------------------------------------- NLOS


def test_nlos_with_p_zero_equals_the_base_model_for_the_same_seed():
    d = true_ranges()
    assert np.array_equal(NLOS(0.0, 2.0, 1.0).sample(d, rng(4)), Gaussian(0.5).sample(d, rng(4)))
    custom = UniformPercent(2.0)
    assert np.array_equal(NLOS(0.0, 2.0, 1.0, base=custom).sample(d, rng(4)), custom.sample(d, rng(4)))


def test_nlos_with_p_one_biases_every_anchor_positively():
    d = np.full(500, 20.0)
    extra = NLOS(1.0, 2.0, 1.0).sample(d, rng(8)) - Gaussian(0.5).sample(d, rng(8))
    assert np.all(extra > 0.0)
    assert 1.0 < extra.mean() < 3.0  # |N(2, 1)| has mean ~2.0


@pytest.mark.parametrize("p", [0.2, 0.7])
def test_nlos_fraction_of_biased_anchors_matches_p(p):
    d = np.full(2000, 20.0)
    extra = NLOS(p, 3.0, 1.0).sample(d, rng(13)) - Gaussian(0.5).sample(d, rng(13))
    assert np.all(extra >= 0.0)
    fraction = float(np.mean(extra > 0.0))
    assert abs(fraction - p) <= 0.05


def test_nlos_defaults_and_validation():
    model = NLOS(0.2, 2.0, 1.0)
    assert model.base == Gaussian(0.5)
    assert model.name == "nlos"
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            NLOS(bad, 2.0, 1.0)
    with pytest.raises(ValueError):
        NLOS(0.2, 2.0, -1.0)


# --------------------------------------------------------------------------- measure()


def test_measure_with_fixed_baseline_offsets_matches_the_scenario():
    sc = Scenario()
    m = measure(sc.anchors, sc.true_position, FixedOffsets(sc.noise), rng(0))
    assert m.shape == (4,)
    assert np.allclose(m, sc.measured(), atol=1e-12)
    assert abs(m[0] - 40.21519810323691) < 1e-9


def test_measure_equals_true_ranges_plus_a_seeded_sample():
    sc = Scenario()
    model = Gaussian(0.5)
    m = measure(sc.anchors, sc.true_position, model, rng(99))
    expected = sc.true_ranges() + model.sample(sc.true_ranges(), rng(99))
    assert np.allclose(m, expected, atol=1e-12)


def test_measure_accepts_lists_specs_and_floats_and_validates_shapes():
    sc = Scenario()
    anchors = sc.anchors.tolist()
    p = sc.true_position.tolist()
    assert np.allclose(measure(anchors, p, FixedOffsets(BASELINE_OFFSETS), rng(0)), sc.measured())
    assert np.allclose(measure(anchors, p, list(BASELINE_OFFSETS), rng(0)), sc.measured())
    assert np.allclose(measure(anchors, p, 0.5, rng(5)), measure(anchors, p, Gaussian(0.5), rng(5)))
    assert np.allclose(measure(anchors, p, "gaussian:0.5", 5), measure(anchors, p, Gaussian(0.5), rng(5)))
    with pytest.raises(ValueError):
        measure(np.ones((4, 2)), p, Gaussian(0.5), rng(0))
    with pytest.raises(ValueError):
        measure(anchors, [1.0, 2.0], Gaussian(0.5), rng(0))


# --------------------------------------------------------------------------- from_spec()


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("gaussian:0.5", Gaussian(0.5)),
        ("percent:2", UniformPercent(2.0)),
        ("nlos:0.2", NLOS(0.2, 2.0, 1.0)),
        ("nlos:0.2:3.0:1.0", NLOS(0.2, 3.0, 1.0)),
        ("nlos:0.2:3.0:1.0:0.7", NLOS(0.2, 3.0, 1.0, base=Gaussian(0.7))),
        ("nlos:0.2:2.0:1.0@percent:2", NLOS(0.2, 2.0, 1.0, base=UniformPercent(2.0))),
        ("shadowing:4", LogNormalShadowing(4.0)),
        ("shadowing:4:3.0:-40", LogNormalShadowing(4.0, path_loss_exponent=3.0, reference_power_dbm=-40.0)),
        ("fixed:0.4,-0.3,0.5,-0.4", FixedOffsets(BASELINE_OFFSETS)),
        ("  Gaussian:0.5 ", Gaussian(0.5)),
        ("0.5", Gaussian(0.5)),
        ("none", Gaussian(0.0)),
    ],
)
def test_from_spec_parses_supported_specs(spec, expected):
    model = from_spec(spec)
    assert type(model) is type(expected)
    assert model == expected


def test_from_spec_nlos_default_base_is_gaussian_half_metre():
    model = from_spec("nlos:0.2")
    assert isinstance(model, NLOS)
    assert (model.p_nlos, model.bias_mean, model.bias_sigma) == (0.2, 2.0, 1.0)
    assert model.base == Gaussian(0.5)


@pytest.mark.parametrize(
    "spec",
    ["laplace:1", "gaussian", "gaussian:abc", "gaussian:0.5:1", "percent:-2", "nlos:1.5", "nlos", "fixed", "", ":", "shadowing:-1",
     "nlos:0.2:2.0:1.0:0.7@percent:2", "nlos:0.2@laplace:1"],
)
def test_from_spec_rejects_unknown_or_malformed_specs(spec):
    with pytest.raises(ValueError):
        from_spec(spec)


def test_describe_round_trips_through_from_spec():
    for model in (Gaussian(0.5), UniformPercent(2.0), NLOS(0.2, 3.0, 1.0), LogNormalShadowing(4.0),
                  FixedOffsets(BASELINE_OFFSETS), NLOS(0.25, 2.0, 1.0, base=Gaussian(0.7)),
                  NLOS(0.2, 2.0, 1.0, base=UniformPercent(2.0))):
        text = model.describe()
        assert isinstance(text, str) and text
        assert from_spec(text) == model


def test_as_noise_model_coercions():
    model = Gaussian(0.5)
    assert as_noise_model(model) is model
    assert as_noise_model(0.25) == Gaussian(0.25)
    assert as_noise_model("percent:2") == UniformPercent(2.0)
    assert as_noise_model(BASELINE_OFFSETS) == FixedOffsets(BASELINE_OFFSETS)
    assert as_noise_model(np.asarray(BASELINE_OFFSETS)) == FixedOffsets(BASELINE_OFFSETS)
    with pytest.raises(TypeError):
        as_noise_model({"sigma": 0.5})


# --------------------------------------------------------------------------- shared fixture


def build_noise_cases() -> dict:
    """Deterministic content of ``tests/fixtures/noise_cases.json`` (seeded samples for every model)."""
    sc = Scenario()
    d = sc.true_ranges()
    cases = []
    for key, factory in MODEL_FACTORIES.items():
        model = factory()
        cases.append({
            "id": key,
            "spec": model.describe(),
            "seed": FIXTURE_SEED,
            "sample": model.sample(d, rng(FIXTURE_SEED)).tolist(),
        })
    cases.append({
        "id": "measure_fixed_baseline",
        "spec": FixedOffsets(BASELINE_OFFSETS).describe(),
        "seed": FIXTURE_SEED,
        "measured": measure(sc.anchors, sc.true_position, FixedOffsets(BASELINE_OFFSETS), rng(FIXTURE_SEED)).tolist(),
    })
    cases.append({
        "id": "rssi_p03",
        "true_ranges": P03_TRUE,
        "noise_db": P03_NOISE_DB,
        "path_loss_exponent": 2.8,
        "reference_power_dbm": -45.0,
        "estimated": rssi_round_trip(P03_TRUE, P03_NOISE_DB, 2.8, -45.0).tolist(),
    })
    return {
        "description": "Seeded samples of the pso3d noise models for the slide-16 scenario "
                       "(numpy default_rng(seed) per case) plus the p03 RSSI round-trip example.",
        "anchors": sc.anchors.tolist(),
        "true_position": sc.true_position.tolist(),
        "true_ranges": d.tolist(),
        "cases": cases,
    }


def _assert_json_close(actual, expected, path="root", tol=1e-12):
    if isinstance(expected, dict):
        assert isinstance(actual, dict), path
        assert set(actual) == set(expected), path
        for key in expected:
            _assert_json_close(actual[key], expected[key], f"{path}.{key}", tol)
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_json_close(a, e, f"{path}[{i}]", tol)
    elif isinstance(expected, (str, bool)) or expected is None:
        assert actual == expected, path
    else:
        assert abs(float(actual) - float(expected)) <= tol, f"{path}: {actual!r} != {expected!r}"


def test_fixture_file_round_trips_seeded_samples():
    cases = build_noise_cases()
    by_id = {c["id"]: c for c in cases["cases"]}
    # sanity of the generated content before anything is written
    assert np.allclose(by_id["fixed"]["sample"], BASELINE_OFFSETS)
    assert np.std(by_id["gaussian"]["sample"]) > 0.0
    assert np.array_equal(np.round(by_id["rssi_p03"]["estimated"], 2), P03_ESTIMATED)
    assert abs(by_id["measure_fixed_baseline"]["measured"][0] - 40.21519810323691) < 1e-9
    if not FIXTURE_PATH.exists():
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    loaded = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert [c["id"] for c in loaded["cases"]] == list(by_id)
    _assert_json_close(loaded, cases)
    # the stored samples are reproducible from their spec + seed alone
    for case in loaded["cases"]:
        if "sample" in case:
            model = from_spec(case["spec"])
            regenerated = model.sample(np.asarray(loaded["true_ranges"]), rng(case["seed"]))
            assert np.allclose(regenerated, case["sample"], atol=1e-12)

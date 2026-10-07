"""Tests for ``pso3d.geometry`` - anchor-geometry diagnostics (rank, GDOP, CRLB, flip ambiguity, coverage)."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pso3d.geometry import (
    CRLB,
    anchor_plane,
    anchor_rank,
    coverage_grid,
    crlb,
    flip_ambiguity_risk,
    flip_risk_hint,
    gdop,
    is_non_coplanar,
    mirror_point,
    range_jacobian,
    scenario_report,
    tetrahedron_volume,
)
from pso3d.trilateration import gauss_newton_refine, least_squares_trilateration

# Slide-16 baseline scenario (the same numbers as pso3d.config.Scenario, repeated here so that the
# geometry tests do not depend on the optimiser modules).
ANCHORS = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 5.0], [0.0, 60.0, 6.0], [60.0, 60.0, 20.0]])
TRUE_NODE = np.array([37.0, 12.0, 8.5])
NOISE = np.array([0.4, -0.3, 0.5, -0.4])
SIGMA = 0.5
PSO_ESTIMATE = np.array([37.43, 11.90, 9.29])      # frozen Review-1 result (tests/test_pso.py)
PROBES = [[37.0, 12.0, 8.5], [30.0, 30.0, 10.0], [5.0, 55.0, 2.0]]

FLAT = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [60.0, 60.0, 0.0]])        # all z = 0
NEAR_FLAT = np.array([[0.0, 0.0, 0.0], [60.0, 0.0, 0.0], [0.0, 60.0, 0.0], [60.0, 60.0, 0.01]])
LINE = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]])

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "geometry_cases.json"

# Reference values computed independently from the closed-form definitions in docs/geometry.md.
REF = {
    "gdop": 5.292718244567838,
    "crlb_rmse": 2.646359122283919,
    "per_axis": [0.43803374, 0.59573751, 2.54095255],
    "mirror": [37.84469293, 12.93360797, 3.19351495],
    "mirror_distance": 5.453798089577663,
    "normal": [-0.15488159, -0.17118492, 0.97298891],
    "probe_gdop": [5.292718244567838, 6.9632830895677, 2.2943122836946936],
    # flip_ambiguity_risk = clip(f(p) / max(f(mirror), 1e-12), 0, 1) with the slide-16 measured ranges
    "risk_at_estimate": 0.060680566,      # candidate (37.43, 11.90, 9.29): 0.060680566058...
    "risk_at_truth": 0.735599124,         # candidate (37, 12, 8.5): 0.735599124272...  (2.73 m from the plane)
    "risk_far_from_plane": 0.045014391,   # node (10, 10, 19), 17.5 m from the plane, same noise vector: 0.045014390795...
    "hint_at_truth": 0.719753327,         # 4 sigma^2 / (4 sigma^2 + f0(mirror)) = 1.0 / (1.0 + 0.389364887)
}


def true_ranges(anchors, p):
    return np.linalg.norm(np.asarray(anchors, float) - np.asarray(p, float), axis=1)


def measured_baseline():
    return true_ranges(ANCHORS, TRUE_NODE) + NOISE


# --------------------------------------------------------------------------- volume / rank / coplanarity

def test_regular_tetrahedron_volume():
    verts = np.array([[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    edge = np.linalg.norm(verts[1] - verts[0])
    assert tetrahedron_volume(*verts) == pytest.approx(edge ** 3 / (6.0 * math.sqrt(2.0)), rel=1e-12)
    assert tetrahedron_volume(*verts) == pytest.approx(8.0 / 3.0, rel=1e-12)


def test_tetrahedron_volume_unit_corner_order_and_degenerate():
    corner = ([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1])
    assert tetrahedron_volume(*corner) == pytest.approx(1.0 / 6.0, rel=1e-12)
    a, b, c, d = corner
    assert tetrahedron_volume(d, c, b, a) == pytest.approx(1.0 / 6.0, rel=1e-12)
    assert tetrahedron_volume(*FLAT) == 0.0
    assert tetrahedron_volume(*ANCHORS) == pytest.approx(5400.0, rel=1e-12)   # |det| = 32400


def test_anchor_rank_classifies_layouts():
    assert anchor_rank(ANCHORS) == 3
    assert anchor_rank(FLAT) == 2
    assert anchor_rank(LINE) == 1
    assert anchor_rank(ANCHORS[:1]) == 0
    assert anchor_rank([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) == 0


def test_anchor_rank_tolerance_is_relative_to_the_largest_singular_value():
    assert anchor_rank(NEAR_FLAT) == 3                 # 1 cm of height is far above 1e-9 x 60 m
    assert anchor_rank(NEAR_FLAT, tol=1e-3) == 2       # ... but counts as flat at a 0.1 % tolerance
    assert anchor_rank(1e-6 * ANCHORS) == 3            # scale invariant


def test_is_non_coplanar_baseline_true_flat_false_and_too_few_false():
    assert is_non_coplanar(ANCHORS) is True
    assert is_non_coplanar(FLAT) is False
    assert is_non_coplanar(LINE) is False
    assert is_non_coplanar(ANCHORS[:3]) is False
    assert is_non_coplanar(np.zeros((0, 3))) is False


def test_is_non_coplanar_searches_every_four_anchor_subset():
    five = np.vstack([FLAT, [[30.0, 30.0, 15.0]]])    # four flat anchors plus one above them
    assert is_non_coplanar(five) is True
    assert is_non_coplanar(five, min_volume=1e9) is False
    assert is_non_coplanar(NEAR_FLAT) is True          # volume 6 m^3 > 1e-6
    assert is_non_coplanar(NEAR_FLAT, min_volume=10.0) is False


# --------------------------------------------------------------------------- Jacobian / GDOP

def test_range_jacobian_rows_are_unit_vectors_pointing_from_anchor_to_node():
    J = range_jacobian(ANCHORS, TRUE_NODE)
    assert J.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(J, axis=1), 1.0, atol=1e-12)
    expected0 = (TRUE_NODE - ANCHORS[0]) / np.linalg.norm(TRUE_NODE - ANCHORS[0])
    np.testing.assert_allclose(J[0], expected0, atol=1e-12)
    assert J[0, 0] > 0 and J[1, 0] < 0     # node is +x of anchor 1 (origin) and -x of anchor 2 (x = 60)


def test_range_jacobian_guards_zero_distance():
    J = range_jacobian(ANCHORS, ANCHORS[2])            # node sitting exactly on anchor 3
    assert J.shape == (4, 3)
    assert np.isfinite(J).all()
    np.testing.assert_allclose(J[2], 0.0)
    np.testing.assert_allclose(np.linalg.norm(J[[0, 1, 3]], axis=1), 1.0, atol=1e-12)


def test_gdop_baseline_node_is_finite_and_matches_reference():
    g = gdop(ANCHORS, TRUE_NODE)
    assert math.isfinite(g)
    assert g == pytest.approx(REF["gdop"], rel=1e-9)
    diff = TRUE_NODE - ANCHORS
    J = diff / np.linalg.norm(diff, axis=1)[:, None]
    assert g == pytest.approx(math.sqrt(np.trace(np.linalg.inv(J.T @ J))), rel=1e-12)
    assert gdop(ANCHORS.tolist(), TRUE_NODE.tolist()) == pytest.approx(g)     # list inputs accepted


def test_gdop_is_infinite_for_coplanar_anchors_in_their_plane():
    assert gdop(FLAT, [30.0, 30.0, 0.0]) == float("inf")
    assert gdop(FLAT, [10.0, 50.0, 0.0]) == float("inf")
    assert math.isfinite(gdop(FLAT, [20.0, 40.0, 12.0]))   # above the plane the geometry is fine


def test_gdop_is_infinite_without_three_independent_directions():
    assert gdop(ANCHORS[:2], TRUE_NODE) == float("inf")
    assert gdop(LINE, TRUE_NODE) == float("inf")           # collinear anchors: directions span a plane only
    assert gdop(LINE, [5.0, 5.0, 5.0]) == float("inf")     # node on the anchor line


# --------------------------------------------------------------------------- CRLB

def test_crlb_baseline_vertical_axis_is_the_weakest():
    b = crlb(ANCHORS, TRUE_NODE, SIGMA)
    assert isinstance(b, CRLB)
    assert b.cov.shape == (3, 3) and b.per_axis.shape == (3,)
    assert b.per_axis[2] > b.per_axis[1] and b.per_axis[2] > b.per_axis[0]
    assert b.per_axis[2] > 4 * b.per_axis[0]           # the README's delta-z effect: ~2.5 m vs ~0.44 m
    assert b.rmse_bound == pytest.approx(math.sqrt(float(np.sum(b.per_axis ** 2))), rel=1e-12)
    assert b.rmse_bound == pytest.approx(REF["crlb_rmse"], rel=1e-9)
    np.testing.assert_allclose(b.per_axis, REF["per_axis"], atol=1e-6)
    np.testing.assert_allclose(b.cov, b.cov.T, atol=1e-12)
    assert b.rmse_bound == pytest.approx(SIGMA * gdop(ANCHORS, TRUE_NODE), rel=1e-12)


def test_crlb_scales_linearly_with_sigma():
    b1 = crlb(ANCHORS, TRUE_NODE, 0.5)
    b2 = crlb(ANCHORS, TRUE_NODE, 1.0)
    assert b1.rmse_bound > 0
    assert b2.rmse_bound == pytest.approx(2.0 * b1.rmse_bound, rel=1e-12)
    np.testing.assert_allclose(b2.per_axis, 2.0 * b1.per_axis, rtol=1e-12)
    np.testing.assert_allclose(b2.cov, 4.0 * b1.cov, rtol=1e-12)


def test_crlb_matches_explicit_fisher_information():
    diff = TRUE_NODE - ANCHORS
    J = diff / np.linalg.norm(diff, axis=1)[:, None]
    expected = np.linalg.inv(J.T @ J / SIGMA ** 2)
    b = crlb(ANCHORS, TRUE_NODE, SIGMA)
    np.testing.assert_allclose(b.cov, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(b.per_axis, np.sqrt(np.diag(expected)), rtol=1e-10)


def test_crlb_singular_geometry_gives_infinite_bounds():
    b = crlb(FLAT, [30.0, 30.0, 0.0], SIGMA)
    assert b.rmse_bound == float("inf")
    assert np.isinf(b.per_axis).all() and np.isinf(b.cov).all()


def test_crlb_rejects_non_positive_sigma():
    with pytest.raises(ValueError):
        crlb(ANCHORS, TRUE_NODE, 0.0)
    with pytest.raises(ValueError):
        crlb(ANCHORS, TRUE_NODE, -0.5)


def test_crlb_is_consistent_with_monte_carlo_lsq_gn():
    """20 noisy trials of LSQ + Gauss-Newton at sigma = 0.1 m: the sample RMSE must not undercut the bound by much."""
    sigma = 0.1
    bound = crlb(ANCHORS, TRUE_NODE, sigma).rmse_bound
    rng = np.random.default_rng(6)
    ranges = true_ranges(ANCHORS, TRUE_NODE)
    squared = []
    for _ in range(20):
        d = ranges + rng.normal(0.0, sigma, size=ranges.shape[0])
        p0 = least_squares_trilateration(ANCHORS, d)
        p1, _ = gauss_newton_refine(ANCHORS, d, p0)
        squared.append(float(np.sum((p1 - TRUE_NODE) ** 2)))
    rmse = math.sqrt(sum(squared) / len(squared))
    assert rmse >= 0.8 * bound
    assert rmse <= 1.5 * bound     # GN from the LSQ start is close to efficient: no large gap either


# --------------------------------------------------------------------------- plane / mirror / flip risk

def test_anchor_plane_centroid_and_unit_normal():
    c, n = anchor_plane(ANCHORS)
    np.testing.assert_allclose(c, [30.0, 30.0, 7.75], atol=1e-12)
    assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(n, REF["normal"], atol=1e-6)
    assert np.abs((ANCHORS - c) @ n).max() < 4.0        # anchors stay within a few metres of the plane
    c2, n2 = anchor_plane(FLAT)
    np.testing.assert_allclose(c2, [30.0, 30.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(n2, [0.0, 0.0, 1.0], atol=1e-12)


def test_mirror_of_point_on_anchor_plane_is_itself():
    centroid, _ = anchor_plane(ANCHORS)
    m = mirror_point(ANCHORS, centroid)
    assert m is not None
    np.testing.assert_allclose(m, centroid, atol=1e-9)
    on_flat = np.array([12.0, 48.0, 0.0])
    m2 = mirror_point(FLAT, on_flat)
    assert m2 is not None
    np.testing.assert_allclose(m2, on_flat, atol=1e-9)


def test_mirror_of_true_node_is_on_the_other_side_and_reflects_back():
    m = mirror_point(ANCHORS, TRUE_NODE)
    assert m is not None
    np.testing.assert_allclose(m, REF["mirror"], atol=1e-6)
    assert np.linalg.norm(m - TRUE_NODE) == pytest.approx(REF["mirror_distance"], abs=1e-6)
    back = mirror_point(ANCHORS, m)
    assert back is not None
    np.testing.assert_allclose(back, TRUE_NODE, atol=1e-9)
    c, n = anchor_plane(ANCHORS)
    assert float(np.dot(m - c, n)) == pytest.approx(-float(np.dot(TRUE_NODE - c, n)), abs=1e-9)
    # the mirror produces almost the same ranges as the true node: the root of the flip ambiguity
    delta = np.linalg.norm(m - ANCHORS, axis=1) - true_ranges(ANCHORS, TRUE_NODE)
    assert np.abs(delta).max() < 0.6


def test_mirror_point_is_none_without_a_plane():
    assert mirror_point(LINE, TRUE_NODE) is None
    assert mirror_point(ANCHORS[:2], TRUE_NODE) is None
    assert mirror_point(ANCHORS[:1], TRUE_NODE) is None


def test_flip_risk_stays_within_unit_interval():
    meas = measured_baseline()
    for q in ([37.0, 12.0, 8.5], [30.0, 30.0, 10.0], [5.0, 55.0, 2.0], [0.0, 0.0, 0.0],
              [60.0, 60.0, 20.0], [30.0, 30.0, 7.75]):
        r = flip_ambiguity_risk(ANCHORS, q, meas)
        assert 0.0 <= r <= 1.0
    assert flip_ambiguity_risk(LINE, TRUE_NODE, true_ranges(LINE, TRUE_NODE)) == 1.0   # no plane -> 1


def test_flip_risk_is_one_for_perfectly_coplanar_anchors():
    p = np.array([20.0, 40.0, 12.0])
    meas = true_ranges(FLAT, p) + np.array([0.2, -0.1, 0.3, -0.2])
    # the mirror (20, 40, -12) has exactly the same ranges to every anchor -> f(p) == f(mirror) -> 1.0
    assert flip_ambiguity_risk(FLAT, p, meas) == 1.0
    assert flip_ambiguity_risk(FLAT, [45.0, 15.0, 18.0], meas) == 1.0


def test_flip_risk_baseline_candidates():
    meas = measured_baseline()
    # noise-free ranges: the true node fits perfectly, its mirror does not
    assert flip_ambiguity_risk(ANCHORS, TRUE_NODE, true_ranges(ANCHORS, TRUE_NODE)) < 1e-12
    # the fitted Review-1 estimate is clearly better than its mirror image
    at_estimate = flip_ambiguity_risk(ANCHORS, PSO_ESTIMATE, meas)
    assert at_estimate < 0.5
    assert at_estimate == pytest.approx(REF["risk_at_estimate"], abs=1e-9)
    # the wrong side is detected: candidate = mirror of the estimate -> ratio > 1 -> clipped to 1
    wrong_side = mirror_point(ANCHORS, PSO_ESTIMATE)
    assert wrong_side is not None
    assert flip_ambiguity_risk(ANCHORS, wrong_side, meas) == 1.0
    # At the exact true node the fitness sits on the noise floor (sum n_j^2 = 0.66) while the mirror,
    # only 5.45 m away on the other side of the anchor plane, scores 0.897 -> ratio 0.7356 < 1.
    # The node is just 2.73 m from the least-squares anchor plane, so the 0-20 m height spread of the
    # anchors resolves the flip only weakly (the README's "weak vertical geometry" remark).
    at_truth = flip_ambiguity_risk(ANCHORS, TRUE_NODE, meas)
    assert at_truth < 1.0
    assert at_truth == pytest.approx(REF["risk_at_truth"], abs=1e-9)
    assert at_estimate < at_truth


def test_flip_risk_is_low_for_a_node_far_from_the_anchor_plane():
    """The metric discriminates: a node 17.5 m from the anchor plane (same noise vector) is well resolved."""
    far = np.array([10.0, 10.0, 19.0])
    meas_far = true_ranges(ANCHORS, far) + NOISE
    centroid, normal = anchor_plane(ANCHORS)
    assert abs(float(np.dot(far - centroid, normal))) > 15.0
    risk_far = flip_ambiguity_risk(ANCHORS, far, meas_far)
    assert risk_far < 0.5
    assert risk_far == pytest.approx(REF["risk_far_from_plane"], abs=1e-9)
    assert risk_far < flip_ambiguity_risk(ANCHORS, TRUE_NODE, measured_baseline())
    # a second off-plane node for good measure: (30, 30, 19) is 10.9 m above the plane
    top = np.array([30.0, 30.0, 19.0])
    assert flip_ambiguity_risk(ANCHORS, top, true_ranges(ANCHORS, top) + NOISE) < 0.5


def test_flip_risk_hint_baseline_and_limits():
    hint = flip_risk_hint(ANCHORS, TRUE_NODE, SIGMA)
    assert hint == pytest.approx(REF["hint_at_truth"], abs=1e-9)
    assert 0.0 < hint < 1.0
    assert flip_risk_hint(ANCHORS, TRUE_NODE, 0.1) < hint            # less noise -> less ambiguity
    assert flip_risk_hint(ANCHORS, [10.0, 10.0, 19.0], SIGMA) < 0.1  # far from the plane -> low
    assert flip_risk_hint(FLAT, [20.0, 40.0, 12.0], SIGMA) == 1.0    # exactly coplanar: identical ranges
    assert flip_risk_hint(LINE, TRUE_NODE, SIGMA) == 1.0             # no plane at all


# --------------------------------------------------------------------------- coverage grid / report

def test_coverage_grid_shape_matches_axes_and_points():
    xs, ys, zs, grid = coverage_grid(ANCHORS, (60.0, 60.0, 20.0), SIGMA, 20.0)
    assert xs.tolist() == [0.0, 20.0, 40.0, 60.0]
    assert ys.tolist() == [0.0, 20.0, 40.0, 60.0]
    assert zs.tolist() == [0.0, 20.0]
    assert grid.shape == (len(zs), len(ys), len(xs))
    assert np.isfinite(grid).all() and (grid > 0).all()
    # grid[iz, iy, ix] is the CRLB bound at (xs[ix], ys[iy], zs[iz])
    assert grid[1, 1, 2] == pytest.approx(crlb(ANCHORS, [40.0, 20.0, 20.0], SIGMA).rmse_bound, rel=1e-9)
    assert grid[0, 3, 0] == pytest.approx(crlb(ANCHORS, [0.0, 60.0, 0.0], SIGMA).rmse_bound, rel=1e-9)


def test_coverage_grid_bounds_grow_with_sigma():
    *_, g1 = coverage_grid(ANCHORS, (60, 60, 20), 0.5, 10.0)
    *_, g2 = coverage_grid(ANCHORS, (60, 60, 20), 1.0, 10.0)
    assert g1.shape == (3, 7, 7)
    assert np.isfinite(g1).all()
    assert (g2 > g1).all()
    np.testing.assert_allclose(g2, 2.0 * g1, rtol=1e-12)


def test_coverage_grid_is_infinite_in_the_anchor_plane_only():
    xs, ys, zs, grid = coverage_grid(FLAT, (60, 60, 20), SIGMA, 10.0)
    assert grid.shape == (3, 7, 7)
    assert np.isinf(grid[0]).all()          # z = 0: node in the anchor plane, no vertical information
    assert np.isfinite(grid[1:]).all()      # above the plane the bound is finite again


def test_coverage_grid_accepts_scalar_field_size():
    xs, ys, zs, grid = coverage_grid(ANCHORS, 60.0, SIGMA, 30.0)
    assert xs.tolist() == ys.tolist() == zs.tolist() == [0.0, 30.0, 60.0]
    assert grid.shape == (3, 3, 3)


def test_scenario_report_keys_values_and_json():
    rep = scenario_report(ANCHORS, TRUE_NODE, SIGMA)
    assert set(rep) == {"rank", "non_coplanar", "gdop", "crlb_rmse", "crlb_per_axis", "mirror", "flip_risk_hint"}
    assert rep["rank"] == 3 and rep["non_coplanar"] is True
    assert rep["gdop"] == pytest.approx(REF["gdop"], rel=1e-9)
    assert rep["crlb_rmse"] == pytest.approx(REF["crlb_rmse"], rel=1e-9)
    assert rep["crlb_per_axis"] == pytest.approx(REF["per_axis"], abs=1e-6)
    assert rep["mirror"] == pytest.approx(REF["mirror"], abs=1e-6)
    assert 0.0 <= rep["flip_risk_hint"] <= 1.0
    json.dumps(rep)                                           # plain Python types only
    with_meas = scenario_report(ANCHORS, PSO_ESTIMATE, SIGMA, measured=measured_baseline())
    assert with_meas["flip_risk"] == pytest.approx(flip_ambiguity_risk(ANCHORS, PSO_ESTIMATE, measured_baseline()))
    flat = scenario_report(FLAT, [30.0, 30.0, 0.0], SIGMA)
    assert flat["rank"] == 2 and flat["non_coplanar"] is False
    assert flat["gdop"] == float("inf") and flat["crlb_rmse"] == float("inf")
    assert flat["flip_risk_hint"] == 1.0


def test_geometry_fixture_roundtrip():
    """Writes tests/fixtures/geometry_cases.json once (if absent) and checks the stored numbers afterwards."""
    meas = measured_baseline()
    cases = []
    for point in PROBES:
        q = np.array(point)
        b = crlb(ANCHORS, q, SIGMA)
        m = mirror_point(ANCHORS, q)
        assert m is not None
        cases.append({
            "point": point,
            "gdop": gdop(ANCHORS, q),
            "crlb_rmse": b.rmse_bound,
            "crlb_per_axis": [float(v) for v in b.per_axis],
            "mirror": [float(v) for v in m],
            "flip_risk": flip_ambiguity_risk(ANCHORS, q, meas),
            "flip_risk_hint": flip_risk_hint(ANCHORS, q, SIGMA),
        })
    # sanity before anything is written: these must be the real diagnostics, not placeholders
    assert [c["gdop"] for c in cases] == pytest.approx(REF["probe_gdop"], rel=1e-9)
    assert all(math.isfinite(c["crlb_rmse"]) and c["crlb_rmse"] > 0 for c in cases)
    centroid, normal = anchor_plane(ANCHORS)
    payload = {
        "version": 1,
        "description": "Geometry diagnostics of the slide-16 baseline anchors at three probe points: GDOP, CRLB "
                       "(sigma = 0.5 m), mirror point across the least-squares anchor plane, flip-ambiguity risk "
                       "for the baseline measured ranges and the noise-only risk hint. Written by "
                       "tests/test_geometry.py and compared on every run (tolerance 1e-9).",
        "anchors": ANCHORS.tolist(),
        "true_position": TRUE_NODE.tolist(),
        "noise": NOISE.tolist(),
        "measured": meas.tolist(),
        "sigma": SIGMA,
        "rank": anchor_rank(ANCHORS),
        "non_coplanar": is_non_coplanar(ANCHORS),
        "tetrahedron_volume": tetrahedron_volume(*ANCHORS),
        "plane": {"centroid": centroid.tolist(), "normal": normal.tolist()},
        "cases": cases,
    }
    if not FIXTURE.exists():
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    stored = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert stored["rank"] == 3 and stored["non_coplanar"] is True
    assert stored["tetrahedron_volume"] == pytest.approx(payload["tetrahedron_volume"], rel=1e-9)
    np.testing.assert_allclose(stored["plane"]["centroid"], centroid, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(stored["plane"]["normal"], normal, rtol=1e-9, atol=1e-9)
    assert len(stored["cases"]) == len(cases)
    for got, exp in zip(cases, stored["cases"]):
        assert got["point"] == exp["point"]
        for key in ("gdop", "crlb_rmse", "flip_risk", "flip_risk_hint"):
            assert got[key] == pytest.approx(exp[key], rel=1e-9, abs=1e-9)
        for key in ("crlb_per_axis", "mirror"):
            np.testing.assert_allclose(got[key], exp[key], rtol=1e-9, atol=1e-9)
    assert FIXTURE.exists()

"""Tests for ``pso3d.centroid``: plain and inverse-distance weighted centroid estimators.

The pinned numbers come from the weighted-centroid example (code 5) of the repository's
"10 Codes with Summaries" document: five anchors, true node (16, 38, 5), perfect ranges.
"""
from __future__ import annotations

import numpy as np

from pso3d.centroid import centroid, weighted_centroid

P05_ANCHORS = np.array([[0.0, 0.0, 2.0], [60.0, 0.0, 4.0], [0.0, 60.0, 3.0], [60.0, 60.0, 6.0], [30.0, 30.0, 18.0]])
P05_TRUE = np.array([16.0, 38.0, 5.0])


def p05_ranges() -> np.ndarray:
    return np.linalg.norm(P05_ANCHORS - P05_TRUE, axis=1)


def test_plain_centroid_matches_code5_example():
    c = centroid(P05_ANCHORS)
    assert c.shape == (3,)
    assert np.allclose(np.round(c, 2), [30.0, 30.0, 6.6])
    assert round(float(np.linalg.norm(c - P05_TRUE)), 2) == 16.20


def test_weighted_centroid_matches_code5_example():
    wc = weighted_centroid(P05_ANCHORS, p05_ranges())
    assert wc.shape == (3,)
    assert np.allclose(np.round(wc, 2), [25.23, 33.19, 8.31])
    assert round(float(np.linalg.norm(wc - P05_TRUE)), 2) == 10.92


def test_weighted_centroid_beats_plain_centroid_on_code5_example():
    d = p05_ranges()
    err_plain = np.linalg.norm(centroid(P05_ANCHORS) - P05_TRUE)
    err_weighted = np.linalg.norm(weighted_centroid(P05_ANCHORS, d) - P05_TRUE)
    assert err_weighted < err_plain


def test_weighted_centroid_default_power_is_one():
    d = p05_ranges()
    assert np.allclose(weighted_centroid(P05_ANCHORS, d), weighted_centroid(P05_ANCHORS, d, power=1.0), atol=1e-12)


def test_weighted_centroid_power_zero_is_the_plain_centroid():
    assert np.allclose(weighted_centroid(P05_ANCHORS, p05_ranges(), power=0.0), centroid(P05_ANCHORS), atol=1e-12)


def test_weighted_centroid_is_the_explicit_inverse_distance_average():
    d = p05_ranges()
    for power in (0.5, 1.0, 2.0):
        w = 1.0 / d ** power
        expected = (P05_ANCHORS * w[:, None]).sum(axis=0) / w.sum()
        assert np.allclose(weighted_centroid(P05_ANCHORS, d, power=power), expected, atol=1e-12)


def test_large_power_moves_the_estimate_to_the_nearest_anchor():
    d = p05_ranges()
    nearest = P05_ANCHORS[np.argmin(d)]
    assert np.allclose(weighted_centroid(P05_ANCHORS, d, power=100.0), nearest, atol=1e-6)


def test_zero_range_is_guarded_and_selects_that_anchor():
    d = p05_ranges().copy()
    d[2] = 0.0                                   # node sits exactly on anchor 3 -> 1/d would be infinite
    wc = weighted_centroid(P05_ANCHORS, d)
    assert np.isfinite(wc).all()
    assert np.allclose(wc, P05_ANCHORS[2], atol=1e-6)


def test_tiny_or_negative_ranges_never_produce_non_finite_values():
    d = np.array([1e-15, -0.3, 10.0, 20.0, 30.0])
    wc = weighted_centroid(P05_ANCHORS, d)
    assert np.isfinite(wc).all()
    # both guarded ranges get the same (huge) weight -> midpoint of anchors 1 and 2
    assert np.allclose(wc, (P05_ANCHORS[0] + P05_ANCHORS[1]) / 2, atol=1e-6)


def test_centroids_lie_inside_the_anchor_bounding_box():
    rng = np.random.default_rng(5)
    for _ in range(10):
        anchors = rng.uniform([0, 0, 0], [60, 60, 20], size=(6, 3))
        true_p = rng.uniform([0, 0, 0], [60, 60, 20])
        d = np.linalg.norm(anchors - true_p, axis=1) + rng.normal(0.0, 0.5, 6)
        for est in (centroid(anchors), weighted_centroid(anchors, d), weighted_centroid(anchors, d, power=2.0)):
            assert (est >= anchors.min(axis=0) - 1e-9).all()
            assert (est <= anchors.max(axis=0) + 1e-9).all()


def test_accepts_python_lists_and_returns_float_vectors():
    tetra = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]
    c = centroid(tetra)
    assert c.dtype == float and c.shape == (3,)
    assert np.allclose(c, [0.5, 0.5, 0.5])
    wc = weighted_centroid(tetra, [1, 1, 1, 1])
    assert wc.dtype == float and wc.shape == (3,)
    assert np.allclose(wc, [0.5, 0.5, 0.5])

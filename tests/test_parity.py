"""Python <-> JavaScript parity on shared fixtures.

The browser simulator (``web/pso.js``, ``web/geometry.js``) re-implements the closed-form
solvers, the range-error fitness, the geometry diagnostics and the AMCMPSO coefficient
schedule in plain JavaScript.  ``tests/js/dump.js`` evaluates all of them on the cases in
``tests/fixtures/parity_inputs.json``; this test computes the same quantities with the
Python package and compares them.

Random streams (numpy ``default_rng`` vs. mulberry32) are *not* expected to match and are
not compared here.
"""
from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE = os.path.join(ROOT, "tests", "fixtures", "parity_inputs.json")
DUMP = os.path.join(ROOT, "tests", "js", "dump.js")
REL = 1e-6
CASE_IDS = ["baseline", "near_coplanar", "six_anchors"]


@pytest.fixture(scope="module")
def py() -> SimpleNamespace:
    """Python reference implementations (imported lazily so collection never fails)."""
    amcmpso = importlib.import_module("pso3d.amcmpso")
    geometry = importlib.import_module("pso3d.geometry")
    trilateration = importlib.import_module("pso3d.trilateration")
    fitness = importlib.import_module("pso3d.fitness")
    return SimpleNamespace(
        centre_of_mass=amcmpso.centre_of_mass,
        coefficient_schedule=amcmpso.coefficient_schedule,
        crlb=geometry.crlb,
        gdop=geometry.gdop,
        is_non_coplanar=geometry.is_non_coplanar,
        mirror_point=geometry.mirror_point,
        least_squares_trilateration=trilateration.least_squares_trilateration,
        gauss_newton_refine=trilateration.gauss_newton_refine,
        RangeErrorFitness=fitness.RangeErrorFitness,
    )


@pytest.fixture(scope="module")
def dump() -> dict:
    node = shutil.which("node")
    assert node, "node is required for the parity test"
    assert os.path.exists(DUMP), f"missing {DUMP}"
    out = subprocess.run([node, DUMP, FIXTURE], capture_output=True, text=True, check=True, timeout=120).stdout
    return json.loads(out)


@pytest.fixture(scope="module")
def cases() -> dict:
    with open(FIXTURE, encoding="utf-8") as fh:
        data = json.load(fh)
    return {c["id"]: c for c in data["cases"]}


def _is_infinite(value) -> bool:
    if value is None or value == "Infinity":
        return True
    return isinstance(value, (int, float)) and (value > 1e12 or value != value)


def test_dump_covers_all_cases(dump, cases):
    assert set(dump) == set(cases) == set(CASE_IDS)


@pytest.mark.parametrize("cid", CASE_IDS)
def test_measured_ranges_match(dump, cases, cid):
    c = cases[cid]
    a, p, n = (np.asarray(c[k], float) for k in ("anchors", "true_position", "noise"))
    measured = np.linalg.norm(a - p, axis=1) + n
    np.testing.assert_allclose(dump[cid]["measured"], measured, rtol=REL)


@pytest.mark.parametrize("cid", ["baseline", "six_anchors"])
def test_closed_form_matches(py, dump, cases, cid):
    c = cases[cid]
    a = np.asarray(c["anchors"], float)
    measured = np.asarray(dump[cid]["measured"], float)
    lsq = py.least_squares_trilateration(a, measured)
    np.testing.assert_allclose(dump[cid]["lsq"], lsq, rtol=REL, atol=1e-6)
    gn, used = py.gauss_newton_refine(a, measured, lsq, iterations=10, tol=1e-9)
    np.testing.assert_allclose(dump[cid]["gn"]["p"], gn, rtol=REL, atol=1e-6)
    assert dump[cid]["gn"]["iterations"] == used


@pytest.mark.parametrize("cid", CASE_IDS)
def test_fitness_matches(py, dump, cases, cid):
    c = cases[cid]
    a = np.asarray(c["anchors"], float)
    measured = np.asarray(dump[cid]["measured"], float)
    fit = py.RangeErrorFitness(a, measured)
    ours = fit(np.asarray(c["probe_points"], float))
    np.testing.assert_allclose(dump[cid]["fitness"], ours, rtol=REL, atol=1e-9)


@pytest.mark.parametrize("cid", CASE_IDS)
def test_geometry_matches(py, dump, cases, cid):
    c = cases[cid]
    a = np.asarray(c["anchors"], float)
    sigma = float(c["sigma"])
    assert bool(dump[cid]["non_coplanar"]) == bool(py.is_non_coplanar(a))
    for k, q in enumerate(np.asarray(c["probe_points"], float)):
        g_py = py.gdop(a, q)
        g_js = dump[cid]["gdop"][k]
        if np.isinf(g_py):
            assert _is_infinite(g_js), f"{cid}[{k}] gdop: py=inf js={g_js}"
        else:
            assert g_js == pytest.approx(g_py, rel=1e-5), f"{cid}[{k}] gdop"
        b_py = py.crlb(a, q, sigma).rmse_bound
        b_js = dump[cid]["crlb_rmse"][k]
        if np.isinf(b_py):
            assert _is_infinite(b_js), f"{cid}[{k}] crlb: py=inf js={b_js}"
        else:
            assert b_js == pytest.approx(b_py, rel=1e-5), f"{cid}[{k}] crlb"
        m_py = py.mirror_point(a, q)
        m_js = dump[cid]["mirror"][k]
        if m_py is None:
            assert m_js is None
        else:
            np.testing.assert_allclose(m_js, m_py, rtol=1e-6, atol=1e-6)


def test_amcmpso_schedule_and_centre_of_mass_match(py, dump, cases):
    sched = py.coefficient_schedule(60)
    js = dump["baseline"]["schedule"]
    for key in ("w", "c1", "c2"):
        np.testing.assert_allclose(js[key], sched[key], rtol=0, atol=1e-12)
    c = cases["baseline"]
    a = np.asarray(c["anchors"], float)
    measured = np.asarray(dump["baseline"]["measured"], float)
    pts = np.asarray(c["probe_points"], float)
    f = py.RangeErrorFitness(a, measured)(pts)
    np.testing.assert_allclose(dump["baseline"]["centre_of_mass"], py.centre_of_mass(pts, f), rtol=REL)

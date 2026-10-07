"""Packaging / CLI tests: ``python -m pso3d`` sub-commands, the baseline script entry point,
``scripts/check.sh`` portability, ``pyproject.toml`` metadata, the Vercel headers and the CI workflow.

All sub-process runs use ``sys.executable`` from the repository root and write only to ``tmp_path``.
"""
from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tomllib

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pso3d  # noqa: E402  (repo root on sys.path)

COMMITTED_BASELINE = os.path.join(ROOT, "results", "baseline.json")
CHECK_SH = os.path.join(ROOT, "scripts", "check.sh")
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
VERCEL_JSON = os.path.join(ROOT, "web", "vercel.json")
CI_YML = os.path.join(ROOT, ".github", "workflows", "ci.yml")
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")
RUN_BASELINE = os.path.join(ROOT, "scripts", "run_baseline.py")

# The production policy served by Vercel and enforced by the browser gate (tests/e2e/web_smoke.js).
PRODUCTION_CSP = ("default-src 'self'; script-src 'self' https://cdn.plot.ly; style-src 'self' 'unsafe-inline'; "
                  "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; "
                  "base-uri 'self'; form-action 'self'; frame-ancestors 'none'")


def run_cli(*args: str, cwd: str = ROOT, timeout: float = 120.0) -> subprocess.CompletedProcess:
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return subprocess.run([sys.executable, "-m", "pso3d", *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=timeout)


def _close(a, b, tol=1e-9) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


# --------------------------------------------------------------------------------------------
# baseline sub-command (S1: numbers identical to the committed results/baseline.json)
# --------------------------------------------------------------------------------------------
def test_baseline_cli_reproduces_committed_numbers(tmp_path):
    proc = run_cli("baseline", "--out", str(tmp_path), "--figures", str(tmp_path), "--no-gif", "--quiet")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == "", "--quiet must silence the run summary"
    out_json = tmp_path / "baseline.json"
    assert out_json.exists(), "baseline --out DIR must write DIR/baseline.json"
    got = json.loads(out_json.read_text(encoding="utf-8"))
    with open(COMMITTED_BASELINE, encoding="utf-8") as fh:
        ref = json.load(fh)
    assert _close(got["simplified_pso"]["estimate"], ref["simplified_pso"]["estimate"])
    assert got["simplified_pso"]["fitness_evaluations"] == 1200
    assert got["simplified_pso"]["distance_computations"] == 4800
    assert got["simplified_pso"]["swarm_state_floats"] == 120
    assert abs(got["simplified_pso"]["error_m"] - ref["simplified_pso"]["error_m"]) <= 1e-9
    assert _close(got["least_squares"]["estimate"], ref["least_squares"]["estimate"])
    assert _close(got["least_squares_gauss_newton"]["estimate"], ref["least_squares_gauss_newton"]["estimate"])
    assert got["least_squares_gauss_newton"]["iterations"] == ref["least_squares_gauss_newton"]["iterations"]
    assert _close(got["standard_pso"]["estimate"], ref["standard_pso"]["estimate"])
    assert got["standard_pso"]["fitness_evaluations"] == ref["standard_pso"]["fitness_evaluations"]
    # keys: identical to the committed file except that the AMCMPSO block is now labelled as an interpretation
    # (its values are not compared: the lead regenerates results/baseline.json after integration)
    expected_keys = (set(ref) - {"amcmpso_in_progress"}) | {"amcmpso_interpretation"}
    assert set(got) == expected_keys, "baseline.json keys must stay identical"
    assert set(got["amcmpso_interpretation"]) == {"estimate", "error_m", "fitness_evaluations", "swarm_state_floats"}
    assert set(got["simplified_pso"]) == set(ref["simplified_pso"])
    assert (tmp_path / "convergence.png").stat().st_size > 5000
    assert (tmp_path / "scenario_3d.png").stat().st_size > 5000
    assert not (tmp_path / "convergence.gif").exists(), "--no-gif must skip the GIF"


def test_run_baseline_script_main_prints_todays_summary(tmp_path, capsys):
    src = open(RUN_BASELINE, encoding="utf-8").read()
    assert "def main(" in src, "scripts/run_baseline.py must expose main(argv) (and not run at import time)"
    assert 'if __name__ == "__main__"' in src
    spec = importlib.util.spec_from_file_location("run_baseline_under_test", RUN_BASELINE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    captured = capsys.readouterr()
    assert captured.out == "", "importing the script must not run the baseline"
    # the figures themselves are covered by the sub-process test above; here only the text/JSON contract is
    # exercised, so the (slow) matplotlib renderers are replaced by no-ops that just create the files
    touch = lambda *args, **kwargs: open([a for a in args if isinstance(a, str)][-1], "wb").close()  # noqa: E731
    mod.plot_convergence = touch
    mod.plot_scenario_3d = touch
    mod.make_convergence_gif = touch
    rc = mod.main(["--out", str(tmp_path), "--figures", str(tmp_path), "--no-gif"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.startswith("measured ranges d_hat:")
    assert re.search(r"PSO estimate: \[37\.43\s+11\.9\s+9\.29\]\s+error: 0\.90 m\s+\(dx,dy,dz\)=\(\+0\.43,-0\.10,\+0\.79\)", out)
    assert "fitness evaluations: 1200  distance computations: 4800" in out
    assert "swarm state: 120 floats = 960 bytes (float64)" in out
    assert "within 1 % of final value from iteration 48" in out
    assert "LSQ  estimate: [37.13 11.44 11.65]  error: 3.20 m" in out
    assert "LSQ+GN estimate: [37.42 11.88  9.36]  error: 0.96 m  (10 GN iterations)" in out
    assert "AMCMPSO (interpretation of the base paper, not a reproduction):" in out
    assert "IN PROGRESS" not in out and "scaffold" not in out
    assert re.search(r"^saved .*baseline\.json, .*convergence\.png, .*scenario_3d\.png$", out, re.M)
    assert "convergence.gif" not in out
    assert (tmp_path / "baseline.json").exists()


# --------------------------------------------------------------------------------------------
# help / version / dispatch
# --------------------------------------------------------------------------------------------
def test_top_level_help_lists_all_subcommands():
    proc = run_cli("--help")
    assert proc.returncode == 0, proc.stderr
    for name in ("baseline", "experiments", "network", "geometry", "version"):
        assert re.search(rf"^\s*{name}\b", proc.stdout, re.M), f"--help must list the {name} command:\n{proc.stdout}"


@pytest.mark.parametrize("command", ["baseline", "experiments", "network", "geometry"])
def test_each_subcommand_has_help(command):
    proc = run_cli(command, "--help")
    assert proc.returncode == 0, f"{command} --help failed:\n{proc.stdout}{proc.stderr}"
    assert "usage" in proc.stdout.lower()
    assert "--out" in proc.stdout or command == "geometry"


def test_version_prints_package_version():
    proc = run_cli("version")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == pso3d.__version__


def test_unknown_command_and_missing_command_exit_2():
    proc = run_cli("frobnicate")
    assert proc.returncode == 2
    assert "unknown command" in proc.stderr.lower()
    proc = run_cli()
    assert proc.returncode == 2
    assert "usage" in (proc.stdout + proc.stderr).lower()


def test_main_is_importable_and_returns_int(capsys):
    from pso3d.cli import main
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == pso3d.__version__
    assert main(["--help"]) == 0
    assert "geometry" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------
# geometry sub-command (needs pso3d.geometry; the test fails, never skips, while it is absent)
# --------------------------------------------------------------------------------------------
def test_geometry_json_reports_default_scenario():
    proc = run_cli("geometry", "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert "gdop" in report
    assert isinstance(report["gdop"], (int, float)) and math.isfinite(report["gdop"]) and report["gdop"] > 0
    assert report.get("non_coplanar") is True
    assert report.get("anchor_rank") == 3
    assert report.get("crlb_rmse_bound", 0) > 0


def test_geometry_text_and_scenario_file(tmp_path):
    proc = run_cli("geometry")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "GDOP" in proc.stdout and "CRLB" in proc.stdout and "non-coplanar" in proc.stdout
    scenario = {"anchors": [[0, 0, 0], [60, 0, 5], [0, 60, 6], [60, 60, 20], [30, 30, 15]],
                "position": [42.0, 18.0, 6.0], "sigma": 0.3}
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    proc = run_cli("geometry", "--scenario", str(path), "--json")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["n_anchors"] == 5 and abs(report["sigma"] - 0.3) < 1e-12
    assert report["gdop"] > 0 and report["non_coplanar"] is True


# --------------------------------------------------------------------------------------------
# repository plumbing: check.sh, pyproject.toml, vercel.json, CI workflow
# --------------------------------------------------------------------------------------------
def test_check_sh_is_portable_and_complete():
    assert os.path.exists(CHECK_SH), "scripts/check.sh is missing"
    text = open(CHECK_SH, encoding="utf-8").read()
    assert "/home/" not in text, "check.sh must not contain machine-specific paths"
    assert "/Users/" not in text and "C:\\" not in text
    assert text.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in text
    assert "compileall" in text
    assert "node --check" in text
    assert "pytest" in text and "--ignore=tests/e2e" in text
    assert "node --test tests/js" in text
    assert re.search(r'echo\s+"?OK', text), "check.sh must print a final OK line"


def test_pyproject_metadata():
    assert os.path.exists(PYPROJECT)
    with open(PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    project = data["project"]
    assert project["name"] == "pso3d"
    assert "version" in project["dynamic"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {"attr": "pso3d.__version__"}
    assert project["requires-python"] == ">=3.10"
    assert project["dependencies"] == ["numpy>=1.24"]
    assert project["optional-dependencies"]["plots"] == ["matplotlib>=3.7", "pillow>=9.0"]
    assert project["optional-dependencies"]["dev"] == ["pytest>=7.0", "setuptools>=77"]
    assert project["scripts"]["pso3d"] == "pso3d.cli:main"
    assert project["readme"] == "README.md"
    assert data["build-system"]["build-backend"] == "setuptools.build_meta"
    assert data["tool"]["setuptools"]["packages"]["find"]["include"] == ["pso3d*"]
    ini = data["tool"]["pytest"]["ini_options"]
    assert ini["testpaths"] == ["tests"]
    assert {"e2e", "js"} <= set(ini["norecursedirs"])


def _requirement_floor(requirements, name):
    """The `>=` floor of requirement `name` in a list of PEP 508 strings (None when absent)."""
    for req in requirements:
        m = re.match(rf"^\s*{name}\s*(?:\[[^\]]*\])?\s*>=\s*([0-9][0-9A-Za-z.\-]*)\s*$", req)
        if m:
            return m.group(1)
    return None


def test_dev_extra_can_build_the_archives_with_the_declared_backend():
    """tests/test_sdist.py and scripts/check_sdist.py import setuptools.build_meta in the interpreter that runs the
    tests. `pip install -e .` only provides the [build-system] requirement inside pip's isolated build environment
    (and Python 3.12 runners ship no setuptools at all), so both documented install paths — the dev extra and
    requirements.txt — must carry the same setuptools floor as [build-system].requires."""
    with open(PYPROJECT, "rb") as fh:
        data = tomllib.load(fh)
    build_floor = _requirement_floor(data["build-system"]["requires"], "setuptools")
    assert build_floor is not None, data["build-system"]["requires"]
    dev = data["project"]["optional-dependencies"]["dev"]
    assert any(re.match(r"^\s*pytest\b", r) for r in dev), f"pytest must stay in the dev extra: {dev}"
    dev_floor = _requirement_floor(dev, "setuptools")
    assert dev_floor is not None, f"the dev extra must require setuptools>={build_floor} (the [build-system] floor): {dev}"
    assert dev_floor == build_floor, f"dev extra setuptools floor {dev_floor} differs from the build-system floor {build_floor}"
    ci = open(CI_YML, encoding="utf-8").read()
    install_lines = [ln for ln in ci.splitlines() if ".[plots,dev]" in ln and not ln.lstrip().startswith("#")]
    assert install_lines, "ci.yml must install the project with the plots and dev extras"
    # the README's alternative `pip install -r requirements.txt` must be able to build the archives as well
    reqs = [ln.strip() for ln in open(REQUIREMENTS, encoding="utf-8").read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    assert {"numpy>=1.24", "matplotlib>=3.7", "pillow>=9.0", "pytest>=7.0"} <= set(reqs), reqs
    req_floor = _requirement_floor(reqs, "setuptools")
    assert req_floor is not None, f"requirements.txt must list setuptools>={build_floor} (the [build-system] floor): {reqs}"
    assert req_floor == build_floor, f"requirements.txt setuptools floor {req_floor} differs from the build-system floor {build_floor}"


def test_vercel_json_serves_the_production_security_headers():
    with open(VERCEL_JSON, encoding="utf-8") as fh:
        cfg = json.load(fh)
    assert cfg.get("cleanUrls") is True
    rules = {rule["source"]: {h["key"]: h["value"] for h in rule["headers"]} for rule in cfg.get("headers", [])}
    assert "/(.*)" in rules, "catch-all header rule missing"
    catch_all = rules["/(.*)"]
    assert catch_all["Content-Security-Policy"] == PRODUCTION_CSP
    assert catch_all["X-Content-Type-Options"] == "nosniff"
    assert catch_all["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert catch_all["Permissions-Policy"] == "camera=(), microphone=(), geolocation=()"
    assert catch_all["X-Frame-Options"] == "DENY"
    assert rules["/(.*)\\.(js|css)"]["Cache-Control"] == "public, max-age=3600, must-revalidate"
    assert rules["/"]["Cache-Control"] == "no-cache"
    assert "Content-Security-Policy-Report-Only" not in catch_all, "the policy must be enforced, not report-only"


def test_ci_workflow_runs_check_sh_on_python_311_and_312():
    assert os.path.exists(CI_YML), ".github/workflows/ci.yml is missing"
    text = open(CI_YML, encoding="utf-8").read()
    assert "ubuntu-latest" in text
    assert '"3.11"' in text and '"3.12"' in text
    assert "actions/setup-node" in text and 'node-version: "20"' in text
    assert "pip install -e" in text and "[plots,dev]" in text
    assert "bash scripts/check.sh" in text
    assert "/home/" not in text

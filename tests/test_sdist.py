"""Source-distribution completeness and self-sufficiency.

A source archive built with the standard backend (``setuptools.build_meta``) must contain everything needed to
re-run the published checks from the extracted archive alone: the test fixtures, the Node tests, the scripts
behind ``python -m pso3d baseline|experiments|network``, the web simulator with its deployment headers, the
docs and the frozen reference results. It must never ship build output, caches, byte-code, logs, the generated
figures or the reference papers / personal documents kept at the repository root.

The artefacts are built from a temporary copy of the checkout (the working tree is not modified) and extracted
under ``tmp_path``. Every sub-process below runs with ``PYTHONPATH`` unset and with the extracted archive as its
working directory, so nothing can fall back to this checkout. The complete test suite is *not* executed here
(that is ``scripts/check_sdist.py``); this file stays within a few seconds of build time.
"""
from __future__ import annotations

import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CHECK_SDIST = ROOT / "scripts" / "check_sdist.py"

# Files that must ship in the source archive (paths relative to the archive's top-level directory).
REQUIRED_FILES = [
    "LICENSE", "README.md", "pyproject.toml", "MANIFEST.in", "CONTRIBUTING.md", "CHANGELOG.md", "requirements.txt",
    "tests/conftest.py",
    "tests/fixtures/parity_inputs.json", "tests/fixtures/amcmpso_cases.json", "tests/fixtures/closed_form_cases.json",
    "tests/fixtures/geometry_cases.json", "tests/fixtures/noise_cases.json",
    "tests/js/dump.js", "tests/js/pso.test.js", "tests/js/geometry.test.js", "tests/js/closed_form.test.js",
    "tests/js/share.test.js", "tests/js/export.test.js",
    "tests/e2e/serve.js", "tests/e2e/serve.test.js", "tests/e2e/web_smoke.js", "tests/e2e/README.md",
    "scripts/run_baseline.py", "scripts/run_experiments.py", "scripts/run_network.py", "scripts/check.sh",
    "scripts/check_sdist.py",
    "web/index.html", "web/app.js", "web/pso.js", "web/geometry.js", "web/share.js", "web/export.js",
    "web/style.css", "web/vercel.json",
    ".github/workflows/ci.yml",
    "results/baseline.json", "results/sweeps.csv", "results/results.md", "results/extended/sweeps.csv",
    "results/network.md",
    "docs/amcmpso.md", "docs/closed_form.md", "docs/experiments.md", "docs/geometry.md", "docs/js_parity.md",
    "docs/network.md", "docs/noise_models.md", "docs/stopping_and_warm_start.md", "docs/web_simulator.md",
]
# Every file of the checkout matching these patterns must ship too (future modules, tests and docs included).
REQUIRED_GLOBS = ["pso3d/*.py", "tests/test_*.py", "docs/*.md"]
# Nothing matching these may ship.
FORBIDDEN_RULES = [
    ("build output", lambda p: p.startswith(("build/", "dist/"))),
    ("node_modules", lambda p: "node_modules/" in p),
    ("caches", lambda p: ".pytest_cache/" in p or ".ruff_cache/" in p or ".mypy_cache/" in p),
    ("git metadata", lambda p: p.startswith(".git/")),
    ("byte-code", lambda p: "__pycache__" in p or p.endswith((".pyc", ".pyo"))),
    ("PDF / Word documents (reference papers, personal files)", lambda p: p.lower().endswith((".pdf", ".docx"))),
    ("generated figures", lambda p: p.startswith("figures/")),
    ("logs", lambda p: p.endswith(".log")),
    (".DS_Store", lambda p: p.endswith(".DS_Store")),
]
# Member names must be relative to the archive and free of directories of the machine that built it ("/<dir>/").
MACHINE_PATH_FRAGMENTS = tuple(f"/{d}/" for d in ("home", "Users", "tmp", "private", "var", "state"))
COPY_IGNORE = (".git", "build", "dist", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", "*.egg-info", "*.pyc")
MIN_COLLECTED = 307            # Python tests collected from the extracted archive (tests/js and tests/e2e excluded)
WHEEL_DIST_INFO = {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt", "licenses/LICENSE"}


def clean_env() -> dict[str, str]:
    """The caller's environment without any Python path overrides (no fallback to this checkout)."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "PYTEST_ADDOPTS")}
    env.update(PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
    return env


def run_in(cwd: Path, args: list[str], timeout: float = 300.0) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=str(cwd), env=clean_env(), capture_output=True, text=True, timeout=timeout)


def collected_count(stdout: str) -> int:
    """Number of collected tests from `pytest --collect-only` output in any of its quiet formats."""
    ids = sum(1 for line in stdout.splitlines() if "::" in line and not line.startswith((" ", "<")))
    per_file = sum(int(m.group(1)) for m in re.finditer(r"^\S+\.py: (\d+)$", stdout, re.M))
    summary = re.search(r"(\d+) tests? collected", stdout)
    return max(ids, per_file, int(summary.group(1)) if summary else 0)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Build sdist + wheel from a clean copy of the checkout, extract both, return their contents."""
    base = tmp_path_factory.mktemp("sdist")
    src, dist = base / "src", base / "dist"
    shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns(*COPY_IGNORE))
    dist.mkdir()
    # setuptools.build_meta drives distutils by rewriting sys.argv, so the output directory is captured
    # before the backend is imported or called (reading sys.argv[1] after build_sdist yields a command word)
    code = ("import sys\nout = sys.argv[1]\nimport setuptools.build_meta as backend\n"
            "backend.build_sdist(out)\nbackend.build_wheel(out)\n")
    proc = run_in(src, [sys.executable, "-c", code, str(dist)], timeout=600)
    listing = "\n  ".join(sorted(p.relative_to(dist).as_posix() for p in dist.rglob("*") if p.is_file())) or "(empty)"
    diag = (f"build exit {proc.returncode}\n--- dist/ (recursive) ---\n  {listing}\n--- build output tail ---\n"
            + (proc.stdout + proc.stderr)[-3000:])
    assert proc.returncode == 0, "building the sdist/wheel failed:\n" + diag
    sdists, wheels = sorted(dist.glob("*.tar.gz")), sorted(dist.glob("*.whl"))
    assert len(sdists) == 1 and len(wheels) == 1, "expected exactly one sdist and one wheel in dist/:\n" + diag
    sdist_path, wheel_path = sdists[0], wheels[0]

    extract = base / "extract"
    with tarfile.open(sdist_path) as tar:
        raw = [m.name for m in tar.getmembers() if m.isfile()]
        try:
            tar.extractall(extract, filter="data")
        except TypeError:                    # Python without the extraction-filter backport
            tar.extractall(extract)
    prefixes = {n.split("/", 1)[0] for n in raw}
    assert len(prefixes) == 1, prefixes
    prefix = prefixes.pop()
    members = sorted(n.split("/", 1)[1] for n in raw if "/" in n)
    sdist_root = extract / prefix

    wheel_dir = base / "wheel"
    with zipfile.ZipFile(wheel_path) as zf:
        wheel_members = sorted(i.filename for i in zf.infolist() if not i.is_dir())
        zf.extractall(wheel_dir)

    init = (sdist_root / "pso3d" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__\s*=\s*"([^"]+)"', init, re.M)
    assert m, "pso3d/__init__.py in the archive has no __version__"
    return SimpleNamespace(sdist_path=sdist_path, wheel_path=wheel_path, raw_names=raw, prefix=prefix, members=members,
                           sdist_root=sdist_root, wheel_members=wheel_members, wheel_dir=wheel_dir, version=m.group(1))


# --------------------------------------------------------------------------------------------
# archive contents
# --------------------------------------------------------------------------------------------
def test_archive_name_prefix_and_metadata(built):
    assert built.sdist_path.name == f"pso3d-{built.version}.tar.gz"
    assert built.prefix == f"pso3d-{built.version}"
    pkg_info = (built.sdist_root / "PKG-INFO").read_text(encoding="utf-8")
    assert re.search(r"^Name: pso3d$", pkg_info, re.M)
    assert re.search(rf"^Version: {re.escape(built.version)}$", pkg_info, re.M)


def test_sdist_contains_every_required_file(built):
    expected = set(REQUIRED_FILES)
    for pattern in REQUIRED_GLOBS:
        found = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern) if p.is_file())
        assert found, f"nothing matches {pattern} in the checkout"
        expected.update(found)
    missing = sorted(expected - set(built.members))
    assert not missing, (f"{len(missing)} required file(s) missing from {built.sdist_path.name} "
                         f"({len(built.members)} members):\n  " + "\n  ".join(missing))


def test_sdist_ships_nothing_forbidden(built):
    offenders = [f"{p}  ({label})" for p in built.members for label, rule in FORBIDDEN_RULES if rule(p)]
    assert not offenders, "forbidden member(s) in the archive:\n  " + "\n  ".join(offenders)


def test_sdist_member_paths_are_relative_and_portable(built):
    bad = [n for n in built.raw_names
           if n.startswith("/") or ".." in n.split("/") or "\\" in n or not n.startswith(built.prefix + "/")
           or any(frag in "/" + n for frag in MACHINE_PATH_FRAGMENTS)]
    assert not bad, bad[:10]


# --------------------------------------------------------------------------------------------
# behaviour inside the extracted archive (cwd = archive root, PYTHONPATH unset)
# --------------------------------------------------------------------------------------------
def test_extracted_archive_reports_its_version(built):
    proc = run_in(built.sdist_root, [sys.executable, "-m", "pso3d", "version"])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == built.version


def test_extracted_archive_runs_the_checkout_only_commands(built):
    """`baseline --help` needs scripts/run_baseline.py next to the package: exit 0 only when scripts/ ships."""
    proc = run_in(built.sdist_root, [sys.executable, "-m", "pso3d", "baseline", "--help"])
    assert proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr.strip()}"
    assert "usage" in proc.stdout.lower() and "--no-gif" in proc.stdout


def test_extracted_archive_collects_the_complete_python_suite(built):
    proc = run_in(built.sdist_root, [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests",
                                     "--ignore=tests/e2e", "-p", "no:cacheprovider"])
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-3000:]
    n = collected_count(proc.stdout)
    assert n >= MIN_COLLECTED, f"only {n} tests collected from the archive (expected >= {MIN_COLLECTED})"


# --------------------------------------------------------------------------------------------
# wheel and licence
# --------------------------------------------------------------------------------------------
def test_wheel_contains_only_the_package(built):
    expected_py = sorted("pso3d/" + p.name for p in (ROOT / "pso3d").glob("*.py"))
    assert sorted(m for m in built.wheel_members if m.startswith("pso3d/")) == expected_py
    info = f"pso3d-{built.version}.dist-info/"
    dist_info = {m[len(info):] for m in built.wheel_members if m.startswith(info)}
    assert WHEEL_DIST_INFO <= dist_info, dist_info
    assert [m for m in built.wheel_members if not m.startswith(("pso3d/", info))] == []
    assert not any(m.startswith(("tests/", "scripts/", "web/", "docs/", "results/")) for m in built.wheel_members)


def test_license_ships_unchanged_in_both_artefacts(built):
    original = (ROOT / "LICENSE").read_bytes()
    assert b"MIT License" in original
    assert (built.sdist_root / "LICENSE").read_bytes() == original
    assert (built.wheel_dir / f"pso3d-{built.version}.dist-info" / "licenses" / "LICENSE").read_bytes() == original


# --------------------------------------------------------------------------------------------
# the archive check is part of the repository checks
# --------------------------------------------------------------------------------------------
def test_archive_check_is_wired_into_the_repository_checks():
    assert CHECK_SDIST.exists(), "scripts/check_sdist.py is missing"
    spec = importlib.util.spec_from_file_location("check_sdist_under_test", CHECK_SDIST)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert set(REQUIRED_FILES) <= set(mod.REQUIRED_FILES), sorted(set(REQUIRED_FILES) - set(mod.REQUIRED_FILES))
    assert set(REQUIRED_GLOBS) <= set(mod.REQUIRED_GLOBS)
    assert callable(getattr(mod, "main", None))

    def active(text: str, needle: str) -> bool:
        return any(needle in line and not line.lstrip().startswith("#") for line in text.splitlines())

    assert active((ROOT / "scripts" / "check.sh").read_text(encoding="utf-8"), "scripts/check_sdist.py"), \
        "scripts/check.sh must run the archive check"
    assert "check_sdist" in (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "## Source distribution" in (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")

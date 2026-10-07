#!/usr/bin/env python3
"""Build the source distribution and the wheel in isolation, verify their contents and run the complete test
suite from the extracted source archive.

    python scripts/check_sdist.py [--out DIR] [--skip-node] [--keep-temp]

Steps
  0. precondition: setuptools with its build backend is importable in this interpreter at the floor declared in
     pyproject.toml [build-system] (the `dev` extra provides it: pip install -e ".[dev]"); otherwise exit 1 at once
  1. copy the checkout to a temporary directory (without .git, build output, caches, byte-code, node_modules)
  2. build pso3d-<version>.tar.gz and the wheel there with setuptools.build_meta
     (standard library + setuptools only: no `build` module, no pip, no network, no upload anywhere)
  3. extract both; check the required and forbidden member lists and that member paths are relative and portable
  4. wheel: import pso3d from the extracted wheel only; `python -m pso3d version` and `geometry --help` exit 0;
     `baseline --help` exits 2 with the checkout-only guidance (the documented contract of the wheel)
  5. source archive: with the extracted archive as working directory and PYTHONPATH unset run the complete,
     unmodified suite: `python -m pytest tests --ignore=tests/e2e` (quiet through the project's addopts),
     `node --test tests/js/` and `node --test tests/e2e/serve.test.js` (the Node part is skipped only with
     --skip-node, and then it says so)
  6. write <out>/report.json (hashes, member lists, exit codes, test counts) and copy the artefacts to <out>/dist/

Exit code 0 only when every check passed; 1 otherwise; 2 for usage errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "pso3d"
COPY_IGNORE = (".git", "build", "dist", "__pycache__", "node_modules", ".pytest_cache", ".ruff_cache", ".mypy_cache",
               "*.egg-info", "*.pyc", "*.pyo", ".DS_Store")

# Files that must ship in the source archive (relative to the archive's top-level directory).
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
# Every checkout file matching these patterns must ship as well (new modules, tests and docs included).
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
# directories of the machine that built the archive must not appear in member names ("/<dir>/")
MACHINE_PATH_FRAGMENTS = tuple(f"/{d}/" for d in ("home", "Users", "tmp", "private", "var", "state"))
WHEEL_DIST_INFO = {"METADATA", "WHEEL", "RECORD", "entry_points.txt", "top_level.txt", "licenses/LICENSE"}
# `python -m pso3d baseline` from a wheel has no scripts/ next to the package: exit 2 plus this guidance.
CHECKOUT_ONLY_EXIT = 2
CHECKOUT_ONLY_MESSAGE = "repository checkout"
# no extra -q: pyproject.toml already sets addopts = "-q"; a second -q would suppress the "N passed" summary line
PYTEST_ARGS = ["-m", "pytest", "tests", "--ignore=tests/e2e", "-p", "no:cacheprovider"]
NODE_TARGETS = ["tests/js/", "tests/e2e/serve.test.js"]
# setuptools.build_meta drives distutils by rewriting sys.argv: capture the output directory before the backend
# is imported or called (reading sys.argv[1] after build_sdist would hand build_wheel a command word as outdir).
BUILD_CODE = ("import sys\nout = sys.argv[1]\nimport setuptools.build_meta as backend\n"
              "print('sdist', backend.build_sdist(out))\nprint('wheel', backend.build_wheel(out))\n")


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
class Report:
    def __init__(self) -> None:
        self.checks: list[dict] = []
        self.data: dict = {}

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        detail = str(detail)
        self.checks.append({"name": name, "ok": ok, "detail": detail[:6000]})
        short = detail.replace("\n", " | ")
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {short[:220]}" if short else ""))
        return ok

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(c["ok"] for c in self.checks)


def clean_env(**overrides: str | None) -> dict[str, str]:
    """The caller's environment without Python path overrides (nothing may fall back to this checkout)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSAFEPATH", "PYTEST_ADDOPTS", "PYTHONSTARTUP")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    env.update({k: v for k, v in overrides.items() if v is not None})
    return env


def run(args: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: float = 600.0) -> dict:
    t0 = time.perf_counter()
    rec = {"args": args, "cwd": cwd.name, "exit": None, "stdout": "", "stderr": "", "seconds": 0.0}
    try:
        proc = subprocess.run(args, cwd=str(cwd), env=env or clean_env(), capture_output=True, text=True,
                              errors="replace", timeout=timeout)
        rec.update(exit=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        rec.update(exit=124, stdout=out, stderr=err + f"\n[timeout after {timeout:.0f} s]")
    except FileNotFoundError as exc:
        rec.update(exit=127, stderr=f"command not found: {exc}")
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    return rec


def tail(rec: dict, n: int = 12) -> str:
    text = (rec["stdout"] + "\n" + rec["stderr"]).strip()
    return f"exit {rec['exit']}: " + "\n".join(text.splitlines()[-n:])


def summary_line(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def pytest_counts(text: str) -> dict[str, int]:
    return {key: int(m.group(1)) for key in ("passed", "failed", "error", "errors", "skipped", "xfailed", "xpassed")
            for m in [re.search(rf"(\d+) {key}\b", text)] if m}


def tap_counts(text: str) -> dict[str, int]:
    return {key: int(m.group(1)) for key in ("tests", "pass", "fail", "skipped", "todo", "cancelled")
            for m in [re.search(rf"^# {key} (\d+)$", text, re.M)] if m}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------------
# steps
# --------------------------------------------------------------------------------------------
def declared_setuptools_floor() -> int | None:
    """The `setuptools>=N` floor of [build-system].requires in pyproject.toml (None when it cannot be parsed)."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"\[build-system\](.*?)(?:\n\[|\Z)", text, re.S)
    m = re.search(r'"setuptools\s*>=\s*(\d+)', block.group(1)) if block else None
    return int(m.group(1)) if m else None


def check_backend(report: Report) -> bool:
    """The build backend must be importable in *this* interpreter with at least the declared floor.

    `pip install -e .` satisfies [build-system].requires only inside pip's isolated build environment, and some
    runners ship no setuptools at all; the `dev` extra carries the same floor so `pip install -e ".[dev]"` suffices."""
    floor = declared_setuptools_floor()
    if floor is None:
        return report.check("setuptools floor declared in pyproject.toml [build-system].requires", False, "cannot parse `setuptools>=N`")
    rec = run([sys.executable, "-c", "import setuptools, setuptools.build_meta; print(setuptools.__version__)"], cwd=ROOT)
    version = rec["stdout"].strip().splitlines()[-1].strip() if rec["stdout"].strip() else ""
    m = re.match(r"(\d+)", version)
    major = int(m.group(1)) if m else -1
    ok = rec["exit"] == 0 and major >= floor
    report.data["setuptools"] = {"version": version or None, "floor": floor, "exit": rec["exit"], "python": sys.executable}
    return report.check(f"setuptools >= {floor} available in this interpreter (needed to build the archives; "
                        f"pip install -e '.[dev]' or pip install 'setuptools>={floor}')", ok,
                        f"setuptools {version} in {sys.executable}" if version else tail(rec))


def build_artifacts(src: Path, dist: Path, report: Report) -> tuple[Path | None, Path | None]:
    rec = run([sys.executable, "-c", BUILD_CODE, str(dist)], cwd=src, timeout=900)
    report.data["build"] = {"exit": rec["exit"], "seconds": rec["seconds"], "tail": tail(rec, 20)}
    report.check("build sdist + wheel with setuptools.build_meta from a clean copy of the checkout", rec["exit"] == 0, tail(rec))
    sdists, wheels = sorted(dist.glob("*.tar.gz")), sorted(dist.glob("*.whl"))
    listing = ", ".join(sorted(p.relative_to(dist).as_posix() for p in dist.rglob("*") if p.is_file())) or "(nothing)"
    report.check("exactly one sdist and one wheel were produced", len(sdists) == 1 and len(wheels) == 1,
                 f"dist/ (recursive): {listing}" + ("" if len(sdists) == 1 and len(wheels) == 1 else "\n" + tail(rec, 30)))
    return (sdists[0] if len(sdists) == 1 else None), (wheels[0] if len(wheels) == 1 else None)


def extract_sdist(sdist_path: Path, dest: Path) -> tuple[str, list[str], list[str]]:
    with tarfile.open(sdist_path) as tar:
        raw = [m.name for m in tar.getmembers() if m.isfile()]
        try:
            tar.extractall(dest, filter="data")
        except TypeError:                      # Python without the extraction-filter backport
            tar.extractall(dest)
    prefixes = sorted({n.split("/", 1)[0] for n in raw})
    prefix = prefixes[0] if len(prefixes) == 1 else ""
    members = sorted(n.split("/", 1)[1] for n in raw if "/" in n)
    return prefix, raw, members


def check_sdist_members(prefix: str, raw: list[str], members: list[str], version: str, report: Report) -> None:
    report.check(f"sdist has the single top-level directory {PACKAGE}-{version}/", prefix == f"{PACKAGE}-{version}", prefix or "several prefixes")
    expected = set(REQUIRED_FILES)
    for pattern in REQUIRED_GLOBS:
        found = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern) if p.is_file())
        report.check(f"checkout has files matching {pattern}", bool(found), f"{len(found)} file(s)")
        expected.update(found)
    missing = sorted(expected - set(members))
    report.data["required"] = {"expected": len(expected), "missing": missing}
    report.check(f"sdist contains every required file ({len(expected)} expected, {len(members)} members)", not missing,
                 "missing: " + ", ".join(missing) if missing else "")
    offenders = [f"{p} ({label})" for p in members for label, rule in FORBIDDEN_RULES if rule(p)]
    report.data["forbidden"] = offenders
    report.check("sdist ships no build output, caches, byte-code, logs, figures, PDFs or Word documents", not offenders, ", ".join(offenders[:20]))
    bad = [n for n in raw if n.startswith("/") or ".." in n.split("/") or "\\" in n or (prefix and not n.startswith(prefix + "/"))
           or any(frag in "/" + n for frag in MACHINE_PATH_FRAGMENTS)]
    report.check("sdist member paths are relative and free of machine-specific directories", not bad, ", ".join(bad[:10]))


def check_wheel(wheel_path: Path, wheel_dir: Path, version: str, report: Report) -> None:
    with zipfile.ZipFile(wheel_path) as zf:
        members = sorted(i.filename for i in zf.infolist() if not i.is_dir())
        zf.extractall(wheel_dir)
    report.data["wheel_members"] = members
    expected_py = sorted(f"{PACKAGE}/" + p.name for p in (ROOT / PACKAGE).glob("*.py"))
    got_py = sorted(m for m in members if m.startswith(PACKAGE + "/"))
    report.check(f"wheel contains exactly the {len(expected_py)} package modules", got_py == expected_py,
                 f"unexpected: {sorted(set(got_py) ^ set(expected_py))}")
    info = f"{PACKAGE}-{version}.dist-info/"
    dist_info = {m[len(info):] for m in members if m.startswith(info)}
    report.check("wheel dist-info has METADATA, WHEEL, RECORD, entry_points.txt, top_level.txt and licenses/LICENSE",
                 WHEEL_DIST_INFO <= dist_info, ", ".join(sorted(dist_info)))
    extra = [m for m in members if not m.startswith((PACKAGE + "/", info))]
    report.check("wheel ships nothing besides the package and its metadata (no tests/, scripts/, web/)", not extra, ", ".join(extra[:10]))
    lic = wheel_dir / info / "licenses" / "LICENSE"
    report.check("wheel LICENSE is byte-identical to the repository LICENSE", lic.is_file() and lic.read_bytes() == (ROOT / "LICENSE").read_bytes())

    env = clean_env(PYTHONPATH=str(wheel_dir))
    probe = run([sys.executable, "-c", f"import {PACKAGE}; print({PACKAGE}.__version__); print({PACKAGE}.__file__)"], cwd=wheel_dir, env=env)
    lines = probe["stdout"].split()
    from_wheel = len(lines) >= 2 and Path(lines[1]).resolve().is_relative_to(wheel_dir.resolve())
    report.check(f"wheel imports as {PACKAGE} {version} from the extracted wheel only",
                 probe["exit"] == 0 and len(lines) >= 2 and lines[0] == version and from_wheel, tail(probe))
    rec = run([sys.executable, "-m", PACKAGE, "version"], cwd=wheel_dir, env=env)
    report.check("wheel: python -m pso3d version exits 0 and prints the version", rec["exit"] == 0 and rec["stdout"].strip() == version, tail(rec))
    rec = run([sys.executable, "-m", PACKAGE, "geometry", "--help"], cwd=wheel_dir, env=env)
    report.check("wheel: python -m pso3d geometry --help exits 0", rec["exit"] == 0 and "usage" in rec["stdout"].lower(), tail(rec))
    rec = run([sys.executable, "-m", PACKAGE, "baseline", "--help"], cwd=wheel_dir, env=env)
    report.data["wheel_baseline_help"] = {"exit": rec["exit"], "stderr": rec["stderr"].strip()}
    report.check(f"wheel: python -m pso3d baseline --help exits {CHECKOUT_ONLY_EXIT} with the checkout-only guidance (documented contract)",
                 rec["exit"] == CHECKOUT_ONLY_EXIT and CHECKOUT_ONLY_MESSAGE in rec["stderr"] and "scripts/run_baseline.py" in rec["stderr"], tail(rec))


def run_suite(sdist_root: Path, skip_node: bool, report: Report) -> None:
    env = clean_env()
    probe = run([sys.executable, "-c", f"import {PACKAGE}; print({PACKAGE}.__file__)"], cwd=sdist_root, env=env)
    where = probe["stdout"].strip()
    inside = bool(where) and Path(where).resolve().is_relative_to(sdist_root.resolve())
    report.check("extracted archive: `import pso3d` resolves inside the archive (PYTHONPATH unset)", probe["exit"] == 0 and inside, where or tail(probe))

    rec = run([sys.executable, *PYTEST_ARGS], cwd=sdist_root, env=env, timeout=2400)
    counts = pytest_counts(summary_line(rec["stdout"]))
    report.data["pytest"] = {"exit": rec["exit"], "seconds": rec["seconds"], "summary": summary_line(rec["stdout"]), "counts": counts,
                             "tail": tail(rec, 40)}
    report.check("extracted archive: complete Python suite passes (python -m pytest tests --ignore=tests/e2e)",
                 rec["exit"] == 0 and counts.get("passed", 0) > 0 and not any(counts.get(k) for k in ("failed", "error", "errors")),
                 (summary_line(rec["stdout"]) + f" (exit {rec['exit']}, {rec['seconds']} s)") if rec["exit"] == 0 else tail(rec))

    if skip_node:
        print("SKIPPED  extracted archive: node --test (requested with --skip-node; the Node suites were NOT run)")
        report.data["node"] = {"skipped": True}
        return
    report.data["node"] = {"skipped": False, "runs": []}
    for target in NODE_TARGETS:
        rec = run(["node", "--test", target], cwd=sdist_root, env=env, timeout=900)
        counts = tap_counts(rec["stdout"])
        report.data["node"]["runs"].append({"target": target, "exit": rec["exit"], "seconds": rec["seconds"], "counts": counts,
                                            "tail": tail(rec, 20)})
        report.check(f"extracted archive: node --test {target} passes",
                     rec["exit"] == 0 and counts.get("pass", 0) > 0 and counts.get("fail", 0) == 0,
                     f"pass {counts.get('pass', '?')} fail {counts.get('fail', '?')}" if counts else tail(rec))


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="check_sdist.py", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", metavar="DIR", default=str(ROOT / "build" / "sdist-check"),
                    help="where report.json and copies of the artefacts go (default: build/sdist-check)")
    ap.add_argument("--skip-node", action="store_true", help="do not run the Node suites inside the archive (reported as skipped)")
    ap.add_argument("--keep-temp", action="store_true", help="keep the temporary build/extraction directory (printed at the end)")
    ns = ap.parse_args(argv)
    out = Path(ns.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "dist").mkdir(exist_ok=True)

    started = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    report = Report()
    report.data["environment"] = {"python": sys.version.split()[0], "platform": platform.platform(), "node": summary_line(run(["node", "--version"], cwd=ROOT)["stdout"])}
    tmp = Path(tempfile.mkdtemp(prefix=f"{PACKAGE}-sdist-check-"))
    print(f"check_sdist: temporary directory {tmp}")
    print("[0/6] checking that this interpreter can build the archives")
    if not check_backend(report):
        return finish(report, out, started, t0, tmp, ns.keep_temp)
    try:
        src, dist, sdist_extract, wheel_dir = tmp / "src", tmp / "dist", tmp / "sdist", tmp / "wheel"
        print("[1/6] copying the checkout (without .git, build output, caches, byte-code, node_modules)")
        shutil.copytree(ROOT, src, ignore=shutil.ignore_patterns(*COPY_IGNORE), symlinks=True)
        report.check("checkout copied to a temporary directory", (src / "pyproject.toml").is_file() and (src / "MANIFEST.in").is_file(),
                     "" if (src / "MANIFEST.in").is_file() else "MANIFEST.in is missing")

        print("[2/6] building the sdist and the wheel")
        dist.mkdir()
        sdist_path, wheel_path = build_artifacts(src, dist, report)
        if sdist_path is None or wheel_path is None:
            return finish(report, out, started, t0, tmp, ns.keep_temp)
        for p in (sdist_path, wheel_path):
            shutil.copy2(p, out / "dist" / p.name)

        print("[3/6] extracting and checking the archive contents")
        prefix, raw, members = extract_sdist(sdist_path, sdist_extract)
        sdist_root = sdist_extract / prefix
        version = ""
        m = re.search(r'^__version__\s*=\s*"([^"]+)"', (sdist_root / PACKAGE / "__init__.py").read_text(encoding="utf-8"), re.M) if prefix else None
        if m:
            version = m.group(1)
        report.check("sdist contains pso3d/__init__.py with __version__", bool(version), version)
        report.data["sdist"] = {"file": sdist_path.name, "sha256": sha256(sdist_path), "bytes": sdist_path.stat().st_size,
                                "member_count": len(members), "members": members}
        report.data["wheel"] = {"file": wheel_path.name, "sha256": sha256(wheel_path), "bytes": wheel_path.stat().st_size}
        report.data["version"] = version
        report.check(f"sdist file name is {PACKAGE}-{version}.tar.gz", sdist_path.name == f"{PACKAGE}-{version}.tar.gz", sdist_path.name)
        check_sdist_members(prefix, raw, members, version, report)
        report.check("sdist LICENSE is byte-identical to the repository LICENSE",
                     (sdist_root / "LICENSE").is_file() and (sdist_root / "LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes())

        print("[4/6] checking the wheel")
        check_wheel(wheel_path, wheel_dir, version, report)

        print("[5/6] running the complete suite inside the extracted source archive")
        run_suite(sdist_root, ns.skip_node, report)
    finally:
        pass
    return finish(report, out, started, t0, tmp, ns.keep_temp)


def finish(report: Report, out: Path, started: datetime, t0: float, tmp: Path, keep: bool) -> int:
    print("[6/6] writing the report")
    ok = report.ok
    failed = [c for c in report.checks if not c["ok"]]
    payload = {"ok": ok, "exit_code": 0 if ok else 1, "started": started.isoformat(timespec="seconds"),
               "seconds": round(time.perf_counter() - t0, 1), "totals": {"checks": len(report.checks), "failed": len(failed)},
               "checks": report.checks, **report.data}
    (out / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sd, wh = report.data.get("sdist"), report.data.get("wheel")
    print("")
    print("== check_sdist summary ==")
    if sd:
        print(f"  sdist  {sd['file']}  {sd['member_count']} members  {sd['bytes']} bytes  sha256 {sd['sha256']}")
    if wh:
        print(f"  wheel  {wh['file']}  {wh['bytes']} bytes  sha256 {wh['sha256']}")
    py = report.data.get("pytest")
    if py:
        print(f"  pytest inside the archive: {py['summary']}")
    node = report.data.get("node")
    if node and node.get("skipped"):
        print("  node tests inside the archive: SKIPPED (--skip-node)")
    elif node:
        for r in node["runs"]:
            print(f"  node --test {r['target']}: pass {r['counts'].get('pass', '?')} fail {r['counts'].get('fail', '?')} (exit {r['exit']})")
    for c in failed:
        print(f"  FAILED: {c['name']}" + (f" -- {c['detail'].splitlines()[-1][:200]}" if c['detail'] else ""))
    print(f"  checks: {len(report.checks) - len(failed)}/{len(report.checks)} passed  report: {out / 'report.json'}  {payload['seconds']} s")
    if keep:
        print(f"  temporary directory kept: {tmp}")
    else:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"RESULT: {'OK: source archive complete and self-sufficient' if ok else 'FAIL: source archive check failed'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

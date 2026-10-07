"""Command-line interface: ``python -m pso3d <command> [options]`` (or the ``pso3d`` console script).

Commands
--------
baseline      reproduce the Review-1 baseline run (delegates to scripts/run_baseline.py)
experiments   Monte-Carlo parameter study (delegates to scripts/run_experiments.py)
network       multi-node iterative auto-localization (delegates to scripts/run_network.py)
geometry      anchor-geometry diagnostics: rank, non-coplanarity, GDOP, CRLB (pso3d.geometry)
version       print the package version

The three script commands need a repository checkout (``scripts/`` next to the ``pso3d`` package,
e.g. after ``pip install -e .``); every option after the command name is forwarded unchanged to
the script's ``main(argv)``.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

PACKAGE_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = PACKAGE_DIR.parent / "scripts"

COMMANDS: dict[str, str] = {
    "baseline": "reproduce the Review-1 baseline run and its comparisons (scripts/run_baseline.py)",
    "experiments": "Monte-Carlo parameter study: sweeps, error CDF, results.md (scripts/run_experiments.py)",
    "network": "multi-node iterative auto-localization and coverage (scripts/run_network.py)",
    "geometry": "anchor-geometry diagnostics: rank, non-coplanarity, GDOP, CRLB bound",
    "version": "print the package version",
}
SCRIPT_OF = {"baseline": "run_baseline.py", "experiments": "run_experiments.py", "network": "run_network.py"}
DEFAULT_SIGMA = 0.5   # ranging-noise std [m] for the CRLB when a scenario gives none (Monte-Carlo base setting)


def usage() -> str:
    width = max(len(k) for k in COMMANDS)
    lines = ["usage: pso3d <command> [options]",
             "       python -m pso3d <command> [options]", "",
             "PSO-based 3D node localization for wireless sensor networks.", "", "commands:"]
    lines += [f"  {name.ljust(width)}  {text}" for name, text in COMMANDS.items()]
    lines += ["",
              "Run 'pso3d <command> --help' for the options of a command. The baseline, experiments and",
              "network commands forward their options to scripts/<name>.py and need a repository checkout.",
              "", "examples:",
              "  pso3d baseline --out results --figures figures --no-gif",
              "  pso3d experiments --trials 50 --out build/exp --figures build/exp",
              "  pso3d network --nodes 50 --beacons 10 --rounds 10",
              "  pso3d geometry --json",
              "  pso3d geometry --scenario scenario.json --sigma 0.3"]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# script delegation
# --------------------------------------------------------------------------------------------
def _exit_code(code: Any) -> int:
    """Normalise a main() return value or SystemExit code to a process exit code."""
    if code is None:
        return 0
    if isinstance(code, bool):
        return 0 if code else 1
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _load_script_main(command: str) -> Callable[..., Any] | None:
    """Import scripts/<name>.py by file path and return its main(argv); None (after a message) if unavailable."""
    path = SCRIPTS_DIR / SCRIPT_OF[command]
    if not path.is_file():
        print(f"pso3d {command}: cannot find {path}\n"
              f"  This command runs scripts/{SCRIPT_OF[command]} from a repository checkout; clone the repository "
              "and install it with 'pip install -e .' (or run the script directly).", file=sys.stderr)
        return None
    source = path.read_text(encoding="utf-8")
    if "def main(" not in source:
        print(f"pso3d {command}: {path} defines no main(argv) entry point", file=sys.stderr)
        return None
    name = f"pso3d_script_{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        print(f"pso3d {command}: cannot load {path}", file=sys.stderr)
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module          # dataclasses inside the script resolve their module through sys.modules
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        print(f"pso3d {command}: {path} has no callable main(argv)", file=sys.stderr)
        return None
    return main


def _run_script(command: str, args: list[str]) -> int:
    main = _load_script_main(command)
    if main is None:
        return 2
    try:
        return _exit_code(main(list(args)))
    except SystemExit as exc:           # argparse --help / usage errors raised inside the script
        return _exit_code(exc.code)


# --------------------------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------------------------
def _jsonable(obj: Any, strict: bool = False) -> Any:
    """Plain JSON types from numpy scalars/arrays, dataclasses, namedtuples (strict: non-finite -> None)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name), strict) for f in dataclasses.fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v, strict) for k, v in obj.items()}
    if hasattr(obj, "tolist"):          # numpy arrays and scalars
        return _jsonable(obj.tolist(), strict)
    if hasattr(obj, "_asdict"):         # namedtuple
        return _jsonable(obj._asdict(), strict)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v, strict) for v in obj]
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if (math.isfinite(obj) or not strict) else None
    return str(obj)


# keys of geometry.scenario_report() that the CLI presents under its canonical names
_REPORT_ALIASES = {"rank": "anchor_rank", "crlb_rmse": "crlb_rmse_bound", "crlb_rmse_m": "crlb_rmse_bound",
                   "is_non_coplanar": "non_coplanar"}


def _geometry_report(geo: Any, anchors: Any, position: Any, sigma: float) -> dict[str, Any]:
    """scenario_report() from pso3d.geometry, completed with the contract functions when keys are missing."""
    report: dict[str, Any] = {}
    scenario_report = getattr(geo, "scenario_report", None)
    if callable(scenario_report):
        full = _jsonable(scenario_report(anchors, position, sigma))
        if isinstance(full, dict):
            report.update(full)
    for old, new in _REPORT_ALIASES.items():
        if old in report and new not in report:
            report[new] = report.pop(old)
    report.setdefault("n_anchors", int(len(anchors)))
    report.setdefault("position", [float(v) for v in position])
    report.setdefault("sigma", float(sigma))
    if "anchor_rank" not in report and hasattr(geo, "anchor_rank"):
        report["anchor_rank"] = int(geo.anchor_rank(anchors))
    if "non_coplanar" not in report and hasattr(geo, "is_non_coplanar"):
        report["non_coplanar"] = bool(geo.is_non_coplanar(anchors))
    if "gdop" not in report and hasattr(geo, "gdop"):
        report["gdop"] = float(geo.gdop(anchors, position))
    if hasattr(geo, "crlb") and ("crlb_rmse_bound" not in report or "crlb_per_axis" not in report):
        bound = geo.crlb(anchors, position, sigma)
        report.setdefault("crlb_rmse_bound", float(bound.rmse_bound))
        report.setdefault("crlb_per_axis", [float(v) for v in bound.per_axis])
    return report


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return "inf" if math.isinf(value) else ("nan" if math.isnan(value) else f"{value:.4g}")
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (list, tuple)):
        return "(" + ", ".join(_fmt(v) for v in value) + ")"
    return str(value)


def _print_geometry_report(report: dict[str, Any]) -> None:
    shown = {"n_anchors", "anchor_rank", "non_coplanar", "position", "sigma", "gdop", "crlb_rmse_bound", "crlb_per_axis"}
    print(f"anchors: {report.get('n_anchors')}   rank: {report.get('anchor_rank', '?')}   "
          f"non-coplanar: {_fmt(report.get('non_coplanar', '?'))}")
    print(f"node position: {_fmt(report.get('position'))} m   sigma: {_fmt(report.get('sigma'))} m")
    print(f"GDOP at the node: {_fmt(report.get('gdop', '?'))}")
    print(f"CRLB RMSE bound: {_fmt(report.get('crlb_rmse_bound', '?'))} m   per axis (x, y, z): {_fmt(report.get('crlb_per_axis', '?'))} m")
    for key in sorted(k for k in report if k not in shown):
        print(f"{key}: {json.dumps(_jsonable(report[key], strict=True))}")


def cmd_geometry(args: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pso3d geometry",
                                 description="Anchor-geometry diagnostics for a range-based 3D localization scenario: anchor rank, "
                                             "non-coplanarity, GDOP and the Cramer-Rao lower bound (RMSE) at the node position.")
    ap.add_argument("--scenario", metavar="FILE",
                    help='JSON file with keys "anchors" (M x 3), "position" (3) and optional "sigma" (ranging-noise std, m); '
                         "default: the Review-1 baseline scenario")
    ap.add_argument("--sigma", type=float, metavar="M", help=f"ranging-noise sigma for the CRLB (default: file value or {DEFAULT_SIGMA})")
    ap.add_argument("--json", action="store_true", help="print the report as JSON (non-finite numbers become null)")
    ns = ap.parse_args(args)
    try:
        from pso3d import geometry as geo
    except ImportError as exc:
        print(f"pso3d geometry: the pso3d.geometry module is not available ({exc})", file=sys.stderr)
        return 2
    import numpy as np

    sigma = DEFAULT_SIGMA
    if ns.scenario:
        try:
            with open(ns.scenario, encoding="utf-8") as fh:
                data = json.load(fh)
            anchors = np.asarray(data["anchors"], dtype=float)
            position = np.asarray(data["position"] if "position" in data else data["true_position"], dtype=float)
            sigma = float(data.get("sigma", DEFAULT_SIGMA))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"pso3d geometry: cannot read scenario {ns.scenario!r}: {exc!r} "
                  '(expected JSON keys "anchors", "position" and optional "sigma")', file=sys.stderr)
            return 2
    else:
        from pso3d import Scenario
        sc = Scenario()
        anchors, position = sc.anchors, sc.true_position
    if ns.sigma is not None:
        sigma = float(ns.sigma)
    if anchors.ndim != 2 or anchors.shape[1] != 3 or anchors.shape[0] < 3 or position.shape != (3,):
        print("pso3d geometry: anchors must be an M x 3 array (M >= 3) and position a 3-vector", file=sys.stderr)
        return 2
    report = _geometry_report(geo, anchors, position, sigma)
    if ns.json:
        print(json.dumps(_jsonable(report, strict=True), indent=2))
    else:
        _print_geometry_report(report)
    return 0


# --------------------------------------------------------------------------------------------
# version / dispatch
# --------------------------------------------------------------------------------------------
def cmd_version(args: list[str]) -> int:
    if any(a in ("-h", "--help") for a in args):
        print("usage: pso3d version\n\nPrint the pso3d package version.")
        return 0
    import pso3d
    print(pso3d.__version__)
    return 0


_HANDLERS: dict[str, Callable[[list[str]], int]] = {
    "baseline": lambda args: _run_script("baseline", args),
    "experiments": lambda args: _run_script("experiments", args),
    "network": lambda args: _run_script("network", args),
    "geometry": cmd_geometry,
    "version": cmd_version,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(usage(), file=sys.stderr)
        return 2
    command, rest = args[0], args[1:]
    if command in ("-h", "--help", "help"):
        print(usage())
        return 0
    if command in ("-V", "--version"):
        command = "version"
    handler = _HANDLERS.get(command)
    if handler is None:
        print(f"pso3d: unknown command {command!r}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":   # pragma: no cover - `python -m pso3d` goes through __main__.py
    raise SystemExit(main())

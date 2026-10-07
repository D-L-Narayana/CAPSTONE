"""Monte-Carlo parameter study: noise, number of anchors, swarm size, iterations.

    python scripts/run_experiments.py [--trials 200] [--out DIR] [--figures DIR]
                                      [--methods pso,std,lsq,lsq_gn] [--extended] [--no-plots]

Writes <out>/sweeps.csv and <out>/results.md plus <figures>/sweep_*.png and <figures>/error_cdf.png.
With the default arguments the CSV has exactly the Phase-1 header and rows (the four legacy methods).
Requesting other methods (``--methods`` with keys from ``pso3d.experiments.METHODS`` or ``--extended``,
which adds amcmpso, std_tol, std_ws and amcmpso_ws) appends the columns ``crlb_rmse_m`` and
``iterations_mean``, adds an "Extended methods" table and a CRLB column to results.md and renders
<figures>/method_bars.png.
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pso3d.experiments import METHODS, LEGACY_METHODS, EXTENDED_METHODS, resolve_methods, sweep, monte_carlo  # noqa: E402
from pso3d.plots import plot_sweep, plot_error_cdf, plot_method_bars  # noqa: E402

BASE = dict(noise_sigma=0.5, n_anchors=4, n_particles=20, n_iterations=60)
SWEEPS = (
    ("noise_sigma", [0.1, 0.25, 0.5, 1.0, 2.0], "ranging noise sigma [m]", "sweep_noise.png", True),
    ("n_anchors", [4, 5, 6, 8, 10], "number of anchors", "sweep_anchors.png", False),
    ("n_particles", [5, 10, 20, 40, 80], "swarm size N", "sweep_particles.png", True),
    ("n_iterations", [10, 20, 30, 60, 120], "iterations T", "sweep_iterations.png", True),
)
NAMES = {"noise_sigma": "Ranging noise sigma [m]", "n_anchors": "Number of anchors",
         "n_particles": "Swarm size N", "n_iterations": "Iterations T"}
LEGACY_HEADER = ["parameter", "value", "method", "rmse_m", "mean_error_m", "p90_error_m",
                 "fitness_evaluations", "runtime_ms", "memory_floats", "trials"]
EXTRA_COLUMNS = ["crlb_rmse_m", "iterations_mean"]


def parse_args(argv):
    ap = argparse.ArgumentParser(description="Monte-Carlo parameter study of the 3D localizers.")
    ap.add_argument("--trials", type=int, default=200, help="random nodes per setting (default 200)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"), help="directory for sweeps.csv and results.md")
    ap.add_argument("--figures", default=os.path.join(ROOT, "figures"), help="directory for the PNG figures")
    ap.add_argument("--methods", default=",".join(LEGACY_METHODS),
                    help="comma-separated method keys (default: the legacy four; 'all' selects every method): "
                         + ", ".join(METHODS))
    ap.add_argument("--extended", action="store_true",
                    help="also run " + ", ".join(EXTENDED_METHODS) + " and add the CRLB/iterations columns")
    ap.add_argument("--no-plots", action="store_true", help="skip the PNG figures (write sweeps.csv and results.md only)")
    return ap.parse_args(argv)


def cdf_label(method: str) -> str:
    spec = METHODS[method]
    if spec.closed_form:
        return spec.label
    return f"{spec.label} ({BASE['n_particles']} x {BASE['n_iterations']})"


def write_csv(rows, path, extended: bool) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(LEGACY_HEADER + (EXTRA_COLUMNS if extended else []))
        for r in rows:
            record = [r.label, r.value, r.method, f"{r.rmse:.3f}", f"{r.mean_error:.3f}", f"{r.p90_error:.3f}",
                      f"{r.evaluations:.0f}", f"{r.runtime_ms:.3f}", r.memory_floats, r.trials]
            if extended:
                record += [f"{r.crlb_rmse:.3f}", f"{r.iterations_mean:.1f}"]
            w.writerow(record)


def build_report(rows, mc, keys, trials: int, extended: bool) -> list[str]:
    b = BASE
    lines = ["# Monte-Carlo parameter study", "",
             f"{trials} random nodes per setting, uniform in the 60 x 60 x 20 m field; Gaussian ranging noise; "
             "anchors 1-4 are the fixed non-coplanar corners of the baseline scenario, extra anchors are random. "
             f"Base setting: sigma = {b['noise_sigma']} m, {b['n_anchors']} anchors, N = {b['n_particles']}, T = {b['n_iterations']}. "
             "Both PSO variants clip particles to the field; LSQ is the linearised trilateration solve, LSQ + Gauss-Newton refines it (<= 10 steps).", ""]
    if extended:
        lines += ["Extended methods: AMCMPSO (an interpretation of the adaptive mean / centre-of-mass idea of the base paper, "
                  "not a reproduction of it), Standard PSO with a relative tolerance stop (1 % over 10 iterations) and the "
                  "warm-started variants (particle 0 seeded with the LSQ + Gauss-Newton estimate). The CRLB column is the "
                  "mean Cramer-Rao lower bound on the RMS position error at the sampled true positions for Gaussian noise of the "
                  "given sigma; it depends on the anchor geometry only and is a reference, not a method.", ""]
    if extended:
        header = "| value | method | RMSE [m] | mean err [m] | p90 err [m] | CRLB [m] | fitness evals | iterations | runtime [ms] | memory [floats] |"
        rule = "|---|---|---|---|---|---|---|---|---|---|"
    else:
        header = "| value | method | RMSE [m] | mean err [m] | p90 err [m] | fitness evals | runtime [ms] | memory [floats] |"
        rule = "|---|---|---|---|---|---|---|---|"
    for lab in NAMES:
        selected = [r for r in rows if r.label == lab]
        if not selected:              # closed-form methods only: the swarm-parameter sweeps have no rows
            lines += [f"## {NAMES[lab]}", "", "No swarm method selected; this sweep applies to swarm methods only.", ""]
            continue
        lines += [f"## {NAMES[lab]}", "", header, rule]
        for r in selected:
            cells = [f"{r.value:g}", METHODS[r.method].label, f"{r.rmse:.3f}", f"{r.mean_error:.3f}", f"{r.p90_error:.3f}"]
            if extended:
                cells.append(f"{r.crlb_rmse:.3f}")
            cells.append(f"{r.evaluations:.0f}")
            if extended:
                cells.append(f"{r.iterations_mean:.1f}")
            cells += [f"{r.runtime_ms:.3f}", f"{r.memory_floats}"]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"## Error CDF (sigma = {b['noise_sigma']} m, {b['n_anchors']} anchors)", "",
              "| method | RMSE [m] | median [m] | p90 [m] | fitness evals | memory [floats] |", "|---|---|---|---|---|---|"]
    for m in keys:
        r = mc[m]
        lines.append(f"| {cdf_label(m)} | {r['rmse']:.3f} | {np.median(r['errors']):.3f} | {r['p90_error']:.3f} | "
                     f"{r['evaluations']:.0f} | {r['memory_floats']} |")
    lines.append("")
    if extended:
        lines += [f"## Extended methods (sigma = {b['noise_sigma']} m, {b['n_anchors']} anchors, N = {b['n_particles']}, T = {b['n_iterations']})", "",
                  "| method | RMSE [m] | median [m] | p90 [m] | CRLB [m] | fitness evals | iterations | memory [floats] |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in keys:
            r = mc[m]
            lines.append(f"| {METHODS[m].label} | {r['rmse']:.3f} | {np.median(r['errors']):.3f} | {r['p90_error']:.3f} | "
                         f"{r['crlb_rmse']:.3f} | {r['evaluations']:.0f} | {r['iterations_mean']:.1f} | {r['memory_floats']} |")
        lines += ["",
                  "Notes: fitness evaluations count swarm evaluations only (the warm start adds one least-squares solve and "
                  "<= 10 Gauss-Newton steps per node); iterations are the optimiser iterations actually run (the tolerance stop "
                  "ends a run early, closed-form rows report Gauss-Newton steps). AMCMPSO numbers are those of this repository's "
                  "interpretation and must not be quoted as results of the base paper.", ""]
    return lines


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        keys = list(resolve_methods(args.methods))
        if args.extended:
            keys = list(resolve_methods(keys + list(EXTENDED_METHODS)))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.trials < 1:
        print("error: --trials must be >= 1", file=sys.stderr)
        return 2
    extended = args.extended or any(k not in LEGACY_METHODS for k in keys)
    plots = not args.no_plots
    os.makedirs(args.out, exist_ok=True)
    if plots:
        os.makedirs(args.figures, exist_ok=True)
    T = args.trials

    rows = []
    for label, values, _xlabel, _fname, _logx in SWEEPS:
        rows += sweep(label, values, T, methods=keys, **BASE)
    write_csv(rows, os.path.join(args.out, "sweeps.csv"), extended)

    mc = {m: monte_carlo(m, T, BASE["noise_sigma"], BASE["n_anchors"], BASE["n_particles"], BASE["n_iterations"])
          for m in keys}
    if plots:
        for label, _values, xlabel, fname, logx in SWEEPS:
            plot_sweep(rows, label, xlabel, os.path.join(args.figures, fname), logx=logx)
        plot_error_cdf({cdf_label(m): mc[m]["errors"] for m in mc}, os.path.join(args.figures, "error_cdf.png"),
                       title=f"Error CDF, Monte-Carlo (sigma = {BASE['noise_sigma']} m, {BASE['n_anchors']} anchors)")
        if extended:
            plot_method_bars(rows, os.path.join(args.figures, "method_bars.png"), "noise_sigma", BASE["noise_sigma"])

    lines = build_report(rows, mc, keys, T, extended)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

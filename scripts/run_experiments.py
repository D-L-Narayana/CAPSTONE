"""Monte-Carlo parameter study: noise, number of anchors, swarm size, iterations.

    python scripts/run_experiments.py [--trials 200]

Writes results/sweeps.csv, results/results.md and figures/sweep_*.png, figures/error_cdf.png.
"""
from __future__ import annotations
import argparse, csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pso3d.experiments import sweep, monte_carlo
from pso3d.plots import plot_sweep, plot_error_cdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

ap = argparse.ArgumentParser(); ap.add_argument("--trials", type=int, default=200); args = ap.parse_args()
T = args.trials
base = dict(noise_sigma=0.5, n_anchors=4, n_particles=20, n_iterations=60)

label_of = {"pso": "Simplified PSO (20 x 60)", "std": "Standard PSO (20 x 60)", "lsq": "Least-squares trilateration", "lsq_gn": "LSQ + Gauss-Newton"}
rows = []
rows += sweep("noise_sigma", [0.1, 0.25, 0.5, 1.0, 2.0], T, **base)
rows += sweep("n_anchors", [4, 5, 6, 8, 10], T, **base)
rows += sweep("n_particles", [5, 10, 20, 40, 80], T, **base)
rows += sweep("n_iterations", [10, 20, 30, 60, 120], T, **base)

with open(os.path.join(RES, "sweeps.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["parameter", "value", "method", "rmse_m", "mean_error_m", "p90_error_m",
                                   "fitness_evaluations", "runtime_ms", "memory_floats", "trials"])
    for r in rows:
        w.writerow([r.label, r.value, r.method, f"{r.rmse:.3f}", f"{r.mean_error:.3f}", f"{r.p90_error:.3f}",
                    f"{r.evaluations:.0f}", f"{r.runtime_ms:.3f}", r.memory_floats, r.trials])

plot_sweep(rows, "noise_sigma", "ranging noise sigma [m]", os.path.join(FIG, "sweep_noise.png"), logx=True)
plot_sweep(rows, "n_anchors", "number of anchors", os.path.join(FIG, "sweep_anchors.png"))
plot_sweep(rows, "n_particles", "swarm size N", os.path.join(FIG, "sweep_particles.png"), logx=True)
plot_sweep(rows, "n_iterations", "iterations T", os.path.join(FIG, "sweep_iterations.png"), logx=True)

mc = {m: monte_carlo(m, T, 0.5, 4, 20, 60) for m in ("pso", "std", "lsq", "lsq_gn")}
plot_error_cdf({label_of[m]: mc[m]["errors"] for m in mc}, os.path.join(FIG, "error_cdf.png"))

names = {"noise_sigma": "Ranging noise sigma [m]", "n_anchors": "Number of anchors", "n_particles": "Swarm size N", "n_iterations": "Iterations T"}
lines = ["# Monte-Carlo parameter study", "",
         f"{T} random nodes per setting, uniform in the 60 x 60 x 20 m field; Gaussian ranging noise; "
         "anchors 1-4 are the fixed non-coplanar corners of the baseline scenario, extra anchors are random. "
         f"Base setting: sigma = {base['noise_sigma']} m, {base['n_anchors']} anchors, N = {base['n_particles']}, T = {base['n_iterations']}. "
         "Both PSO variants clip particles to the field; LSQ is the linearised trilateration solve, LSQ + Gauss-Newton refines it (<= 10 steps).", ""]
for lab in ("noise_sigma", "n_anchors", "n_particles", "n_iterations"):
    lines += [f"## {names[lab]}", "", "| value | method | RMSE [m] | mean err [m] | p90 err [m] | fitness evals | runtime [ms] | memory [floats] |", "|---|---|---|---|---|---|---|---|"]
    for r in [r for r in rows if r.label == lab]:
        lines.append(f"| {r.value:g} | {label_of[r.method].split(' (')[0]} | {r.rmse:.3f} | {r.mean_error:.3f} | {r.p90_error:.3f} | {r.evaluations:.0f} | {r.runtime_ms:.3f} | {r.memory_floats} |")
    lines.append("")
lines += ["## Error CDF (sigma = 0.5 m, 4 anchors)", "", "| method | RMSE [m] | median [m] | p90 [m] | fitness evals | memory [floats] |", "|---|---|---|---|---|---|"]
for m in mc:
    lines.append(f"| {label_of[m]} | {mc[m]['rmse']:.3f} | {np.median(mc[m]['errors']):.3f} | {mc[m]['p90_error']:.3f} | {mc[m]['evaluations']:.0f} | {mc[m]['memory_floats']} |")
lines.append("")
open(os.path.join(RES, "results.md"), "w").write("\n".join(lines))
print("\n".join(lines))

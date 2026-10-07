"""Reproduce the Review-1 baseline (slide 16) and compare with least squares.

    python scripts/run_baseline.py [--out DIR] [--figures DIR] [--no-gif] [--quiet]
    python -m pso3d baseline [same options]

Writes baseline.json to --out (default results/) and convergence.png, scenario_3d.png and
convergence.gif to --figures (default figures/). The printed numbers are the Phase-1 regression
gate: estimate (37.43, 11.90, 9.29) m, error 0.90 m, 1200 fitness evaluations, 120 floats.
"""
from __future__ import annotations
import argparse, inspect, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from pso3d import (Scenario, RangeErrorFitness, SimplifiedPSO, StandardPSO, AMCMPSO,
                   least_squares_trilateration, gauss_newton_refine)
from pso3d.plots import plot_convergence, plot_scenario_3d, make_convergence_gif


def _display(path: str) -> str:
    """Repository-relative path for the summary line (identical to the historical output for the defaults)."""
    rel = os.path.relpath(path, ROOT)
    return path if rel.startswith("..") else rel.replace(os.sep, "/")


def run_baseline(out_dir: str, fig_dir: str, gif: bool = True, quiet: bool = False) -> dict:
    """Run the baseline study, write baseline.json + figures and return the results dict."""
    say = (lambda *a, **k: None) if quiet else print
    os.makedirs(out_dir, exist_ok=True); os.makedirs(fig_dir, exist_ok=True)

    sc = Scenario()
    meas = sc.measured()
    say("measured ranges d_hat:", np.round(meas, 3))

    # --- 1. simplified PSO, exact Review-1 settings -------------------------------------------
    fit = RangeErrorFitness(sc.anchors, meas)
    pso = SimplifiedPSO(n_particles=20, n_iterations=60, w=0.7, c=1.4, seed=1, clip=False, record_positions=True)
    res = pso.run(fit, sc.lower, sc.upper)
    err = res.error(sc.true_position)
    dz = res.estimate - sc.true_position
    say(f"PSO estimate: {np.round(res.estimate, 2)}  error: {err:.2f} m  (dx,dy,dz)=({dz[0]:+.2f},{dz[1]:+.2f},{dz[2]:+.2f})")
    say(f"fitness evaluations: {res.fitness_evaluations}  distance computations: {res.distance_computations}")
    say(f"swarm state: {res.swarm_state_floats} floats = {res.swarm_state_bytes} bytes (float64)  runtime: {res.runtime_s*1e3:.2f} ms")
    # plateau: first iteration after which best fitness never improves by > 1 %
    bh = np.array(res.best_history)
    plateau = int(np.argmax(bh <= bh[-1] * 1.01)) + 1
    say(f"best fitness {bh[-1]:.4f} m^2; within 1 % of final value from iteration {plateau}")

    # --- 2. least-squares trilateration (+ Gauss-Newton refinement) ----------------------------
    t0 = time.perf_counter(); p_ls = least_squares_trilateration(sc.anchors, meas); t_ls = time.perf_counter() - t0
    p_gn, gn_it = gauss_newton_refine(sc.anchors, meas, p_ls)
    say(f"LSQ  estimate: {np.round(p_ls, 2)}  error: {np.linalg.norm(p_ls - sc.true_position):.2f} m  ({t_ls*1e6:.0f} us)")
    say(f"LSQ+GN estimate: {np.round(p_gn, 2)}  error: {np.linalg.norm(p_gn - sc.true_position):.2f} m  ({gn_it} GN iterations)")

    # --- 3. reference PSO variants (same budget); AMCMPSO is an interpretation of Alhasan et al. 2023 -----
    fit2 = RangeErrorFitness(sc.anchors, meas)
    res_std = StandardPSO(20, 60, seed=1).run(fit2, sc.lower, sc.upper)
    fit3 = RangeErrorFitness(sc.anchors, meas)
    res_am = AMCMPSO(20, 60, seed=1).run(fit3, sc.lower, sc.upper)
    say(f"Standard PSO (pbest/gbest): {np.round(res_std.estimate, 2)} error {res_std.error(sc.true_position):.2f} m, {res_std.fitness_evaluations} evals, {res_std.swarm_state_floats} floats")
    say(f"AMCMPSO (interpretation of the base paper, not a reproduction): {np.round(res_am.estimate, 2)} error {res_am.error(sc.true_position):.2f} m, {res_am.fitness_evaluations} evals, {res_am.swarm_state_floats} floats")

    # --- 4. early stopping variant ---------------------------------------------------------------
    fit4 = RangeErrorFitness(sc.anchors, meas)
    res_es = SimplifiedPSO(20, 60, seed=1, clip=False, early_stop_patience=10).run(fit4, sc.lower, sc.upper)
    say(f"Simplified PSO + early stop (patience 10): {np.round(res_es.estimate, 2)} error {res_es.error(sc.true_position):.2f} m after {res_es.iterations_run} iterations, {res_es.fitness_evaluations} evals")

    out = {
        "scenario": {"field": sc.field_size, "anchors": sc.anchors.tolist(), "true_position": sc.true_position.tolist(),
                     "noise": sc.noise.tolist(), "measured": meas.tolist()},
        "simplified_pso": {"estimate": res.estimate.tolist(), "error_m": err, "delta": dz.tolist(),
                            "fitness_evaluations": res.fitness_evaluations, "distance_computations": res.distance_computations,
                            "swarm_state_floats": res.swarm_state_floats, "swarm_state_bytes": res.swarm_state_bytes,
                            "runtime_ms": res.runtime_s * 1e3, "best_fitness": bh[-1], "plateau_iteration": plateau,
                            "best_history": bh.tolist()},
        "least_squares": {"estimate": p_ls.tolist(), "error_m": float(np.linalg.norm(p_ls - sc.true_position)), "runtime_us": t_ls * 1e6},
        "least_squares_gauss_newton": {"estimate": p_gn.tolist(), "error_m": float(np.linalg.norm(p_gn - sc.true_position)), "iterations": gn_it},
        "standard_pso": {"estimate": res_std.estimate.tolist(), "error_m": res_std.error(sc.true_position),
                         "fitness_evaluations": res_std.fitness_evaluations, "swarm_state_floats": res_std.swarm_state_floats},
        "amcmpso_interpretation": {"estimate": res_am.estimate.tolist(), "error_m": res_am.error(sc.true_position),
                                   "fitness_evaluations": res_am.fitness_evaluations, "swarm_state_floats": res_am.swarm_state_floats},
        "simplified_pso_early_stop": {"estimate": res_es.estimate.tolist(), "error_m": res_es.error(sc.true_position),
                                      "iterations_run": res_es.iterations_run, "fitness_evaluations": res_es.fitness_evaluations},
    }
    json_path = os.path.join(out_dir, "baseline.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    conv_path = os.path.join(fig_dir, "convergence.png")
    # the x = 30 "plateau region" annotation is an argument of newer plots.plot_convergence; keep today's figure either way
    conv_kwargs = {"plateau_iteration": 30} if "plateau_iteration" in inspect.signature(plot_convergence).parameters else {}
    plot_convergence(res.best_history, conv_path, **conv_kwargs)
    scen_path = os.path.join(fig_dir, "scenario_3d.png")
    plot_scenario_3d(sc, res.estimate, res.positions_history, scen_path)
    written = [json_path, conv_path, scen_path]
    if gif:
        gif_path = os.path.join(fig_dir, "convergence.gif")
        make_convergence_gif(sc, res.positions_history, res.estimate_history, gif_path)
        written.append(gif_path)
    say("saved " + ", ".join(_display(p) for p in written))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="run_baseline.py",
                                 description="Reproduce the Review-1 baseline run (simplified PSO, 20 x 60, seed 1) and compare it "
                                             "with least squares, Gauss-Newton, standard PSO and AMCMPSO. Also: python -m pso3d baseline")
    ap.add_argument("--out", metavar="DIR", default=os.path.join(ROOT, "results"), help="directory for baseline.json (default: results/)")
    ap.add_argument("--figures", metavar="DIR", default=os.path.join(ROOT, "figures"),
                    help="directory for convergence.png, scenario_3d.png and convergence.gif (default: figures/)")
    ap.add_argument("--no-gif", action="store_true", help="skip the convergence GIF (the slowest step)")
    ap.add_argument("--quiet", action="store_true", help="do not print the run summary")
    args = ap.parse_args(sys.argv[1:] if argv is None else argv)
    run_baseline(args.out, args.figures, gif=not args.no_gif, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

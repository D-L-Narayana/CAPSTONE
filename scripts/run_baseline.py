"""Reproduce the Review-1 baseline (slide 16) and compare with least squares.

    python scripts/run_baseline.py
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pso3d import (Scenario, RangeErrorFitness, SimplifiedPSO, StandardPSO, AMCMPSO,
                   least_squares_trilateration, gauss_newton_refine)
from pso3d.plots import plot_convergence, plot_scenario_3d, make_convergence_gif

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures")
os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)

sc = Scenario()
meas = sc.measured()
print("measured ranges d_hat:", np.round(meas, 3))

# --- 1. simplified PSO, exact Review-1 settings -------------------------------------------
fit = RangeErrorFitness(sc.anchors, meas)
pso = SimplifiedPSO(n_particles=20, n_iterations=60, w=0.7, c=1.4, seed=1, clip=False, record_positions=True)
res = pso.run(fit, sc.lower, sc.upper)
err = res.error(sc.true_position)
dz = res.estimate - sc.true_position
print(f"PSO estimate: {np.round(res.estimate, 2)}  error: {err:.2f} m  (dx,dy,dz)=({dz[0]:+.2f},{dz[1]:+.2f},{dz[2]:+.2f})")
print(f"fitness evaluations: {res.fitness_evaluations}  distance computations: {res.distance_computations}")
print(f"swarm state: {res.swarm_state_floats} floats = {res.swarm_state_bytes} bytes (float64)  runtime: {res.runtime_s*1e3:.2f} ms")
# plateau: first iteration after which best fitness never improves by > 1 %
bh = np.array(res.best_history)
plateau = int(np.argmax(bh <= bh[-1] * 1.01)) + 1
print(f"best fitness {bh[-1]:.4f} m^2; within 1 % of final value from iteration {plateau}")

# --- 2. least-squares trilateration (+ Gauss-Newton refinement) ----------------------------
t0 = time.perf_counter(); p_ls = least_squares_trilateration(sc.anchors, meas); t_ls = time.perf_counter() - t0
p_gn, gn_it = gauss_newton_refine(sc.anchors, meas, p_ls)
print(f"LSQ  estimate: {np.round(p_ls, 2)}  error: {np.linalg.norm(p_ls - sc.true_position):.2f} m  ({t_ls*1e6:.0f} us)")
print(f"LSQ+GN estimate: {np.round(p_gn, 2)}  error: {np.linalg.norm(p_gn - sc.true_position):.2f} m  ({gn_it} GN iterations)")

# --- 3. reference PSO variants (same budget) -----------------------------------------------
fit2 = RangeErrorFitness(sc.anchors, meas)
res_std = StandardPSO(20, 60, seed=1).run(fit2, sc.lower, sc.upper)
fit3 = RangeErrorFitness(sc.anchors, meas)
res_am = AMCMPSO(20, 60, seed=1).run(fit3, sc.lower, sc.upper)
print(f"Standard PSO (pbest/gbest): {np.round(res_std.estimate, 2)} error {res_std.error(sc.true_position):.2f} m, {res_std.fitness_evaluations} evals, {res_std.swarm_state_floats} floats")
print(f"AMCMPSO scaffold (IN PROGRESS): {np.round(res_am.estimate, 2)} error {res_am.error(sc.true_position):.2f} m, {res_am.fitness_evaluations} evals, {res_am.swarm_state_floats} floats")

# --- 4. early stopping variant ---------------------------------------------------------------
fit4 = RangeErrorFitness(sc.anchors, meas)
res_es = SimplifiedPSO(20, 60, seed=1, clip=False, early_stop_patience=10).run(fit4, sc.lower, sc.upper)
print(f"Simplified PSO + early stop (patience 10): {np.round(res_es.estimate, 2)} error {res_es.error(sc.true_position):.2f} m after {res_es.iterations_run} iterations, {res_es.fitness_evaluations} evals")

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
    "amcmpso_in_progress": {"estimate": res_am.estimate.tolist(), "error_m": res_am.error(sc.true_position),
                            "fitness_evaluations": res_am.fitness_evaluations, "swarm_state_floats": res_am.swarm_state_floats},
    "simplified_pso_early_stop": {"estimate": res_es.estimate.tolist(), "error_m": res_es.error(sc.true_position),
                                  "iterations_run": res_es.iterations_run, "fitness_evaluations": res_es.fitness_evaluations},
}
with open(os.path.join(RES, "baseline.json"), "w") as f:
    json.dump(out, f, indent=2)
plot_convergence(res.best_history, os.path.join(FIG, "convergence.png"))
plot_scenario_3d(sc, res.estimate, res.positions_history, os.path.join(FIG, "scenario_3d.png"))
make_convergence_gif(sc, res.positions_history, res.estimate_history, os.path.join(FIG, "convergence.gif"))
print("saved results/baseline.json, figures/convergence.png, figures/scenario_3d.png, figures/convergence.gif")

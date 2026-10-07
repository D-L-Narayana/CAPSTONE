"""Multi-node iterative auto-localization study in 3D.

Distributed iterative scheme of Kulkarni et al. (2009, Section IV) transposed to a 3-D field:
nodes that get localized in a round serve as references in the next one.

    python scripts/run_network.py --nodes 50 --beacons 10 --range 25 --field 100 100 30 \
        --noise gaussian:0.5 --localizer std --particles 20 --iterations 60 --rounds 10 --seed 1 \
        [--min-refs 4 --max-refs 6 --allow-coplanar --out DIR --figures DIR --no-plots --quiet]

Noise specs: gaussian:SIGMA | percent:P | nlos:p[:bias[:sigma]] | shadowing:dB  (the last three need
``pso3d.noise``).  Writes <out>/network.csv (one row per round), <out>/network_nodes.csv (one row per
node), <out>/network.md (summary) and, unless --no-plots, <figures>/network_rounds.png and
<figures>/network_map.png.  Exit code 0 on success, 2 on a bad argument.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pso3d.network import (NetworkScenario, iterative_localize, lsq_gn_localizer, pso_localizer,  # noqa: E402
                           summary_table, warm_started_localizer)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCALIZERS = ("std", "lsq_gn", "amcmpso", "std_ws", "simplified")
NOISE_GRAMMAR = "gaussian:SIGMA | percent:P | nlos:p[:bias[:sigma]] | shadowing:dB"
NOISE_KINDS = ("gaussian", "percent", "nlos", "shadowing")


def parse_noise_spec(spec: str):
    """Turn a noise spec string into a noise model (object with ``.sample``) or a Gaussian sigma.

    ``gaussian:SIGMA`` (or a bare number) is handled here and yields a float sigma, which the
    network simulator turns into zero-mean Gaussian ranging noise.  The other kinds are delegated
    to ``pso3d.noise.from_spec`` when that module is available.  Unknown kinds raise ValueError.
    """
    spec = str(spec).strip()
    if not spec:
        raise ValueError(f"empty noise spec; expected {NOISE_GRAMMAR}")
    kind, sep, rest = spec.partition(":")
    kind = kind.strip().lower()
    if not sep and _is_number(kind):
        kind, rest = "gaussian", kind
    if kind not in NOISE_KINDS:
        raise ValueError(f"unknown noise spec {spec!r}; expected {NOISE_GRAMMAR}")
    if kind == "gaussian":
        try:
            sigma = float(rest)
        except ValueError:
            raise ValueError(f"bad Gaussian noise spec {spec!r}; expected gaussian:SIGMA") from None
        if sigma < 0:
            raise ValueError("Gaussian sigma must be non-negative")
        return sigma
    try:
        from pso3d.noise import from_spec  # optional module
    except ImportError:
        from_spec = None
    if from_spec is None:
        raise ValueError(f"noise spec {spec!r} needs the pso3d.noise module, which this checkout does not provide")
    model = from_spec(spec)
    if not hasattr(model, "sample"):
        raise ValueError(f"pso3d.noise.from_spec returned no noise model for {spec!r}")
    return model


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def make_localizer(name: str, particles: int, iterations: int):
    """Localizer factory for the --localizer choices."""
    if name == "lsq_gn":
        return lsq_gn_localizer()
    swarm = dict(n_particles=int(particles), n_iterations=int(iterations), clip=True)
    if name == "std":
        from pso3d.standard_pso import StandardPSO
        return pso_localizer(StandardPSO, **swarm)
    if name == "std_ws":
        from pso3d.standard_pso import StandardPSO
        return warm_started_localizer(StandardPSO, **swarm)
    if name == "simplified":
        from pso3d.pso import SimplifiedPSO
        return pso_localizer(SimplifiedPSO, **swarm)
    if name == "amcmpso":
        from pso3d.amcmpso import AMCMPSO
        return pso_localizer(AMCMPSO, **swarm)
    raise ValueError(f"unknown localizer {name!r}; choose from {', '.join(LOCALIZERS)}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Multi-node iterative auto-localization in a 3-D field "
                                             "(settled nodes become references for the next round).")
    ap.add_argument("--nodes", type=int, default=50, help="number of unknown nodes N (default 50)")
    ap.add_argument("--beacons", type=int, default=10, help="number of beacons M (default 10)")
    ap.add_argument("--range", dest="radio_range", type=float, default=25.0, help="radio range r in m (default 25)")
    ap.add_argument("--field", type=float, nargs=3, default=(100.0, 100.0, 30.0), metavar=("X", "Y", "Z"),
                    help="field size in m (default 100 100 30)")
    ap.add_argument("--noise", default="gaussian:0.5", help=f"ranging noise spec: {NOISE_GRAMMAR} (default gaussian:0.5)")
    ap.add_argument("--localizer", choices=LOCALIZERS, default="std",
                    help="per-node estimator: std (standard PSO), lsq_gn (closed form), amcmpso (interpretation), "
                         "std_ws (standard PSO warm-started from LSQ+GN), simplified (Review-1 PSO); default std")
    ap.add_argument("--particles", type=int, default=20, help="swarm size for PSO localizers (default 20)")
    ap.add_argument("--iterations", type=int, default=60, help="iterations for PSO localizers (default 60)")
    ap.add_argument("--rounds", type=int, default=10, help="maximum number of rounds (default 10)")
    ap.add_argument("--min-refs", type=int, default=4, help="references needed to be localizable (default 4)")
    ap.add_argument("--max-refs", type=int, default=6, help="nearest references used at most (default 6)")
    ap.add_argument("--allow-coplanar", action="store_true", help="do not require a non-coplanar reference set")
    ap.add_argument("--seed", type=int, default=1, help="deployment seed; noise/PSO seeds derive from seed + 1 (default 1)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"), help="directory for network.csv / network.md")
    ap.add_argument("--figures", default=os.path.join(ROOT, "figures"), help="directory for the PNG figures")
    ap.add_argument("--no-plots", action="store_true", help="skip the figures")
    ap.add_argument("--quiet", action="store_true", help="do not print the summary")
    return ap


def write_rounds_csv(result, path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["round", "newly_settled", "settled_total", "refs_available_mean", "evaluations", "coverage_fraction"])
        n = max(result.n_nodes, 1)
        for r in result.rounds:
            w.writerow([r.round, r.newly_settled, r.settled_total, f"{r.refs_available_mean:.3f}", r.evaluations,
                        f"{r.settled_total / n:.4f}"])


def write_nodes_csv(result, nodes, path: str) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["node", "x", "y", "z", "settled", "round", "refs_used", "x_hat", "y_hat", "z_hat", "error_m"])
        for i, p in enumerate(np.asarray(nodes, float)):
            s = bool(result.settled[i])
            est = result.estimates[i]
            w.writerow([i, f"{p[0]:.3f}", f"{p[1]:.3f}", f"{p[2]:.3f}", int(s),
                        int(result.settled_round[i]) if result.settled_round is not None else "",
                        int(result.n_refs_used[i]) if result.n_refs_used is not None else "",
                        f"{est[0]:.3f}" if s else "", f"{est[1]:.3f}" if s else "", f"{est[2]:.3f}" if s else "",
                        f"{result.errors[i]:.3f}" if s and result.errors is not None else ""])


def render_markdown(result, args, localizer) -> str:
    swarm = "" if args.localizer == "lsq_gn" else f" ({args.particles} particles x {args.iterations} iterations)"
    warm = ""
    if args.localizer == "std_ws":
        warm = (" - warm start from LSQ+GN" if getattr(localizer, "warm_start_supported", False)
                else " - the installed optimiser does not accept x0, so the swarm ran without a warm start")
    lines = ["# Multi-node iterative auto-localization (3D)", "",
             "Iterative scheme (Kulkarni et al. 2009, Section IV, in 3-D): every unknown node that has at least "
             f"{args.min_refs} references (beacons or already-settled nodes) within radio range runs the single-node "
             f"localizer on its noisy ranges to the {args.max_refs} nearest references; settled nodes serve as "
             "references (through their *estimated* positions) in the next round, until no new node settles.", "",
             "| setting | value |", "|---|---|",
             f"| localizer | `{args.localizer}`{swarm}{warm} |",
             f"| noise | `{args.noise}` |",
             f"| field [m] | {args.field[0]:g} x {args.field[1]:g} x {args.field[2]:g} |",
             f"| nodes / beacons / radio range | {args.nodes} / {args.beacons} / {args.radio_range:g} m |",
             f"| min / max references | {args.min_refs} / {args.max_refs}"
             f"{' (coplanar reference sets allowed)' if args.allow_coplanar else ' (non-coplanar reference sets only)'} |",
             f"| max rounds | {args.rounds} |",
             f"| seed | {args.seed} |", "",
             "## Results", "", summary_table(result), "",
             "Metrics: N_NL = nodes not localized, E_l = mean squared error over the localized nodes (eq. 4 of the paper "
             "with the z term), RMSE = sqrt(E_l), coverage = settled / N. Ranges are simulated from the true positions; "
             "errors of settled nodes propagate to the nodes that use them as references, so later rounds are usually "
             "less accurate than the first one.", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        noise = parse_noise_spec(args.noise)
        localizer = make_localizer(args.localizer, args.particles, args.iterations)
        scenario = NetworkScenario(field_size=tuple(args.field), n_nodes=args.nodes, n_beacons=args.beacons,
                                   radio_range=args.radio_range, seed=args.seed)
        result = iterative_localize(scenario, localizer, noise=noise, max_rounds=args.rounds, min_refs=args.min_refs,
                                    max_refs=args.max_refs, require_non_coplanar=not args.allow_coplanar)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    _, nodes = scenario.generate()
    os.makedirs(args.out, exist_ok=True)
    write_rounds_csv(result, os.path.join(args.out, "network.csv"))
    write_nodes_csv(result, nodes, os.path.join(args.out, "network_nodes.csv"))
    text = render_markdown(result, args, localizer)
    with open(os.path.join(args.out, "network.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    written = ["network.csv", "network_nodes.csv", "network.md"]
    if not args.no_plots:
        os.makedirs(args.figures, exist_ok=True)
        from pso3d.network_plots import plot_network_map, plot_rounds  # matplotlib only when needed
        plot_rounds(result, os.path.join(args.figures, "network_rounds.png"))
        plot_network_map(scenario, result, os.path.join(args.figures, "network_map.png"))
        written += ["network_rounds.png", "network_map.png"]
    if not args.quiet:
        print(text)
        print(f"saved {', '.join(written)} to {args.out}" + ("" if args.no_plots else f" / {args.figures}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Changelog

All notable changes to the `pso3d` package, the experiment scripts and the web simulator.

## 0.3.0 — Phase-2 toolkit (unreleased)

The Review-1 baseline, the Monte-Carlo numbers for the four original methods and the static web
deployment are unchanged; everything below is additive.

### Core optimiser (`pso3d`)
- `pso3d.stopping`: `Patience`, `Tolerance`, `MaxEvaluations`, `TargetFitness`, `AnyOf` stopping rules
  usable with every PSO class (`stopping=`); `early_stop_patience` is kept as a deprecated alias.
- Warm start (`x0=`) for `SimplifiedPSO`, `StandardPSO` and `AMCMPSO`; the first particle is placed
  at the clipped start point without consuming random numbers, so seeded runs stay reproducible.
- `PSOResult` gains `stopped_by`, `warm_start` and `method`.
- `RangeErrorFitness` accepts Python lists/tuples (previously raised `AttributeError`) and has `evaluate_one`.
- `AMCMPSO`: configurable linear/cosine coefficient schedules (`coefficient_schedule`), `centre_of_mass`,
  diagnostics (`diversity_history`, `improvement_rate`). It remains an *interpretation* of the base
  paper's idea; its numbers are not a reproduction of Alhasan et al. (2023).

### New modules
- `pso3d.noise`: `FixedOffsets`, `Gaussian`, `UniformPercent`, `LogNormalShadowing` (RSSI round trip),
  `NLOS`; `measure()` and `from_spec("gaussian:0.5" | "percent:2" | "nlos:0.2" | "shadowing:4")`.
- `pso3d.robust`: `HuberRangeFitness`, `WeightedRangeFitness`.
- `pso3d.geometry`: non-coplanarity, anchor rank, GDOP, CRLB (`crlb`), mirror point / flip-ambiguity
  risk, CRLB coverage grid, `scenario_report`.
- `pso3d.trilateration`: weighted least squares, `gauss_newton` / `levenberg_marquardt` returning
  `GNResult`, `warm_start_estimate`; `pso3d.centroid`: `centroid`, `weighted_centroid`.
- `pso3d.network`: multi-node iterative auto-localization in 3D (settled nodes become references),
  with N_NL / E_l / RMSE / coverage metrics and per-round statistics; `pso3d.network_plots`.
- `pso3d.experiments`: method registry (`METHODS`) with `amcmpso`, `std_tol`, `std_ws`, `amcmpso_ws`,
  CRLB reference column, pluggable noise models; legacy rows are byte-for-byte protected by tests.

### Command line
- `python -m pso3d baseline|experiments|network|geometry|version`; all scripts expose `main(argv)` and
  accept `--out` / `--figures`; `scripts/run_network.py` is new; `scripts/check.sh` runs every check.
- Packaging via `pyproject.toml` (`pip install -e .[plots,dev]`); GitHub Actions workflow.

### Web simulator (`web/`, still static and build-free)
- AMCMPSO variant, stopping rules, warm start, noise models (Gaussian / uniform % / NLOS) in the browser.
- New KPIs: Gauss-Newton error, GDOP, CRLB bound, geometry badge, flip-ambiguity risk, stopped-by.
- Inline validation instead of `alert()`, labelled controls, live regions, keyboard shortcuts,
  shareable links (`#s=…`), CSV / JSON export, offline fallback when Plotly cannot load, dark colour scheme.
- `vercel.json` sends a Content-Security-Policy and other security headers; `tests/e2e/` contains a
  Node/Playwright smoke test that enforces those headers while exercising the new workflows.
- `web/geometry.js` mirrors `pso3d.geometry`; `tests/test_parity.py` checks Python/JavaScript agreement.

### Tests
- From 3 tests to 318 Python tests (`python -m pytest -q tests`) and 75 JavaScript tests (`node --test tests/js/`),
  including characterisation tests that pin the Review-1 numbers and the legacy Monte-Carlo results, a Python/JavaScript
  parity test on shared fixtures, and a browser gate (`tests/e2e/web_smoke.js`) that runs the simulator workflows in
  headless Chromium under the production Content-Security-Policy.

### Source distribution
- `MANIFEST.in` makes the source archive (`sdist`) complete: it now ships `scripts/`, `tests/` (including `conftest.py`,
  fixtures, the Node and browser tests), `web/`, `docs/`, `results/`, the CI workflow, `CONTRIBUTING.md`, `CHANGELOG.md`
  and `requirements.txt`, so the documented checkout commands and the complete test suite run from the extracted
  archive. Generated figures and the root PDF/DOCX documents are deliberately excluded; the wheel is unchanged
  (package code only; `pso3d baseline|experiments|network` remain checkout-only commands and say so).
- `scripts/check_sdist.py` builds the sdist and wheel offline, extracts them, verifies their contents and the wheel
  imports, and runs the complete test suite inside the extracted archive; `tests/test_sdist.py` guards the archive
  contents, and both run in `scripts/check.sh` / CI.
- The `dev` extra and `requirements.txt` declare `setuptools>=77` (the `[build-system]` floor, kept equal by a test)
  because the archive check and `tests/test_sdist.py` build with `setuptools.build_meta` in the test interpreter, which a
  plain `pip install -e .` does not provide; `check_sdist.py` fails early with guidance when the backend is missing.

### Fixes
- `scripts/run_experiments.py` and `scripts/run_network.py` write their CSV files with LF line endings
  (`csv.writer(..., lineterminator="\n")`); the committed CSV artefacts were normalised accordingly without changing
  any cell value.

### Regenerated artefacts
- `results/baseline.json` (identical numbers; the AMCMPSO block is now keyed `amcmpso_interpretation`),
  `results/sweeps.csv` / `results/results.md` (legacy rows identical to three decimals, runtime column aside),
  `results/extended/` (eight methods with the CRLB column), `results/network*/` and the corresponding figures.

## 0.2.0 — Phase 1 (Review 1)
- `pso3d` package: simplified PSO (bit-exact Review-1 reproduction), standard PSO, AMCMPSO scaffold,
  least-squares trilateration + Gauss-Newton, Monte-Carlo study, figures, static web simulator on Vercel.

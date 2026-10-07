# Contributing

Thanks for helping with the capstone toolkit. This page explains how to set up a development environment,
which checks must pass, which numbers are frozen, and the rules for the web simulator and for claims about
the base-paper algorithm.

## Development setup

```bash
git clone https://github.com/D-L-Narayana/CAPSTONE.git && cd CAPSTONE
python -m venv .venv && source .venv/bin/activate        # optional
pip install -e ".[plots,dev]"                              # numpy + matplotlib/pillow (figures) + pytest
```

Python ≥ 3.10 and Node 20 (for the browser scripts and their tests). The `pso3d` core package depends on
NumPy only; matplotlib and Pillow are imported solely in `pso3d/plots.py`, `pso3d/network_plots.py` and the
scripts (Agg backend). Do not add runtime dependencies to the core.

Command-line entry points (`pso3d …` or `python -m pso3d …`):

```bash
pso3d baseline --out results --figures figures [--no-gif] [--quiet]   # Review-1 baseline + comparisons
pso3d experiments --trials 200 --out results --figures figures        # Monte-Carlo study (scripts/run_experiments.py)
pso3d network --nodes 50 --beacons 10 --rounds 10                     # multi-node auto-localization (scripts/run_network.py)
pso3d geometry [--scenario file.json] [--sigma 0.5] [--json]          # rank / non-coplanarity / GDOP / CRLB
pso3d version
```

The `baseline`, `experiments` and `network` commands run the scripts in `scripts/`, so they need a repository
checkout (an editable install); they forward every option to the script's `main(argv)`.

## Running the checks

```bash
bash scripts/check.sh            # everything CI runs, from any directory
```

`check.sh` runs, in order: `python -m compileall pso3d scripts tests`, `node --check` on every `web/*.js`,
`pytest tests --ignore=tests/e2e`, `node --test tests/js/`, and `python scripts/check_sdist.py` (the source-archive
check described below). CI (`.github/workflows/ci.yml`) runs the same script on Ubuntu with Python 3.11 and 3.12
and Node 20 and attaches the source-archive report to the workflow run. Individual pieces:

```bash
python -m pytest -q tests                         # Python unit tests (tests/js and tests/e2e are not collected)
python -m pytest -q tests/test_pso.py             # the frozen baseline gate alone
node --test tests/js/                             # browser algorithms, share codec, exporters
node --test tests/e2e/serve.test.js               # static server + header rules (no browser needed)
python scripts/check_sdist.py                     # sdist + wheel rebuilt, verified and tested in isolation
node tests/e2e/web_smoke.js                       # browser gate, see below
```

Tests must stay fast (the whole Python suite well under two minutes, each file within seconds): Monte-Carlo
inside tests uses ≤ 20 trials, no GIFs, Agg backend, and files are written only to `tmp_path`. New behaviour
comes with tests written first; do not add `skip`/`xfail` markers to make a gate pass.

## Source distribution

The project is packaged with setuptools (`pyproject.toml`); `MANIFEST.in` defines what the **source archive**
(`pso3d-<version>.tar.gz`) contains. The archive is self-sufficient: the complete test suite and every documented
checkout command run from the extracted archive, without the git repository.

| Ships in the source archive | Deliberately excluded |
|---|---|
| `pso3d/` (the package), `pyproject.toml`, `MANIFEST.in`, `LICENSE`, `README.md`, `CHANGELOG.md`, `CONTRIBUTING.md`, `requirements.txt` | `figures/` — generated binaries; regenerate them with `pso3d baseline`, `pso3d experiments` and `pso3d network` |
| `tests/` including `conftest.py`, `fixtures/*.json`, the Node tests in `tests/js/` and the browser gate in `tests/e2e/` | the PDF and Word documents at the repository root (reference papers and personal documents) — never shipped |
| `scripts/` (`run_baseline.py`, `run_experiments.py`, `run_network.py`, `check.sh`, `check_sdist.py`) | build output, caches, byte-code, logs, `node_modules/`, git metadata |
| `web/` (the simulator with its `vercel.json` headers), `docs/`, `results/` (frozen reference numbers), `.github/workflows/ci.yml` | |

The **wheel** contains only the `pso3d` package and its metadata (including `licenses/LICENSE`). From a wheel,
`python -m pso3d version` and `python -m pso3d geometry …` work; `baseline`, `experiments` and `network` need the
`scripts/` directory next to the package, so they exit with code 2 and a message pointing to a repository
checkout (or the source archive). This limitation is deliberate and is asserted by the check below.

The archive check needs setuptools ≥ 77 in the active interpreter — both `pip install -e ".[plots,dev]"` (through
the `dev` extra) and `pip install -r requirements.txt` provide it (a plain `pip install -e .` uses setuptools only
inside pip's isolated build environment, and some Python 3.12+ installations ship none); the `build` module is
still not required. Build and verify both artefacts offline:

```bash
python scripts/check_sdist.py                      # report and artefacts in build/sdist-check/
python scripts/check_sdist.py --skip-node          # Python only; prints that the Node suites were skipped
python -c "import setuptools.build_meta as b; b.build_sdist('dist'); b.build_wheel('dist')"   # artefacts only
```

`check_sdist.py` copies the checkout to a temporary directory (without `.git`, build output and caches), builds
the sdist and the wheel there with `setuptools.build_meta`, extracts both, checks the required and forbidden
member lists and the portability of the member paths, imports the package from the extracted wheel only
(`version` and `geometry --help` exit 0, `baseline --help` exits 2 with the guidance), and finally runs the
complete, unmodified suite inside the extracted source archive with `PYTHONPATH` unset:
`python -m pytest tests --ignore=tests/e2e`, `node --test tests/js/` and `node --test tests/e2e/serve.test.js`.
It writes `report.json` (hashes, member lists, exit codes, test counts) and exits non-zero on any failure. It is
step 5 of `scripts/check.sh` and a CI step; `tests/test_sdist.py` keeps the member lists and the wheel contents
under test. When you add a file that tests or documented commands need, add it to `MANIFEST.in` and to the
required list in `scripts/check_sdist.py` / `tests/test_sdist.py`; never add the documents or the figures.
No script uploads anything to a package registry.

## Frozen regression gates

These numbers are the project's audit trail. Changing them is not a refactor; it is a different experiment and
needs its own documentation.

1. **Review-1 baseline** (`tests/test_pso.py`, bit-exact). Simplified PSO, 20 particles × 60 iterations,
   `w = 0.7`, `c = 1.4`, seed 1 (`numpy.random.default_rng(1)`), no clipping, on the slide-16 scenario:
   estimate **(37.43, 11.90, 9.29) m**, error **0.90 m**, **1 200** fitness evaluations, **4 800** distance
   computations, **120** swarm-state floats. The order of random draws in `SimplifiedPSO.run` must not change;
   new features are opt-in parameters with defaults that reproduce the old path.
   `python -m pso3d baseline --out build/b --figures build/b --no-gif` must reproduce the `simplified_pso`,
   `least_squares`, `least_squares_gauss_newton` and `standard_pso` blocks of `results/baseline.json`
   (`tests/test_cli.py` checks the estimates to 1e-9).
2. **Legacy Monte-Carlo rows** (`results/sweeps.csv`). For the methods `pso`, `std`, `lsq` and `lsq_gn`
   the rows of `python scripts/run_experiments.py --trials 200` must stay identical to three decimals
   (the runtime column excluded). This pins the RNG consumption order in `pso3d.experiments.monte_carlo`:
   anchors → true position → noise → per-trial PSO seed, with `seed = 123`. New methods get new keys;
   they never reorder the draws of the legacy ones.
3. **Closed-form pins.** `least_squares_trilateration` gives (37.13, 11.44, 11.65) m and
   `gauss_newton_refine` (37.42, 11.88, 9.36) m after 10 iterations on the baseline scenario.

Regenerate `results/` and `figures/` only through the CLI (`pso3d baseline`, `pso3d experiments`,
`pso3d network`) and commit the regenerated files together with the code that changed them. Never hand-edit them.

## Web simulator constraints

`web/` is a static, build-free site deployed on Vercel with project root `web/`: plain browser scripts,
Plotly 2.35.2 from `https://cdn.plot.ly`, no bundler, no TypeScript, no `package.json` in `web/`.
Every script ends with a `module.exports` shim so Node can `require` it for tests; `share.js`, `export.js`,
`pso.js` and `geometry.js` must not touch the DOM at load time.

Production responses carry the security headers defined in `web/vercel.json` — an **enforced**
Content-Security-Policy:

```
default-src 'self'; script-src 'self' https://cdn.plot.ly; style-src 'self' 'unsafe-inline';
img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self';
form-action 'self'; frame-ancestors 'none'
```

plus `X-Content-Type-Options: nosniff`, `Referrer-Policy: strict-origin-when-cross-origin`,
`Permissions-Policy: camera=(), microphone=(), geolocation=()`, `X-Frame-Options: DENY`, and `Cache-Control`
(`no-cache` for the document, one hour for `*.js`/`*.css`). Consequences for every change under `web/`:

* no inline `<script>` blocks, no inline event-handler attributes (`onclick="…"`), no `javascript:` URLs,
  no `eval` / `new Function` — wire behaviour with `addEventListener` in `app.js`;
* no remote fonts or images, no `fetch`/XHR to other origins; the only external resource is the Plotly script;
* inline `style="…"` attributes and dynamically created `<style>` elements are allowed;
* downloads go through Blob URLs on an `<a download>`; sharing uses `location.hash` (`#s=v1.…`);
* validation messages go to the inline `role="alert"` region, never `alert()`;
* the page must keep working without Plotly (offline notice, KPIs still update) and in the dark colour scheme.

Preview locally with the production headers: `node tests/e2e/serve.js --port 8000` (or any static server such as
`python -m http.server 8000 --directory web`, which serves the files without the headers).

## Browser gate

`node tests/e2e/web_smoke.js` serves `web/` with the headers above and drives headless Chromium through the
real workflows (run to end, AMCMPSO with an evaluation budget, warm start, CSV/JSON downloads, share link,
validation, offline fallback, accessibility smoke, screenshots) while collecting `securitypolicyviolation`
events, console CSP refusals, console errors and requests to unexpected origins. It exits 0 only when every
check passes and all collectors are empty; 3 when Playwright or its Chromium is unavailable (reported, never a
silent pass). It needs `require('playwright')` to resolve and the Chromium build from
`npx playwright install chromium`, which is why it is **not** part of `scripts/check.sh` or CI — run it
before a deployment and attach `build/e2e/summary.json` to the pull request. Details: `tests/e2e/README.md`.

## Honesty rules for AMCMPSO claims

The base paper is Alhasan et al., 2023 (doi:10.1016/j.jksuci.2023.101782). Its text is not part of this
repository, and `pso3d/amcmpso.py` (and the `amcmpso` variant in `web/pso.js`) is an **interpretation** of the
described adaptive mean / centre-of-mass PSO, built from the description available to the team.

* Call it "AMCMPSO (interpretation of the base paper)"; never a "reproduction" or "implementation of the paper".
* Do not quote the paper's reported numbers (improvement rate, centimetre errors, coverage) as results of this
  code, and do not present this code's numbers as the paper's.
* Every number in `README.md`, `results/` and `docs/` must come from a command in this repository that anyone
  can re-run; state the command and the seed.
* Keep the existing honesty notes in the README (the mulberry32 note, "does not generalise", the AMCMPSO status)
  when editing nearby text.

The same applies to other references: cite, do not paraphrase as if measured here.

## Pull request checklist

- [ ] `bash scripts/check.sh` passes locally (and the browser gate when `web/` changed).
- [ ] Frozen gates untouched: `tests/test_pso.py` unchanged, legacy `sweeps.csv` rows identical, baseline numbers reproduced.
- [ ] New behaviour has tests; no skipped or weakened assertions.
- [ ] `results/` and `figures/` regenerated through the CLI if the numbers legitimately changed, with the reason in the PR.
- [ ] Documentation updated (`README.md`, `docs/`), AMCMPSO wording as above.
- [ ] No machine-specific paths, credentials or generated-by banners in committed files; `web/` stays static and CSP-compliant.
- [ ] New files that tests or documented commands need are listed in `MANIFEST.in` (`python scripts/check_sdist.py` passes).

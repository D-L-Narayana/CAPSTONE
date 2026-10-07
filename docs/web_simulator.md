# Web simulator — user guide

The simulator in `web/` is a static, build-free page (plain browser scripts plus Plotly 2.35.2 from the CDN).
It runs the same range-based 3D localization problem as the Python package: minimise
f(p) = Σ_j (‖p − a_j‖ − d̂_j)² over the field with a particle swarm, and compares the result with the
closed-form least-squares (LSQ) and LSQ + Gauss-Newton estimates and with the geometric limits of the anchor layout.

Run it locally from the repository root with any static file server, e.g.
`python3 -m http.server 8000 --directory web`, then open `http://localhost:8000/`.
Production is a Vercel static deployment with project root `web/` (no build step).

## Controls

### Scenario (left panel)

| Control | Meaning |
|---|---|
| Field X / Y / Z | Search box `[0, X] × [0, Y] × [0, Z]` in metres. Anchors and the true node must lie inside it. |
| Anchor table | Anchor coordinates `a_j` and the per-anchor ranging offset `n_j` (used by the *fixed* noise model). `×` removes a row, **+ anchor** appends one at a seeded random position. At least four non-coplanar anchors are required. |
| Noise model | How the measured ranges `d̂_j = ‖p − a_j‖ + noise_j` are generated (see below). |
| True node p | The position to be localized; **random node (seeded)** draws one inside the field. |

Noise models (all random draws come from `mulberry32(seed ^ 0x9E3779B9)`, so a given seed always yields the same ranges):

| Model | Fields | noise_j |
|---|---|---|
| fixed per-anchor `n_j` (default) | table column `n_j` | `n_j` exactly (the Review-1 scenario uses 0.4, −0.3, 0.5, −0.4 m) |
| Gaussian σ | σ | `σ · N(0, 1)` |
| uniform ± % of range | `P_n` | `d_j · (P_n / 100) · U(−1, 1)` |
| NLOS | σ (line-of-sight), NLOS probability, bias mean, bias σ | `σ · N(0,1)` plus, with the given probability, a positive bias `abs(mean + biasσ · N(0,1))` |

### Swarm

| Control | Meaning |
|---|---|
| Variant | *Simplified PSO* (best-of-iteration attractor, no particle memory — the Review-1 algorithm), *Standard PSO* (pbest / gbest, synchronous updates) or *AMCMPSO (interpretation of the base paper)*. |
| AMCMPSO coefficient schedule | `linear` or `cosine` interpolation of w, c1, c2 between their start and end values (shown only for AMCMPSO). The w / c2 / c1 fields are ignored by this variant. |
| Seed | Seed of the swarm generator *and* of the noise, random-node and +anchor streams. |
| Particles N, Iterations T, Speed | Swarm size (≥ 2), iteration budget (≥ 1) and playback delay per iteration. |
| Inertia w, c2 social, c1 cognitive | Velocity coefficients (c1 only for the standard variant). |
| Stopping | *none* (always T iterations), *patience* (stop after `patience` iterations without best-ever improvement), *tolerance* (stop when the best-ever fitness improved by ≤ tol · f_old over `window` iterations), *budget* (stop once the fitness evaluations reach the budget). |
| clip particles to the field | Clamp particle positions to the field after every update. |
| warm start from LSQ + Gauss-Newton | Replace particle 0 of the initial swarm by `x₀ = clip(GN(LSQ))`; `x₀` is drawn as an open square in the 3D view. |

**AMCMPSO status.** The AMCMPSO variant is an *interpretation* of Alhasan et al. 2023 (adaptive mean / centre-of-mass PSO),
not a reproduction of the paper. Its numbers must not be quoted as the paper's results.

### Playback

**Reset** rebuilds the scenario and the swarm from the controls (every change of a control does this automatically),
**Step** runs one iteration, **Play / Pause** animates at the chosen speed, **Run to end** finishes the run immediately.
When the configuration is invalid these buttons are disabled and the reasons are listed in the red alert box above the plot
(fewer than four anchors, coplanar anchors, anchors or node outside the field, N < 2, T < 1, non-numeric field sizes).

## KPIs (updated after every iteration)

| KPI | Definition |
|---|---|
| estimate p̂, error ‖p̂ − p‖ | Current best position and its Euclidean distance to the true node. |
| best fitness | f(p̂) in m². |
| fitness evaluations, distance computations | Counters of the fitness object (one evaluation = M distance computations for M anchors). |
| swarm memory | Floats held by the optimiser (N·6 simplified, N·9 + 3 standard, N·9 + 6 AMCMPSO). |
| iterations run · stopped by | `t / T`; the label names the stopping rule when a rule ended the run early (also shown in the 3D title). |
| least-squares error | Error of the linearised LSQ trilateration (open circle). |
| GN error | Error of the Gauss-Newton refinement started at the LSQ solution (open diamond); the label shows the iterations used. |
| warm start x₀ error | Error of the warm-start point when the checkbox is on. |
| GDOP @ p | `GDOP = sqrt(trace((JᵀJ)⁻¹))` with `J` the M×3 Jacobian of the ranges at p (rows = unit vectors from each anchor to p); ∞ when the geometry is singular. Highlighted above 10. |
| CRLB bound @ p (σ) | `cov = σ² (JᵀJ)⁻¹`, bound = `sqrt(trace(cov))` — the smallest RMSE any unbiased estimator can reach with range noise σ. σ is the model σ for Gaussian/NLOS, the RMS of the `n_j` offsets for *fixed* and the RMS uniform σ (`d·P_n/100/√3`) for the ± % model. |
| geometry | Badge *non-coplanar ✓* / *coplanar ⚠* (tetrahedron-volume test on the anchors). |
| flip-ambiguity risk @ p | Value in [0, 1] from the ratio of the range-error fitness at p and at its mirror image p′ (reflection of p across the anchors' least-squares plane); 1 means both fit the ranges equally well. Highlighted above 0.5; the mirror image is drawn when the risk is ≥ 0.05. |
| measured ranges d̂ | The noisy ranges used by the fitness function. |

A KPI shows "–" when the corresponding helper is not available (for example when `geometry.js` did not load).

## Share links

**Copy link** writes `…/#s=v1.<base64url JSON>` to the clipboard (when clipboard access is refused the link is shown in a
read-only field, already selected). Opening such a link restores every control before the first run, so the recipient sees
the same scenario, swarm configuration and noise. The encoded object is

```
{v:1, field:[x,y,z], anchors:[[x,y,z,n_j], …], p:[x,y,z], variant, seed, N, T, w, c, c1, clip,
 stop:{type, patience, tol, window, budget}, warm,
 noise:{model, sigma, pn, nlosP, nlosBias, nlosSigma}, amcmpso?:{schedule}}
```

Links are validated before use: the version prefix must be `v1.`, the payload must be valid JSON of an object, anchors must be
4–64 rows of four finite numbers, `p` and `field` three finite numbers (field sizes positive), enumerations must be known,
numbers finite, booleans boolean. Anything else is ignored and the page starts with its defaults. Missing optional keys are
completed with the defaults listed above; unknown keys are dropped.

## Exports

* **Export CSV** → `pso3d_run.csv` with the header `iteration,best_fitness,est_x,est_y,est_z,error_m` and one row per
  recorded iteration (6 significant digits; `error_m` is the distance of the iteration's best estimate to the true node).
  The simplified variant records iterations 1…t; the standard and AMCMPSO variants also record the evaluated initial swarm as iteration 0.
* **Export JSON** → `pso3d_run.json` with `scenario` (field, anchors, noise, noise model, p, measured), `config` (the swarm
  configuration without functions), `result` (estimate, error_m, best_fitness, iterations_run, stopped_by, fitness_evaluations,
  distance_computations, swarm_state_floats), `history` (best_fitness[], estimates[]) and `extras` (LSQ, Gauss-Newton,
  warm start, geometry diagnostics, true ranges and the share-link payload of the run).

Downloads are created client-side from a Blob; nothing is uploaded.

## Keyboard

`Space` play / pause · `→` one step · `R` reset. Shortcuts are ignored while an input, select, text area or button has focus,
so typing in a field never triggers them.

## Offline fallback

If Plotly cannot be loaded (offline, blocked CDN) a notice appears instead of the plots and the page keeps working:
the KPIs update as usual and a text line shows the iteration counter and the last ten best-fitness values.

## Appearance and accessibility

* Light and dark colour schemes follow the operating-system preference (`prefers-color-scheme`); the light tokens are unchanged.
  The plot markers adapt to the scheme as well: in the dark scheme the anchor squares with their `a_j` labels and the
  estimate cross switch to light tones so they stay readable on the dark scene background (the least-squares /
  Gauss-Newton open markers and the mirror-image marker keep their colours, which are readable in both schemes).
* Every input has a label; anchor-table cells carry `aria-label="anchor N x|y|z|noise"`; validation messages live in a
  `role="alert"` region, the KPI grid is an `aria-live="polite"` region (paused while the animation runs) and all
  focusable controls have a visible focus outline.
* Random numbers come from a seeded JavaScript generator (mulberry32), so a run is repeatable in the browser but does not
  reproduce the NumPy `default_rng(1)` numbers of the Python baseline bit-for-bit.

## For contributors

The production site ships a strict Content-Security-Policy (`script-src 'self' https://cdn.plot.ly`, no `unsafe-inline`
for scripts, `connect-src 'self'`). Keep the page compliant:

* no inline `<script>` blocks, no inline `on*=` handler attributes, no `javascript:` URLs, no `eval` / `new Function`;
  wire behaviour with `addEventListener` in `app.js`;
* no remote fonts or images, no fetches to other origins; downloads go through Blob URLs on an `<a download>`;
* inline `style=` attributes and dynamically created `<style>` elements are allowed;
* keep the script order at the end of `index.html`: `pso.js → geometry.js → share.js → export.js → app.js`
  (Plotly stays in `<head>`); every script ends with a `module.exports` shim so Node can `require` it;
* `share.js` and `export.js` are pure (no DOM access at load time) and are tested with
  `node --test tests/js/share.test.js tests/js/export.test.js`; run `node --check web/*.js` before committing;
* `app.js` must never throw when an optional helper is missing — use `typeof name === 'function'` checks and show "–".

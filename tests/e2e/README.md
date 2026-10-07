# Browser gate for the web simulator (`tests/e2e/`)

The Python and Node unit tests cover the algorithms and the pure helpers. This directory checks the
*deployed artefact*: the static page in `web/` served with the **production security headers** from
`web/vercel.json` (an enforced Content-Security-Policy, not a report-only one) and driven by a real
headless Chromium.

| File | Purpose |
|---|---|
| `serve.js` | Dependency-free static server that mimics the Vercel deployment: `cleanUrls`, correct `Content-Type`, every `headers` rule of `web/vercel.json` (`/(.*)`, `/(.*)\\.(js\|css)`, `/`). Exports `createServer(root, vercelJsonPath)`. |
| `serve.test.js` | `node --test` unit tests for the server: routes, 404/traversal, Content-Types, CSP and the other headers on every 200 response, `Cache-Control` differing between the document and the assets, the source-pattern matcher. Runs without a browser. |
| `web_smoke.js` | The browser gate (Playwright + Chromium). |

## Running

```bash
node --test tests/e2e/serve.test.js                      # server only, no browser needed
node tests/e2e/web_smoke.js                              # serves web/ on a free port, artefacts in build/e2e/
node tests/e2e/web_smoke.js --screenshot-dir build/e2e --keep
node tests/e2e/web_smoke.js --url https://capstone-pso3d.vercel.app   # same checks against a deployment
node tests/e2e/serve.js --port 8000                      # manual preview with the production headers
```

`web_smoke.js` needs Playwright resolvable through `require('playwright')` (installed in the repository
or any parent directory) and its Chromium build (`npx playwright install chromium`; the browsers live under
`~/.cache/ms-playwright`). Nothing is downloaded by the gate itself. The Plotly library is fetched from
`https://cdn.plot.ly` exactly as in production, so the run needs network access; without it the
`window.Plotly` check **fails** (it is never skipped).

It is deliberately **not** part of `scripts/check.sh` / CI: those must stay installable with
`pip install -e .[plots,dev]` plus a stock Node 20, whereas the gate needs a browser download.

## Exit codes

| Code | Meaning |
|---|---|
| 0 | every check passed **and** zero CSP violations / console CSP refusals / unexpected foreign requests / console errors / uncaught page errors |
| 1 | at least one check failed (details in the printed summary and `summary.json`) |
| 2 | usage error |
| 3 | Playwright is not resolvable or Chromium cannot be launched — reported, nothing was checked (never a silent pass) |

## What is checked

Response headers (fetched outside the browser for `/`, `/app.js`, `/pso.js`, `/style.css`, and from the
navigation response inside the browser): `Content-Security-Policy` equal to the production policy,
`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`,
`Cache-Control` (`no-cache` for the document, `public, max-age=3600, must-revalidate` for assets).

Collectors active on every page: `securitypolicyviolation` DOM events (forwarded by an init script to
`console.error("CSP-VIOLATION", …)` and to an exposed binding), console messages (errors and any
"Refused to …" / "Content Security Policy" text), `pageerror`, requests to origins other than the
server and `https://cdn.plot.ly`, native dialogs (the page must use its inline `role="alert"` region, not `alert()`).

Workflows, all under the enforced policy:

* **A** defaults → *Run to end*: estimate / error / fitness-evaluations KPIs populated (1 200 evaluations), title contains `60 / 60`.
* **B** `#variant = amcmpso`, `#stopType = budget`, `#stopBudget = 600` → evaluations ≤ 620 and a "stopped by" KPI mentioning the budget.
* **C** `#warm` checked → the warm-start KPI (< 2 m) or an estimate error < 2 m.
* **D** real downloads: CSV header `iteration,best_fitness,est_x,est_y,est_z,error_m` with ≥ 10 rows; JSON with `scenario/config/result/history`. Copies are kept in `<dir>/downloads/`.
* **E** *Copy link* → URL containing `#s=v1.` (clipboard or `#shareUrl`), opened in a fresh navigation → `#variant` restored to `amcmpso`.
* **F** validation: `N = 1` and "three anchors left" each show `#alerts` text and disable `#play`.
* **G** offline fallback: a page with `https://cdn.plot.ly/**` blocked shows `#plotFallback` and the KPIs still populate.
* **a11y smoke** (plain DOM queries, no extra instrumentation injected into the application pages): every input/select in the panel has an accessible name (`label[for]`, wrapping label, `aria-label`/`aria-labelledby`), every button has a name, `#alerts` has `role="alert"`, `#stats` has `aria-live`, the document has `lang` and a title.
* **CSP enforcement probe** on a separate page: an inline `<script>` injected into the served document must be blocked with disposition `enforce`. Its single expected violation is reported separately and excluded from the application-flow counts.
* Screenshots: `initial.png`, `after-run.png`, `dark.png` (`prefers-color-scheme: dark`), `mobile.png` (420 × 900).

The printed summary lists PASS/FAIL per check and the collector counts; `summary.json` holds the same data
plus the captured console output per page.

#!/usr/bin/env node
'use strict';
/* Production-policy browser gate for the static simulator in web/.
 *
 *   node tests/e2e/web_smoke.js [--screenshot-dir DIR] [--keep] [--url ORIGIN] [--timeout MS]
 *
 * Serves web/ in-process through tests/e2e/serve.js — which applies every header rule of web/vercel.json, i.e.
 * the *enforced* Content-Security-Policy and the other security headers of the production deployment — then
 * drives headless Chromium (Playwright) through the real workflows under that policy while collecting
 * securitypolicyviolation events, console CSP refusals, console errors, uncaught page errors and requests to
 * origins other than the server and https://cdn.plot.ly.
 *
 * Exit codes: 0 every check passed and the collectors are empty · 1 at least one check failed ·
 *             2 usage error · 3 Playwright (or its Chromium build) is not available — reported, never a silent pass.
 * A structured summary is printed and written to <screenshot-dir>/summary.json (default build/e2e).
 * See tests/e2e/README.md.
 */
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const https = require('node:https');
const { createServer } = require('./serve.js');

const REPO = path.resolve(__dirname, '..', '..');
const WEB = path.join(REPO, 'web');
const VERCEL = path.join(WEB, 'vercel.json');
const PLOTLY_ORIGIN = 'https://cdn.plot.ly';
// The production policy (web/vercel.json). The gate fails if the served header differs from it.
const REQUIRED_CSP = "default-src 'self'; script-src 'self' https://cdn.plot.ly; style-src 'self' 'unsafe-inline'; "
  + "img-src 'self' data: blob:; connect-src 'self'; font-src 'self'; object-src 'none'; base-uri 'self'; "
  + "form-action 'self'; frame-ancestors 'none'";
const REQUIRED_HEADERS = {
  'x-content-type-options': 'nosniff',
  'referrer-policy': 'strict-origin-when-cross-origin',
  'permissions-policy': 'camera=(), microphone=(), geolocation=()',
  'x-frame-options': 'DENY',
};
const CACHE_DOCUMENT = 'no-cache';
const CACHE_ASSET = 'public, max-age=3600, must-revalidate';
const CSV_HEADER = 'iteration,best_fitness,est_x,est_y,est_z,error_m';
const DEFAULT_EVALUATIONS = 1200;       // simplified PSO, N = 20, T = 60
const DEFAULT_N = 20;

function usage() {
  return [
    'usage: node tests/e2e/web_smoke.js [--screenshot-dir DIR] [--keep] [--url ORIGIN] [--timeout MS]',
    '',
    '  --screenshot-dir DIR  where screenshots, export copies and summary.json go (default build/e2e)',
    '  --keep                do not clear previous artefacts in that directory',
    '  --url ORIGIN          check a deployed origin (e.g. https://capstone-pso3d.vercel.app) instead of serving web/ locally',
    '  --timeout MS          wait for window.Plotly / navigations (default 20000)',
  ].join('\n');
}

function parseArgs(argv) {
  const opts = { dir: path.join(REPO, 'build', 'e2e'), keep: false, url: null, timeout: 20000 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`missing value for ${a}`); return argv[++i]; };
    if (a === '--screenshot-dir') opts.dir = path.resolve(next());
    else if (a === '--keep') opts.keep = true;
    else if (a === '--url') opts.url = next().replace(/\/+$/, '');
    else if (a === '--timeout') opts.timeout = Number(next());
    else if (a === '-h' || a === '--help') opts.help = true;
    else throw new Error(`unknown option ${a}`);
  }
  if (!Number.isFinite(opts.timeout) || opts.timeout <= 0) throw new Error('--timeout must be a positive number of milliseconds');
  if (opts.url && !/^https?:\/\/[^/]+$/.test(opts.url)) throw new Error('--url must be an origin such as https://example.com');
  return opts;
}

// ------------------------------------------------------------------------------------------------
// check registry and per-page collectors
// ------------------------------------------------------------------------------------------------
const checks = [];
function check(group, name, pass, detail) {
  const rec = { group, name, pass: !!pass, detail: detail === undefined || detail === null ? '' : String(detail) };
  checks.push(rec);
  console.log(`${rec.pass ? 'PASS' : 'FAIL'} [${group}] ${name}${rec.detail ? ' — ' + rec.detail : ''}`);
  return rec.pass;
}

const pageLogs = {};
const labelOf = new WeakMap();
function newLog(label) {
  const log = { label, console: [], consoleErrors: [], toleratedErrors: [], cspRefusals: [], cspViolationEvents: [],
    cspViolationConsole: [], pageErrors: [], foreignRequests: [], failedRequests: [], dialogs: [] };
  pageLogs[label] = log;
  return log;
}
function attach(page, label, allowedOrigins, tolerate) {
  const log = pageLogs[label] || newLog(label);
  labelOf.set(page, label);
  page.on('console', (msg) => {
    const text = msg.text();
    const type = msg.type();
    log.console.push({ type, text });
    if (/^CSP-VIOLATION\b/.test(text)) { log.cspViolationConsole.push(text); return; }
    if (/Refused to|Content Security Policy/i.test(text)) log.cspRefusals.push(text);
    if (type === 'error') {
      if (tolerate && tolerate.test(text)) log.toleratedErrors.push(text); else log.consoleErrors.push(text);
    }
  });
  page.on('pageerror', (err) => log.pageErrors.push(String(err && err.message ? err.message : err)));
  page.on('request', (req) => {
    const url = req.url();
    if (/^(data|blob|about|chrome|chrome-error|devtools):/.test(url)) return;
    let origin;
    try { origin = new URL(url).origin; } catch (e) { return; }
    if (!allowedOrigins.has(origin)) log.foreignRequests.push(url);
  });
  page.on('requestfailed', (req) => log.failedRequests.push(`${req.url()} ${(req.failure() || {}).errorText || ''}`.trim()));
  page.on('dialog', (dialog) => { log.dialogs.push(`${dialog.type()}: ${dialog.message()}`); dialog.dismiss().catch(() => {}); });
  return log;
}
const violationCount = (log) => Math.max(log.cspViolationEvents.length, log.cspViolationConsole.length);

// ------------------------------------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------------------------------------
function fetchHeaders(url) {
  return new Promise((resolve, reject) => {
    const mod = url.startsWith('https:') ? https : http;
    const req = mod.request(url, { method: 'GET', headers: { 'user-agent': 'pso3d-web-smoke' } }, (res) => {
      res.resume();
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers }));
    });
    req.on('error', reject);
    req.setTimeout(15000, () => req.destroy(new Error('timeout')));
    req.end();
  });
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const parseNum = (s) => {
  if (typeof s !== 'string') return NaN;
  const m = s.replace(/[\s,  ]/g, '').match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/i);
  return m ? Number(m[0]) : NaN;
};
const TRIPLE = /^-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?,\s*-?\d+(\.\d+)?$/;
async function readKpis(page) {
  return page.$$eval('#stats .kpi', (nodes) => nodes.map((k) => ({
    label: ((k.querySelector('.l') || {}).textContent || '').trim(),
    value: ((k.querySelector('.v') || {}).textContent || '').trim(),
  })));
}
const findKpi = (kpis, re) => kpis.find((k) => re.test(k.label));
const kpiText = (kpis) => kpis.map((k) => `${k.label}: ${k.value}`).join(' | ');
async function statusText(page) {
  return page.evaluate(() => {
    const run = document.getElementById('runStatus');
    const title = document.querySelector('#plot3d .gtitle');
    return { runStatus: run ? run.textContent.trim() : '', plotTitle: title ? title.textContent.trim() : '' };
  });
}

// ------------------------------------------------------------------------------------------------
// main
// ------------------------------------------------------------------------------------------------
async function main(argv) {
  let opts;
  try { opts = parseArgs(argv); } catch (e) { console.error(`web_smoke: ${e.message}\n${usage()}`); return 2; }
  if (opts.help) { console.log(usage()); return 0; }

  let playwright;
  try {
    playwright = require('playwright');
  } catch (e) {
    console.error('web_smoke: Playwright is not resolvable via require("playwright") — ' + (e && e.code ? e.code : e));
    console.error('  Install it in the repository or a parent directory (npm install playwright && npx playwright install chromium).');
    console.error('  This browser gate is therefore not part of scripts/check.sh / CI; nothing was checked (exit 3).');
    return 3;
  }

  const started = new Date();
  const t0 = Date.now();
  if (!opts.keep && fs.existsSync(opts.dir)) {
    for (const f of fs.readdirSync(opts.dir)) {
      if (/\.(png|json)$/.test(f) || f === 'downloads') fs.rmSync(path.join(opts.dir, f), { recursive: true, force: true });
    }
  }
  fs.mkdirSync(path.join(opts.dir, 'downloads'), { recursive: true });

  // --- server -------------------------------------------------------------------------------------
  let server = null, origin = opts.url;
  if (!origin) {
    server = createServer(WEB, VERCEL);
    await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
    origin = `http://127.0.0.1:${server.address().port}`;
  }
  console.log(`web_smoke: origin ${origin}${server ? ` (serving ${WEB} with ${server.rules.length} header rules from vercel.json)` : ' (remote)'}`);
  const allowedOrigins = new Set([origin, PLOTLY_ORIGIN]);

  // --- vercel.json pin ----------------------------------------------------------------------------
  const vercel = JSON.parse(fs.readFileSync(VERCEL, 'utf8'));
  let vercelCsp = null;
  for (const rule of vercel.headers || []) for (const h of rule.headers || []) if (h.key.toLowerCase() === 'content-security-policy') vercelCsp = h.value;
  check('config', 'web/vercel.json Content-Security-Policy equals the production policy', vercelCsp === REQUIRED_CSP, vercelCsp || '(missing)');
  check('config', 'web/vercel.json has no Content-Security-Policy-Report-Only (policy is enforced)',
    !(vercel.headers || []).some((r) => (r.headers || []).some((h) => h.key.toLowerCase() === 'content-security-policy-report-only')));

  // --- response headers (outside the browser) -----------------------------------------------------
  for (const p of ['/', '/app.js', '/pso.js', '/style.css']) {
    try {
      const r = await fetchHeaders(origin + p);
      const h = r.headers;
      check('headers', `${p} responds 200`, r.status === 200, `status ${r.status}`);
      check('headers', `${p} Content-Security-Policy equals the production policy`, h['content-security-policy'] === REQUIRED_CSP, h['content-security-policy'] || '(missing)');
      for (const [key, value] of Object.entries(REQUIRED_HEADERS)) check('headers', `${p} ${key}`, h[key] === value, h[key] || '(missing)');
      const expectedCache = p === '/' ? CACHE_DOCUMENT : CACHE_ASSET;
      check('headers', `${p} Cache-Control`, h['cache-control'] === expectedCache, h['cache-control'] || '(missing)');
    } catch (e) {
      check('headers', `${p} fetched`, false, e.message);
    }
  }

  // --- browser ------------------------------------------------------------------------------------
  let browser;
  try {
    browser = await playwright.chromium.launch({ headless: true, args: ['--enable-unsafe-swiftshader'] });
  } catch (e) {
    console.error('web_smoke: Playwright is installed but headless Chromium could not be launched — ' + (e && e.message ? e.message.split('\n')[0] : e));
    console.error('  Run `npx playwright install chromium` (browsers live under ~/.cache/ms-playwright). Nothing was checked (exit 3).');
    if (server) await new Promise((r) => server.close(r));
    return 3;
  }
  const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 900 }, colorScheme: 'light', bypassCSP: false });
  await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin });
  context.setDefaultTimeout(10000);
  await context.exposeBinding('__pso3dCspViolation', (source, info) => {
    const label = labelOf.get(source.page) || 'unknown';
    (pageLogs[label] || newLog(label)).cspViolationEvents.push(info);
  });
  await context.addInitScript(() => {
    document.addEventListener('securitypolicyviolation', (e) => {
      const info = { violatedDirective: e.violatedDirective, effectiveDirective: e.effectiveDirective, blockedURI: e.blockedURI,
        disposition: e.disposition, sourceFile: e.sourceFile, lineNumber: e.lineNumber, sample: e.sample };
      console.error('CSP-VIOLATION', info.violatedDirective, info.blockedURI || '', info.sourceFile || '', String(info.lineNumber || 0));
      if (typeof window.__pso3dCspViolation === 'function') { try { window.__pso3dCspViolation(info); } catch (err) { /* console line above suffices */ } }
    });
  });
  const shot = async (page, name) => {
    const file = path.join(opts.dir, `${name}.png`);
    try {
      await page.screenshot({ path: file, fullPage: true });
      const size = fs.statSync(file).size;
      check('screenshots', `${name}.png written`, size > 1000, `${size} bytes`);
    } catch (e) { check('screenshots', `${name}.png written`, false, e.message); }
  };
  const runStage = async (group, fn) => {
    try { await fn(); } catch (e) { check(group, 'stage completed without an exception', false, (e && e.message ? e.message.split('\n')[0] : String(e))); }
  };

  const page = await context.newPage();
  attach(page, 'main', allowedOrigins);
  let plotlyLoaded = false;

  // ---- load + initial screenshot + a11y smoke ----
  await runStage('load', async () => {
    const resp = await page.goto(origin + '/', { waitUntil: 'load', timeout: opts.timeout });
    check('load', 'GET / in the browser responds 200', !!resp && resp.status() === 200, resp ? `status ${resp.status()}` : 'no response');
    const h = resp ? resp.headers() : {};
    check('headers', 'document as rendered by the browser carries the enforced production CSP', h['content-security-policy'] === REQUIRED_CSP, h['content-security-policy'] || '(missing)');
    try {
      await page.waitForFunction(() => typeof window.Plotly !== 'undefined', null, { timeout: opts.timeout });
      plotlyLoaded = true;
    } catch (e) { plotlyLoaded = false; }
    check('load', 'window.Plotly is available', plotlyLoaded, plotlyLoaded ? `loaded from ${PLOTLY_ORIGIN}` : 'Plotly CDN script did not load — network blocked?');
    if (plotlyLoaded) {
      let rendered = true;
      try { await page.waitForSelector('#plot3d .main-svg', { timeout: 10000 }); } catch (e) { rendered = false; }
      check('load', '#plot3d rendered by Plotly', rendered, rendered ? '' : 'no Plotly content inside #plot3d');
    }
    await page.waitForSelector('#stats .kpi');
    const alerts = (await page.textContent('#alerts')) || '';
    check('load', 'default configuration is valid (#alerts empty, #play enabled)', alerts.trim() === '' && !(await page.isDisabled('#play')), alerts.trim());
    await shot(page, 'initial');
  });

  await runStage('a11y', async () => {
    const a = await page.evaluate(() => {
      const panel = document.querySelector('aside.panel') || document.querySelector('main') || document.body;
      const esc = (s) => (window.CSS && CSS.escape ? CSS.escape(s) : s.replace(/([^\w-])/g, '\\$1'));
      const nameOf = (el) => {
        const aria = el.getAttribute('aria-label'); if (aria && aria.trim()) return aria.trim();
        const by = el.getAttribute('aria-labelledby');
        if (by) { const t = by.split(/\s+/).map((id) => (document.getElementById(id) || {}).textContent || '').join(' ').trim(); if (t) return t; }
        if (el.id) { const l = document.querySelector(`label[for="${esc(el.id)}"]`); if (l && l.textContent.trim()) return l.textContent.trim(); }
        const wrap = el.closest('label');
        if (wrap) { const c = wrap.cloneNode(true); c.querySelectorAll('input,select,textarea').forEach((n) => n.remove()); const t = c.textContent.trim(); if (t) return t; }
        return '';
      };
      const describe = (el) => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') + (el.type ? `[${el.type}]` : '');
      const controls = [...panel.querySelectorAll('input, select, textarea')];
      const buttons = [...panel.querySelectorAll('button')];
      const alerts = document.getElementById('alerts');
      const stats = document.getElementById('stats');
      return {
        controls: controls.length,
        unnamed: controls.filter((el) => !nameOf(el)).map(describe),
        buttons: buttons.length,
        unnamedButtons: buttons.filter((b) => !(b.textContent.trim() || b.getAttribute('aria-label') || b.getAttribute('aria-labelledby') || b.getAttribute('title'))).map(describe),
        alertsRole: alerts ? alerts.getAttribute('role') : null,
        statsLive: stats ? stats.getAttribute('aria-live') : null,
        lang: document.documentElement.getAttribute('lang'),
        title: document.title,
      };
    });
    check('a11y', `every input/select in the panel has an accessible name (${a.controls} controls)`, a.controls > 0 && a.unnamed.length === 0, a.unnamed.length ? 'unnamed: ' + a.unnamed.join(', ') : '');
    check('a11y', `every button in the panel has an accessible name (${a.buttons} buttons)`, a.buttons > 0 && a.unnamedButtons.length === 0, a.unnamedButtons.join(', '));
    check('a11y', '#alerts has role="alert"', a.alertsRole === 'alert', `role=${a.alertsRole}`);
    check('a11y', '#stats has aria-live', !!a.statsLive, `aria-live=${a.statsLive}`);
    check('a11y', 'document has lang and a title', !!a.lang && !!a.title, `lang=${a.lang} title=${JSON.stringify(a.title)}`);
  });

  // ---- workflow A: defaults, run to end ----
  await runStage('A', async () => {
    await page.click('#finish');
    const kpis = await readKpis(page);
    const est = findKpi(kpis, /^estimate/i), err = findKpi(kpis, /^error/i), ev = findKpi(kpis, /fitness evaluations/i);
    check('A', 'estimate KPI shows three numbers', !!est && TRIPLE.test(est.value), est ? est.value : kpiText(kpis));
    const e = err ? parseNum(err.value) : NaN;
    check('A', 'error KPI is a finite distance (< 10 m)', Number.isFinite(e) && e < 10, err ? err.value : '(missing)');
    check('A', `fitness evaluations KPI == ${DEFAULT_EVALUATIONS} for the defaults`, !!ev && parseNum(ev.value) === DEFAULT_EVALUATIONS, ev ? ev.value : '(missing)');
    const st = await statusText(page);
    const titleText = plotlyLoaded && st.plotTitle ? st.plotTitle : st.runStatus;
    check('A', 'run title contains "60 / 60"', /60\s*\/\s*60/.test(titleText), `${plotlyLoaded && st.plotTitle ? 'plot title' : '#runStatus'}: ${titleText}`);
    await shot(page, 'after-run');
  });

  // ---- workflow B: AMCMPSO + evaluation budget ----
  await runStage('B', async () => {
    await page.selectOption('#variant', 'amcmpso');
    await page.selectOption('#stopType', 'budget');
    await page.fill('#stopBudget', '600');
    await page.click('#reset');
    await page.click('#finish');
    const kpis = await readKpis(page);
    const ev = findKpi(kpis, /fitness evaluations/i);
    const n = ev ? parseNum(ev.value) : NaN;
    check('B', `AMCMPSO with budget 600: fitness evaluations <= ${600 + DEFAULT_N}`, Number.isFinite(n) && n > 0 && n <= 600 + DEFAULT_N, ev ? ev.value : '(missing)');
    const stop = kpis.find((k) => /stopped by/i.test(k.label + ' ' + k.value));
    const st = await statusText(page);
    const mentions = (stop && /budget/i.test(stop.label + ' ' + stop.value)) || /stopped.*budget/i.test(st.runStatus);
    check('B', 'a "stopped by" KPI mentions the budget', mentions, stop ? `${stop.label}: ${stop.value}` : st.runStatus);
    const err = findKpi(kpis, /^error/i);
    check('B', 'AMCMPSO error KPI is a finite distance', !!err && Number.isFinite(parseNum(err.value)), err ? err.value : '(missing)');
  });

  // ---- workflow C: warm start ----
  await runStage('C', async () => {
    await page.check('#warm');
    await page.click('#reset');
    await page.click('#finish');
    const kpis = await readKpis(page);
    const warm = findKpi(kpis, /warm start/i);
    const err = findKpi(kpis, /^error/i);
    const e = err ? parseNum(err.value) : NaN;
    const w = warm ? parseNum(warm.value) : NaN;
    check('C', 'warm start: a KPI reports the warm-start point (< 2 m) or the estimate error is < 2 m',
      (Number.isFinite(w) && w < 2) || (Number.isFinite(e) && e < 2), `${warm ? warm.label + ': ' + warm.value : 'no warm-start KPI'}; error ${err ? err.value : '(missing)'}`);
  });

  // ---- workflow D: exports (real downloads under the enforced CSP) ----
  await runStage('D', async () => {
    const [csvDl] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('#exportCsv')]);
    const csvPath = await csvDl.path();
    const csv = fs.readFileSync(csvPath, 'utf8');
    fs.copyFileSync(csvPath, path.join(opts.dir, 'downloads', csvDl.suggestedFilename() || 'pso3d_run.csv'));
    const lines = csv.trim().split(/\r?\n/);
    check('D', `CSV export header is "${CSV_HEADER}"`, lines[0] === CSV_HEADER, lines[0]);
    check('D', 'CSV export has >= 10 data rows with 6 numeric columns', lines.length - 1 >= 10 && lines.slice(1).every((l) => l.split(',').length === 6 && l.split(',').every((c) => Number.isFinite(Number(c)))), `${lines.length - 1} rows (${csvDl.suggestedFilename()})`);
    const [jsonDl] = await Promise.all([page.waitForEvent('download', { timeout: 10000 }), page.click('#exportJson')]);
    const jsonPath = await jsonDl.path();
    const text = fs.readFileSync(jsonPath, 'utf8');
    fs.copyFileSync(jsonPath, path.join(opts.dir, 'downloads', jsonDl.suggestedFilename() || 'pso3d_run.json'));
    let data = null;
    try { data = JSON.parse(text); } catch (e) { data = null; }
    const keys = data ? Object.keys(data) : [];
    check('D', 'JSON export parses and has scenario/config/result/history', !!data && ['scenario', 'config', 'result', 'history'].every((k) => keys.includes(k)), keys.join(','));
    check('D', 'JSON export result has estimate and fitness_evaluations', !!data && data.result && Array.isArray(data.result.estimate) && Number.isFinite(data.result.fitness_evaluations), data && data.result ? `evaluations ${data.result.fitness_evaluations}` : '');
  });

  // ---- workflow E: share link round trip ----
  await runStage('E', async () => {
    await page.click('#share');
    await sleep(300);
    let clip = null;
    try { clip = await page.evaluate(() => navigator.clipboard.readText()); } catch (e) { clip = null; }
    let field = null;
    try { field = await page.inputValue('#shareUrl'); } catch (e) { field = null; }
    const url = (clip && clip.includes('#s=')) ? clip : field;
    check('E', 'share produces a URL containing "#s=v1." (clipboard or #shareUrl)', !!url && url.includes('#s=v1.'), url ? `${clip && clip.includes('#s=') ? 'clipboard' : '#shareUrl'}: ${url.slice(0, 80)}…` : 'no URL');
    if (url && url.includes('#s=v1.')) {
      const target = url.startsWith('http') ? url : origin + '/' + url.replace(/^.*?#/, '#');
      await page.goto('about:blank');
      await page.goto(target, { waitUntil: 'load', timeout: opts.timeout });
      await page.waitForSelector('#stats .kpi');
      const variant = await page.inputValue('#variant');
      const stopType = await page.inputValue('#stopType');
      const warm = await page.isChecked('#warm');
      check('E', 'opening the share link restores #variant = amcmpso', variant === 'amcmpso', `variant=${variant} stopType=${stopType} warm=${warm}`);
    }
  });

  // ---- workflow F: inline validation ----
  await runStage('F', async () => {
    await page.fill('#np', '1');
    await page.click('#reset');
    await sleep(100);
    const alerts = ((await page.textContent('#alerts')) || '').trim();
    const disabled = await page.isDisabled('#play');
    check('F', 'N = 1 shows an inline #alerts message and disables #play', alerts.length > 0 && disabled, `${alerts.slice(0, 80)} | play disabled=${disabled}`);
    await page.fill('#np', '20');
    await page.click('#reset');
    let rows = await page.$$('#anchors tbody tr');
    let guard = 20;
    while (rows.length > 3 && guard-- > 0) {
      const btn = await rows[rows.length - 1].$('button');
      if (!btn) break;
      await btn.click();
      rows = await page.$$('#anchors tbody tr');
    }
    const alerts2 = ((await page.textContent('#alerts')) || '').trim();
    const disabled2 = await page.isDisabled('#play');
    check('F', 'three anchors left shows an inline #alerts message and disables #play', rows.length === 3 && alerts2.length > 0 && disabled2, `${rows.length} rows | ${alerts2.slice(0, 80)} | play disabled=${disabled2}`);
  });

  // ---- dark scheme screenshot (main page, fresh load) ----
  await runStage('screenshots', async () => {
    await page.goto(origin + '/', { waitUntil: 'load', timeout: opts.timeout });
    await page.waitForSelector('#stats .kpi');
    await page.emulateMedia({ colorScheme: 'dark' });
    await sleep(500);
    await shot(page, 'dark');
    await page.emulateMedia({ colorScheme: 'light' });
  });

  // ---- mobile viewport ----
  await runStage('screenshots', async () => {
    const pm = await context.newPage();
    attach(pm, 'mobile', allowedOrigins);
    await pm.setViewportSize({ width: 420, height: 900 });
    await pm.goto(origin + '/', { waitUntil: 'load', timeout: opts.timeout });
    await pm.waitForSelector('#stats .kpi');
    await sleep(300);
    await shot(pm, 'mobile');
    await pm.close();
  });

  // ---- workflow G: offline fallback (Plotly CDN blocked) ----
  await runStage('G', async () => {
    const po = await context.newPage();
    attach(po, 'offline', allowedOrigins, /cdn\.plot\.ly|net::ERR_(FAILED|BLOCKED_BY_CLIENT|ABORTED)/);
    await po.route(`${PLOTLY_ORIGIN}/**`, (route) => route.abort('blockedbyclient'));
    await po.goto(origin + '/', { waitUntil: 'load', timeout: opts.timeout });
    let visible = false;
    try { await po.waitForSelector('#plotFallback', { state: 'visible', timeout: 5000 }); visible = true; } catch (e) { visible = false; }
    check('G', '#plotFallback is visible when the Plotly CDN is blocked', visible, visible ? (await po.textContent('#plotFallback') || '').trim() : 'not visible');
    await po.waitForSelector('#stats .kpi');
    await po.click('#finish');
    const kpis = await readKpis(po);
    const est = findKpi(kpis, /^estimate/i), ev = findKpi(kpis, /fitness evaluations/i);
    check('G', 'KPIs still populate without Plotly (estimate + evaluations)', !!est && TRIPLE.test(est.value) && !!ev && parseNum(ev.value) === DEFAULT_EVALUATIONS, `${est ? est.value : '-'} / ${ev ? ev.value : '-'}`);
    check('G', 'offline page raised no uncaught errors', pageLogs.offline.pageErrors.length === 0, pageLogs.offline.pageErrors.join('; '));
    await po.close();
  });

  // ---- CSP enforcement probe (separate page; its one expected violation is excluded from the app-flow counts) ----
  await runStage('csp-probe', async () => {
    const pp = await context.newPage();
    attach(pp, 'csp-probe', allowedOrigins);
    await pp.goto(origin + '/', { waitUntil: 'load', timeout: opts.timeout });
    const result = await pp.evaluate(() => new Promise((resolve) => {
      const seen = [];
      const onViolation = (e) => seen.push({ violatedDirective: e.violatedDirective, blockedURI: e.blockedURI, disposition: e.disposition });
      document.addEventListener('securitypolicyviolation', onViolation);
      window.__pso3dProbe = 0;
      const s = document.createElement('script');
      s.textContent = 'window.__pso3dProbe = 1;';
      document.head.appendChild(s);
      setTimeout(() => { document.removeEventListener('securitypolicyviolation', onViolation); resolve({ executed: window.__pso3dProbe === 1, seen }); }, 400);
    }));
    check('csp-probe', 'an inline <script> injected into the served page is blocked and reported with disposition "enforce"',
      !result.executed && result.seen.length > 0 && result.seen.every((v) => v.disposition === 'enforce' && /^script-src/.test(v.violatedDirective)),
      result.executed ? 'inline script executed — the policy is not enforced' : JSON.stringify(result.seen[0] || {}));
    await pp.close();
  });

  // --- collectors --------------------------------------------------------------------------------
  const appLabels = Object.keys(pageLogs).filter((l) => l !== 'csp-probe');
  const sum = (labels, key) => labels.reduce((n, l) => n + pageLogs[l][key].length, 0);
  const counts = {
    consoleErrors: sum(appLabels, 'consoleErrors'),
    toleratedOfflineErrors: sum(appLabels, 'toleratedErrors'),
    cspViolations: appLabels.reduce((n, l) => n + violationCount(pageLogs[l]), 0),
    cspRefusals: sum(appLabels, 'cspRefusals'),
    pageErrors: sum(appLabels, 'pageErrors'),
    foreignRequests: sum(Object.keys(pageLogs), 'foreignRequests'),
    dialogs: sum(appLabels, 'dialogs'),
    probeViolations: pageLogs['csp-probe'] ? violationCount(pageLogs['csp-probe']) : 0,
  };
  const list = (key) => appLabels.flatMap((l) => pageLogs[l][key].map((x) => `[${l}] ${typeof x === 'string' ? x : JSON.stringify(x)}`));
  check('csp', 'zero securitypolicyviolation events in the application pages', counts.cspViolations === 0, list('cspViolationEvents').concat(list('cspViolationConsole')).slice(0, 5).join('; '));
  check('csp', 'zero console CSP refusals in the application pages', counts.cspRefusals === 0, list('cspRefusals').slice(0, 5).join('; '));
  check('network', 'zero requests to origins other than the server and https://cdn.plot.ly', counts.foreignRequests === 0, Object.keys(pageLogs).flatMap((l) => pageLogs[l].foreignRequests).slice(0, 5).join('; '));
  check('console', 'zero console errors in the application pages', counts.consoleErrors === 0, list('consoleErrors').slice(0, 5).join('; '));
  check('console', 'zero uncaught page errors', counts.pageErrors === 0, list('pageErrors').slice(0, 5).join('; '));
  check('ui', 'no native alert()/confirm() dialogs', counts.dialogs === 0, list('dialogs').slice(0, 5).join('; '));

  // --- teardown ------------------------------------------------------------------------------------
  const versions = { playwright: (() => { try { return require('playwright/package.json').version; } catch (e) { return null; } })(), chromium: browser.version() };
  await context.close();
  await browser.close();
  if (server) {
    if (typeof server.closeAllConnections === 'function') server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
  }

  // --- summary -------------------------------------------------------------------------------------
  const failed = checks.filter((c) => !c.pass);
  const ok = failed.length === 0;
  const summary = {
    ok, exitCode: ok ? 0 : 1, startedAt: started.toISOString(), finishedAt: new Date().toISOString(), durationMs: Date.now() - t0,
    origin, remote: !!opts.url, screenshotDir: opts.dir, versions, requiredCsp: REQUIRED_CSP,
    totals: { checks: checks.length, passed: checks.length - failed.length, failed: failed.length },
    counts,
    checks,
    pages: Object.fromEntries(Object.entries(pageLogs).map(([label, log]) => [label, {
      consoleErrors: log.consoleErrors, toleratedErrors: log.toleratedErrors, cspRefusals: log.cspRefusals,
      cspViolationEvents: log.cspViolationEvents, cspViolationConsole: log.cspViolationConsole, pageErrors: log.pageErrors,
      foreignRequests: log.foreignRequests, failedRequests: log.failedRequests, dialogs: log.dialogs,
      console: log.console.slice(0, 200),
    }])),
  };
  fs.writeFileSync(path.join(opts.dir, 'summary.json'), JSON.stringify(summary, null, 2));
  console.log('');
  console.log('== web_smoke summary ==');
  const groups = [...new Set(checks.map((c) => c.group))];
  for (const g of groups) {
    const gs = checks.filter((c) => c.group === g);
    console.log(`  ${g.padEnd(12)} ${gs.filter((c) => c.pass).length}/${gs.length} passed`);
  }
  if (failed.length) { console.log('failed checks:'); for (const f of failed) console.log(`  - [${f.group}] ${f.name}${f.detail ? ' — ' + f.detail : ''}`); }
  console.log(`console errors: ${counts.consoleErrors} (tolerated offline resource errors: ${counts.toleratedOfflineErrors}) · CSP violations: ${counts.cspViolations} · CSP refusals: ${counts.cspRefusals} · foreign requests: ${counts.foreignRequests} · page errors: ${counts.pageErrors} · dialogs: ${counts.dialogs} · probe violations (expected >= 1): ${counts.probeViolations}`);
  console.log(`checks: ${checks.length - failed.length}/${checks.length} passed · summary: ${path.join(opts.dir, 'summary.json')} · ${summary.durationMs} ms`);
  console.log(`RESULT: ${ok ? 'PASS' : 'FAIL'} (exit ${ok ? 0 : 1})`);
  return ok ? 0 : 1;
}

main(process.argv.slice(2)).then((code) => { process.exitCode = code; }, (err) => {
  console.error('web_smoke: unexpected error — ' + (err && err.stack ? err.stack : err));
  process.exitCode = 1;
});

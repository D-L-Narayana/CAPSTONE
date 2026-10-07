/* PSO 3D localization simulator — UI layer.
   Algorithms live in pso.js / geometry.js, the share-link codec in share.js and the exporters in export.js.
   Every optional helper is feature-checked, so a missing function degrades to "–" in the KPIs instead of an error.
   No inline scripts or handlers: everything is wired with addEventListener (content-security-policy friendly). */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const DEFAULT_ANCHORS = [[0, 0, 0, 0.4], [60, 0, 5, -0.3], [0, 60, 6, 0.5], [60, 60, 20, -0.4]];
  // Python AMCMPSO constructor defaults (pso3d/amcmpso.py); only the schedule is user-selectable here.
  const AMCMPSO_DEFAULTS = { wMax: 0.9, wMin: 0.4, c1Start: 2.0, c1End: 0.5, c2Start: 0.5, c2End: 2.0, cMean: 0.5, cCom: 0.5 };
  const COLOR = { anchors: '#1B474D', particles: '#20808D', node: '#A84B2F', estimate: '#111', trail: '#7A7974', lsq: '#FFC553' };
  // Dark colour scheme: the deep-teal anchor squares and their a_j labels vanish on the dark scene background,
  // so they switch to the dark-scheme --primary-hover tone; every light-scheme colour above stays as it is.
  const DARK = { anchors: '#7CCBD1', estimate: '#E8E6E1', text: '#E8E6E1', surface: '#22272A', grid: '#3A4044' };
  const AXES = ['x', 'y', 'z', 'noise'];
  const HINTS = {
    variant: {
      simplified: 'Best-of-iteration attractor without particle memory (the Review-1 algorithm, N·6 floats).',
      standard: 'Personal / global best with synchronous updates (N·9 + 3 floats).',
      amcmpso: 'Interpretation of Alhasan et al. 2023: velocity guided by the swarm mean and its fitness-weighted centre of mass with an adaptive w / c1 / c2 schedule. The w, c2, c1 fields are not used by this variant. Not a reproduction of the paper’s numbers.',
    },
    noise: {
      fixed: 'Each measured range is the true range plus the per-anchor offset n_j from the table.',
      gaussian: 'Each range gets independent zero-mean Gaussian noise with standard deviation σ (seeded from the seed field).',
      uniformPercent: 'Each range gets uniform noise within ± P_n % of the true range (seeded).',
      nlos: 'Gaussian line-of-sight noise σ plus, with the given probability, a positive bias |mean + bias σ · N(0, 1)| (seeded).',
    },
  };

  // ---- local fallbacks (used only when pso.js does not provide the helper) ----
  function localMulberry32(seed) {
    let a = (seed >>> 0) || 1;
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function localGaussian(rand) {
    let u = 0, v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  function localDist(p, q) { return Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]); }

  // ---- optional algorithm API; identifiers are tested with typeof so a missing script never throws ----
  const api = {
    Swarm: typeof Swarm === 'function' ? Swarm : null,
    Fitness: typeof Fitness === 'function' ? Fitness : null,
    mulberry32: typeof mulberry32 === 'function' ? mulberry32 : localMulberry32,
    gaussian: typeof gaussian === 'function' ? gaussian : localGaussian,
    dist: typeof dist === 'function' ? dist : localDist,
    leastSquares: typeof leastSquares === 'function' ? leastSquares : null,
    gaussNewton: typeof gaussNewton === 'function' ? gaussNewton : null,
    warmStart: typeof warmStart === 'function' ? warmStart : null,
    sampleNoise: typeof sampleNoise === 'function' ? sampleNoise : null,
    isNonCoplanar: typeof isNonCoplanar === 'function' ? isNonCoplanar : null,
    gdop: typeof gdop === 'function' ? gdop : null,
    crlb: typeof crlb === 'function' ? crlb : null,
    mirrorPoint: typeof mirrorPoint === 'function' ? mirrorPoint : null,
    flipAmbiguityRisk: typeof flipAmbiguityRisk === 'function' ? flipAmbiguityRisk : null,
    encodeState: typeof encodeState === 'function' ? encodeState : null,
    decodeState: typeof decodeState === 'function' ? decodeState : null,
    applyHash: typeof applyHash === 'function' ? applyHash : null,
    historyToCSV: typeof historyToCSV === 'function' ? historyToCSV : null,
    runToJSON: typeof runToJSON === 'function' ? runToJSON : null,
    downloadText: typeof downloadText === 'function' ? downloadText : null,
  };

  function localSampleNoise(model, trueRanges, rand) {
    const g = () => api.gaussian(rand);
    switch (model.type) {
      case 'gaussian': return trueRanges.map(() => model.sigma * g());
      case 'uniformPercent': return trueRanges.map((d) => d * (model.pn / 100) * (2 * rand() - 1));
      case 'nlos': return trueRanges.map(() => {
        const base = model.sigma * g();
        return rand() < model.p ? base + Math.abs(model.biasMean + model.biasSigma * g()) : base;
      });
      default: return trueRanges.map((_, j) => { const v = Number(model.offsets && model.offsets[j]); return Number.isFinite(v) ? v : 0; });
    }
  }

  // ---- state ----
  let swarm = null, fitness = null, scenario = null, timer = null, valid = false;
  let lsq = null, gn = null, x0 = null, geo = null;
  let randSeed = null, nodeRand = null, anchorRand = null;

  // ---- small helpers ----
  const safe = (fn, fallback) => { try { return fn(); } catch (e) { return fallback === undefined ? null : fallback; } };
  const num = (id, fallback) => { const v = parseFloat($(id).value); return Number.isFinite(v) ? v : (fallback === undefined ? NaN : fallback); };
  const int = (id, fallback) => { const v = parseInt($(id).value, 10); return Number.isFinite(v) ? v : (fallback === undefined ? NaN : fallback); };
  const vec3 = (v) => (Array.isArray(v) && v.length === 3 && v.every(Number.isFinite) ? v.map(Number) : null);
  const fmt = (v, d) => (Number.isFinite(v) ? v.toFixed(d === undefined ? 2 : d) : '–');
  const fmtInf = (v, d) => (v === Infinity ? '∞' : fmt(v, d));
  const unit = (txt, u) => (txt === '–' ? txt : txt + ' ' + u);
  const mean = (a) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN);
  const setVal = (id, v) => { $(id).value = String(v); };
  const plotlyOk = () => typeof Plotly !== 'undefined' && Plotly !== null && typeof Plotly.react === 'function';
  const isDark = () => !!(window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches);

  // ---- anchor table ----
  function renderAnchorRows(rows) {
    const tb = $('anchors').querySelector('tbody');
    tb.textContent = '';
    rows.forEach((r, i) => {
      const tr = document.createElement('tr');
      const th = document.createElement('th');
      th.scope = 'row'; th.textContent = 'a' + (i + 1);
      tr.appendChild(th);
      for (let k = 0; k < 4; k++) {
        const td = document.createElement('td');
        const inp = document.createElement('input');
        inp.type = 'number'; inp.step = k === 3 ? '0.1' : '1';
        inp.value = String(Number.isFinite(r[k]) ? r[k] : 0);
        inp.dataset.k = String(k);
        inp.setAttribute('aria-label', 'anchor ' + (i + 1) + ' ' + AXES[k]);
        inp.addEventListener('change', reset);
        td.appendChild(inp); tr.appendChild(td);
      }
      const td = document.createElement('td');
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = '×'; b.title = 'remove anchor ' + (i + 1);
      b.setAttribute('aria-label', 'remove anchor ' + (i + 1));
      b.dataset.i = String(i);
      b.addEventListener('click', () => removeAnchor(i));
      td.appendChild(b); tr.appendChild(td);
      tb.appendChild(tr);
    });
  }
  function readAnchorRows() {
    return [...$('anchors').querySelectorAll('tbody tr')].map((tr) =>
      [...tr.querySelectorAll('input')].map((i) => { const v = parseFloat(i.value); return Number.isFinite(v) ? v : 0; }));
  }
  function removeAnchor(i) {
    const rows = readAnchorRows();
    if (rows.length <= 1) return;
    rows.splice(i, 1);
    renderAnchorRows(rows);
    reset();
  }

  // ---- seeded helpers for "random node" / "+ anchor" (repeatable: derived from the seed field) ----
  function seededStreams() {
    const seed = int('seed', 1);
    if (randSeed !== seed || !nodeRand || !anchorRand) {
      randSeed = seed;
      nodeRand = api.mulberry32((seed ^ 0x51ED270B) >>> 0);
      anchorRand = api.mulberry32((seed ^ 0x2545F491) >>> 0);
    }
  }
  function randomNode() {
    seededStreams();
    const r = nodeRand;
    setVal('px', (r() * num('fx', 60)).toFixed(1));
    setVal('py', (r() * num('fy', 60)).toFixed(1));
    setVal('pz', (r() * num('fz', 20)).toFixed(1));
    reset();
  }
  function addAnchor() {
    seededStreams();
    const r = anchorRand;
    const rows = readAnchorRows();
    rows.push([Math.round(r() * num('fx', 60)), Math.round(r() * num('fy', 60)), Math.round(r() * num('fz', 20)), 0]);
    renderAnchorRows(rows);
    reset();
  }

  // ---- scenario ----
  function readNoiseModel(offsets) {
    const model = $('noiseModel').value;
    if (model === 'gaussian') return { type: 'gaussian', sigma: Math.max(0, num('sigma', 0.5)) };
    if (model === 'uniformPercent') return { type: 'uniformPercent', pn: Math.max(0, num('pn', 2)) };
    if (model === 'nlos') {
      return { type: 'nlos', p: Math.min(1, Math.max(0, num('nlosP', 0.2))), biasMean: num('nlosBias', 2), biasSigma: Math.max(0, num('nlosSigma', 1)), sigma: Math.max(0, num('sigma', 0.5)) };
    }
    return { type: 'fixed', offsets: offsets.slice() };
  }
  /** σ used for the CRLB: the model σ for Gaussian/NLOS, the RMS of the offsets for fixed, the RMS uniform σ for ± %. */
  function effectiveSigma(s) {
    const m = s.noiseModel;
    if (m.type === 'gaussian' || m.type === 'nlos') return m.sigma;
    if (m.type === 'uniformPercent') return Math.sqrt(mean(s.trueRanges.map((d) => (d * m.pn / 100) * (d * m.pn / 100) / 3)));
    return Math.sqrt(mean(m.offsets.map((o) => o * o)));
  }
  function readScenario() {
    const upper = [num('fx'), num('fy'), num('fz')];
    const rows = readAnchorRows();
    const anchors = rows.map((r) => r.slice(0, 3));
    const offsets = rows.map((r) => r[3]);
    const p = [num('px', 0), num('py', 0), num('pz', 0)];
    const seed = int('seed', 1);
    const trueRanges = anchors.map((a) => api.dist(p, a));
    const noiseModel = readNoiseModel(offsets);
    let noise = api.sampleNoise ? safe(() => api.sampleNoise(noiseModel, trueRanges, api.mulberry32(seed ^ 0x9E3779B9))) : null;
    let noiseSource = 'pso.js sampleNoise';
    if (!Array.isArray(noise) || noise.length !== anchors.length || !noise.every(Number.isFinite)) {
      noise = localSampleNoise(noiseModel, trueRanges, api.mulberry32(seed ^ 0x9E3779B9));
      noiseSource = 'app.js fallback';
    }
    const measured = trueRanges.map((d, j) => d + noise[j]);
    return { field: upper.slice(), upper, lower: [0, 0, 0], anchors, offsets, p, seed, trueRanges, noiseModel, noise, noiseSource, measured };
  }

  function validate(s, N, T) {
    const msgs = [];
    const [fx, fy, fz] = s.upper;
    const fieldOk = fx > 0 && fy > 0 && fz > 0;
    if (!fieldOk) msgs.push('Field sizes X, Y and Z must be positive numbers.');
    if (s.anchors.length < 4) {
      msgs.push('3D localization needs at least four anchors (' + s.anchors.length + ' given): press “+ anchor”.');
    } else if (api.isNonCoplanar && safe(() => api.isNonCoplanar(s.anchors), true) === false) {
      msgs.push('The anchors are coplanar (or nearly so), so a position above and one below their plane fit the ranges equally well. Move at least one anchor to a different height.');
    }
    if (fieldOk) {
      const inside = (q) => q.every((v, k) => Number.isFinite(v) && v >= 0 && v <= s.upper[k]);
      const box = '0…' + fx + ' × 0…' + fy + ' × 0…' + fz + ' m';
      s.anchors.forEach((a, j) => { if (!inside(a)) msgs.push('Anchor a' + (j + 1) + ' = (' + a.join(', ') + ') lies outside the field ' + box + '.'); });
      if (!inside(s.p)) msgs.push('The true node p = (' + s.p.join(', ') + ') lies outside the field ' + box + '.');
    }
    if (!(N >= 2)) msgs.push('Particles N must be an integer ≥ 2.');
    if (!(T >= 1)) msgs.push('Iterations T must be an integer ≥ 1.');
    return msgs;
  }
  function showAlerts(msgs) {
    const box = $('alerts');
    box.textContent = '';
    msgs.forEach((m) => { const d = document.createElement('div'); d.className = 'alert'; d.textContent = m; box.appendChild(d); });
  }
  function setEnabled(ok) {
    ['step', 'play', 'finish', 'exportCsv', 'exportJson'].forEach((id) => { $(id).disabled = !ok; });
  }

  // ---- swarm configuration ----
  function readStopping() {
    const type = $('stopType').value;
    if (type === 'patience') return { type: 'patience', patience: Math.max(1, int('stopPatience', 10)) };
    if (type === 'tolerance') return { type: 'tolerance', tol: Math.max(0, num('stopTol', 0.01)), window: Math.max(1, int('stopWindow', 10)), relative: true };
    if (type === 'budget') return { type: 'budget', evaluations: Math.max(1, int('stopBudget', 600)) };
    return null;
  }
  function buildCfg(s, N, T) {
    const variant = $('variant').value;
    const cfg = {
      N, T, w: num('w', 0.7), c: num('c', 1.4), c1: num('c1', 1.4), seed: s.seed, clip: $('clip').checked,
      lower: s.lower.slice(), upper: s.upper.slice(), variant, stopping: readStopping(), x0: x0 ? x0.slice() : null,
    };
    if (variant === 'amcmpso') cfg.amcmpso = Object.assign({}, AMCMPSO_DEFAULTS, { schedule: $('amcSchedule').value });
    return cfg;
  }
  function computeGeometry(s) {
    if (s.anchors.length < 4) return null;
    const sigma = effectiveSigma(s);
    const g = {
      sigma,
      nonCoplanar: api.isNonCoplanar ? safe(() => api.isNonCoplanar(s.anchors)) : null,
      gdop: api.gdop ? safe(() => api.gdop(s.anchors, s.p), NaN) : NaN,
      crlb: api.crlb ? safe(() => api.crlb(s.anchors, s.p, sigma)) : null,
      flipRisk: api.flipAmbiguityRisk ? safe(() => api.flipAmbiguityRisk(s.anchors, s.p, s.measured), NaN) : NaN,
      mirror: api.mirrorPoint ? vec3(safe(() => api.mirrorPoint(s.anchors, s.p))) : null,
    };
    if (typeof g.nonCoplanar !== 'boolean') g.nonCoplanar = null;
    if (typeof g.gdop !== 'number') g.gdop = NaN;
    if (typeof g.flipRisk !== 'number') g.flipRisk = NaN;
    if (!g.crlb || typeof g.crlb.rmseBound !== 'number') g.crlb = null;
    return g;
  }

  // ---- run control ----
  function reset() {
    stop();
    scenario = readScenario();
    const N = int('np'), T = int('nt');
    const problems = validate(scenario, N, T);
    if (!api.Swarm || !api.Fitness) problems.push('pso.js did not load, so the optimiser is unavailable. Reload the page.');
    swarm = null; fitness = null; lsq = null; gn = null; x0 = null;
    geo = safe(() => computeGeometry(scenario));
    valid = problems.length === 0;
    if (valid) {
      const s = scenario;
      fitness = new api.Fitness(s.anchors, s.measured);
      lsq = api.leastSquares ? vec3(safe(() => api.leastSquares(s.anchors, s.measured))) : null;
      gn = api.gaussNewton && lsq ? safe(() => api.gaussNewton(s.anchors, s.measured, lsq)) : null;
      if (gn && !vec3(gn.p)) gn = null;
      if ($('warm').checked && api.warmStart) x0 = vec3(safe(() => api.warmStart(s.anchors, s.measured, s.lower, s.upper)));
      swarm = safe(() => new api.Swarm(buildCfg(s, N, T), fitness));
      if (!swarm) {
        problems.push('The optimiser rejected this configuration (the variant or stopping rule may be unsupported by the loaded pso.js).');
        valid = false; fitness = null;
      }
    }
    showAlerts(problems);
    setEnabled(valid);
    draw();
  }
  const isDone = () => !swarm || swarm.done === true || swarm.t >= swarm.cfg.T;
  function safeStep() {
    if (!swarm) return false;
    let r;
    try {
      r = swarm.step();
    } catch (e) {
      stop();
      showAlerts(['The optimiser failed at iteration ' + (swarm.t + 1) + ': ' + (e && e.message ? e.message : String(e)) + '. This variant may be unsupported by the loaded pso.js.']);
      valid = false; setEnabled(false);
      return false;
    }
    return !(r === false || isDone());
  }
  function stepOnce() {
    if (!valid || isDone()) { stop(); return false; }
    const more = safeStep();
    draw();
    if (!more) stop();
    return more;
  }
  function play() {
    if (timer) { stop(); return; }
    if (!valid) return;
    if (isDone()) { reset(); if (!valid) return; }
    $('play').textContent = 'Pause';
    $('play').setAttribute('aria-pressed', 'true');
    $('stats').setAttribute('aria-busy', 'true');
    const ms = Math.max(0, num('speed', 120));
    timer = setInterval(() => { if (!stepOnce()) stop(); }, ms);
  }
  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
    $('play').textContent = 'Play';
    $('play').setAttribute('aria-pressed', 'false');
    $('stats').removeAttribute('aria-busy');
  }
  function finish() {
    stop();
    if (!valid || !swarm) return;
    let guard = (swarm.cfg.T | 0) + 2;
    while (!isDone() && guard-- > 0) { if (!safeStep()) break; }
    draw();
  }

  // ---- drawing ----
  function draw() {
    const s = scenario, sw = swarm;
    const est = sw && sw.best ? vec3(sw.best) : null;
    const T = sw ? sw.cfg.T : int('nt', 0);
    const t = sw ? sw.t : 0;
    const stoppedBy = sw && sw.stoppedBy ? String(sw.stoppedBy) : null;
    const title = 'iteration ' + t + ' / ' + T + (stoppedBy ? ' · stopped early: ' + stoppedBy : (sw && t >= T ? ' · completed' : ''));
    $('runStatus').textContent = sw ? title : 'no run (fix the configuration above)';
    const havePlotly = plotlyOk();
    let plotted = false;
    if (havePlotly) {
      try { drawPlots(s, sw, est, title, T); plotted = true; } catch (e) { plotted = false; }
    }
    $('plotFallback').hidden = plotted;
    // Text stand-in only when Plotly is absent; when Plotly exists but a render failed, leave its DOM alone.
    if (!havePlotly) renderTextFallback(s, sw, est, title);
    renderKpis(s, sw, est, T, t, stoppedBy);
  }
  function drawPlots(s, sw, est, title, T) {
    const dark = isDark();
    const font = { color: dark ? DARK.text : '#28251D' };
    const grid = dark ? DARK.grid : '#E5E3DD';
    const anchorColor = dark ? DARK.anchors : COLOR.anchors;   // marker AND label colour of the anchors trace
    const up = s.upper.map((v) => (v > 0 ? v : 1));
    const axis = (text, max) => {
      const a = { range: [0, max], title: { text } };
      if (dark) Object.assign(a, { backgroundcolor: DARK.surface, gridcolor: DARK.grid, zerolinecolor: DARK.grid, showbackground: true });
      return a;
    };
    const marker = (symbol, size, color) => ({ type: 'scatter3d', mode: 'markers', marker: { symbol, size, color } });
    const point = (name, q, symbol, size, color) => Object.assign(marker(symbol, size, color), { name, x: [q[0]], y: [q[1]], z: [q[2]] });
    const data = [
      { type: 'scatter3d', mode: 'markers+text', name: 'anchors', x: s.anchors.map((a) => a[0]), y: s.anchors.map((a) => a[1]), z: s.anchors.map((a) => a[2]),
        text: s.anchors.map((_, j) => 'a' + (j + 1)), textposition: 'top center', textfont: { color: anchorColor }, marker: { symbol: 'square', size: 7, color: anchorColor } },
    ];
    if (sw && Array.isArray(sw.x)) {
      data.push({ type: 'scatter3d', mode: 'markers', name: 'particles', x: sw.x.map((q) => q[0]), y: sw.x.map((q) => q[1]), z: sw.x.map((q) => q[2]), marker: { size: 3.5, color: COLOR.particles, opacity: 0.85 } });
    }
    data.push(point('true node p', s.p, 'diamond', 8, COLOR.node));
    if (est) data.push(point('estimate p̂', est, 'x', 7, dark ? DARK.estimate : COLOR.estimate));
    const trail = sw && Array.isArray(sw.estHistory) ? sw.estHistory : [];
    if (trail.length > 1) {
      data.push({ type: 'scatter3d', mode: 'lines', name: 'best-estimate trail', x: trail.map((q) => q[0]), y: trail.map((q) => q[1]), z: trail.map((q) => q[2]), line: { color: COLOR.trail, width: 2 }, opacity: 0.6 });
    }
    if (lsq) data.push(point('least-squares', lsq, 'circle-open', 7, COLOR.lsq));
    if (gn) data.push(point('LSQ + Gauss-Newton', gn.p, 'diamond-open', 7, COLOR.lsq));
    if (x0) data.push(point('warm start x₀', x0, 'square-open', 8, COLOR.particles));
    if (geo && geo.mirror && geo.flipRisk >= 0.05) data.push(point('mirror image p′ (flip)', geo.mirror, 'diamond-open', 8, COLOR.node));
    const layout = {
      margin: { l: 0, r: 0, t: 30, b: 0 }, paper_bgcolor: 'rgba(0,0,0,0)', font, legend: { x: 0, y: 1, font: { size: 11 } },
      title: { text: title, font: { size: 14 }, x: 0.02 }, uirevision: 'keep',
      scene: { xaxis: axis('x [m]', up[0]), yaxis: axis('y [m]', up[1]), zaxis: axis('z [m]', up[2]),
        aspectmode: 'manual', aspectratio: { x: 1, y: up[1] / up[0], z: up[2] / up[0] } },
    };
    Plotly.react('plot3d', data, layout, { displaylogo: false, responsive: true });
    const hist = sw && Array.isArray(sw.history) ? sw.history : [];
    const first = sw ? Math.max(0, sw.t - hist.length + 1) : 1;
    const conv = [{ x: hist.map((_, i) => i + first), y: hist, type: 'scatter', mode: 'lines', line: { color: COLOR.particles, width: 2 }, name: 'best fitness' }];
    Plotly.react('plotConv', conv, {
      margin: { l: 55, r: 10, t: 28, b: 38 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font,
      title: { text: 'best fitness f(x_best) per iteration (log scale)', font: { size: 13 }, x: 0.02 },
      xaxis: { title: { text: 'iteration' }, range: [0, T > 0 ? T : 1], gridcolor: grid }, yaxis: { type: 'log', title: { text: 'f [m²]' }, gridcolor: grid }, uirevision: 'keep',
    }, { displaylogo: false, responsive: true });
    $('plot3d').classList.remove('textplot'); $('plotConv').classList.remove('textplot');
  }
  /** Text stand-in for the plots when Plotly is unavailable: KPIs and convergence numbers stay visible. */
  function renderTextFallback(s, sw, est, title) {
    const p3 = $('plot3d'), pc = $('plotConv');
    p3.classList.add('textplot'); pc.classList.add('textplot');
    const v = (q) => q.map((x) => fmt(x, 1)).join(', ');
    p3.textContent = 'anchors: ' + s.anchors.map((a, j) => 'a' + (j + 1) + ' (' + v(a) + ')').join(' · ') + ' · true node p (' + v(s.p) + ')' + (est ? ' · estimate p̂ (' + v(est) + ')' : '');
    const hist = sw && Array.isArray(sw.history) ? sw.history : [];
    const first = sw ? Math.max(0, sw.t - hist.length + 1) : 1;
    const n = hist.length, k = Math.min(10, n), parts = [];
    for (let i = n - k; i < n; i++) parts.push('it ' + (first + i) + ': ' + (Number.isFinite(hist[i]) ? hist[i].toPrecision(4) : '–'));
    pc.textContent = title + (parts.length ? ' · best fitness (last ' + k + '): ' + parts.join(', ') : ' · no iterations yet');
  }
  function renderKpis(s, sw, est, T, t, stoppedBy) {
    const box = $('stats');
    box.textContent = '';
    const add = (label, value, cls) => {
      const k = document.createElement('div'); k.className = 'kpi' + (cls ? ' ' + cls : '');
      const v = document.createElement('div'); v.className = 'v';
      if (value && typeof value === 'object' && value.nodeType === 1) v.appendChild(value); else v.textContent = String(value);
      const l = document.createElement('div'); l.className = 'l'; l.textContent = label;
      k.appendChild(v); k.appendChild(l); box.appendChild(k);
    };
    const badge = (text, ok) => { const b = document.createElement('span'); b.className = 'badge ' + (ok ? 'ok' : 'warn'); b.textContent = text; return b; };
    const g = geo;
    const err = est ? api.dist(est, s.p) : NaN;
    const mem = sw && typeof sw.memoryFloats === 'function' ? safe(() => sw.memoryFloats(), NaN) : NaN;
    add('estimate p̂ (m)', est ? fmt(est[0]) + ', ' + fmt(est[1]) + ', ' + fmt(est[2]) : '–');
    add('error ‖p̂ − p‖', unit(fmt(err), 'm'));
    add('best fitness', unit(sw ? fmt(sw.bestF, 4) : '–', 'm²'));
    add('fitness evaluations', fitness && Number.isFinite(fitness.evals) ? fitness.evals.toLocaleString() : '–');
    add('distance computations', fitness && Number.isFinite(fitness.distances) ? fitness.distances.toLocaleString() : '–');
    add('swarm memory', Number.isFinite(mem) ? mem + ' floats · ' + (mem * 8) + ' B' : '–');
    add('iterations run' + (stoppedBy ? ' · stopped by ' + stoppedBy : (sw && t >= T ? ' · completed' : '')), sw ? t + ' / ' + T : '–');
    add('least-squares error', unit(lsq ? fmt(api.dist(lsq, s.p)) : '–', 'm'));
    add('GN error (LSQ + Gauss-Newton' + (gn && Number.isFinite(gn.iterations) ? ', ' + gn.iterations + ' it.' : '') + ')', unit(gn ? fmt(api.dist(gn.p, s.p)) : '–', 'm'));
    if ($('warm').checked) add('warm start x₀ error', unit(x0 ? fmt(api.dist(x0, s.p)) : '–', 'm'));
    add('GDOP @ p', g ? fmtInf(g.gdop) : '–', g && (g.gdop > 10 || g.gdop === Infinity) ? 'warn' : '');
    add('CRLB bound @ p (σ = ' + (g ? fmt(g.sigma) : '–') + ' m)', unit(g && g.crlb ? fmtInf(g.crlb.rmseBound) : '–', 'm'));
    add('geometry', g && g.nonCoplanar !== null ? badge(g.nonCoplanar ? 'non-coplanar ✓' : 'coplanar ⚠', g.nonCoplanar) : '–');
    add('flip-ambiguity risk @ p', g && Number.isFinite(g.flipRisk) ? fmt(g.flipRisk) : '–', g && g.flipRisk > 0.5 ? 'warn' : '');
    add('measured ranges d̂ (m)', s.measured.map((v) => fmt(v)).join(', '));
  }

  // ---- visibility of dependent controls ----
  function syncVisibility() {
    const st = $('stopType').value;
    $('stopPatienceWrap').hidden = st !== 'patience';
    $('stopToleranceWrap').hidden = st !== 'tolerance';
    $('stopBudgetWrap').hidden = st !== 'budget';
    const nm = $('noiseModel').value;
    $('sigmaWrap').hidden = !(nm === 'gaussian' || nm === 'nlos');
    $('pnWrap').hidden = nm !== 'uniformPercent';
    $('nlosPWrap').hidden = nm !== 'nlos';
    $('nlosBiasWrap').hidden = nm !== 'nlos';
    $('nlosSigmaWrap').hidden = nm !== 'nlos';
    $('noiseHint').textContent = HINTS.noise[nm] || '';
    const variant = $('variant').value;
    $('amcmpsoWrap').hidden = variant !== 'amcmpso';
    $('variantHint').textContent = HINTS.variant[variant] || '';
  }

  // ---- share links ----
  function currentState() {
    const st = {
      v: 1,
      field: [num('fx'), num('fy'), num('fz')],
      anchors: readAnchorRows(),
      p: [num('px', 0), num('py', 0), num('pz', 0)],
      variant: $('variant').value,
      seed: int('seed', 1),
      N: int('np'), T: int('nt'), w: num('w'), c: num('c'), c1: num('c1'),
      clip: $('clip').checked,
      stop: { type: $('stopType').value, patience: int('stopPatience', 10), tol: num('stopTol', 0.01), window: int('stopWindow', 10), budget: int('stopBudget', 600) },
      warm: $('warm').checked,
      noise: { model: $('noiseModel').value, sigma: num('sigma', 0.5), pn: num('pn', 2), nlosP: num('nlosP', 0.2), nlosBias: num('nlosBias', 2), nlosSigma: num('nlosSigma', 1) },
    };
    if (st.variant === 'amcmpso') st.amcmpso = { schedule: $('amcSchedule').value };
    return st;
  }
  function applyState(st) {
    renderAnchorRows(st.anchors);
    setVal('fx', st.field[0]); setVal('fy', st.field[1]); setVal('fz', st.field[2]);
    setVal('px', st.p[0]); setVal('py', st.p[1]); setVal('pz', st.p[2]);
    $('variant').value = st.variant;
    setVal('seed', st.seed); setVal('np', st.N); setVal('nt', st.T); setVal('w', st.w); setVal('c', st.c); setVal('c1', st.c1);
    $('clip').checked = !!st.clip;
    $('warm').checked = !!st.warm;
    $('stopType').value = st.stop.type;
    setVal('stopPatience', st.stop.patience); setVal('stopTol', st.stop.tol); setVal('stopWindow', st.stop.window); setVal('stopBudget', st.stop.budget);
    $('noiseModel').value = st.noise.model;
    setVal('sigma', st.noise.sigma); setVal('pn', st.noise.pn); setVal('nlosP', st.noise.nlosP); setVal('nlosBias', st.noise.nlosBias); setVal('nlosSigma', st.noise.nlosSigma);
    if (st.amcmpso && st.amcmpso.schedule) $('amcSchedule').value = st.amcmpso.schedule;
    syncVisibility();
  }
  function encodedState() {
    if (!api.encodeState || !api.decodeState) return null;
    const enc = api.encodeState(currentState());
    return api.decodeState(enc) ? enc : null;
  }
  function shareLink() {
    const status = $('shareStatus');
    if (!api.encodeState || !api.decodeState) { status.textContent = 'Sharing is unavailable (share.js did not load).'; return; }
    const enc = encodedState();
    if (!enc) { status.textContent = 'Fix the highlighted inputs before creating a link.'; return; }
    const base = location.origin && location.origin !== 'null' ? location.origin + location.pathname : location.href.split('#')[0];
    const url = base + '#s=' + enc;
    safe(() => history.replaceState(null, '', '#s=' + enc));
    const field = $('shareUrl');
    field.value = url;
    $('shareUrlWrap').hidden = false;
    const fallback = () => {
      field.focus(); field.select();
      let copied = false;
      try { copied = typeof document.execCommand === 'function' && document.execCommand('copy'); } catch (e) { copied = false; }
      status.textContent = copied ? 'Link copied to the clipboard.' : 'Clipboard access was refused — the link is selected below; press Ctrl+C (⌘C) to copy it.';
    };
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      navigator.clipboard.writeText(url).then(() => { status.textContent = 'Link copied to the clipboard.'; }, fallback);
    } else {
      fallback();
    }
  }

  // ---- exports ----
  function exportCsv() {
    const status = $('shareStatus');
    if (!swarm || !api.historyToCSV || !api.downloadText) { status.textContent = 'CSV export is unavailable.'; return; }
    const ok = api.downloadText('pso3d_run.csv', api.historyToCSV(swarm, scenario), 'text/csv;charset=utf-8');
    status.textContent = ok ? 'Downloaded pso3d_run.csv (' + (swarm.history ? swarm.history.length : 0) + ' iterations).' : 'The browser refused the download.';
  }
  function exportJson() {
    const status = $('shareStatus');
    if (!swarm || !api.runToJSON || !api.downloadText) { status.textContent = 'JSON export is unavailable.'; return; }
    const s = scenario, g = geo;
    const extras = {
      lsq, lsq_error_m: lsq ? api.dist(lsq, s.p) : null,
      gauss_newton: gn ? { p: gn.p, iterations: gn.iterations, converged: gn.converged, error_m: api.dist(gn.p, s.p) } : null,
      warm_start_x0: x0, warm_start_error_m: x0 ? api.dist(x0, s.p) : null,
      geometry: g ? { non_coplanar: g.nonCoplanar, gdop: g.gdop, crlb_sigma: g.sigma, crlb_rmse_bound: g.crlb ? g.crlb.rmseBound : null, crlb_per_axis: g.crlb ? g.crlb.perAxis : null, flip_ambiguity_risk: g.flipRisk, mirror_point: g.mirror } : null,
      true_ranges: s.trueRanges, noise_source: s.noiseSource, share: encodedState(),
      rng: 'mulberry32 (seeded; repeatable in the browser, not identical to the NumPy baseline)',
    };
    const text = JSON.stringify(api.runToJSON(swarm, scenario, fitness, extras), null, 2);
    const ok = api.downloadText('pso3d_run.json', text, 'application/json');
    status.textContent = ok ? 'Downloaded pso3d_run.json.' : 'The browser refused the download.';
  }

  // ---- keyboard ----
  function onKey(e) {
    if (e.defaultPrevented || e.ctrlKey || e.metaKey || e.altKey) return;
    const el = e.target;
    const tag = el && el.tagName ? el.tagName.toLowerCase() : '';
    if (tag === 'input' || tag === 'select' || tag === 'textarea' || tag === 'button' || (el && el.isContentEditable)) return;
    if (e.key === ' ' || e.code === 'Space') { e.preventDefault(); if (!$('play').disabled) play(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); if (!$('step').disabled) stepOnce(); }
    else if (e.key === 'r' || e.key === 'R') { e.preventDefault(); reset(); }
  }

  // ---- wiring ----
  function init() {
    ['fx', 'fy', 'fz', 'px', 'py', 'pz', 'seed', 'np', 'nt', 'w', 'c', 'c1', 'clip', 'warm', 'sigma', 'pn', 'nlosP', 'nlosBias', 'nlosSigma',
      'stopPatience', 'stopTol', 'stopWindow', 'stopBudget', 'amcSchedule'].forEach((id) => $(id).addEventListener('change', reset));
    ['variant', 'stopType', 'noiseModel'].forEach((id) => $(id).addEventListener('change', () => { syncVisibility(); reset(); }));
    $('speed').addEventListener('change', () => { if (timer) { stop(); play(); } });
    $('addAnchor').addEventListener('click', addAnchor);
    $('randomNode').addEventListener('click', randomNode);
    $('reset').addEventListener('click', reset);
    $('step').addEventListener('click', () => { stepOnce(); });
    $('play').addEventListener('click', play);
    $('finish').addEventListener('click', finish);
    $('share').addEventListener('click', shareLink);
    $('exportCsv').addEventListener('click', exportCsv);
    $('exportJson').addEventListener('click', exportJson);
    document.addEventListener('keydown', onKey);
    window.addEventListener('hashchange', () => {
      const st = api.applyHash ? api.applyHash(location.hash) : null;
      if (st) { applyState(st); reset(); }
    });
    if (window.matchMedia) {
      const mq = window.matchMedia('(prefers-color-scheme: dark)');
      const onScheme = () => { if (scenario) draw(); };
      if (typeof mq.addEventListener === 'function') mq.addEventListener('change', onScheme);
      else if (typeof mq.addListener === 'function') mq.addListener(onScheme);
    }
    $('plotFallback').hidden = plotlyOk();
    const shared = api.applyHash ? api.applyHash(location.hash) : null;
    if (shared) applyState(shared); else renderAnchorRows(DEFAULT_ANCHORS);
    syncVisibility();
    reset();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();

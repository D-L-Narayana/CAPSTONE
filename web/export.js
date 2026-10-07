/* CSV / JSON export of a simulator run. historyToCSV and runToJSON are pure (testable in Node);
   downloadText touches the DOM only when it is called and only when a document exists. */
(function (root) {
  'use strict';

  const HEADER = 'iteration,best_fitness,est_x,est_y,est_z,error_m';

  const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
  const arr = (v) => (Array.isArray(v) ? v : []);
  const finite = (v) => typeof v === 'number' && Number.isFinite(v);
  const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  /** 6 significant digits, trailing zeros removed; empty cell for non-finite values. */
  const sig6 = (v) => (finite(v) ? String(Number(v.toPrecision(6))) : '');
  /** JSON-safe deep copy (drops functions/undefined, maps NaN/Infinity to null). */
  function plain(v) {
    if (v === undefined) return null;
    try { return JSON.parse(JSON.stringify(v)); } catch (e) { return null; }
  }
  /** Iteration index of history[0]: 1 for the simplified variant (no initial evaluation), 0 when the
      initial swarm was evaluated (standard / AMCMPSO), derived from swarm.t and the history length. */
  function firstIteration(swarm, n) {
    const t = swarm && finite(swarm.t) ? swarm.t : n;
    return Math.max(0, t - n + 1);
  }

  function historyToCSV(swarm, scenario) {
    const hist = arr(swarm && swarm.history);
    const ests = arr(swarm && swarm.estHistory);
    const p = scenario && Array.isArray(scenario.p) && scenario.p.length === 3 ? scenario.p : null;
    const first = firstIteration(swarm, hist.length);
    const lines = [HEADER];
    for (let i = 0; i < hist.length; i++) {
      const e = Array.isArray(ests[i]) && ests[i].length === 3 ? ests[i] : [NaN, NaN, NaN];
      const err = p && e.every(finite) ? dist(e, p) : NaN;
      lines.push([String(first + i), sig6(hist[i]), sig6(e[0]), sig6(e[1]), sig6(e[2]), sig6(err)].join(','));
    }
    return lines.join('\n') + '\n';
  }

  function configWithoutFunctions(cfg) {
    const out = {};
    if (!isObj(cfg)) return out;
    for (const k of Object.keys(cfg)) {
      if (cfg[k] === undefined || typeof cfg[k] === 'function') continue;
      out[k] = plain(cfg[k]);
    }
    return out;
  }

  function runToJSON(swarm, scenario, fitness, extras) {
    const sw = isObj(swarm) ? swarm : {};
    const sc = isObj(scenario) ? scenario : {};
    const fit = isObj(fitness) ? fitness : {};
    const hist = arr(sw.history);
    const ests = arr(sw.estHistory);
    const est = Array.isArray(sw.best) && sw.best.length === 3 ? sw.best.slice() : null;
    const p = Array.isArray(sc.p) && sc.p.length === 3 ? sc.p : null;
    const errM = est && p && est.every(finite) && p.every(finite) ? dist(est, p) : null;
    let floats = null;
    if (typeof sw.memoryFloats === 'function') {
      try { floats = plain(sw.memoryFloats()); } catch (e) { floats = null; }
    }
    return {
      scenario: {
        field: plain(sc.field !== undefined ? sc.field : sc.upper),
        anchors: plain(sc.anchors),
        noise: plain(sc.noise),
        noise_model: plain(sc.noiseModel),
        p: plain(sc.p),
        measured: plain(sc.measured),
      },
      config: configWithoutFunctions(sw.cfg),
      result: {
        estimate: plain(est),
        error_m: errM,
        best_fitness: finite(sw.bestF) ? sw.bestF : null,
        iterations_run: finite(sw.t) ? sw.t : hist.length,
        stopped_by: typeof sw.stoppedBy === 'string' ? sw.stoppedBy : null,
        fitness_evaluations: finite(fit.evals) ? fit.evals : null,
        distance_computations: finite(fit.distances) ? fit.distances : null,
        swarm_state_floats: floats,
      },
      history: { best_fitness: plain(hist), estimates: plain(ests) },
      extras: isObj(extras) ? plain(extras) : {},
    };
  }

  /** Trigger a client-side download through a Blob URL on a temporary <a download>. Returns true when started. */
  function downloadText(filename, text, mime) {
    if (typeof document === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined' || typeof URL.createObjectURL !== 'function') return false;
    try {
      const blob = new Blob([String(text)], { type: mime || 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.rel = 'noopener';
      a.style.display = 'none';
      document.body.appendChild(a);
      a.click();
      setTimeout(() => { document.body.removeChild(a); URL.revokeObjectURL(url); }, 0);
      return true;
    } catch (e) {
      return false;
    }
  }

  const api = { historyToCSV, runToJSON, downloadText };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis !== 'undefined' ? globalThis : this);

/* PSO 3D localizer — same algorithms as the Python package pso3d (browser port).
 *
 * Plain browser script (no module syntax, no bundler). Node can load it through the
 * module.exports shim at the end of the file (used by tests/js and tests/js/dump.js).
 * Random numbers come from the seeded mulberry32 generator, so a run is repeatable in the
 * browser but does not reproduce the NumPy default_rng streams of the Python package
 * bit-for-bit; everything that does not consume random numbers (fitness, least squares,
 * Gauss-Newton, AMCMPSO coefficient schedule, centre of mass) matches Python (docs/js_parity.md).
 */
'use strict';

/** Seeded PRNG (mulberry32) returning floats in [0,1). */
function mulberry32(seed) {
  let a = (seed >>> 0) || 1;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gaussian(rand) { // Box-Muller
  let u = 0, v = 0; while (u === 0) u = rand(); while (v === 0) v = rand();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

/** Range-error fitness f(p) = sum_j (||p - a_j|| - d_j)^2 with evaluation counting. */
class Fitness {
  constructor(anchors, measured) { this.a = anchors; this.d = measured; this.evals = 0; this.distances = 0; }
  f(p) {
    this.evals++; this.distances += this.a.length;
    let s = 0;
    for (let j = 0; j < this.a.length; j++) { const e = dist(p, this.a[j]) - this.d[j]; s += e * e; }
    return s;
  }
}

/** Default constants of the AMCMPSO interpretation (same values as pso3d/amcmpso.py). */
const AMCMPSO_DEFAULTS = Object.freeze({
  wMax: 0.9, wMin: 0.4, c1Start: 2.0, c1End: 0.5, c2Start: 0.5, c2End: 2.0, cMean: 0.5, cCom: 0.5, schedule: 'linear',
});

/** Coefficients of AMCMPSO iteration t (0-based) out of T: frac = t / max(T, 1); cosine uses s = (1 - cos(pi*frac)) / 2. */
function amcmpsoCoefficients(t, T, opts) {
  const frac = t / Math.max(T, 1);
  const s = opts.schedule === 'cosine' ? (1 - Math.cos(Math.PI * frac)) / 2 : frac;
  return {
    w: opts.wMax - (opts.wMax - opts.wMin) * s,
    c1: opts.c1Start + (opts.c1End - opts.c1Start) * s,
    c2: opts.c2Start + (opts.c2End - opts.c2Start) * s,
  };
}

function clampPoint(p, lower, upper) {
  return [0, 1, 2].map(k => Math.min(upper[k], Math.max(lower[k], p[k])));
}

/** Validates and normalises the stopping-rule descriptor of a Swarm (null / undefined / {type:'none'} = no rule). */
function normaliseStopping(rule) {
  if (rule == null) return null;
  if (typeof rule !== 'object' || typeof rule.type !== 'string') throw new TypeError('stopping must be null or an object with a string "type"');
  const num = (v, name) => {   // numbers or numeric strings (form inputs); anything else is a programming error
    const n = typeof v === 'string' && v.trim() !== '' ? Number(v) : v;
    if (typeof n !== 'number' || !Number.isFinite(n)) throw new TypeError(`stopping rule "${rule.type}" needs a finite number "${name}"`);
    return n;
  };
  const atLeast = (v, min, name) => {   // same validity ranges as pso3d/stopping.py
    if (v < min) throw new RangeError(`stopping rule "${rule.type}": "${name}" must be >= ${min}, got ${v}`);
    return v;
  };
  switch (rule.type) {
    case 'none': return null;
    case 'patience': return { type: 'patience', patience: atLeast(Math.floor(rule.patience == null ? 10 : num(rule.patience, 'patience')), 0, 'patience') };
    case 'tolerance': return {
      type: 'tolerance',
      tol: atLeast(rule.tol == null ? 1e-2 : num(rule.tol, 'tol'), 0, 'tol'),
      window: atLeast(Math.floor(rule.window == null ? 10 : num(rule.window, 'window')), 1, 'window'),
      relative: rule.relative !== false,
    };
    case 'budget': return { type: 'budget', evaluations: atLeast(num(rule.evaluations, 'evaluations'), 0, 'evaluations') };
    case 'target': return { type: 'target', fitness: num(rule.fitness, 'fitness') };
    default: throw new RangeError(`unknown stopping rule type "${rule.type}"`);
  }
}

/**
 * Swarm optimiser driven step by step.
 *
 * cfg: { N, T, w, c, c1, seed, clip, lower, upper,
 *        variant: 'simplified' | 'standard' | 'amcmpso'          (default 'simplified')
 *        stopping: null | {type:'patience', patience} | {type:'tolerance', tol, window, relative}
 *                       | {type:'budget', evaluations} | {type:'target', fitness}
 *        x0: null | [x, y, z]      warm start: particle 0 starts at clip(x0) with zero velocity,
 *                                  after the uniform initial draw (the RNG stream is untouched)
 *        amcmpso: { wMax, wMin, c1Start, c1End, c2Start, c2End, cMean, cCom, schedule } (defaults as in Python) }
 *
 * variants
 *   simplified  Review-1 algorithm: v = w v + c r (x_best - x) with the best particle of the *current* iteration
 *               (no memory); history has one entry per iteration; N*T evaluations for T iterations.
 *   standard    pbest/gbest PSO, synchronous like pso3d/standard_pso.py: all particles move, then all are
 *               evaluated, then pbest and gbest are updated; r1 then r2 are drawn per coordinate.
 *               history[0] is the initial evaluation; N*(T+1) evaluations.
 *   amcmpso     interpretation of Alhasan et al. 2023 (pso3d/amcmpso.py): standard PSO plus attraction to the
 *               swarm mean and the inverse-fitness centre of mass with adaptive w, c1, c2; r1..r4 per coordinate.
 *
 * step() runs one iteration and returns false when the run is finished (t >= T or a stopping rule fired):
 *   while (swarm.step()) {}   runs to the end.  Fields: t, best, bestF, history, estHistory, done, stoppedBy.
 * Stopping rules are checked after each iteration: patience = best-ever fitness not improved by more than 1e-12
 * for `patience` consecutive iterations (patience 0 fires after the first iteration, as pso3d.stopping.Patience(0));
 * tolerance = best-ever fitness improved by <= tol*f_old (relative) or <= tol (absolute) over the last `window`
 * iterations; budget = fitness.evals >= evaluations; target = bestF <= fitness.
 * Malformed descriptors (unknown type, non-numeric or out-of-range parameters) throw TypeError / RangeError.
 */
class Swarm {
  constructor(cfg, fitness) {
    this.cfg = cfg; this.fit = fitness; this.rand = mulberry32(cfg.seed);
    const { N, lower, upper } = cfg;
    const variant = cfg.variant == null ? 'simplified' : cfg.variant;
    if (variant !== 'simplified' && variant !== 'standard' && variant !== 'amcmpso') throw new RangeError(`unknown swarm variant "${variant}"`);
    this.variant = variant;
    this.x = []; this.v = [];
    for (let i = 0; i < N; i++) {
      this.x.push([0, 1, 2].map(k => lower[k] + this.rand() * (upper[k] - lower[k])));
      this.v.push([0, 0, 0]);
    }
    if (cfg.x0 != null && N > 0) {
      if (!Array.isArray(cfg.x0) || cfg.x0.length !== 3 || !cfg.x0.every(Number.isFinite)) throw new TypeError('x0 must be an array of three finite numbers');
      this.x[0] = clampPoint(cfg.x0, lower, upper); this.v[0] = [0, 0, 0];
    }
    this.t = 0; this.best = null; this.bestF = Infinity; this.history = []; this.estHistory = [];
    this.done = !(cfg.T > 0); this.stoppedBy = null;
    this.rule = normaliseStopping(cfg.stopping);
    this.bestEver = Infinity; this.stall = 0; this.bestEverHistory = [];
    this.amc = Object.assign({}, AMCMPSO_DEFAULTS, cfg.amcmpso || {});
    if (variant !== 'simplified') {
      const f = this.x.map(p => this.fit.f(p));
      this.fx = f;
      this.pbest = this.x.map(p => p.slice()); this.pbestF = f.slice();
      const g = argmin(f); this.best = this.x[g].slice(); this.bestF = f[g];
      this.history.push(this.bestF); this.estHistory.push(this.best.slice());
    }
  }
  /** Floats of swarm state per node: simplified 6N (x, v); standard 9N + 3 (+ pbest, gbest); amcmpso 9N + 6 (+ mean, centre of mass). */
  memoryFloats() {
    if (this.variant === 'amcmpso') return this.cfg.N * 9 + 6;
    if (this.variant === 'standard') return this.cfg.N * 9 + 3;
    return this.cfg.N * 6;
  }
  /** One iteration; returns false when finished (no-op once done). */
  step() {
    if (this.done) return false;
    if (this.variant === 'simplified') this.stepSimplified();
    else if (this.variant === 'standard') this.stepStandard();
    else this.stepAmcmpso();
    this.t++;
    this.checkStopping();
    if (!this.done && this.t >= this.cfg.T) this.done = true;
    return !this.done;
  }
  stepSimplified() {
    const { N, w, c, clip, lower, upper } = this.cfg;
    const f = this.x.map(p => this.fit.f(p));
    const b = argmin(f); this.best = this.x[b].slice(); this.bestF = f[b];
    this.history.push(this.bestF); this.estHistory.push(this.best.slice());
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) {
      this.v[i][k] = w * this.v[i][k] + c * this.rand() * (this.best[k] - this.x[i][k]);
      this.x[i][k] += this.v[i][k];
      if (clip) this.x[i][k] = Math.min(upper[k], Math.max(lower[k], this.x[i][k]));
    }
  }
  stepStandard() {
    const { N, w, c, c1, clip, lower, upper } = this.cfg;
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) {
      const r1 = this.rand(), r2 = this.rand();
      this.v[i][k] = w * this.v[i][k] + c1 * r1 * (this.pbest[i][k] - this.x[i][k]) + c * r2 * (this.best[k] - this.x[i][k]);
      this.x[i][k] += this.v[i][k];
      if (clip) this.x[i][k] = Math.min(upper[k], Math.max(lower[k], this.x[i][k]));
    }
    this.evaluateAndRecord();
  }
  stepAmcmpso() {
    const { N, clip, lower, upper } = this.cfg;
    const a = this.amc;
    const { w, c1, c2 } = amcmpsoCoefficients(this.t, this.cfg.T, a);
    const mean = swarmMean(this.x);
    const com = centreOfMass(this.x, this.fx);
    for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) {
      const r1 = this.rand(), r2 = this.rand(), r3 = this.rand(), r4 = this.rand();
      this.v[i][k] = w * this.v[i][k] + c1 * r1 * (this.pbest[i][k] - this.x[i][k]) + c2 * r2 * (this.best[k] - this.x[i][k])
        + a.cMean * r3 * (mean[k] - this.x[i][k]) + a.cCom * r4 * (com[k] - this.x[i][k]);
      this.x[i][k] += this.v[i][k];
      if (clip) this.x[i][k] = Math.min(upper[k], Math.max(lower[k], this.x[i][k]));
    }
    this.evaluateAndRecord();
  }
  /** Synchronous evaluation of all particles, then pbest, then gbest (as in the Python optimisers). */
  evaluateAndRecord() {
    const f = this.x.map(p => this.fit.f(p));
    this.fx = f;
    for (let i = 0; i < this.cfg.N; i++) if (f[i] < this.pbestF[i]) { this.pbestF[i] = f[i]; this.pbest[i] = this.x[i].slice(); }
    const g = argmin(this.pbestF);
    if (this.pbestF[g] < this.bestF) { this.bestF = this.pbestF[g]; this.best = this.pbest[g].slice(); }
    this.history.push(this.bestF); this.estHistory.push(this.best.slice());
  }
  checkStopping() {
    const improved = this.bestF < this.bestEver - 1e-12;
    if (improved) { this.bestEver = this.bestF; this.stall = 0; } else this.stall++;
    this.bestEverHistory.push(this.bestEver);
    const rule = this.rule;
    if (!rule) return;
    let fired = false;
    if (rule.type === 'patience') {
      fired = this.stall >= rule.patience;
    } else if (rule.type === 'tolerance') {
      const h = this.bestEverHistory, n = h.length;
      if (n > rule.window) {
        const fOld = h[n - 1 - rule.window], fNew = h[n - 1];
        fired = fOld - fNew <= (rule.relative ? rule.tol * fOld : rule.tol);
      }
    } else if (rule.type === 'budget') {
      fired = this.fit.evals >= rule.evaluations;
    } else if (rule.type === 'target') {
      fired = this.bestF <= rule.fitness;
    }
    if (fired) { this.done = true; this.stoppedBy = rule.type; }
  }
}
function argmin(arr) { let b = 0; for (let i = 1; i < arr.length; i++) if (arr[i] < arr[b]) b = i; return b; }

function swarmMean(x) {
  const m = [0, 0, 0];
  for (const p of x) for (let k = 0; k < 3; k++) m[k] += p[k];
  return m.map(v => v / x.length);
}

/** Inverse-fitness weighted centre of mass (weights 1 / (f_i + 1e-9), as in pso3d/amcmpso.py). */
function centreOfMass(x, f) {
  let wsum = 0; const c = [0, 0, 0];
  for (let i = 0; i < x.length; i++) {
    const w = 1 / (f[i] + 1e-9);
    wsum += w;
    for (let k = 0; k < 3; k++) c[k] += w * x[i][k];
  }
  return [c[0] / wsum, c[1] / wsum, c[2] / wsum];
}

/** AMCMPSO coefficient schedule for T iterations (t = 0..T-1): {w, c1, c2} arrays, opts as cfg.amcmpso (defaults merged). */
function amcmpsoSchedule(T, opts) {
  const a = Object.assign({}, AMCMPSO_DEFAULTS, opts || {});
  const out = { w: [], c1: [], c2: [] };
  for (let t = 0; t < T; t++) {
    const c = amcmpsoCoefficients(t, T, a);
    out.w.push(c.w); out.c1.push(c.c1); out.c2.push(c.c2);
  }
  return out;
}

/**
 * Additive ranging noise for the true ranges, one value per anchor, drawn from rand() (mulberry32).
 * model: {type:'fixed', offsets}             n_j = offsets[j] (0 when missing)
 *        {type:'gaussian', sigma}            n_j = sigma * N(0,1)                (Box-Muller via gaussian(rand))
 *        {type:'uniformPercent', pn}         n_j = d_j * (2u - 1) * pn / 100     (Kulkarni: d +- d*Pn/100)
 *        {type:'nlos', p, biasMean, biasSigma, sigma}
 *                                            n_j = sigma * N(0,1) + (with probability p) |biasMean + biasSigma * N(0,1)|
 * Draw order per anchor for nlos: base gaussian, then the uniform for the NLOS decision, then the bias gaussian.
 */
function sampleNoise(model, trueRanges, rand) {
  const type = model && model.type;
  const M = trueRanges.length;
  const out = new Array(M);
  if (type === 'fixed') {
    const off = model.offsets || [];
    for (let j = 0; j < M; j++) out[j] = Number.isFinite(off[j]) ? off[j] : 0;
  } else if (type === 'gaussian') {
    const s = model.sigma;
    for (let j = 0; j < M; j++) out[j] = s ? s * gaussian(rand) : 0;
  } else if (type === 'uniformPercent') {
    const pn = model.pn;
    for (let j = 0; j < M; j++) out[j] = trueRanges[j] * (2 * rand() - 1) * pn / 100;
  } else if (type === 'nlos') {
    const s = model.sigma == null ? 0.5 : model.sigma;
    const p = model.p == null ? 0 : model.p;
    const bm = model.biasMean == null ? 0 : model.biasMean;
    const bs = model.biasSigma == null ? 0 : model.biasSigma;
    for (let j = 0; j < M; j++) {
      const base = s ? s * gaussian(rand) : 0;
      const u = rand();
      const bias = u < p ? Math.abs(bm + (bs ? bs * gaussian(rand) : 0)) : 0;
      out[j] = base + bias;
    }
  } else {
    throw new RangeError(`unknown noise model type "${type}"`);
  }
  return out;
}

/** Linearised least-squares trilateration (for comparison). */
function leastSquares(anchors, d) {
  const a0 = anchors[0], m = anchors.length; const A = [], b = [];
  for (let j = 1; j < m; j++) {
    A.push([2 * (anchors[j][0] - a0[0]), 2 * (anchors[j][1] - a0[1]), 2 * (anchors[j][2] - a0[2])]);
    b.push(sq(anchors[j]) - sq(a0) - d[j] * d[j] + d[0] * d[0]);
  }
  // normal equations (A^T A) p = A^T b, 3x3 solve
  const AtA = [[0,0,0],[0,0,0],[0,0,0]], Atb = [0,0,0];
  for (let r = 0; r < A.length; r++) for (let i = 0; i < 3; i++) { Atb[i] += A[r][i] * b[r]; for (let k = 0; k < 3; k++) AtA[i][k] += A[r][i] * A[r][k]; }
  return solve3(AtA, Atb);
}
const sq = p => p[0]*p[0] + p[1]*p[1] + p[2]*p[2];
function solve3(M, y) { // Gaussian elimination with partial pivoting
  const A = M.map((r, i) => [...r, y[i]]);
  for (let c = 0; c < 3; c++) {
    let piv = c; for (let r = c + 1; r < 3; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]];
    if (Math.abs(A[c][c]) < 1e-12) return [NaN, NaN, NaN];
    for (let r = 0; r < 3; r++) if (r !== c) { const f = A[r][c] / A[c][c]; for (let k = c; k < 4; k++) A[r][k] -= f * A[c][k]; }
  }
  return [A[0][3] / A[0][0], A[1][3] / A[1][1], A[2][3] / A[2][2]];
}

/**
 * Gauss-Newton refinement of sum_j (||p - a_j|| - d_j)^2 from p0 (same maths and iteration counting as
 * pso3d.trilateration.gauss_newton_refine): for k = 0..iterations-1 compute residuals r_j = ||p - a_j|| - d_j and
 * the Jacobian rows (p - a_j)/||p - a_j|| (distances below 1e-12 are clamped), solve the normal equations
 * J^T J s = -J^T r, set p += s and used = k + 1, and stop once |s| < tol.
 * Returns {p, iterations: used, converged: |last step| < tol}; a singular system ends the loop early.
 */
function gaussNewton(anchors, d, p0, iterations = 10, tol = 1e-9) {
  let p = [p0[0], p0[1], p0[2]];
  let used = 0, lastStep = Infinity;
  for (let k = 0; k < iterations; k++) {
    const JtJ = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], Jtr = [0, 0, 0];
    for (let j = 0; j < anchors.length; j++) {
      const diff = [p[0] - anchors[j][0], p[1] - anchors[j][1], p[2] - anchors[j][2]];
      let r = Math.hypot(diff[0], diff[1], diff[2]);
      if (r < 1e-12) r = 1e-12;
      const res = r - d[j];
      const u = [diff[0] / r, diff[1] / r, diff[2] / r];
      for (let i = 0; i < 3; i++) {
        Jtr[i] += u[i] * res;
        for (let l = 0; l < 3; l++) JtJ[i][l] += u[i] * u[l];
      }
    }
    const step = solve3(JtJ, [-Jtr[0], -Jtr[1], -Jtr[2]]);
    if (!step.every(Number.isFinite)) break;
    p = [p[0] + step[0], p[1] + step[1], p[2] + step[2]];
    used = k + 1;
    lastStep = Math.hypot(step[0], step[1], step[2]);
    if (lastStep < tol) break;
  }
  return { p, iterations: used, converged: lastStep < tol };
}

function anchorCentroid(anchors) {
  const c = [0, 0, 0];
  for (const a of anchors) for (let k = 0; k < 3; k++) c[k] += a[k];
  return c.map(v => v / anchors.length);
}

/** Warm-start estimate: least squares -> Gauss-Newton (10 iterations) -> clip to [lower, upper]; the plain anchor centroid replaces a singular least-squares start. */
function warmStart(anchors, d, lower, upper) {
  const lsq = leastSquares(anchors, d);
  const start = lsq.every(Number.isFinite) ? lsq : anchorCentroid(anchors);
  return clampPoint(gaussNewton(anchors, d, start, 10, 1e-9).p, lower, upper);
}

if (typeof module !== 'undefined' && module.exports) module.exports = { mulberry32, gaussian, dist, Fitness, Swarm, argmin, leastSquares, solve3, gaussNewton, warmStart, sampleNoise, amcmpsoSchedule, centreOfMass };

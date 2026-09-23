/* PSO 3D localizer — same algorithms as the Python package pso3d (browser port). */
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

/** Range-error fitness with evaluation counting. */
class Fitness {
  constructor(anchors, measured) { this.a = anchors; this.d = measured; this.evals = 0; this.distances = 0; }
  f(p) {
    this.evals++; this.distances += this.a.length;
    let s = 0;
    for (let j = 0; j < this.a.length; j++) { const e = dist(p, this.a[j]) - this.d[j]; s += e * e; }
    return s;
  }
}

/** Swarm optimiser; variant = 'simplified' (best-of-iteration, no memory) or 'standard' (pbest/gbest). */
class Swarm {
  constructor(cfg, fitness) {
    this.cfg = cfg; this.fit = fitness; this.rand = mulberry32(cfg.seed);
    const { N, lower, upper } = cfg;
    this.x = []; this.v = [];
    for (let i = 0; i < N; i++) {
      this.x.push([0, 1, 2].map(k => lower[k] + this.rand() * (upper[k] - lower[k])));
      this.v.push([0, 0, 0]);
    }
    this.t = 0; this.best = null; this.bestF = Infinity; this.history = []; this.estHistory = [];
    if (cfg.variant === 'standard') {
      const f = this.x.map(p => this.fit.f(p));
      this.pbest = this.x.map(p => p.slice()); this.pbestF = f.slice();
      const g = argmin(f); this.best = this.x[g].slice(); this.bestF = f[g];
      this.history.push(this.bestF); this.estHistory.push(this.best.slice());
    }
  }
  memoryFloats() { return this.cfg.variant === 'standard' ? this.cfg.N * 9 + 3 : this.cfg.N * 6; }
  step() {
    const { N, w, c, c1, clip, lower, upper, variant } = this.cfg;
    if (variant === 'simplified') {
      const f = this.x.map(p => this.fit.f(p));
      const b = argmin(f); this.best = this.x[b].slice(); this.bestF = f[b];
      this.history.push(this.bestF); this.estHistory.push(this.best.slice());
      for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) {
        this.v[i][k] = w * this.v[i][k] + c * this.rand() * (this.best[k] - this.x[i][k]);
        this.x[i][k] += this.v[i][k];
        if (clip) this.x[i][k] = Math.min(upper[k], Math.max(lower[k], this.x[i][k]));
      }
    } else {
      for (let i = 0; i < N; i++) {
        for (let k = 0; k < 3; k++) {
          this.v[i][k] = w * this.v[i][k] + c1 * this.rand() * (this.pbest[i][k] - this.x[i][k]) + c * this.rand() * (this.best[k] - this.x[i][k]);
          this.x[i][k] += this.v[i][k];
          if (clip) this.x[i][k] = Math.min(upper[k], Math.max(lower[k], this.x[i][k]));
        }
        const fi = this.fit.f(this.x[i]);
        if (fi < this.pbestF[i]) { this.pbestF[i] = fi; this.pbest[i] = this.x[i].slice(); if (fi < this.bestF) { this.bestF = fi; this.best = this.x[i].slice(); } }
      }
      this.history.push(this.bestF); this.estHistory.push(this.best.slice());
    }
    this.t++;
  }
}
function argmin(arr) { let b = 0; for (let i = 1; i < arr.length; i++) if (arr[i] < arr[b]) b = i; return b; }

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

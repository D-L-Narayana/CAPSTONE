'use strict';
// Node tests for web/pso.js (plain browser script with a module.exports shim).
// Run: node --test tests/js/
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');

const PSO_JS = path.join(__dirname, '..', '..', 'web', 'pso.js');
const pso = require(PSO_JS);
const { mulberry32, gaussian, dist, Fitness, Swarm, argmin, leastSquares, solve3,
  gaussNewton, warmStart, sampleNoise, amcmpsoSchedule, centreOfMass } = pso;

// Slide-16 baseline scenario (same numbers as pso3d.config.DEFAULT_SCENARIO).
const ANCHORS = [[0, 0, 0], [60, 0, 5], [0, 60, 6], [60, 60, 20]];
const TRUE_P = [37, 12, 8.5];
const NOISE = [0.4, -0.3, 0.5, -0.4];
const LOWER = [0, 0, 0];
const UPPER = [60, 60, 20];
const MEASURED = ANCHORS.map((a, j) => dist(TRUE_P, a) + NOISE[j]);

function makeSwarm(overrides, fit) {
  const cfg = Object.assign({ N: 20, T: 60, w: 0.7, c: 1.4, c1: 1.4, seed: 1, clip: false, lower: LOWER, upper: UPPER, variant: 'simplified' }, overrides || {});
  return new Swarm(cfg, fit || new Fitness(ANCHORS, MEASURED));
}
function runAll(sw) {
  let guard = 0;
  while (sw.step()) { if (++guard > 100000) throw new Error('step() never returned false'); }
  return sw;
}
function nonIncreasing(arr) {
  for (let i = 1; i < arr.length; i++) if (arr[i] > arr[i - 1]) return false;
  return true;
}
const close = (a, b, tol) => Math.abs(a - b) <= tol;

// ---------------------------------------------------------------------------------------------
// Pins captured from web/pso.js before the W08 changes (seed 1, N=20, T=60, w=0.7, c=1.4,
// clip=false, baseline scenario). They must never change: the simplified variant is the browser
// twin of the frozen Review-1 algorithm and its random-draw order is part of the contract.
// ---------------------------------------------------------------------------------------------
const PIN_SIMPLIFIED = {
  historyLength: 60,
  bestF: 0.06639094051757627,
  best: [37.41797054891532, 11.945411712185436, 9.43580577056008],
  evals: 1200,
  distances: 4800,
  history0: 149.2419938499143,
  history10: 4.693774036709874,
};
// captured from web/pso.js before the W08 changes: first five draws of mulberry32(1) and gaussian(mulberry32(1))
const PIN_RNG = {
  draws: [0.6270739405881613, 0.002735721180215478, 0.5274470399599522, 0.9810509674716741, 0.9683778982143849],
  gaussianSeed1: 0.9659740590152261,
};

test('module exports exactly the contract names', () => {
  const expected = ['mulberry32', 'gaussian', 'dist', 'Fitness', 'Swarm', 'argmin', 'leastSquares', 'solve3',
    'gaussNewton', 'warmStart', 'sampleNoise', 'amcmpsoSchedule', 'centreOfMass'];
  assert.deepStrictEqual(Object.keys(pso).sort(), expected.slice().sort());
  for (const name of expected) assert.strictEqual(typeof pso[name], 'function', name);
});

test('pso.js has no DOM access and stays a plain script', () => {
  const src = fs.readFileSync(PSO_JS, 'utf8');
  assert.ok(!/\b(document|window)\s*\./.test(src), 'no document./window. access');
  assert.ok(!/^\s*(import|export)\s/m.test(src), 'no ES module syntax');
  assert.ok(/^'use strict';/m.test(src));
});

test('mulberry32 and gaussian are unchanged (pinned draws)', () => {
  const r = mulberry32(1);
  for (let i = 0; i < PIN_RNG.draws.length; i++) assert.strictEqual(r(), PIN_RNG.draws[i]);
  assert.strictEqual(gaussian(mulberry32(1)), PIN_RNG.gaussianSeed1);
  assert.strictEqual(mulberry32(0)(), mulberry32(1)(), 'seed 0 aliases to 1 (legacy behaviour)');
});

test('pinned simplified run is unchanged (history length, final bestF, estimate, counters)', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  const sw = makeSwarm({}, fit);
  while (sw.t < sw.cfg.T) sw.step();          // legacy driving loop (as web/app.js uses today)
  assert.strictEqual(sw.history.length, PIN_SIMPLIFIED.historyLength);
  assert.ok(close(sw.bestF, PIN_SIMPLIFIED.bestF, 1e-9), `bestF ${sw.bestF}`);
  for (let k = 0; k < 3; k++) assert.ok(close(sw.best[k], PIN_SIMPLIFIED.best[k], 1e-9), `best[${k}] ${sw.best[k]}`);
  assert.strictEqual(fit.evals, PIN_SIMPLIFIED.evals);
  assert.strictEqual(fit.distances, PIN_SIMPLIFIED.distances);
  assert.ok(close(sw.history[0], PIN_SIMPLIFIED.history0, 1e-9));
  assert.ok(close(sw.history[10], PIN_SIMPLIFIED.history10, 1e-9));
  assert.strictEqual(sw.estHistory.length, 60);
  assert.strictEqual(sw.memoryFloats(), 6 * 20);
});

test('pinned simplified run is identical when driven by the step() return value', () => {
  const sw = runAll(makeSwarm({}));
  assert.strictEqual(sw.t, 60);
  assert.strictEqual(sw.history.length, 60);
  assert.ok(close(sw.bestF, PIN_SIMPLIFIED.bestF, 1e-9));
  for (let k = 0; k < 3; k++) assert.ok(close(sw.best[k], PIN_SIMPLIFIED.best[k], 1e-9));
});

test('step() returns a boolean, sets done/stoppedBy and becomes a no-op once finished', () => {
  const sw = makeSwarm({ T: 5 });
  assert.strictEqual(sw.done, false);
  assert.strictEqual(sw.stoppedBy, null);
  const flags = [];
  for (let i = 0; i < 5; i++) flags.push(sw.step());
  assert.deepStrictEqual(flags, [true, true, true, true, false]);
  assert.strictEqual(sw.t, 5);
  assert.strictEqual(sw.done, true);
  assert.strictEqual(sw.stoppedBy, null, 'natural completion leaves stoppedBy null');
  const evalsBefore = sw.fit.evals;
  assert.strictEqual(sw.step(), false);
  assert.strictEqual(sw.t, 5, 'no further iteration after done');
  assert.strictEqual(sw.fit.evals, evalsBefore);
  assert.strictEqual(sw.history.length, 5);
});

test('Fitness keeps the legacy fields (a, d, evals, distances) and f(p)', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  assert.strictEqual(fit.a, ANCHORS);
  assert.strictEqual(fit.d, MEASURED);
  const f0 = fit.f(TRUE_P);
  assert.ok(close(f0, NOISE.reduce((s, n) => s + n * n, 0), 1e-12));
  assert.strictEqual(fit.evals, 1);
  assert.strictEqual(fit.distances, 4);
});

test('argmin, solve3 and leastSquares behave as before', () => {
  assert.strictEqual(argmin([3, 1, 2]), 1);
  assert.strictEqual(argmin([1, 1, 0.5]), 2);
  const s = solve3([[2, 0, 0], [0, 4, 0], [0, 0, 8]], [2, 4, 8]);
  assert.deepStrictEqual(s, [1, 1, 1]);
  assert.ok(solve3([[1, 2, 3], [2, 4, 6], [1, 1, 1]], [1, 2, 3]).every(Number.isNaN), 'singular -> NaN');
  const lsq = leastSquares(ANCHORS, MEASURED);
  // Python least_squares_trilateration on the baseline: (37.134489655966945, 11.444313462124539, 11.648976790180622)
  assert.ok(close(lsq[0], 37.134489655966945, 1e-6));
  assert.ok(close(lsq[1], 11.444313462124539, 1e-6));
  assert.ok(close(lsq[2], 11.648976790180622, 1e-6));
});

// ------------------------------------------------------------------ standard (synchronous) variant
test('standard variant: bestF non-increasing, history T+1, N*(T+1) evaluations, error < 1.5 m (seed 1)', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  const sw = runAll(makeSwarm({ variant: 'standard' }, fit));
  assert.strictEqual(sw.history.length, 61);
  assert.strictEqual(sw.estHistory.length, 61);
  assert.ok(nonIncreasing(sw.history), 'gbest fitness never increases');
  assert.ok(dist(sw.best, TRUE_P) < 1.5, `error ${dist(sw.best, TRUE_P)}`);
  assert.strictEqual(fit.evals, 20 * 61);
  assert.strictEqual(sw.memoryFloats(), 9 * 20 + 3);
  assert.ok(Array.isArray(sw.pbest) && sw.pbest.length === 20);
  assert.ok(Array.isArray(sw.pbestF) && sw.pbestF.length === 20);
});

test('standard variant is synchronous: every particle has moved before any is evaluated', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  const sw = makeSwarm({ variant: 'standard', N: 6 }, fit);
  const before = sw.x.map(p => p.slice());
  // the initial gbest particle has pbest == x == gbest, so its first velocity is exactly zero: it legitimately stays put
  const g = sw.x.findIndex(p => p[0] === sw.best[0] && p[1] === sw.best[1] && p[2] === sw.best[2]);
  assert.ok(g >= 0);
  let snapshot = null;
  const orig = fit.f.bind(fit);
  fit.f = p => { if (snapshot === null) snapshot = sw.x.map(q => q.slice()); return orig(p); };
  sw.step();
  assert.ok(snapshot !== null, 'particles were evaluated');
  for (let i = 0; i < 6; i++) {
    if (i !== g) assert.notDeepStrictEqual(snapshot[i], before[i], `particle ${i} had already moved at the first evaluation`);
    assert.deepStrictEqual(snapshot[i], sw.x[i], `particle ${i} did not move again after the evaluation`);
  }
  assert.strictEqual(fit.evals, 6 * 2, 'initial evaluation plus one synchronous sweep');
});

test('standard variant draws r1 then r2 per coordinate (one iteration consumes 2*N*3 draws)', () => {
  const sw = makeSwarm({ variant: 'standard', N: 4, T: 3 });
  const probe = mulberry32(1);
  for (let i = 0; i < 4 * 3; i++) probe();            // initial uniform positions
  sw.step();
  // after one iteration the swarm RNG must be exactly 2*N*3 draws ahead of the probe
  for (let i = 0; i < 2 * 4 * 3; i++) probe();
  assert.strictEqual(sw.rand(), probe());
});

// ------------------------------------------------------------------------------- amcmpso variant
test('amcmpso: error < 1.5 m for seeds 1..5, N*(T+1) evaluations, 9N+6 floats, monotone gbest', () => {
  for (let seed = 1; seed <= 5; seed++) {
    const fit = new Fitness(ANCHORS, MEASURED);
    const sw = runAll(makeSwarm({ variant: 'amcmpso', seed, clip: true }, fit));
    const err = dist(sw.best, TRUE_P);
    assert.ok(err < 1.5, `seed ${seed}: error ${err}`);
    assert.strictEqual(fit.evals, 20 * 61, `seed ${seed}: evaluations`);
    assert.strictEqual(sw.memoryFloats(), 9 * 20 + 6);
    assert.strictEqual(sw.history.length, 61);
    assert.ok(nonIncreasing(sw.history));
    assert.strictEqual(sw.stoppedBy, null);
  }
});

test('amcmpso: cosine schedule and custom constants run and converge (seed 1)', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  const sw = runAll(makeSwarm({ variant: 'amcmpso', clip: true, amcmpso: { schedule: 'cosine', cMean: 0.3, cCom: 0.7 } }, fit));
  assert.ok(dist(sw.best, TRUE_P) < 1.5);
  assert.strictEqual(fit.evals, 20 * 61);
  // defaults are merged, not replaced
  assert.strictEqual(sw.amc.wMax, 0.9);
  assert.strictEqual(sw.amc.cMean, 0.3);
  assert.strictEqual(sw.amc.schedule, 'cosine');
});

test('amcmpso: one iteration consumes 4*N*3 draws (r1..r4 per coordinate)', () => {
  const sw = makeSwarm({ variant: 'amcmpso', N: 3, T: 2 });
  const probe = mulberry32(1);
  for (let i = 0; i < 3 * 3; i++) probe();
  sw.step();
  for (let i = 0; i < 4 * 3 * 3; i++) probe();
  assert.strictEqual(sw.rand(), probe());
});

// ------------------------------------------------------------------------------------- stopping
test('stopping budget: simplified stops at exactly 600 evaluations (30 iterations)', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  const sw = runAll(makeSwarm({ stopping: { type: 'budget', evaluations: 600 } }, fit));
  assert.strictEqual(fit.evals, 600);
  assert.strictEqual(sw.t, 30);
  assert.strictEqual(sw.stoppedBy, 'budget');
  assert.strictEqual(sw.done, true);
  assert.strictEqual(sw.history.length, 30);
});

test('stopping budget: standard stops at exactly 600 evaluations (initial 20 + 29 iterations)', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  const sw = runAll(makeSwarm({ variant: 'standard', stopping: { type: 'budget', evaluations: 600 } }, fit));
  assert.strictEqual(fit.evals, 600);
  assert.strictEqual(sw.t, 29);
  assert.strictEqual(sw.stoppedBy, 'budget');
});

test('stopping patience fires on a plateau of the best-ever fitness', () => {
  const sw = runAll(makeSwarm({ variant: 'standard', T: 400, stopping: { type: 'patience', patience: 10 } }));
  assert.strictEqual(sw.stoppedBy, 'patience');
  assert.ok(sw.t < 400, `stopped at ${sw.t}`);
  assert.ok(sw.t >= 11);
  // the last 10 recorded gbest values did not improve by more than 1e-12 on the value before them
  const h = sw.history;
  const ref = h[h.length - 11];
  for (let i = h.length - 10; i < h.length; i++) assert.ok(h[i] >= ref - 1e-12);
});

test('stopping patience 0 fires after the first iteration (Python Patience(0) semantics)', () => {
  const sw = runAll(makeSwarm({ variant: 'standard', stopping: { type: 'patience', patience: 0 } }));
  assert.strictEqual(sw.t, 1);
  assert.strictEqual(sw.stoppedBy, 'patience');
  const simple = runAll(makeSwarm({ stopping: { type: 'patience', patience: 0 } }));
  assert.strictEqual(simple.t, 1);
  assert.strictEqual(simple.fit.evals, 20);
});

test('stopping tolerance (relative and absolute) fires and sets stoppedBy', () => {
  const rel = runAll(makeSwarm({ variant: 'standard', T: 400, stopping: { type: 'tolerance', tol: 1e-2, window: 10, relative: true } }));
  assert.strictEqual(rel.stoppedBy, 'tolerance');
  assert.ok(rel.t < 400 && rel.t > 10, `relative stopped at ${rel.t}`);
  const h = rel.history;
  const fOld = h[h.length - 11], fNew = h[h.length - 1];
  assert.ok(fOld - fNew <= 1e-2 * fOld + 1e-15, 'improvement over the window <= tol * f_old');
  const abs = runAll(makeSwarm({ variant: 'standard', T: 400, stopping: { type: 'tolerance', tol: 1e-4, window: 5, relative: false } }));
  assert.strictEqual(abs.stoppedBy, 'tolerance');
  assert.ok(abs.t < 400 && abs.t > 5, `absolute stopped at ${abs.t}`);
  const h2 = abs.history;
  assert.ok(h2[h2.length - 6] - h2[h2.length - 1] <= 1e-4 + 1e-15);
});

test('stopping target fires as soon as bestF <= target', () => {
  const sw = runAll(makeSwarm({ variant: 'standard', T: 400, stopping: { type: 'target', fitness: 1.0 } }));
  assert.strictEqual(sw.stoppedBy, 'target');
  assert.ok(sw.bestF <= 1.0);
  const h = sw.history;
  for (let i = 0; i < h.length - 1; i++) assert.ok(h[i] > 1.0, `iteration ${i} was already below target`);
  assert.ok(sw.t < 400);
});

test('stopping rules work for the simplified variant too and never change its draw order', () => {
  const sw = runAll(makeSwarm({ stopping: { type: 'target', fitness: 5.0 } }));
  assert.strictEqual(sw.stoppedBy, 'target');
  assert.ok(sw.bestF <= 5.0);
  // same per-iteration history prefix as the pinned unstopped run
  const ref = runAll(makeSwarm({}));
  for (let i = 0; i < sw.history.length; i++) assert.strictEqual(sw.history[i], ref.history[i]);
});

test('stopping: numeric strings from form inputs are accepted, bad descriptors throw', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  let sw = null;
  assert.doesNotThrow(() => { sw = makeSwarm({ stopping: { type: 'budget', evaluations: '600' } }, fit); }, 'numeric string budget is accepted');
  runAll(sw);
  assert.strictEqual(fit.evals, 600);
  assert.strictEqual(sw.stoppedBy, 'budget');
  let pat = null;
  assert.doesNotThrow(() => { pat = makeSwarm({ variant: 'standard', T: 400, stopping: { type: 'patience', patience: '10' } }); }, 'numeric string patience is accepted');
  runAll(pat);
  assert.strictEqual(pat.stoppedBy, 'patience');
  assert.throws(() => makeSwarm({ stopping: { type: 'budget', evaluations: 'many' } }), TypeError);
  assert.throws(() => makeSwarm({ stopping: { type: 'budget' } }), TypeError);
  // same validity ranges as pso3d/stopping.py (ValueError there): window >= 1, patience >= 0, tol >= 0, budget >= 0
  assert.throws(() => makeSwarm({ stopping: { type: 'tolerance', window: 0 } }), RangeError);
  assert.throws(() => makeSwarm({ stopping: { type: 'tolerance', tol: -1 } }), RangeError);
  assert.throws(() => makeSwarm({ stopping: { type: 'patience', patience: -1 } }), RangeError);
  assert.throws(() => makeSwarm({ stopping: { type: 'budget', evaluations: -5 } }), RangeError);
  assert.throws(() => makeSwarm({ stopping: { type: 'deadline', seconds: 3 } }), RangeError);
  assert.throws(() => makeSwarm({ stopping: 'patience' }), TypeError);
  assert.throws(() => makeSwarm({ variant: 'quantum' }), RangeError);
  assert.throws(() => makeSwarm({ x0: [1, 2] }), TypeError);
  assert.strictEqual(makeSwarm({ stopping: { type: 'none' } }).rule, null);
});

test('stopping: null / undefined means no rule', () => {
  const a = runAll(makeSwarm({ stopping: null }));
  const b = runAll(makeSwarm({}));
  assert.strictEqual(a.t, 60);
  assert.strictEqual(b.t, 60);
  assert.strictEqual(a.stoppedBy, null);
});

// -------------------------------------------------------------------------------- warm start x0
test('x0 = [37.42, 11.88, 9.36]: standard variant error <= 0.97 m within 20 iterations', () => {
  const fit = new Fitness(ANCHORS, MEASURED);
  const sw = makeSwarm({ variant: 'standard', T: 20, x0: [37.42, 11.88, 9.36] }, fit);
  assert.deepStrictEqual(sw.x[0], [37.42, 11.88, 9.36]);
  assert.deepStrictEqual(sw.v[0], [0, 0, 0]);
  runAll(sw);
  assert.strictEqual(sw.t, 20);
  assert.ok(dist(sw.best, TRUE_P) <= 0.97, `error ${dist(sw.best, TRUE_P)}`);
});

test('x0 is clipped to the field and leaves the RNG stream untouched', () => {
  const plain = makeSwarm({ seed: 7 });
  const warm = makeSwarm({ seed: 7, x0: [-5, 70, 10] });
  assert.deepStrictEqual(warm.x[0], [0, 60, 10]);
  for (let i = 1; i < 20; i++) assert.deepStrictEqual(warm.x[i], plain.x[i]);
  assert.strictEqual(warm.rand(), plain.rand(), 'same number of draws consumed');
  const amc = makeSwarm({ seed: 7, variant: 'amcmpso', x0: [10, 10, 10] });
  assert.deepStrictEqual(amc.x[0], [10, 10, 10]);
  assert.deepStrictEqual(amc.pbest[0], [10, 10, 10]);
});

// ------------------------------------------------------------------------------------- sampleNoise
test('sampleNoise: fixed offsets are returned as given', () => {
  const d = [10, 20, 30, 40];
  assert.deepStrictEqual(sampleNoise({ type: 'fixed', offsets: NOISE }, d, mulberry32(1)), NOISE);
  assert.deepStrictEqual(sampleNoise({ type: 'fixed', offsets: [0.1] }, d, mulberry32(1)), [0.1, 0, 0, 0]);
});

test('sampleNoise: gaussian is deterministic per seed, has the right shape and spread', () => {
  const d = [10, 20, 30, 40];
  const g1 = sampleNoise({ type: 'gaussian', sigma: 0.5 }, d, mulberry32(3));
  const g2 = sampleNoise({ type: 'gaussian', sigma: 0.5 }, d, mulberry32(3));
  assert.deepStrictEqual(g1, g2);
  assert.strictEqual(g1.length, 4);
  assert.ok(g1.some(v => v !== 0));
  assert.deepStrictEqual(sampleNoise({ type: 'gaussian', sigma: 0 }, d, mulberry32(3)), [0, 0, 0, 0]);
  const many = sampleNoise({ type: 'gaussian', sigma: 0.5 }, new Array(20000).fill(10), mulberry32(11));
  const mean = many.reduce((s, v) => s + v, 0) / many.length;
  const std = Math.sqrt(many.reduce((s, v) => s + (v - mean) ** 2, 0) / many.length);
  assert.ok(Math.abs(std - 0.5) < 0.05, `std ${std}`);
  assert.ok(Math.abs(mean) < 0.02, `mean ${mean}`);
});

test('sampleNoise: uniformPercent stays within +-d*pn/100', () => {
  const d = [10, 20, 30, 40];
  const u = sampleNoise({ type: 'uniformPercent', pn: 2 }, d, mulberry32(5));
  assert.strictEqual(u.length, 4);
  u.forEach((v, j) => assert.ok(Math.abs(v) <= d[j] * 0.02 + 1e-12, `anchor ${j}`));
  const many = sampleNoise({ type: 'uniformPercent', pn: 2 }, new Array(5000).fill(100), mulberry32(9));
  assert.ok(Math.max(...many) > 1.5 && Math.min(...many) < -1.5, 'spans most of +-2 m');
});

test('sampleNoise: nlos adds a non-negative bias with probability p', () => {
  const d = [10, 20, 30, 40];
  const none = sampleNoise({ type: 'nlos', p: 0, biasMean: 3, biasSigma: 1, sigma: 0 }, d, mulberry32(2));
  assert.deepStrictEqual(none, [0, 0, 0, 0]);
  const all = sampleNoise({ type: 'nlos', p: 1, biasMean: 3, biasSigma: 0, sigma: 0 }, d, mulberry32(2));
  assert.deepStrictEqual(all, [3, 3, 3, 3]);
  const mixed = sampleNoise({ type: 'nlos', p: 1, biasMean: 3, biasSigma: 1, sigma: 0 }, new Array(2000).fill(10), mulberry32(4));
  assert.ok(mixed.every(v => v >= 0), 'bias is always non-negative');
  const half = sampleNoise({ type: 'nlos', p: 0.5, biasMean: 3, biasSigma: 0, sigma: 0 }, new Array(4000).fill(10), mulberry32(6));
  const frac = half.filter(v => v === 3).length / half.length;
  assert.ok(Math.abs(frac - 0.5) < 0.05, `affected fraction ${frac}`);
  const a = sampleNoise({ type: 'nlos', p: 0.3, biasMean: 2, biasSigma: 0.5, sigma: 0.5 }, d, mulberry32(8));
  const b = sampleNoise({ type: 'nlos', p: 0.3, biasMean: 2, biasSigma: 0.5, sigma: 0.5 }, d, mulberry32(8));
  assert.deepStrictEqual(a, b);
});

test('sampleNoise: unknown model type throws', () => {
  assert.throws(() => sampleNoise({ type: 'laplace' }, [1, 2], mulberry32(1)), /noise model/);
});

// ------------------------------------------------------------------------------- amcmpsoSchedule
test('amcmpsoSchedule: linear endpoints, lengths and monotonicity', () => {
  const T = 60;
  const s = amcmpsoSchedule(T, {});
  assert.deepStrictEqual([s.w.length, s.c1.length, s.c2.length], [T, T, T]);
  assert.ok(close(s.w[0], 0.9, 1e-12));
  assert.ok(close(s.c1[0], 2.0, 1e-12));
  assert.ok(close(s.c2[0], 0.5, 1e-12));
  const fracLast = (T - 1) / T;
  assert.ok(close(s.w[T - 1], 0.9 - (0.9 - 0.4) * fracLast, 1e-12));
  assert.ok(close(s.c1[T - 1], 2.0 + (0.5 - 2.0) * fracLast, 1e-12));
  assert.ok(close(s.c2[T - 1], 0.5 + (2.0 - 0.5) * fracLast, 1e-12));
  assert.ok(nonIncreasing(s.w) && nonIncreasing(s.c1));
  assert.ok(nonIncreasing(s.c2.slice().reverse()), 'c2 non-decreasing');
  assert.ok(s.w[1] < s.w[0] && s.c2[1] > s.c2[0], 'strictly changing');
});

test('amcmpsoSchedule: cosine schedule uses s = (1 - cos(pi*frac))/2 and custom constants', () => {
  const T = 10;
  const s = amcmpsoSchedule(T, { schedule: 'cosine', wMax: 0.8, wMin: 0.2, c1Start: 1.5, c1End: 0.5, c2Start: 0.5, c2End: 1.5 });
  assert.ok(close(s.w[0], 0.8, 1e-12));
  const t = 7, frac = t / T, sm = (1 - Math.cos(Math.PI * frac)) / 2;
  assert.ok(close(s.w[t], 0.8 - (0.8 - 0.2) * sm, 1e-12));
  assert.ok(close(s.c1[t], 1.5 + (0.5 - 1.5) * sm, 1e-12));
  assert.ok(close(s.c2[t], 0.5 + (1.5 - 0.5) * sm, 1e-12));
  assert.ok(nonIncreasing(s.w) && nonIncreasing(s.c1) && nonIncreasing(s.c2.slice().reverse()));
  assert.deepStrictEqual(amcmpsoSchedule(0, {}), { w: [], c1: [], c2: [] });
});

// --------------------------------------------------------------------------------- centreOfMass
test('centreOfMass: inverse-fitness weights, equals the better particle when its fitness is ~0', () => {
  const x = [[1, 2, 3], [10, 20, 30]];
  const com = centreOfMass(x, [0, 1e3]);
  for (let k = 0; k < 3; k++) assert.ok(close(com[k], x[0][k], 1e-6), `axis ${k}: ${com[k]}`);
  const even = centreOfMass(x, [2, 2]);
  for (let k = 0; k < 3; k++) assert.ok(close(even[k], [5.5, 11, 16.5][k], 1e-12), 'equal fitness -> plain mean');
  const w1 = 1 / (1 + 1e-9), w2 = 1 / (3 + 1e-9);
  const expect = [(w1 * 1 + w2 * 10) / (w1 + w2), (w1 * 2 + w2 * 20) / (w1 + w2), (w1 * 3 + w2 * 30) / (w1 + w2)];
  const got = centreOfMass(x, [1, 3]);
  for (let k = 0; k < 3; k++) assert.ok(close(got[k], expect[k], 1e-12));
});

// ------------------------------------------------------------------------------------- dump CLI
test('dump.js prints valid JSON for the three parity fixture cases', () => {
  const dumpPath = path.join(__dirname, 'dump.js');
  const fixture = path.join(__dirname, '..', 'fixtures', 'parity_inputs.json');
  assert.ok(fs.existsSync(fixture), 'tests/fixtures/parity_inputs.json must exist');
  const out = execFileSync(process.execPath, [dumpPath, fixture], { encoding: 'utf8', timeout: 60000 });
  const data = JSON.parse(out);
  assert.deepStrictEqual(Object.keys(data).sort(), ['baseline', 'near_coplanar', 'six_anchors']);
  const keys = ['measured', 'lsq', 'gn', 'fitness', 'gdop', 'crlb_rmse', 'mirror', 'non_coplanar', 'schedule', 'centre_of_mass'];
  for (const id of Object.keys(data)) {
    const c = data[id];
    for (const k of keys) assert.ok(Object.prototype.hasOwnProperty.call(c, k), `${id}.${k}`);
    assert.ok(Array.isArray(c.gn.p) && c.gn.p.length === 3, `${id}.gn.p`);
    assert.strictEqual(typeof c.gn.iterations, 'number');
    assert.strictEqual(typeof c.non_coplanar, 'boolean');
    assert.strictEqual(c.schedule.w.length, 60);
    assert.strictEqual(c.schedule.c1.length, 60);
    assert.strictEqual(c.schedule.c2.length, 60);
    assert.strictEqual(c.centre_of_mass.length, 3);
    assert.strictEqual(c.fitness.length, c.gdop.length);
    assert.strictEqual(c.fitness.length, c.crlb_rmse.length);
    assert.strictEqual(c.fitness.length, c.mirror.length);
  }
  assert.ok(close(data.baseline.measured[0], 40.21519810323691, 1e-9), `measured[0] = ${data.baseline.measured[0]}`);
  assert.strictEqual(data.baseline.non_coplanar, true);
  assert.strictEqual(data.six_anchors.non_coplanar, true);
  // near_coplanar: fourth anchor at z = 0.01 gives a tetrahedron volume of 6 m^3 (> 1e-6) -> non-coplanar as computed
  assert.strictEqual(data.near_coplanar.non_coplanar, true);
  assert.ok(data.baseline.gdop.every(v => typeof v === 'number' && v > 0));
  assert.ok(close(data.baseline.fitness[0], NOISE.reduce((s, n) => s + n * n, 0), 1e-9));
});

test('dump.js exits non-zero without an input path', () => {
  const dumpPath = path.join(__dirname, 'dump.js');
  assert.throws(() => execFileSync(process.execPath, [dumpPath], { encoding: 'utf8', stdio: 'pipe' }));
});

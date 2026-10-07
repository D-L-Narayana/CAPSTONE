'use strict';
// Closed-form solvers in web/pso.js: leastSquares (legacy), gaussNewton, warmStart.
const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');

const { dist, leastSquares, gaussNewton, warmStart, solve3 } = require(path.join(__dirname, '..', '..', 'web', 'pso.js'));

const ANCHORS = [[0, 0, 0], [60, 0, 5], [0, 60, 6], [60, 60, 20]];
const TRUE_P = [37, 12, 8.5];
const NOISE = [0.4, -0.3, 0.5, -0.4];
const LOWER = [0, 0, 0];
const UPPER = [60, 60, 20];
const MEASURED = ANCHORS.map((a, j) => dist(TRUE_P, a) + NOISE[j]);
const EXACT = ANCHORS.map(a => dist(TRUE_P, a));
const close = (a, b, tol) => Math.abs(a - b) <= tol;

test('gaussNewton on noise-free ranges converges to the true point (1e-6) in <= 10 iterations', () => {
  const r = gaussNewton(ANCHORS, EXACT, [30, 30, 10], 10, 1e-9);
  assert.strictEqual(r.converged, true);
  assert.ok(r.iterations >= 1 && r.iterations <= 10, `iterations ${r.iterations}`);
  for (let k = 0; k < 3; k++) assert.ok(close(r.p[k], TRUE_P[k], 1e-6), `axis ${k}: ${r.p[k]}`);
});

test('gaussNewton default arguments are iterations=10, tol=1e-9', () => {
  const a = gaussNewton(ANCHORS, MEASURED, [30, 30, 10]);
  const b = gaussNewton(ANCHORS, MEASURED, [30, 30, 10], 10, 1e-9);
  assert.deepStrictEqual(a, b);
});

test('gaussNewton reproduces the Python LSQ -> GN baseline (10 iterations)', () => {
  const lsq = leastSquares(ANCHORS, MEASURED);
  const r = gaussNewton(ANCHORS, MEASURED, lsq, 10, 1e-9);
  // pso3d.trilateration.gauss_newton_refine on the baseline: (37.42108686662121, 11.884316250529604, 9.359200270570119), 10 iterations
  assert.ok(close(r.p[0], 37.42108686662121, 1e-6), `x ${r.p[0]}`);
  assert.ok(close(r.p[1], 11.884316250529604, 1e-6), `y ${r.p[1]}`);
  assert.ok(close(r.p[2], 9.359200270570119, 1e-6), `z ${r.p[2]}`);
  assert.strictEqual(r.iterations, 10);
  assert.ok(close(dist(r.p, TRUE_P), 0.9638059888173359, 1e-6));
});

test('gaussNewton counts iterations like Python (used = k + 1, break when |step| < tol)', () => {
  const atSolution = gaussNewton(ANCHORS, EXACT, TRUE_P.slice(), 10, 1e-9);
  assert.strictEqual(atSolution.iterations, 1, 'one (tiny) step is taken and counted');
  assert.strictEqual(atSolution.converged, true);
  const capped = gaussNewton(ANCHORS, EXACT, [30, 30, 10], 1, 1e-9);
  assert.strictEqual(capped.iterations, 1);
  assert.strictEqual(capped.converged, false);
  const zero = gaussNewton(ANCHORS, EXACT, [30, 30, 10], 0, 1e-9);
  assert.strictEqual(zero.iterations, 0);
  assert.strictEqual(zero.converged, false);
  assert.deepStrictEqual(zero.p, [30, 30, 10]);
});

test('gaussNewton does not mutate p0 and returns a fresh array', () => {
  const p0 = [30, 30, 10];
  const r = gaussNewton(ANCHORS, EXACT, p0);
  assert.deepStrictEqual(p0, [30, 30, 10]);
  assert.notStrictEqual(r.p, p0);
});

test('gaussNewton with a singular Jacobian (collinear anchors, start on the line) returns finite values', () => {
  const line = [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]];
  const d = line.map(a => dist([15, 0, 0], a));
  const r = gaussNewton(line, d, [5, 0, 0], 10, 1e-9);
  assert.ok(r.p.every(Number.isFinite));
  assert.strictEqual(typeof r.converged, 'boolean');
});

test('warmStart lies inside the field and equals the clipped LSQ -> GN estimate', () => {
  const p = warmStart(ANCHORS, MEASURED, LOWER, UPPER);
  assert.strictEqual(p.length, 3);
  for (let k = 0; k < 3; k++) {
    assert.ok(Number.isFinite(p[k]));
    assert.ok(p[k] >= LOWER[k] && p[k] <= UPPER[k], `axis ${k} inside the field`);
  }
  const gn = gaussNewton(ANCHORS, MEASURED, leastSquares(ANCHORS, MEASURED), 10, 1e-9).p;
  for (let k = 0; k < 3; k++) assert.ok(close(p[k], gn[k], 1e-12));
  assert.ok(dist(p, TRUE_P) < 1.0);
});

test('warmStart clips an out-of-field refinement', () => {
  // ranges consistent with a point outside the field (z = -4) -> GN lands outside -> clipped to z = 0
  const outside = [20, 20, -4];
  const d = ANCHORS.map(a => dist(outside, a));
  const p = warmStart(ANCHORS, d, LOWER, UPPER);
  assert.ok(close(p[2], 0, 1e-12), `z ${p[2]}`);
  assert.ok(close(p[0], 20, 1e-6) && close(p[1], 20, 1e-6));
});

test('warmStart falls back to the anchor centroid when the least-squares system is singular', () => {
  const line = [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]];
  const d = line.map(a => dist([15, 3, 2], a));
  assert.ok(leastSquares(line, d).some(Number.isNaN), 'LSQ is singular on collinear anchors');
  const p = warmStart(line, d, LOWER, UPPER);
  assert.ok(p.every(Number.isFinite));
  for (let k = 0; k < 3; k++) assert.ok(p[k] >= LOWER[k] && p[k] <= UPPER[k]);
  assert.ok(close(p[0], 15, 1e-6), 'x of the centroid (15) is kept along the line');
});

test('solve3 solves a well-conditioned system to machine precision', () => {
  const M = [[4, 1, 0.5], [1, 3, 0.2], [0.5, 0.2, 2]];
  const xTrue = [1.5, -2, 0.25];
  const y = M.map(row => row[0] * xTrue[0] + row[1] * xTrue[1] + row[2] * xTrue[2]);
  const x = solve3(M, y);
  for (let k = 0; k < 3; k++) assert.ok(close(x[k], xTrue[k], 1e-12));
});

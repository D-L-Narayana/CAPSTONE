'use strict';
// Node tests for web/geometry.js (JS twin of pso3d/geometry.py).
const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');

const GEO_JS = path.join(__dirname, '..', '..', 'web', 'geometry.js');
const geo = require(GEO_JS);
const { tetrahedronVolume, anchorRank, isNonCoplanar, rangeJacobian, gdop, crlb, mirrorPoint, flipAmbiguityRisk } = geo;

const ANCHORS = [[0, 0, 0], [60, 0, 5], [0, 60, 6], [60, 60, 20]];
const TRUE_P = [37, 12, 8.5];
const NOISE = [0.4, -0.3, 0.5, -0.4];
const COPLANAR = [[0, 0, 0], [60, 0, 0], [0, 60, 0], [60, 60, 0]];
const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
const MEASURED = ANCHORS.map((a, j) => dist(TRUE_P, a) + NOISE[j]);
const close = (a, b, tol) => Math.abs(a - b) <= tol;

test('module exports exactly the contract names and has no DOM access', () => {
  const expected = ['tetrahedronVolume', 'anchorRank', 'isNonCoplanar', 'rangeJacobian', 'gdop', 'crlb', 'mirrorPoint', 'flipAmbiguityRisk'];
  assert.deepStrictEqual(Object.keys(geo).sort(), expected.slice().sort());
  for (const n of expected) assert.strictEqual(typeof geo[n], 'function', n);
  const src = fs.readFileSync(GEO_JS, 'utf8');
  assert.ok(!/\b(document|window)\s*\./.test(src));
  assert.ok(!/^\s*(import|export)\s/m.test(src));
});

test('regular tetrahedron volume = a^3 / (6 sqrt 2)', () => {
  const a = 2.5;
  const s = a / (2 * Math.SQRT2);           // cube-corner construction has edge 2*sqrt(2) before scaling
  const v = [[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]].map(p => p.map(c => c * s));
  assert.ok(close(dist(v[0], v[1]), a, 1e-12));
  const vol = tetrahedronVolume(v[0], v[1], v[2], v[3]);
  assert.ok(close(vol, a ** 3 / (6 * Math.SQRT2), 1e-12), `volume ${vol}`);
  assert.ok(vol > 0);
  // orientation does not change the (unsigned) volume
  assert.ok(close(tetrahedronVolume(v[1], v[0], v[2], v[3]), vol, 1e-12));
  assert.strictEqual(tetrahedronVolume(...COPLANAR), 0);
  // baseline anchors: |det[b-a, c-a, d-a]| / 6 = |60*60*20 - 60*5*60 ... | computed by hand = 60*(60*20-6*60)/6 - ... check against the determinant
  const [A, B, C, D] = ANCHORS;
  const u = [B[0] - A[0], B[1] - A[1], B[2] - A[2]], w = [C[0] - A[0], C[1] - A[1], C[2] - A[2]], z = [D[0] - A[0], D[1] - A[1], D[2] - A[2]];
  const det = u[0] * (w[1] * z[2] - w[2] * z[1]) - u[1] * (w[0] * z[2] - w[2] * z[0]) + u[2] * (w[0] * z[1] - w[1] * z[0]);
  assert.ok(close(tetrahedronVolume(A, B, C, D), Math.abs(det) / 6, 1e-9));
});

test('coplanar anchors have anchorRank 2 and are not non-coplanar; the baseline is rank 3', () => {
  assert.strictEqual(anchorRank(COPLANAR), 2);
  assert.strictEqual(isNonCoplanar(COPLANAR), false);
  assert.strictEqual(anchorRank(ANCHORS), 3);
  assert.strictEqual(isNonCoplanar(ANCHORS), true);
  assert.strictEqual(anchorRank([[0, 0, 0], [10, 0, 0], [20, 0, 0]]), 1);
  assert.strictEqual(anchorRank([[5, 5, 5]]), 0);
  assert.strictEqual(isNonCoplanar([[0, 0, 0], [1, 0, 0], [0, 1, 0]]), false, 'fewer than four anchors');
  // tilted plane (not axis aligned) is still rank 2
  const tilted = [[0, 0, 0], [1, 0, 1], [0, 1, 2], [1, 1, 3], [2, 3, 8]];
  assert.strictEqual(anchorRank(tilted), 2);
  assert.strictEqual(isNonCoplanar(tilted), false);
  // near-coplanar fixture (z = 0.01) is numerically rank 3 and has volume 6 m^3 > 1e-6
  const near = [[0, 0, 0], [60, 0, 0], [0, 60, 0], [60, 60, 0.01]];
  assert.strictEqual(anchorRank(near), 3);
  assert.strictEqual(isNonCoplanar(near), true);
  assert.strictEqual(isNonCoplanar(near, 10), false, 'minVolume is honoured');
  // tol is relative to the largest *singular value* of the centred anchor matrix (numpy.linalg.matrix_rank semantics
  // of pso3d.geometry.anchor_rank): a 1e-5 m height step over a 60 m square is a 1.4e-7 relative spread (rank 3),
  // a 1e-8 m step is 1.4e-10 (rank 2 at the default tol = 1e-9, rank 3 at tol = 1e-12)
  assert.strictEqual(anchorRank([[0, 0, 0], [60, 0, 0], [0, 60, 0], [60, 60, 1e-5]]), 3, 'relative spread 1.4e-7 > 1e-9');
  assert.strictEqual(anchorRank([[0, 0, 0], [60, 0, 0], [0, 60, 0], [60, 60, 1e-8]]), 2, 'relative spread 1.4e-10 < 1e-9');
  assert.strictEqual(anchorRank([[0, 0, 0], [60, 0, 0], [0, 60, 0], [60, 60, 1e-8]], 1e-12), 3, 'tol is honoured');
  assert.strictEqual(anchorRank([]), 0);
});

test('rangeJacobian rows are the unit vectors (p - a_j) / |p - a_j|', () => {
  const J = rangeJacobian(ANCHORS, TRUE_P);
  assert.strictEqual(J.length, 4);
  for (let j = 0; j < 4; j++) {
    const n = Math.hypot(J[j][0], J[j][1], J[j][2]);
    assert.ok(close(n, 1, 1e-12));
    const r = dist(TRUE_P, ANCHORS[j]);
    for (let k = 0; k < 3; k++) assert.ok(close(J[j][k], (TRUE_P[k] - ANCHORS[j][k]) / r, 1e-12));
  }
});

test('gdop is Infinity for coplanar anchors at a point in their plane and finite elsewhere', () => {
  assert.strictEqual(gdop(COPLANAR, [30, 30, 0]), Infinity);
  assert.strictEqual(gdop(COPLANAR, [10, 50, 0]), Infinity);
  const g = gdop(ANCHORS, TRUE_P);
  assert.ok(Number.isFinite(g) && g > 0, `gdop ${g}`);
  const above = gdop(COPLANAR, [30, 30, 10]);
  assert.ok(Number.isFinite(above) && above > 0);
  // GDOP = sqrt(trace((J^T J)^-1)) checked against an explicit 3x3 inverse at the true node
  const J = rangeJacobian(ANCHORS, TRUE_P);
  const G = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (const row of J) for (let i = 0; i < 3; i++) for (let k = 0; k < 3; k++) G[i][k] += row[i] * row[k];
  const det = G[0][0] * (G[1][1] * G[2][2] - G[1][2] * G[2][1]) - G[0][1] * (G[1][0] * G[2][2] - G[1][2] * G[2][0]) + G[0][2] * (G[1][0] * G[2][1] - G[1][1] * G[2][0]);
  const c00 = (G[1][1] * G[2][2] - G[1][2] * G[2][1]) / det;
  const c11 = (G[0][0] * G[2][2] - G[0][2] * G[2][0]) / det;
  const c22 = (G[0][0] * G[1][1] - G[0][1] * G[1][0]) / det;
  assert.ok(close(g, Math.sqrt(c00 + c11 + c22), 1e-9));
  assert.strictEqual(gdop([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]], [15, 0, 0]), Infinity, 'collinear anchors');
});

test('crlb: rmseBound > 0, perAxis z largest at the true node of the baseline, consistent cov', () => {
  const sigma = 0.5;
  const b = crlb(ANCHORS, TRUE_P, sigma);
  assert.ok(b.rmseBound > 0 && Number.isFinite(b.rmseBound));
  assert.strictEqual(b.perAxis.length, 3);
  assert.ok(b.perAxis[2] > b.perAxis[0] && b.perAxis[2] > b.perAxis[1], `z bound largest: ${b.perAxis}`);
  assert.ok(close(b.rmseBound, Math.hypot(...b.perAxis), 1e-9));
  assert.ok(close(b.rmseBound, sigma * gdop(ANCHORS, TRUE_P), 1e-9), 'rmse bound = sigma * GDOP');
  assert.strictEqual(b.cov.length, 3);
  for (let i = 0; i < 3; i++) {
    assert.strictEqual(b.cov[i].length, 3);
    assert.ok(close(Math.sqrt(b.cov[i][i]), b.perAxis[i], 1e-12));
    for (let k = 0; k < 3; k++) assert.ok(close(b.cov[i][k], b.cov[k][i], 1e-9), 'symmetric');
  }
  const twice = crlb(ANCHORS, TRUE_P, 2 * sigma);
  assert.ok(close(twice.rmseBound, 2 * b.rmseBound, 1e-9), 'bound scales linearly with sigma');
  const sing = crlb(COPLANAR, [30, 30, 0], sigma);
  assert.strictEqual(sing.rmseBound, Infinity);
  assert.ok(sing.perAxis.every(v => v === Infinity));
  // singular geometry: cov is a 3x3 of Infinity like the Python CRLB (np.full((3, 3), inf)), not null
  assert.deepStrictEqual(sing.cov, [[Infinity, Infinity, Infinity], [Infinity, Infinity, Infinity], [Infinity, Infinity, Infinity]]);
  // sigma must be a positive noise level (pso3d.geometry.crlb raises ValueError)
  assert.throws(() => crlb(ANCHORS, TRUE_P, 0), RangeError);
  assert.throws(() => crlb(ANCHORS, TRUE_P, -0.5), RangeError);
  assert.throws(() => crlb(ANCHORS, TRUE_P, NaN), RangeError);
});

test('mirrorPoint: a point on the plane maps to itself; reflection is an involution; null below rank 2', () => {
  const mirrorOf = (anchors, p) => {
    const m = mirrorPoint(anchors, p);
    assert.ok(m !== null, 'mirrorPoint returned null');
    assert.ok(Array.isArray(m) && m.length === 3 && m.every(Number.isFinite), 'mirrorPoint returns [x, y, z]');
    return m;
  };
  const m0 = mirrorOf(COPLANAR, [30, 30, 0]);
  for (let k = 0; k < 3; k++) assert.ok(close(m0[k], [30, 30, 0][k], 1e-9));
  const m1 = mirrorOf(COPLANAR, [30, 30, 10]);
  for (let k = 0; k < 3; k++) assert.ok(close(m1[k], [30, 30, -10][k], 1e-9), `axis ${k}: ${m1[k]}`);
  const m2 = mirrorOf(ANCHORS, TRUE_P);
  const back = mirrorOf(ANCHORS, m2);
  for (let k = 0; k < 3; k++) assert.ok(close(back[k], TRUE_P[k], 1e-9), 'involution');
  // the centroid of the anchors lies on the least-squares plane -> fixed point
  const c = [0, 1, 2].map(k => ANCHORS.reduce((s, a) => s + a[k], 0) / ANCHORS.length);
  const mc = mirrorOf(ANCHORS, c);
  for (let k = 0; k < 3; k++) assert.ok(close(mc[k], c[k], 1e-9));
  // the mirror of p and p itself are symmetric about the plane: midpoint is on the plane
  const mid = [0, 1, 2].map(k => (TRUE_P[k] + m2[k]) / 2);
  const mm = mirrorOf(ANCHORS, mid);
  for (let k = 0; k < 3; k++) assert.ok(close(mm[k], mid[k], 1e-9));
  assert.strictEqual(mirrorPoint([[0, 0, 0], [10, 0, 0], [20, 0, 0]], [5, 5, 5]), null, 'collinear anchors');
  assert.strictEqual(mirrorPoint([[1, 1, 1]], [5, 5, 5]), null);
  // three non-collinear anchors define the plane exactly
  const m3 = mirrorOf([[0, 0, 1], [10, 0, 1], [0, 10, 1]], [2, 2, 4]);
  for (let k = 0; k < 3; k++) assert.ok(close(m3[k], [2, 2, -2][k], 1e-9));
});

test('flipAmbiguityRisk is within [0, 1], equals 1 for coplanar anchors and is < 1 for the baseline', () => {
  const noisy = COPLANAR.map((a, j) => dist([30, 30, 10], a) + NOISE[j]);
  const r = flipAmbiguityRisk(COPLANAR, [30, 30, 10], noisy);
  assert.ok(close(r, 1, 1e-9), `coplanar risk ${r}`);
  const rb = flipAmbiguityRisk(ANCHORS, TRUE_P, MEASURED);
  assert.ok(rb >= 0 && rb <= 1);
  assert.ok(rb < 1, `baseline risk ${rb}`);
  // formula pin: f(p) / max(f(mirror), 1e-12) with f = range-error fitness; mirror across the least-squares anchor plane
  const f = q => ANCHORS.reduce((s, a, j) => s + (dist(q, a) - MEASURED[j]) ** 2, 0);
  const expected = Math.min(1, Math.max(0, f(TRUE_P) / Math.max(f(mirrorPoint(ANCHORS, TRUE_P)), 1e-12)));
  assert.ok(close(rb, expected, 1e-9));
  // value computed from the formula on the baseline scenario (documented pin): f(p) = 0.66 (sum of squared noise),
  // f(mirror) ~ 0.897 -> the mirrored node fits the measured ranges almost as well because the anchors are nearly coplanar
  assert.ok(close(rb, 0.7355991242721195, 1e-9), `baseline pin ${rb}`);
  assert.strictEqual(flipAmbiguityRisk([[0, 0, 0], [10, 0, 0], [20, 0, 0]], [5, 5, 5], [7, 5, 7]), 1, 'no plane -> 1');
  // a wrong estimate whose mirror fits better is clipped to 1
  const wrong = mirrorPoint(ANCHORS, TRUE_P);
  assert.strictEqual(flipAmbiguityRisk(ANCHORS, wrong, MEASURED), 1);
  // a probe far from the anchor plane with exact ranges: mirror fits much worse -> near 0
  const exact = ANCHORS.map(a => dist(TRUE_P, a));
  assert.ok(flipAmbiguityRisk(ANCHORS, TRUE_P, exact) < 1e-6);
});

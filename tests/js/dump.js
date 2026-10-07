#!/usr/bin/env node
'use strict';
/* Parity dump: evaluates the browser implementations (web/pso.js, web/geometry.js) on the shared
 * fixture and prints ONE JSON object keyed by case id. The Python side (tests/test_parity.py)
 * computes the same quantities with the pso3d package and compares.
 *
 *   node tests/js/dump.js tests/fixtures/parity_inputs.json
 *
 * Per case: measured (= ||p - a_j|| + noise_j), lsq, gn {p, iterations} (from lsq or the anchor
 * centroid when lsq is NaN; 10 iterations, tol 1e-9), fitness[] / gdop[] / crlb_rmse[] / mirror[]
 * per probe point, non_coplanar, schedule {w, c1, c2} (T = 60, default AMCMPSO constants, linear)
 * and centre_of_mass of the probe points weighted by their fitness.
 * Non-finite numbers are emitted as the strings "Infinity", "-Infinity" or "NaN".
 */
const fs = require('node:fs');
const path = require('node:path');

const WEB = path.join(__dirname, '..', '..', 'web');
const pso = require(path.join(WEB, 'pso.js'));
const geo = require(path.join(WEB, 'geometry.js'));

function centroid(anchors) {
  const c = [0, 0, 0];
  for (const a of anchors) for (let k = 0; k < 3; k++) c[k] += a[k] / anchors.length;
  return c;
}

function dumpCase(c) {
  const anchors = c.anchors, p = c.true_position, noise = c.noise;
  const measured = anchors.map((a, j) => pso.dist(p, a) + noise[j]);
  const lsq = pso.leastSquares(anchors, measured);
  const start = lsq.every(Number.isFinite) ? lsq : centroid(anchors);
  const gn = pso.gaussNewton(anchors, measured, start, 10, 1e-9);
  const fit = new pso.Fitness(anchors, measured);
  const probes = c.probe_points;
  const fitness = probes.map(q => fit.f(q));
  return {
    measured,
    lsq,
    gn: { p: gn.p, iterations: gn.iterations },
    fitness,
    gdop: probes.map(q => geo.gdop(anchors, q)),
    crlb_rmse: probes.map(q => geo.crlb(anchors, q, c.sigma).rmseBound),
    mirror: probes.map(q => geo.mirrorPoint(anchors, q)),
    non_coplanar: geo.isNonCoplanar(anchors),
    schedule: pso.amcmpsoSchedule(60, {}),
    centre_of_mass: pso.centreOfMass(probes, fitness),
  };
}

function main(argv) {
  const input = argv[0];
  if (!input) {
    process.stderr.write('usage: node tests/js/dump.js <parity_inputs.json>\n');
    return 2;
  }
  const data = JSON.parse(fs.readFileSync(path.resolve(input), 'utf8'));
  const out = {};
  for (const c of data.cases) out[c.id] = dumpCase(c);
  const replacer = (key, value) => (typeof value === 'number' && !Number.isFinite(value) ? String(value) : value);
  process.stdout.write(JSON.stringify(out, replacer, 2) + '\n');
  return 0;
}

if (require.main === module) process.exitCode = main(process.argv.slice(2));
module.exports = { dumpCase, main };

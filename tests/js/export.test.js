'use strict';
// Tests for web/export.js — CSV / JSON export of a simulator run (pure functions, no DOM).
// Uses a tiny fake swarm/fitness so the test does not depend on web/pso.js.
// Run: node --test tests/js/export.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const { historyToCSV, runToJSON, downloadText } = require(path.join(__dirname, '..', '..', 'web', 'export.js'));

const HEADER = 'iteration,best_fitness,est_x,est_y,est_z,error_m';
const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);

function fakeRun(variant = 'simplified') {
  const history = [12.3456789, 3.25, 0.5, 0.125];
  const estHistory = [[30, 10, 5], [35, 11, 8], [36.5, 12.2, 8.9], [37.1, 11.9, 8.4]];
  if (variant !== 'simplified') { // synchronous variants also record the initial evaluation (iteration 0)
    history.unshift(40.5);
    estHistory.unshift([20, 20, 10]);
  }
  const swarm = {
    cfg: {
      N: 20, T: 60, w: 0.7, c: 1.4, c1: 1.4, seed: 1, clip: false, lower: [0, 0, 0], upper: [60, 60, 20], variant,
      stopping: { type: 'patience', patience: 10 }, x0: null,
      amcmpso: { wMax: 0.9, wMin: 0.4, schedule: 'linear' },
      onStep: () => {}, // functions must be stripped from the exported config
    },
    t: 4,
    best: [37.1, 11.9, 8.4],
    bestF: 0.125,
    done: true,
    stoppedBy: 'patience',
    history,
    estHistory,
    x: [[1, 2, 3]],
    memoryFloats() { return 120; },
  };
  const scenario = {
    field: [60, 60, 20], upper: [60, 60, 20], lower: [0, 0, 0],
    anchors: [[0, 0, 0], [60, 0, 5], [0, 60, 6], [60, 60, 20]],
    noise: [0.4, -0.3, 0.5, -0.4],
    noiseModel: { type: 'fixed', offsets: [0.4, -0.3, 0.5, -0.4] },
    p: [37, 12, 8.5],
    measured: [40.3, 25.9, 50.1, 55.6],
    seed: 1,
  };
  const fitness = { evals: 80, distances: 320 };
  return { swarm, scenario, fitness };
}

function rows(csv) {
  const lines = csv.trim().split('\n');
  return { header: lines[0], body: lines.slice(1).map(l => l.split(',')) };
}

test('historyToCSV starts with the documented header', () => {
  const { swarm, scenario } = fakeRun();
  const { header } = rows(historyToCSV(swarm, scenario));
  assert.equal(header, HEADER);
});

test('historyToCSV has one row per recorded iteration with six numeric columns', () => {
  const { swarm, scenario } = fakeRun();
  const { body } = rows(historyToCSV(swarm, scenario));
  assert.equal(body.length, swarm.history.length);
  for (const r of body) {
    assert.equal(r.length, 6, 'six columns: ' + r.join(','));
    for (const cell of r) assert.ok(Number.isFinite(Number(cell)), 'numeric cell: ' + JSON.stringify(cell));
  }
});

test('historyToCSV numbers the iterations from 1 when the initial swarm was not evaluated (simplified)', () => {
  const { swarm, scenario } = fakeRun('simplified');
  const { body } = rows(historyToCSV(swarm, scenario));
  assert.deepEqual(body.map(r => r[0]), ['1', '2', '3', '4']);
});

test('historyToCSV numbers the iterations from 0 when the history includes the initial evaluation (standard)', () => {
  const { swarm, scenario } = fakeRun('standard');
  const { body } = rows(historyToCSV(swarm, scenario));
  assert.equal(body.length, 5);
  assert.deepEqual(body.map(r => r[0]), ['0', '1', '2', '3', '4']);
});

test('historyToCSV writes 6 significant digits and the error against scenario.p', () => {
  const { swarm, scenario } = fakeRun();
  const { body } = rows(historyToCSV(swarm, scenario));
  assert.equal(body[0][1], '12.3457');
  assert.equal(body[3][1], '0.125');
  assert.deepEqual(body[3].slice(2, 5).map(Number), [37.1, 11.9, 8.4]);
  const expected = dist([37.1, 11.9, 8.4], [37, 12, 8.5]);
  assert.ok(Math.abs(Number(body[3][5]) - expected) < 1e-4, `error_m ${body[3][5]} ≈ ${expected}`);
});

test('historyToCSV of an un-stepped swarm is the header only', () => {
  const { swarm, scenario } = fakeRun();
  swarm.history = []; swarm.estHistory = []; swarm.t = 0;
  const { header, body } = rows(historyToCSV(swarm, scenario));
  assert.equal(header, HEADER);
  assert.equal(body.length, 0);
});

test('runToJSON contains scenario/config/result/history/extras and copies the counters', () => {
  const { swarm, scenario, fitness } = fakeRun();
  const extras = { lsq: [36.9, 12.1, 8.6], note: 'test' };
  const out = runToJSON(swarm, scenario, fitness, extras);
  assert.deepEqual(Object.keys(out).sort(), ['config', 'extras', 'history', 'result', 'scenario']);
  assert.deepEqual(out.scenario.field, [60, 60, 20]);
  assert.deepEqual(out.scenario.anchors, scenario.anchors);
  assert.deepEqual(out.scenario.noise, scenario.noise);
  assert.deepEqual(out.scenario.p, scenario.p);
  assert.deepEqual(out.scenario.measured, scenario.measured);
  assert.equal(out.result.fitness_evaluations, fitness.evals);
  assert.equal(out.result.distance_computations, fitness.distances);
  assert.equal(out.result.iterations_run, 4);
  assert.equal(out.result.stopped_by, 'patience');
  assert.equal(out.result.swarm_state_floats, 120);
  assert.equal(out.result.best_fitness, 0.125);
  assert.deepEqual(out.result.estimate, [37.1, 11.9, 8.4]);
  assert.ok(Math.abs(out.result.error_m - dist([37.1, 11.9, 8.4], [37, 12, 8.5])) < 1e-12);
  assert.deepEqual(out.history.best_fitness, swarm.history);
  assert.deepEqual(out.history.estimates, swarm.estHistory);
  assert.deepEqual(out.extras, extras);
});

test('runToJSON config is swarm.cfg without functions and is a plain JSON-safe object', () => {
  const { swarm, scenario, fitness } = fakeRun();
  const out = runToJSON(swarm, scenario, fitness, {});
  assert.equal('onStep' in out.config, false, 'functions stripped');
  assert.equal(out.config.N, 20);
  assert.equal(out.config.variant, 'simplified');
  assert.deepEqual(out.config.stopping, { type: 'patience', patience: 10 });
  assert.deepEqual(out.config.amcmpso, { wMax: 0.9, wMin: 0.4, schedule: 'linear' });
  assert.deepEqual(JSON.parse(JSON.stringify(out)), out, 'survives JSON round trip unchanged');
  // the export must not alias live swarm arrays
  out.history.best_fitness.push(999);
  assert.equal(swarm.history.length, 4);
});

test('runToJSON tolerates a swarm that has not stepped yet', () => {
  const { swarm, scenario, fitness } = fakeRun();
  swarm.history = []; swarm.estHistory = []; swarm.t = 0; swarm.best = null; swarm.bestF = Infinity; swarm.stoppedBy = null; swarm.done = false;
  const out = runToJSON(swarm, scenario, fitness);
  assert.equal(out.result.estimate, null);
  assert.equal(out.result.error_m, null);
  assert.equal(out.result.best_fitness, null);
  assert.equal(out.result.iterations_run, 0);
  assert.equal(out.result.stopped_by, null);
  assert.deepEqual(out.extras, {});
  assert.deepEqual(JSON.parse(JSON.stringify(out)), out);
});

test('downloadText is a no-op (returns false) without a document and never throws', () => {
  assert.equal(typeof document, 'undefined', 'node has no DOM');
  assert.equal(downloadText('pso3d_run.csv', HEADER + '\n', 'text/csv'), false);
});

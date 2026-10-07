'use strict';
// Tests for web/share.js — the URL share-state codec (pure functions, no DOM).
// Run: node --test tests/js/share.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');

const { encodeState, decodeState, applyHash } = require(path.join(__dirname, '..', '..', 'web', 'share.js'));

const FULL = Object.freeze({
  v: 1,
  field: [60, 60, 20],
  anchors: [[0, 0, 0, 0.4], [60, 0, 5, -0.3], [0, 60, 6, 0.5], [60, 60, 20, -0.4], [30, 30, 10, 0]],
  p: [37, 12, 8.5],
  variant: 'amcmpso',
  seed: 7,
  N: 25,
  T: 80,
  w: 0.7,
  c: 1.4,
  c1: 1.4,
  clip: true,
  stop: { type: 'tolerance', patience: 10, tol: 0.01, window: 10, budget: 600 },
  warm: true,
  noise: { model: 'nlos', sigma: 0.5, pn: 2, nlosP: 0.2, nlosBias: 2, nlosSigma: 1 },
});

function withPatch(patch) {
  return Object.assign(JSON.parse(JSON.stringify(FULL)), patch);
}

test('encodeState returns a versioned, URL-safe string', () => {
  const s = encodeState(FULL);
  assert.equal(typeof s, 'string');
  assert.ok(s.startsWith('v1.'), 'version prefix v1.');
  assert.match(s, /^v1\.[A-Za-z0-9_-]+$/, 'base64url alphabet only');
  assert.equal(encodeURIComponent(s), s, 'survives percent-encoding unchanged');
});

test('encodeState is deterministic for equal states', () => {
  const a = encodeState(FULL);
  const b = encodeState(JSON.parse(JSON.stringify(FULL)));
  assert.equal(typeof a, 'string');
  assert.ok(a.length > 3, 'non-empty payload');
  assert.equal(a, b);
});

test('decodeState(encodeState(state)) round-trips a full state (deep equality)', () => {
  const decoded = decodeState(encodeState(FULL));
  assert.notEqual(decoded, null);
  assert.deepEqual(decoded, FULL);
});

test('round trip preserves the optional AMCMPSO schedule when present', () => {
  const state = withPatch({ amcmpso: { schedule: 'cosine' } });
  const decoded = decodeState(encodeState(state));
  assert.notEqual(decoded, null);
  assert.deepEqual(decoded, state);
});

test('decodeState returns null for garbage, empty and wrong-version strings', () => {
  for (const bad of ['garbage', 'v0.xxx', 'v2.' + encodeState(FULL).slice(3), '', 'v1.', 'v1.!!!not base64!!!', '#s=']) {
    assert.equal(decodeState(bad), null, JSON.stringify(bad));
  }
});

test('decodeState returns null for non-string input', () => {
  for (const bad of [null, undefined, 42, {}, [], true]) {
    assert.equal(decodeState(bad), null, String(bad));
  }
});

test('decodeState returns null for base64url of non-JSON or non-object JSON', () => {
  const b64 = (s) => 'v1.' + Buffer.from(s, 'utf8').toString('base64url');
  assert.equal(decodeState(b64('not json {')), null, 'invalid JSON');
  assert.equal(decodeState(b64('[1,2,3]')), null, 'array instead of object');
  assert.equal(decodeState(b64('null')), null, 'JSON null');
  assert.equal(decodeState(b64('"string"')), null, 'JSON string');
  assert.equal(decodeState(b64('12')), null, 'JSON number');
});

test('decodeState rejects invalid shapes', () => {
  const cases = {
    'only 3 anchors': withPatch({ anchors: FULL.anchors.slice(0, 3) }),
    'anchors not an array': withPatch({ anchors: 'a1,a2,a3,a4' }),
    'anchor with a non-numeric coordinate': withPatch({ anchors: [['a', 0, 0, 0], ...FULL.anchors.slice(1)] }),
    'anchor with 3 entries instead of 4': withPatch({ anchors: [[0, 0, 0], ...FULL.anchors.slice(1)] }),
    'anchor with a non-finite entry': withPatch({ anchors: [[0, 0, 0, null], ...FULL.anchors.slice(1)] }),
    'too many anchors (65)': withPatch({ anchors: Array.from({ length: 65 }, (_, i) => [i, i, i, 0]) }),
    'p with 2 entries': withPatch({ p: [1, 2] }),
    'p with a string entry': withPatch({ p: [1, 'x', 3] }),
    'field with 2 entries': withPatch({ field: [60, 60] }),
    'field with non-positive size': withPatch({ field: [60, 0, 20] }),
    'unknown variant': withPatch({ variant: 'foo' }),
    'N not numeric': withPatch({ N: 'many' }),
    'T negative': withPatch({ T: -5 }),
    'w not numeric': withPatch({ w: 'heavy' }),
    'clip not boolean': withPatch({ clip: 'yes' }),
    'unknown stop type': withPatch({ stop: { type: 'sometimes', patience: 10, tol: 0.01, window: 10, budget: 600 } }),
    'stop fields not numeric': withPatch({ stop: { type: 'patience', patience: 'x', tol: 0.01, window: 10, budget: 600 } }),
    'unknown noise model': withPatch({ noise: { model: 'laplace', sigma: 0.5, pn: 2, nlosP: 0.2, nlosBias: 2, nlosSigma: 1 } }),
    'noise fields not numeric': withPatch({ noise: { model: 'gaussian', sigma: 'big', pn: 2, nlosP: 0.2, nlosBias: 2, nlosSigma: 1 } }),
    'unknown amcmpso schedule': withPatch({ amcmpso: { schedule: 'random' } }),
    'wrong inner version': withPatch({ v: 2 }),
  };
  for (const [label, bad] of Object.entries(cases)) {
    const enc = encodeState(bad);
    assert.equal(typeof enc, 'string', label + ': encodes');
    assert.equal(decodeState(enc), null, label);
  }
});

test('decodeState completes a minimal valid state with documented defaults', () => {
  const minimal = { v: 1, anchors: FULL.anchors, p: FULL.p };
  const d = decodeState(encodeState(minimal));
  assert.notEqual(d, null);
  assert.deepEqual(d.anchors, FULL.anchors);
  assert.deepEqual(d.p, FULL.p);
  assert.deepEqual(d.field, [60, 60, 20]);
  assert.equal(d.variant, 'simplified');
  assert.equal(d.seed, 1);
  assert.equal(d.N, 20);
  assert.equal(d.T, 60);
  assert.equal(d.w, 0.7);
  assert.equal(d.c, 1.4);
  assert.equal(d.c1, 1.4);
  assert.equal(d.clip, false);
  assert.equal(d.warm, false);
  assert.deepEqual(d.stop, { type: 'none', patience: 10, tol: 0.01, window: 10, budget: 600 });
  assert.deepEqual(d.noise, { model: 'fixed', sigma: 0.5, pn: 2, nlosP: 0.2, nlosBias: 2, nlosSigma: 1 });
  assert.equal('amcmpso' in d, false, 'optional key is not invented');
});

test('decodeState drops unknown top-level keys', () => {
  const d = decodeState(encodeState(withPatch({ evil: '<script>', speed: 120 })));
  assert.notEqual(d, null);
  assert.equal('evil' in d, false);
  assert.equal('speed' in d, false);
  assert.deepEqual(d, FULL);
});

test('applyHash("#s=<encoded>") round-trips and ignores other hashes', () => {
  const enc = encodeState(FULL);
  assert.deepEqual(applyHash('#s=' + enc), FULL);
  assert.deepEqual(applyHash('s=' + enc), FULL, 'leading # is optional');
  assert.deepEqual(applyHash('#x=1&s=' + enc), FULL, 'other params are ignored');
  assert.deepEqual(applyHash('#s=' + encodeURIComponent(enc)), FULL, 'percent-encoded value is accepted');
});

test('applyHash returns null when the share parameter is absent or malformed', () => {
  for (const bad of ['', '#', '#other=1', '#s=', '#s=garbage', '#s=v0.xxx', undefined, null, 42]) {
    assert.equal(applyHash(bad), null, JSON.stringify(bad));
  }
});

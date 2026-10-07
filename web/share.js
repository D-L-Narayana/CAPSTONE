/* URL share-state codec for the PSO 3D simulator (pure functions; no DOM access).
   Format: "v1." + base64url(JSON.stringify(state)).  decodeState() validates the shape and
   returns null for anything malformed, so a tampered link can never reach the UI as-is. */
(function (root) {
  'use strict';

  const VERSION = 1;
  const PREFIX = 'v1.';
  const MAX_ANCHORS = 64;
  const VARIANTS = ['simplified', 'standard', 'amcmpso'];
  const STOP_TYPES = ['none', 'patience', 'tolerance', 'budget'];
  const NOISE_MODELS = ['fixed', 'gaussian', 'uniformPercent', 'nlos'];
  const SCHEDULES = ['linear', 'cosine'];
  const AMCMPSO_NUMERIC = ['wMax', 'wMin', 'c1Start', 'c1End', 'c2Start', 'c2End', 'cMean', 'cCom'];
  const DEFAULTS = Object.freeze({
    field: [60, 60, 20], variant: 'simplified', seed: 1, N: 20, T: 60, w: 0.7, c: 1.4, c1: 1.4, clip: false, warm: false,
    stop: { type: 'none', patience: 10, tol: 0.01, window: 10, budget: 600 },
    noise: { model: 'fixed', sigma: 0.5, pn: 2, nlosP: 0.2, nlosBias: 2, nlosSigma: 1 },
  });

  // ---- base64url (UTF-8 safe; Buffer in Node, TextEncoder/btoa in browsers) ----
  function toBase64url(str) {
    let b64;
    if (typeof Buffer === 'function' && typeof Buffer.from === 'function') {
      b64 = Buffer.from(str, 'utf8').toString('base64');
    } else {
      const bytes = new TextEncoder().encode(str);
      let bin = '';
      for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
      b64 = btoa(bin);
    }
    return b64.replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }
  function fromBase64url(s) {
    if (typeof s !== 'string' || s.length === 0 || s.length % 4 === 1 || !/^[A-Za-z0-9_-]+$/.test(s)) return null;
    const b64 = s.replace(/-/g, '+').replace(/_/g, '/') + '==='.slice(0, (4 - (s.length % 4)) % 4);
    try {
      if (typeof Buffer === 'function' && typeof Buffer.from === 'function') return Buffer.from(b64, 'base64').toString('utf8');
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new TextDecoder('utf-8', { fatal: true }).decode(bytes);
    } catch (e) {
      return null;
    }
  }

  // ---- validators: each returns the normalised value, or null when the input is invalid ----
  const isObj = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
  const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
  const isCount = (v) => Number.isInteger(v) && v >= 0;
  const nonNeg = (v) => v >= 0;
  const positive = (v) => v > 0;

  function pickNum(v, def, pred) {
    if (v === undefined) return def;
    return isNum(v) && (!pred || pred(v)) ? v : null;
  }
  function pickBool(v, def) {
    if (v === undefined) return def;
    return typeof v === 'boolean' ? v : null;
  }
  function pickEnum(v, allowed, def) {
    if (v === undefined) return def;
    return typeof v === 'string' && allowed.indexOf(v) >= 0 ? v : null;
  }
  function parseVec(v, n, pred) {
    if (!Array.isArray(v) || v.length !== n) return null;
    for (const x of v) if (!isNum(x) || (pred && !pred(x))) return null;
    return v.slice();
  }
  function parseAnchors(v) {
    if (!Array.isArray(v) || v.length < 4 || v.length > MAX_ANCHORS) return null;
    const out = [];
    for (const row of v) {
      const r = parseVec(row, 4);
      if (!r) return null;
      out.push(r);
    }
    return out;
  }
  function parseStop(v) {
    if (v === undefined) return Object.assign({}, DEFAULTS.stop);
    if (!isObj(v)) return null;
    const out = {
      type: pickEnum(v.type, STOP_TYPES, DEFAULTS.stop.type),
      patience: pickNum(v.patience, DEFAULTS.stop.patience, nonNeg),
      tol: pickNum(v.tol, DEFAULTS.stop.tol, nonNeg),
      window: pickNum(v.window, DEFAULTS.stop.window, nonNeg),
      budget: pickNum(v.budget, DEFAULTS.stop.budget, nonNeg),
    };
    for (const k of Object.keys(out)) if (out[k] === null) return null;
    return out;
  }
  function parseNoise(v) {
    if (v === undefined) return Object.assign({}, DEFAULTS.noise);
    if (!isObj(v)) return null;
    const out = {
      model: pickEnum(v.model, NOISE_MODELS, DEFAULTS.noise.model),
      sigma: pickNum(v.sigma, DEFAULTS.noise.sigma, nonNeg),
      pn: pickNum(v.pn, DEFAULTS.noise.pn, nonNeg),
      nlosP: pickNum(v.nlosP, DEFAULTS.noise.nlosP, (x) => x >= 0 && x <= 1),
      nlosBias: pickNum(v.nlosBias, DEFAULTS.noise.nlosBias),
      nlosSigma: pickNum(v.nlosSigma, DEFAULTS.noise.nlosSigma, nonNeg),
    };
    for (const k of Object.keys(out)) if (out[k] === null) return null;
    return out;
  }
  function parseAmcmpso(v) {
    if (!isObj(v)) return null;
    const out = {};
    if (v.schedule !== undefined) {
      const s = pickEnum(v.schedule, SCHEDULES, null);
      if (s === null) return null;
      out.schedule = s;
    }
    for (const k of AMCMPSO_NUMERIC) {
      if (v[k] === undefined) continue;
      if (!isNum(v[k])) return null;
      out[k] = v[k];
    }
    return out;
  }

  /** state object -> "v1.<base64url JSON>" (URL-safe, no percent-encoding needed). */
  function encodeState(state) {
    return PREFIX + toBase64url(JSON.stringify(state === undefined ? null : state));
  }

  /** "v1.<base64url JSON>" -> validated, default-completed state object, or null. */
  function decodeState(str) {
    if (typeof str !== 'string' || str.slice(0, PREFIX.length) !== PREFIX) return null;
    const json = fromBase64url(str.slice(PREFIX.length));
    if (json === null) return null;
    let raw;
    try { raw = JSON.parse(json); } catch (e) { return null; }
    if (!isObj(raw)) return null;
    if (raw.v !== undefined && raw.v !== VERSION) return null;

    const anchors = parseAnchors(raw.anchors);
    const p = parseVec(raw.p, 3);
    const field = raw.field === undefined ? DEFAULTS.field.slice() : parseVec(raw.field, 3, positive);
    if (!anchors || !p || !field) return null;

    const out = {
      v: VERSION,
      field,
      anchors,
      p,
      variant: pickEnum(raw.variant, VARIANTS, DEFAULTS.variant),
      seed: pickNum(raw.seed, DEFAULTS.seed),
      N: pickNum(raw.N, DEFAULTS.N, isCount),
      T: pickNum(raw.T, DEFAULTS.T, isCount),
      w: pickNum(raw.w, DEFAULTS.w),
      c: pickNum(raw.c, DEFAULTS.c),
      c1: pickNum(raw.c1, DEFAULTS.c1),
      clip: pickBool(raw.clip, DEFAULTS.clip),
      stop: parseStop(raw.stop),
      warm: pickBool(raw.warm, DEFAULTS.warm),
      noise: parseNoise(raw.noise),
    };
    for (const k of Object.keys(out)) if (out[k] === null) return null;
    if (raw.amcmpso !== undefined) {
      const a = parseAmcmpso(raw.amcmpso);
      if (a === null) return null;
      out.amcmpso = a;
    }
    return out;
  }

  /** location.hash ("#s=<encoded>&…") -> decoded state, or null when absent/invalid. */
  function applyHash(hash) {
    if (typeof hash !== 'string') return null;
    const query = hash.charAt(0) === '#' ? hash.slice(1) : hash;
    for (const part of query.split('&')) {
      const eq = part.indexOf('=');
      if (eq < 0 || part.slice(0, eq) !== 's') continue;
      let value = part.slice(eq + 1);
      try { value = decodeURIComponent(value); } catch (e) { /* keep the raw value */ }
      return decodeState(value);
    }
    return null;
  }

  const api = { encodeState, decodeState, applyHash };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof globalThis !== 'undefined' ? globalThis : this);

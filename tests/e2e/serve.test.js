'use strict';
// Tests for tests/e2e/serve.js — the dependency-free static server that mimics the Vercel deployment
// of web/ (cleanUrls + every "headers" rule of web/vercel.json). Run: node --test tests/e2e/serve.test.js
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const { createServer, sourceToRegExp, headersFor, loadHeaderRules } = require(path.join(__dirname, 'serve.js'));

const WEB = path.resolve(__dirname, '..', '..', 'web');
const VERCEL = path.join(WEB, 'vercel.json');
const vercel = JSON.parse(fs.readFileSync(VERCEL, 'utf8'));

function ruleValue(source, key) {
  const rule = (vercel.headers || []).find(r => r.source === source);
  const h = rule && rule.headers.find(x => x.key.toLowerCase() === key.toLowerCase());
  return h ? h.value : undefined;
}
const CSP = ruleValue('/(.*)', 'Content-Security-Policy');

function request(base, p, method = 'GET') {
  return new Promise((resolve, reject) => {
    const req = http.request(base + p, { method }, res => {
      const chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => resolve({ status: res.statusCode, headers: res.headers, body: Buffer.concat(chunks).toString('utf8') }));
    });
    req.on('error', reject);
    req.end();
  });
}

let server, base;
test.before(async () => {
  server = createServer(WEB, VERCEL);
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  base = `http://127.0.0.1:${server.address().port}`;
});
test.after(async () => { await new Promise(resolve => server.close(resolve)); });

test('vercel.json carries the enforced production policy (precondition for every header check)', () => {
  assert.equal(typeof CSP, 'string');
  assert.match(CSP, /^default-src 'self'; script-src 'self' https:\/\/cdn\.plot\.ly; /);
  assert.match(CSP, /object-src 'none'/);
  assert.match(CSP, /frame-ancestors 'none'/);
  assert.equal(ruleValue('/(.*)', 'Content-Security-Policy-Report-Only'), undefined, 'enforced, not report-only');
});

test('GET / serves index.html with text/html', async () => {
  const r = await request(base, '/');
  assert.equal(r.status, 200);
  assert.match(r.headers['content-type'], /^text\/html/);
  assert.match(r.body, /<title>/);
  assert.equal(Number(r.headers['content-length']), Buffer.byteLength(r.body));
});

test('GET /index.html and the clean URL /index serve the same document', async () => {
  const a = await request(base, '/index.html');
  const b = await request(base, '/index');
  assert.equal(a.status, 200);
  assert.equal(b.status, 200);
  assert.match(a.headers['content-type'], /^text\/html/);
  assert.match(b.headers['content-type'], /^text\/html/);
  assert.equal(a.body, b.body);
});

test('GET /app.js and /style.css have the right Content-Type', async () => {
  const js = await request(base, '/app.js');
  const css = await request(base, '/style.css');
  assert.equal(js.status, 200);
  assert.match(js.headers['content-type'], /javascript/);
  assert.equal(css.status, 200);
  assert.match(css.headers['content-type'], /^text\/css/);
});

test('GET /nope is a 404 and directory traversal is refused', async () => {
  const r = await request(base, '/nope');
  assert.equal(r.status, 404);
  const t = await request(base, '/%2e%2e/pyproject.toml');
  assert.equal(t.status, 404);
  const t2 = await request(base, '/..%2fpyproject.toml');
  assert.equal(t2.status, 404);
});

test('every 200 response carries the production security headers from web/vercel.json', async () => {
  for (const p of ['/', '/index.html', '/index', '/app.js', '/pso.js', '/style.css']) {
    const r = await request(base, p);
    assert.equal(r.status, 200, p);
    assert.equal(r.headers['content-security-policy'], CSP, `CSP on ${p}`);
    assert.equal(r.headers['x-frame-options'], 'DENY', `X-Frame-Options on ${p}`);
    assert.equal(r.headers['x-content-type-options'], 'nosniff', `nosniff on ${p}`);
    assert.equal(r.headers['referrer-policy'], ruleValue('/(.*)', 'Referrer-Policy'), `Referrer-Policy on ${p}`);
    assert.equal(r.headers['permissions-policy'], ruleValue('/(.*)', 'Permissions-Policy'), `Permissions-Policy on ${p}`);
  }
});

test('Cache-Control differs between the document and the assets', async () => {
  const doc = await request(base, '/');
  const js = await request(base, '/app.js');
  const css = await request(base, '/style.css');
  assert.equal(doc.headers['cache-control'], ruleValue('/', 'Cache-Control'));
  assert.equal(js.headers['cache-control'], ruleValue('/(.*)\\.(js|css)', 'Cache-Control'));
  assert.equal(css.headers['cache-control'], js.headers['cache-control']);
  assert.notEqual(doc.headers['cache-control'], js.headers['cache-control']);
});

test('HEAD / returns the headers without a body; other methods are 405', async () => {
  const h = await request(base, '/', 'HEAD');
  assert.equal(h.status, 200);
  assert.equal(h.body, '');
  assert.equal(h.headers['content-security-policy'], CSP);
  const p = await request(base, '/', 'POST');
  assert.equal(p.status, 405);
});

test('sourceToRegExp implements the Vercel source patterns used in vercel.json', () => {
  const all = sourceToRegExp('/(.*)');
  assert.ok(all.test('/'));
  assert.ok(all.test('/app.js'));
  assert.ok(all.test('/deep/path/file.png'));
  const assets = sourceToRegExp('/(.*)\\.(js|css)');
  assert.ok(assets.test('/app.js'));
  assert.ok(assets.test('/style.css'));
  assert.ok(assets.test('/sub/dir/x.css'));
  assert.ok(!assets.test('/index.html'));
  assert.ok(!assets.test('/'));
  assert.ok(!assets.test('/app.json'));
  assert.ok(!assets.test('/appXjs'), 'the escaped dot must stay a literal dot');
  const root = sourceToRegExp('/');
  assert.ok(root.test('/'));
  assert.ok(!root.test('/index'));
  const param = sourceToRegExp('/docs/:slug');
  assert.ok(param.test('/docs/intro'));
  assert.ok(!param.test('/docs/intro/more'));
});

test('headersFor merges every matching rule in order', () => {
  const rules = loadHeaderRules(VERCEL);
  assert.ok(rules.length >= 3);
  const doc = headersFor(rules, '/');
  assert.equal(doc['Content-Security-Policy'], CSP);
  assert.equal(doc['Cache-Control'], 'no-cache');
  const js = headersFor(rules, '/pso.js');
  assert.equal(js['Content-Security-Policy'], CSP);
  assert.equal(js['Cache-Control'], 'public, max-age=3600, must-revalidate');
  const custom = headersFor([
    { source: '/(.*)', regexp: sourceToRegExp('/(.*)'), headers: { 'X-A': '1', 'X-B': 'first' } },
    { source: '/x', regexp: sourceToRegExp('/x'), headers: { 'X-B': 'second' } },
  ], '/x');
  assert.deepEqual(custom, { 'X-A': '1', 'X-B': 'second' }, 'later rules override earlier ones for the same key');
});

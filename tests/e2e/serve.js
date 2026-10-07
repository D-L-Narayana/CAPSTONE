#!/usr/bin/env node
'use strict';
/* Dependency-free static server that mimics the Vercel deployment of web/ for local checks.
 *
 *   node tests/e2e/serve.js [--port 0|N] [--root web] [--vercel web/vercel.json] [--host 127.0.0.1]
 *
 * - serves files below the root with the right Content-Type (html, js, css, png, svg, json, ico, ...)
 * - "/" -> index.html, cleanUrls ("/index" -> index.html), 404 otherwise; no directory listings, no traversal
 * - reads vercel.json and applies every matching "headers" rule (Vercel "source" patterns: "/(.*)",
 *   "/(.*)\\.(js|css)", "/", "/docs/:slug"); later rules override earlier ones for the same key
 * - prints "LISTENING <port>" on stdout once it accepts connections
 * - exports createServer(root, vercelJsonPath) and the matcher helpers for in-process use (tests/e2e/*.js)
 *
 * It does not emulate Vercel's 308 redirects for ".html" requests (a request for /index.html is served directly).
 */
const fs = require('node:fs');
const http = require('node:http');
const path = require('node:path');

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.htm': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.mjs': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.map': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
  '.md': 'text/markdown; charset=utf-8',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.webmanifest': 'application/manifest+json',
};

function contentType(file) {
  return MIME[path.extname(file).toLowerCase()] || 'application/octet-stream';
}

/** Convert the subset of Vercel/path-to-regexp "source" syntax used in this repository to an anchored RegExp. */
function sourceToRegExp(source) {
  let out = '';
  for (let i = 0; i < source.length; i++) {
    const ch = source[i];
    if (ch === '\\' && i + 1 < source.length) {            // escaped literal, e.g. "\\." -> "\."
      out += '\\' + source[i + 1];
      i++;
    } else if (ch === '(') {                                  // regex group such as "(.*)" or "(js|css)": copy verbatim
      let depth = 0, j = i;
      for (; j < source.length; j++) {
        if (source[j] === '\\') { j++; continue; }
        if (source[j] === '(') depth++;
        else if (source[j] === ')') { depth--; if (depth === 0) break; }
      }
      out += source.slice(i, j + 1);
      i = j;
    } else if (ch === ':') {                                  // ":param" or ":param(regex)" -> one path segment
      let j = i + 1;
      while (j < source.length && /[A-Za-z0-9_]/.test(source[j])) j++;
      let group = '([^/]+)';
      if (source[j] === '(') {
        let depth = 0, k = j;
        for (; k < source.length; k++) {
          if (source[k] === '(') depth++;
          else if (source[k] === ')') { depth--; if (depth === 0) break; }
        }
        group = source.slice(j, k + 1);
        j = k + 1;
      }
      out += group;
      i = j - 1;
    } else {
      out += /[.*+?^${}|[\]]/.test(ch) ? '\\' + ch : ch;      // other specials are literals; "/" stays as is
    }
  }
  return new RegExp('^' + out + '$');
}

/** Compile the "headers" entries of a vercel.json into [{source, regexp, headers:{key:value}}]. */
function loadHeaderRules(vercelJsonPath) {
  if (!vercelJsonPath || !fs.existsSync(vercelJsonPath)) return [];
  const cfg = JSON.parse(fs.readFileSync(vercelJsonPath, 'utf8'));
  return (cfg.headers || []).map((rule) => {
    const headers = {};
    for (const h of rule.headers || []) headers[h.key] = h.value;
    return { source: rule.source, regexp: sourceToRegExp(rule.source), headers };
  });
}

/** Headers for a request path: every matching rule applied in order (later rules win on equal keys). */
function headersFor(rules, pathname) {
  const out = {};
  for (const rule of rules) {
    if (rule.regexp.test(pathname)) Object.assign(out, rule.headers);
  }
  return out;
}

/** Resolve a URL path to a file below root (null when missing or outside the root). */
function resolveFile(root, pathname) {
  let decoded;
  try { decoded = decodeURIComponent(pathname); } catch (e) { return null; }
  if (decoded.includes('\0')) return null;
  const abs = path.resolve(root, '.' + path.posix.normalize('/' + decoded.replace(/\\/g, '/')));
  if (abs !== root && !abs.startsWith(root + path.sep)) return null;
  const candidates = [];
  if (decoded.endsWith('/')) candidates.push(path.join(abs, 'index.html'));
  else candidates.push(abs, abs + '.html', path.join(abs, 'index.html'));   // file, cleanUrls, directory index
  for (const c of candidates) {
    try {
      if (fs.statSync(c).isFile() && (c === root || c.startsWith(root + path.sep))) return c;
    } catch (e) { /* try the next candidate */ }
  }
  return null;
}

function createServer(root, vercelJsonPath) {
  root = path.resolve(root || path.join(__dirname, '..', '..', 'web'));
  if (vercelJsonPath === undefined) vercelJsonPath = path.join(root, 'vercel.json');
  const rules = loadHeaderRules(vercelJsonPath);
  const server = http.createServer((req, res) => {
    const pathname = (req.url || '/').split('?')[0].split('#')[0];
    const extra = headersFor(rules, pathname);
    for (const [k, v] of Object.entries(extra)) res.setHeader(k, v);
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.statusCode = 405;
      res.setHeader('Allow', 'GET, HEAD');
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end('method not allowed');
      return;
    }
    const file = resolveFile(root, pathname);
    if (!file) {
      res.statusCode = 404;
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.end('not found');
      return;
    }
    const body = fs.readFileSync(file);
    res.statusCode = 200;
    res.setHeader('Content-Type', contentType(file));
    res.setHeader('Content-Length', body.length);
    if (req.method === 'HEAD') res.end(); else res.end(body);
  });
  server.root = root;
  server.rules = rules;
  return server;
}

function parseArgs(argv) {
  const opts = { port: 0, host: '127.0.0.1', root: path.join(__dirname, '..', '..', 'web'), vercel: undefined };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    const next = () => { if (i + 1 >= argv.length) throw new Error(`missing value for ${a}`); return argv[++i]; };
    if (a === '--port') opts.port = Number(next());
    else if (a === '--host') opts.host = next();
    else if (a === '--root') opts.root = path.resolve(next());
    else if (a === '--vercel') opts.vercel = path.resolve(next());
    else if (a === '-h' || a === '--help') { opts.help = true; }
    else throw new Error(`unknown option ${a}`);
  }
  return opts;
}

function main(argv) {
  let opts;
  try { opts = parseArgs(argv); } catch (e) { console.error(`serve.js: ${e.message}`); return 2; }
  if (opts.help) {
    console.log('usage: node tests/e2e/serve.js [--port 0|N] [--host 127.0.0.1] [--root web] [--vercel web/vercel.json]');
    return 0;
  }
  if (!Number.isInteger(opts.port) || opts.port < 0 || opts.port > 65535) { console.error('serve.js: --port must be 0..65535'); return 2; }
  const server = createServer(opts.root, opts.vercel);
  server.listen(opts.port, opts.host, () => {
    const { port } = server.address();
    console.log(`LISTENING ${port}`);
    console.error(`serving ${server.root} at http://${opts.host}:${port}/ with ${server.rules.length} header rule(s); Ctrl-C to stop`);
  });
  const shutdown = () => server.close(() => process.exit(0));
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
  return 0;
}

module.exports = { createServer, sourceToRegExp, loadHeaderRules, headersFor, resolveFile, contentType, MIME };

if (require.main === module) {
  const code = main(process.argv.slice(2));
  if (code !== 0) process.exit(code);
}

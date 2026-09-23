'use strict';
const $ = id => document.getElementById(id);
const DEFAULT_ANCHORS = [[0,0,0,0.4],[60,0,5,-0.3],[0,60,6,0.5],[60,60,20,-0.4]];
let swarm = null, fitness = null, scenario = null, timer = null, lsq = null;

function renderAnchorRows(rows) {
  const tb = $('anchors').querySelector('tbody'); tb.innerHTML = '';
  rows.forEach((r, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>a${i+1}</td>` + r.map((v, k) => `<td><input type="number" step="${k===3?0.1:1}" value="${v}" data-k="${k}"></td>`).join('') +
      `<td><button title="remove" data-i="${i}">×</button></td>`;
    tb.appendChild(tr);
  });
  tb.querySelectorAll('button').forEach(b => b.onclick = () => { const rows = readAnchorRows(); if (rows.length > 4) { rows.splice(+b.dataset.i, 1); renderAnchorRows(rows); reset(); } else alert('3D needs at least four non-coplanar anchors.'); });
  tb.querySelectorAll('input').forEach(inp => inp.onchange = reset);
}
function readAnchorRows() {
  return [...$('anchors').querySelectorAll('tbody tr')].map(tr => [...tr.querySelectorAll('input')].map(i => parseFloat(i.value) || 0));
}
function readScenario() {
  const upper = [+$('fx').value, +$('fy').value, +$('fz').value];
  const rows = readAnchorRows();
  const anchors = rows.map(r => r.slice(0, 3));
  const p = [+$('px').value, +$('py').value, +$('pz').value];
  const seed = parseInt($('seed').value) || 1;
  let noise = rows.map(r => r[3]);
  if ($('gauss').checked) { const rnd = mulberry32(seed ^ 0x9E3779B9); const s = +$('sigma').value; noise = anchors.map(() => s * gaussian(rnd)); }
  const measured = anchors.map((a, j) => dist(p, a) + noise[j]);
  return { upper, lower: [0,0,0], anchors, p, noise, measured, seed };
}
function reset() {
  stop();
  scenario = readScenario();
  fitness = new Fitness(scenario.anchors, scenario.measured);
  swarm = new Swarm({ N: Math.max(2, parseInt($('np').value) || 20), T: parseInt($('nt').value) || 60, w: +$('w').value, c: +$('c').value, c1: +$('c1').value,
    seed: scenario.seed, clip: $('clip').checked, lower: scenario.lower, upper: scenario.upper, variant: $('variant').value }, fitness);
  lsq = leastSquares(scenario.anchors, scenario.measured);
  draw(true);
}
function stepOnce() { if (!swarm || swarm.t >= swarm.cfg.T) { stop(); return false; } swarm.step(); draw(false); return true; }
function play() { if (timer) { stop(); return; } $('play').textContent = 'Pause'; const ms = Math.max(0, +$('speed').value); timer = setInterval(() => { if (!stepOnce()) stop(); }, ms); }
function stop() { if (timer) clearInterval(timer); timer = null; $('play').textContent = 'Play'; }
function finish() { stop(); while (swarm.t < swarm.cfg.T) swarm.step(); draw(false); }

function draw(init) {
  const s = scenario, sw = swarm;
  const est = sw.best, err = est ? dist(est, s.p) : NaN;
  const trail = sw.estHistory;
  const data = [
    { type: 'scatter3d', mode: 'markers+text', name: 'anchors', x: s.anchors.map(a => a[0]), y: s.anchors.map(a => a[1]), z: s.anchors.map(a => a[2]),
      text: s.anchors.map((_, j) => 'a' + (j+1)), textposition: 'top center', marker: { symbol: 'square', size: 7, color: '#1B474D' } },
    { type: 'scatter3d', mode: 'markers', name: 'particles', x: sw.x.map(p => p[0]), y: sw.x.map(p => p[1]), z: sw.x.map(p => p[2]), marker: { size: 3.5, color: '#20808D', opacity: 0.85 } },
    { type: 'scatter3d', mode: 'markers', name: 'true node p', x: [s.p[0]], y: [s.p[1]], z: [s.p[2]], marker: { symbol: 'diamond', size: 8, color: '#A84B2F' } },
  ];
  if (est) data.push({ type: 'scatter3d', mode: 'markers', name: 'estimate p̂', x: [est[0]], y: [est[1]], z: [est[2]], marker: { symbol: 'x', size: 7, color: '#111' } });
  if (trail.length > 1) data.push({ type: 'scatter3d', mode: 'lines', name: 'best-estimate trail', x: trail.map(p => p[0]), y: trail.map(p => p[1]), z: trail.map(p => p[2]), line: { color: '#7A7974', width: 2 }, opacity: 0.6 });
  if (lsq && lsq.every(Number.isFinite)) data.push({ type: 'scatter3d', mode: 'markers', name: 'least-squares', x: [lsq[0]], y: [lsq[1]], z: [lsq[2]], marker: { symbol: 'circle-open', size: 7, color: '#FFC553' } });
  // spheres of measured range (wireframe circles) for the first four anchors, light
  const layout = { margin: { l: 0, r: 0, t: 30, b: 0 }, paper_bgcolor: 'rgba(0,0,0,0)', legend: { x: 0, y: 1, font: { size: 11 } },
    title: { text: `iteration ${sw.t} / ${sw.cfg.T}`, font: { size: 14 }, x: 0.02 }, uirevision: 'keep',
    scene: { xaxis: { range: [0, s.upper[0]], title: 'x [m]' }, yaxis: { range: [0, s.upper[1]], title: 'y [m]' }, zaxis: { range: [0, s.upper[2]], title: 'z [m]' },
      aspectmode: 'manual', aspectratio: { x: 1, y: s.upper[1] / s.upper[0], z: s.upper[2] / s.upper[0] } } };
  Plotly.react('plot3d', data, layout, { displaylogo: false, responsive: true });
  const conv = [{ x: sw.history.map((_, i) => i + (sw.cfg.variant === 'standard' ? 0 : 1)), y: sw.history, type: 'scatter', mode: 'lines', line: { color: '#20808D', width: 2 }, name: 'best fitness' }];
  Plotly.react('plotConv', conv, { margin: { l: 55, r: 10, t: 28, b: 38 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', title: { text: 'best fitness f(x_best) per iteration (log scale)', font: { size: 13 }, x: 0.02 },
    xaxis: { title: 'iteration', range: [0, sw.cfg.T] }, yaxis: { type: 'log', title: 'f [m²]' }, uirevision: 'keep' }, { displaylogo: false, responsive: true });
  const fmt = (v, d = 2) => Number.isFinite(v) ? v.toFixed(d) : '–';
  const lsqErr = lsq ? dist(lsq, s.p) : NaN;
  const kpis = [
    ['estimate p̂ (m)', est ? `${fmt(est[0])}, ${fmt(est[1])}, ${fmt(est[2])}` : '–'],
    ['error ‖p̂ − p‖', fmt(err) + ' m'],
    ['best fitness', fmt(sw.bestF, 4) + ' m²'],
    ['fitness evaluations', fitness.evals.toLocaleString()],
    ['distance computations', fitness.distances.toLocaleString()],
    ['swarm memory', `${sw.memoryFloats()} floats · ${sw.memoryFloats() * 8} B`],
    ['least-squares error', fmt(lsqErr) + ' m'],
    ['measured ranges d̂ (m)', s.measured.map(v => v.toFixed(2)).join(', ')],
  ];
  $('stats').innerHTML = kpis.map(([l, v]) => `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');
}

renderAnchorRows(DEFAULT_ANCHORS);
['fx','fy','fz','px','py','pz','seed','np','nt','w','c','c1','clip','variant','gauss','sigma'].forEach(id => $(id).onchange = reset);
$('addAnchor').onclick = () => { const rows = readAnchorRows(); const r = mulberry32(Date.now() & 0xffff); rows.push([Math.round(r() * +$('fx').value), Math.round(r() * +$('fy').value), Math.round(r() * +$('fz').value), 0]); renderAnchorRows(rows); reset(); };
$('randomNode').onclick = () => { const r = mulberry32(Date.now() & 0xffff); $('px').value = (r() * +$('fx').value).toFixed(1); $('py').value = (r() * +$('fy').value).toFixed(1); $('pz').value = (r() * +$('fz').value).toFixed(1); reset(); };
$('reset').onclick = reset; $('step').onclick = () => stepOnce(); $('play').onclick = play; $('finish').onclick = finish;
reset();

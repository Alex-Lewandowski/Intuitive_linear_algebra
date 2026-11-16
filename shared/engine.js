// Shared engine for Intuitive Linear Algebra: 2D matrix math, randomness,
// SVG drawing helpers, drag handling and progress storage.
// Plain script (no modules) so pages also work when opened straight from disk.
(function () {
  const LA = (window.LA = {});

  // ---------- 2x2 math (a matrix is {a: column 1, b: column 2}) ----------
  LA.mk = (a, b) => ({ a, b });
  LA.app = (M, v) => [v[0] * M.a[0] + v[1] * M.b[0], v[0] * M.a[1] + v[1] * M.b[1]];
  LA.T = (M) => ({ a: [M.a[0], M.b[0]], b: [M.a[1], M.b[1]] });
  LA.rowsApp = (M, v) => LA.app(LA.T(M), v); // the "read rows as arrows" mistake
  LA.det = (M) => M.a[0] * M.b[1] - M.b[0] * M.a[1];
  LA.same = (p, q) => p[0] === q[0] && p[1] === q[1];
  LA.sameM = (M, N) => LA.same(M.a, N.a) && LA.same(M.b, N.b);
  LA.add = (p, q) => [p[0] + q[0], p[1] + q[1]];
  LA.scale = (k, p) => [k * p[0], k * p[1]];
  LA.inR = (v, r = 6) => Math.abs(v[0]) <= r && Math.abs(v[1]) <= r;
  LA.maxAbs = (M) => Math.max(...M.a.map(Math.abs), ...M.b.map(Math.abs));
  LA.cross = (p, q) => p[0] * q[1] - p[1] * q[0];
  LA.isZero = (v) => v[0] === 0 && v[1] === 0;
  LA.clamp = (x, r = 6) => Math.max(-r, Math.min(r, x));
  LA.gcd = (x, y) => { x = Math.abs(x); y = Math.abs(y); while (y) [x, y] = [y, x % y]; return x; };
  LA.solutions = (M, t, r = 6) => {
    const s = [];
    for (let x = -r; x <= r; x++) for (let y = -r; y <= r; y++) if (LA.same(LA.app(M, [x, y]), t)) s.push([x, y]);
    return s;
  };

  // ---------- formatting ----------
  LA.n = (x) => String(Math.round(x * 100) / 100).replace(/-/g, '−');
  LA.fmt = (v) => `(${LA.n(v[0])}, ${LA.n(v[1])})`;
  LA.matHTML = (M) =>
    `<span class="mat"><span class="br l"></span><span class="col c1"><span>${LA.n(M.a[0])}</span><span>${LA.n(M.a[1])}</span></span><span class="col c2"><span>${LA.n(M.b[0])}</span><span>${LA.n(M.b[1])}</span></span><span class="br r"></span></span>`;
  const term = (c, v, first) => {
    if (c === 0) return '';
    const m = Math.abs(c) === 1 ? '' : Math.abs(c);
    return first ? (c < 0 ? '−' : '') + m + v : (c < 0 ? ' − ' : ' + ') + m + v;
  };
  LA.eqLine = (c1, c2, r) => {
    let s = term(c1, 'x', true);
    s += s ? term(c2, 'y', false) : term(c2, 'y', true);
    return `${s || '0'} = ${LA.n(r)}`;
  };

  // ---------- randomness ----------
  LA.ri = (a, b) => a + Math.floor(Math.random() * (b - a + 1));
  LA.nz = (a, b) => { let x; do x = LA.ri(a, b); while (x === 0); return x; };
  LA.pick = (arr) => arr[Math.floor(Math.random() * arr.length)];
  LA.shuffle = (a) => { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

  // ---------- progress (one record for the whole site) ----------
  const KEY = 'ila-progress';
  LA.loadProgress = () => { try { return JSON.parse(localStorage.getItem(KEY) || '{}') || {}; } catch (e) { return {}; } };
  LA.saveProgress = (p) => { try { localStorage.setItem(KEY, JSON.stringify(p)); } catch (e) {} };

  // ---------- SVG drawing (math coordinates; the board's <g> flips y) ----------
  LA.C = { c1: '#d9581a', c2: '#0f8f84', gold: '#c99700', ink: '#1f1d1a', gray: '#7d776d' };
  const NS = 'vector-effect="non-scaling-stroke"';
  LA.line = (p, q, c, w, x = '') => `<line x1="${p[0]}" y1="${p[1]}" x2="${q[0]}" y2="${q[1]}" stroke="${c}" stroke-width="${w}" stroke-linecap="round" ${NS} ${x}/>`;
  LA.dot = (p, r, f, x = '') => `<circle cx="${p[0]}" cy="${p[1]}" r="${r}" fill="${f}" ${x}/>`;
  LA.ring = (p, c, r = 0.32) => `<circle cx="${p[0]}" cy="${p[1]}" r="${r}" fill="none" stroke="${c}" stroke-width="2.5" ${NS}/>`;
  LA.arrow = (p, q, c, op = 1) => {
    const dx = q[0] - p[0], dy = q[1] - p[1], len = Math.hypot(dx, dy);
    if (len < 1e-6) return LA.dot(p, 0.13, c);
    const ux = dx / len, uy = dy / len, h = Math.min(0.38, len * 0.5), w = 0.2, bx = q[0] - ux * h, by = q[1] - uy * h;
    return `<g opacity="${op}">${LA.line(p, [bx, by], c, 3.5)}<polygon points="${q[0]},${q[1]} ${bx - uy * w},${by + ux * w} ${bx + uy * w},${by - ux * w}" fill="${c}"/></g>`;
  };
  LA.grid = () => {
    let s = '';
    for (let i = -6; i <= 6; i++) {
      const c = i === 0 ? '#b9b2a6' : '#eee9e0', w = i === 0 ? 1.6 : 1;
      s += LA.line([i, -7], [i, 7], c, w) + LA.line([-7, i], [7, i], c, w);
    }
    return s;
  };
  LA.warped = (M) => {
    let s = '';
    for (let i = -8; i <= 8; i++)
      s += LA.line(LA.app(M, [i, -9]), LA.app(M, [i, 9]), 'rgba(15,143,132,.28)', 1) + LA.line(LA.app(M, [-9, i]), LA.app(M, [9, i]), 'rgba(217,88,26,.25)', 1);
    return s;
  };
  LA.square = (M, fill, stroke = 'none', x = '') => {
    const p = [[0, 0], M.a, LA.add(M.a, M.b), M.b];
    return `<polygon points="${p.map((q) => q.join(',')).join(' ')}" fill="${fill}" stroke="${stroke}" stroke-width="2" ${NS} ${x}/>`;
  };
  LA.handle = (p, c) => LA.dot(p, 0.6, c, 'opacity=".18"') + LA.dot(p, 0.3, c, `stroke="#fff" stroke-width="2" ${NS}`);
  LA.star = (p, r = 0.5, fill = '#fff3c4', stroke = '#1f1d1a') => {
    const pts = [];
    for (let i = 0; i < 10; i++) {
      const ang = Math.PI / 2 + (i * Math.PI) / 5, rr = i % 2 ? r * 0.45 : r;
      pts.push([p[0] + rr * Math.cos(ang), p[1] + rr * Math.sin(ang)].map((x) => x.toFixed(3)).join(','));
    }
    return `<polygon points="${pts.join(' ')}" fill="${fill}" stroke="${stroke}" stroke-width="2" stroke-linejoin="round" ${NS}/>`;
  };

  // ---------- dragging (mouse + touch), snaps to whole numbers ----------
  // opts.handles() -> [[key, pos], ...]; opts.onMove(key, point); opts.onDrop()
  LA.board = (svg, g, opts) => {
    const toU = (e) => {
      const pt = svg.createSVGPoint(); pt.x = e.clientX; pt.y = e.clientY;
      const q = pt.matrixTransform(g.getScreenCTM().inverse()); return [q.x, q.y];
    };
    let drag = null, last = null;
    const move = (p) => {
      const q = [LA.clamp(Math.round(p[0])), LA.clamp(Math.round(p[1]))];
      if (last && LA.same(q, last)) return;
      last = q; opts.onMove(drag, q);
    };
    svg.addEventListener('pointerdown', (e) => {
      const p = toU(e), hs = opts.handles(); if (!hs.length) return;
      let best = null, bd = 1.4;
      for (const [k, pos] of hs) { const d = Math.hypot(pos[0] - p[0], pos[1] - p[1]); if (d < bd) { bd = d; best = k; } }
      if (!best && hs.length === 1) best = hs[0][0]; // tap anywhere to place a single dot
      if (!best) return;
      drag = best; last = hs.find((h) => h[0] === best)[1];
      svg.setPointerCapture(e.pointerId); move(p); e.preventDefault();
    });
    svg.addEventListener('pointermove', (e) => { if (drag) move(toU(e)); });
    const end = () => { if (drag) { drag = null; opts.onDrop(); } };
    svg.addEventListener('pointerup', end);
    svg.addEventListener('pointercancel', end);
  };
})();

// Chapter 1: Matrix Machines. Every problem is generated fresh.
(function () {
  const { mk, app, T, rowsApp, det, same, add, scale, inR, maxAbs, cross, isZero, gcd, solutions, n, fmt, ri, nz, pick, shuffle, C } = LA;
  const O = '<span class="c1">orange</span>', TL = '<span class="c2">teal</span>';
  const $ = (id) => document.getElementById(id);

  // ---------- matrix families (entries stay between −3 and 3) ----------
  const F = {
    stretch() { let p, q; do { p = nz(-3, 3); q = nz(-3, 3); } while (Math.abs(p) === 1 && Math.abs(q) === 1); return mk([p, 0], [0, q]); },
    stretchPos() { let p, q; do { p = ri(1, 3); q = ri(1, 3); } while (p === 1 && q === 1); return mk([p, 0], [0, q]); },
    shear() { const k = pick([1, 2, -1, -2]); return Math.random() < 0.6 ? mk([1, 0], [k, 1]) : mk([1, k], [0, 1]); },
    turn() { const k = Math.random() < 0.65 ? 1 : 2; return pick([mk([0, k], [-k, 0]), mk([0, -k], [k, 0]), mk([-k, 0], [0, -k])]); },
    mirror() { return pick([mk([1, 0], [0, -1]), mk([-1, 0], [0, 1]), mk([0, 1], [1, 0]), mk([0, -1], [-1, 0])]); },
    flatten() {
      if (Math.random() < 0.25) return pick([mk([1, 0], [0, 0]), mk([0, 0], [0, 1])]);
      for (;;) {
        const u = pick([[1, 1], [1, 2], [2, 1], [1, -1], [1, 0], [0, 1]]), s = nz(-2, 2), t = nz(-2, 2);
        const M = mk(scale(s, u), scale(t, u));
        if (maxAbs(M) <= 3) return M;
      }
    },
    general() { let M; do M = mk([ri(-2, 3), ri(-2, 3)], [ri(-2, 3), ri(-2, 3)]); while (det(M) === 0 || describe(M)); return M; },
  };
  const INVERTIBLE = [F.stretch, F.shear, F.turn, F.mirror, F.general];
  const DESCRIBABLE = [F.stretchPos, F.shear, F.turn, F.mirror, F.flatten];

  // Plain-words description of a family matrix (null if it has no simple one).
  function describe(M) {
    const [a0, a1] = M.a, [b0, b1] = M.b, d = det(M);
    if (isZero(M.a) && isZero(M.b)) return 'Sends everything to the origin';
    if (d === 0) {
      let u = isZero(M.a) ? M.b : M.a; const g = gcd(u[0], u[1]); u = [u[0] / g, u[1] / g];
      if (u[0] < 0 || (u[0] === 0 && u[1] < 0)) u = [-u[0], -u[1]];
      if (same(u, [1, 0])) return 'Squashes everything flat onto the x-axis';
      if (same(u, [0, 1])) return 'Squashes everything flat onto the y-axis';
      return `Flattens the whole plane onto the line through ${fmt(u)}`;
    }
    if (a1 === 0 && b0 === 0) {
      if (a0 === -1 && b1 === 1) return 'Mirrors left-to-right (across the y-axis)';
      if (a0 === 1 && b1 === -1) return 'Flips upside-down (mirror across the x-axis)';
      if (a0 === b1 && a0 < 0) return `Turns everything a half-turn${a0 < -1 ? ` and makes it ${-a0} times bigger` : ''}`;
      if (a0 > 0 && b1 > 0) {
        if (a0 === b1) return `Makes everything ${a0} times bigger in every direction`;
        if (a0 === 1) return `Stretches things vertically by ${b1}; widths stay the same`;
        if (b1 === 1) return `Stretches things horizontally by ${a0}; heights stay the same`;
        return `Stretches horizontally by ${a0} and vertically by ${b1}`;
      }
      return null;
    }
    if (a0 === 0 && b1 === 0 && a1 === -b0 && a1 !== 0) { const k = Math.abs(a1); return `Turns everything a quarter-turn ${a1 > 0 ? 'counterclockwise' : 'clockwise'}${k > 1 ? ` and makes it ${k} times bigger` : ''}`; }
    if (same(M.a, [0, 1]) && same(M.b, [1, 0])) return 'Mirrors across the diagonal y = x (swaps x and y)';
    if (same(M.a, [0, -1]) && same(M.b, [-1, 0])) return 'Mirrors across the line y = −x';
    if (same(M.a, [1, 0]) && b1 === 1) return `Slides points ${b0 > 0 ? 'right' : 'left'}: the higher they are, the farther they slide (${Math.abs(b0)} per step up)`;
    if (same(M.b, [0, 1]) && a0 === 1) return `Slides points ${a1 > 0 ? 'up' : 'down'}: the farther right they are, the farther they slide (${Math.abs(a1)} per step right)`;
    return null;
  }

  // "(2,1)" written as a recipe of basic steps: "2·(1,0) + 1·(0,1)"
  function recipe(u) {
    const parts = [];
    if (u[0]) parts.push(`${n(u[0])}·(1,0)`);
    if (u[1]) parts.push(`${n(u[1])}·(0,1)`);
    return parts.join(' + ').replace('+ −', '− ');
  }

  // ---------- LAND IT ----------
  const vPos = () => [ri(0, 2), ri(0, 2)], vAny = () => [ri(-2, 2), ri(-2, 2)];
  const vNeg = () => { let v; do v = vAny(); while (v[0] >= 0 && v[1] >= 0); return v; };
  const landQ = {
    stretch: () => 'Can you say what this machine does to a whole picture, in one plain sentence?',
    neg: () => 'What does a negative number of arrows mean? Which way did you walk?',
    shear: () => 'Which points don\'t move at all under this shear? Why not?',
    turn: () => 'Did the square\'s area change? How can you tell from the two arrows?',
    rows: (p) => `The top <b>row</b> is ${fmt([p.M.a[0], p.M.b[0]])}. Compare it with the output's x-coordinate. What does a row tell you?`,
    flat: () => 'Every output of this machine lands on one line. Which line, and why?',
  };
  const landGen = (famFn, vFn, kind, ok) => () => {
    for (;;) {
      const M = famFn(), v = vFn(), ans = app(M, v);
      if (isZero(v) || !inR(ans) || same(ans, v)) continue;
      if (ok && !ok(M, v, ans)) continue;
      return { M, v, ans, kind, key: JSON.stringify([M, v]) };
    }
  };
  const LAND = {
    id: 'land', title: 'Land it', mode: 'land',
    blurb: 'The left column is where (1,0) lands; the right column is where (0,1) lands. Drag the gold dot to where the gray input comes out.',
    tiers: [
      ['Stretches', landGen(F.stretchPos, vPos, 'stretch')],
      ['Negative steps', landGen(() => pick([F.stretch, F.general])(), vNeg, 'neg')],
      ['Shears', landGen(F.shear, vAny, 'shear')],
      ['Turns & mirrors', landGen(() => pick([F.turn, F.mirror])(), vAny, 'turn')],
      ['Columns, not rows', landGen(F.general, vAny, 'rows', (M, v, ans) => !same(rowsApp(M, v), ans))],
      ['Flatteners', landGen(F.flatten, vAny, 'flat')],
    ],
  };

  // ---------- FIND IT ----------
  const PAIRS = {
    cols: [[[1, 0], [0, 1]]],
    sub: [[[1, 0], [1, 1]], [[0, 1], [1, 1]], [[1, 0], [1, -1]], [[0, 1], [-1, 1]], [[0, 1], [1, -1]], [[1, 0], [-1, 1]]],
    half: [[[2, 0], [0, 1]], [[1, 0], [0, 2]], [[2, 0], [0, 2]], [[0, 2], [1, 1]], [[2, 0], [1, 1]]],
    mix: [[[1, 1], [1, -1]], [[1, 1], [2, 1]], [[1, 1], [1, 2]], [[2, 1], [1, 0]], [[1, 2], [0, 1]]],
  };
  const findGen = (kind) => () => {
    for (;;) {
      const M = Math.random() < 0.5 ? F.general() : pick([F.stretch, F.shear, F.turn, F.mirror, F.flatten])();
      let pair, dep = false;
      if (kind === 'enough' && Math.random() < 0.5) {
        const u = pick([[1, 0], [0, 1], [1, 1], [1, -1], [1, 2], [2, 1]]), w = scale(pick([2, -1, -2]), u);
        if (!inR(w, 2)) continue;
        pair = [u, w]; dep = true;
      } else pair = pick(PAIRS[kind === 'enough' ? 'mix' : kind]);
      const clues = shuffle(pair).map((u) => [u, app(M, u)]);
      if (clues.some((c) => !inR(c[1]))) continue;
      return { M, clues, dep, kind, key: JSON.stringify(clues) };
    }
  };
  const FIND = {
    id: 'find', title: 'Find it', mode: 'find',
    blurb: 'Two clues say what a hidden machine does to two inputs. Drag the arrow tips to build a machine that fits both clues.',
    tiers: [
      ['Clues are columns', findGen('cols')],
      ['Subtract', findGen('sub')],
      ['Halve', findGen('half')],
      ['Combine clues', findGen('mix')],
      ['Enough clues?', findGen('enough')],
    ],
  };

  // ---------- SAY IT ----------
  const wordsGen = (famFn) => () => {
    for (;;) {
      const M = famFn(), desc = describe(M);
      if (!desc) continue;
      const opts = [desc], td = describe(T(M));
      const trap = td && td !== desc ? td : null;
      if (trap) opts.push(trap);
      for (let g = 0; opts.length < 4 && g < 300; g++) { const d = describe(pick(DESCRIBABLE)()); if (d && !opts.includes(d)) opts.push(d); }
      return { M, desc, trap, choices: shuffle(opts), key: JSON.stringify(M) };
    }
  };
  const WORDS = {
    id: 'words', title: 'Say it', mode: 'words',
    blurb: 'Read the columns, picture the two arrows, then pick the plain-words description. Predict first, then watch.',
    tiers: [
      ['Stretches', wordsGen(F.stretchPos)],
      ['Flatteners', wordsGen(F.flatten)],
      ['Shears', wordsGen(F.shear)],
      ['Turns & mirrors', wordsGen(() => pick([F.turn, F.mirror])())],
    ],
  };

  // ---------- UNDO IT ----------
  const revOne = (famFn) => () => {
    for (;;) {
      const M = famFn(), v = [ri(-2, 2), ri(-2, 2)];
      if (isZero(v)) continue;
      const t = app(M, v);
      if (!inR(t) || isZero(t)) continue;
      return { M, t, kind: 'one', sol: v, key: JSON.stringify([M, t]) };
    }
  };
  const revMany = () => {
    for (;;) {
      const M = F.flatten(), v = [ri(-2, 2), ri(-2, 2)], t = app(M, v);
      if (isZero(t) || !inR(t)) continue;
      const sols = solutions(M, t);
      if (sols.length < 2) continue;
      let d = [sols[1][0] - sols[0][0], sols[1][1] - sols[0][1]]; const g = gcd(d[0], d[1]); d = [d[0] / g, d[1] / g];
      return { M, t, kind: 'many', sols, nullv: d, key: JSON.stringify([M, t]) };
    }
  };
  const revNone = () => {
    for (;;) {
      const M = F.flatten(), u0 = isZero(M.a) ? M.b : M.a, t = [ri(-4, 4), ri(-4, 4)];
      if (isZero(t) || cross(u0, t) === 0) continue;
      const g = gcd(u0[0], u0[1]); let u = [u0[0] / g, u0[1] / g];
      if (u[0] < 0 || (u[0] === 0 && u[1] < 0)) u = [-u[0], -u[1]];
      return { M, t, kind: 'none', u, key: JSON.stringify([M, t]) };
    }
  };
  const UNDO = {
    id: 'undo', title: 'Undo it', mode: 'undo',
    blurb: 'Run the machine backwards. Drag the gray input until the gold output hits the star, or decide that no input can.',
    tiers: [
      ['Stretches', revOne(F.stretch)],
      ['Shears, turns & mirrors', revOne(() => pick([F.shear, F.turn, F.mirror])())],
      ['Any machine', revOne(F.general)],
      ['Flatteners', () => (Math.random() < 0.5 ? revMany() : revNone())],
    ],
    // endless mix: about 60% one answer, 20% many, 20% none
    mix: () => { const r = Math.random(); return r < 0.6 ? revOne(pick(INVERTIBLE))() : r < 0.8 ? revMany() : revNone(); },
  };

  const SECTIONS = [LAND, FIND, WORDS, UNDO];

  // ---------- progress ----------
  const P = LA.loadProgress();
  P.ch1 = P.ch1 || {};
  const CP = P.ch1;
  const sp = (sec) => (CP[sec.id] = CP[sec.id] || { tier: 0, solved: [], streak: 0, best: 0 });
  const save = () => LA.saveProgress(P);
  const recent = {};

  // ---------- state ----------
  let sec = SECTIONS.find((s) => s.id === CP.sec) || LAND;
  let R = {}; // the current problem and its runtime state

  function newProblem() {
    const s = sp(sec), N = sec.tiers.length;
    const gen = s.tier >= N ? sec.mix || (() => pick(sec.tiers)[1]()) : sec.tiers[s.tier][1];
    const rec = (recent[sec.id] = recent[sec.id] || []);
    let p;
    for (let i = 0; i < 40; i++) { p = gen(); if (!rec.includes(p.key)) break; }
    rec.push(p.key); if (rec.length > 10) rec.shift();
    R = { p, hint: 0, tries: 0, won: false, showPath: false, showWarp: false, found: new Set(), wrong: new Set(), t: 0, anim: 0, levelUp: false };
    if (sec.mode === 'land') R.pt = [0, 0];
    if (sec.mode === 'find') { R.a = [1, 0]; R.b = [0, 1]; }
    if (sec.mode === 'undo') { R.in = [0, 0]; R.trail = [[0, 0]]; }
    hideFb(); ui(); render();
  }
  function goSection(s) { sec = s; CP.sec = s.id; save(); newProblem(); }
  function goTier(i) { sp(sec).tier = i; save(); newProblem(); }

  // ---------- UI ----------
  function btn(label, fn, primary) { const b = document.createElement('button'); b.textContent = label; if (primary) b.className = 'primary'; b.onclick = fn; $('btns').appendChild(b); }
  function showFb(kind, html) { const f = $('fb'); f.className = 'fb show ' + kind; f.innerHTML = html; }
  function hideFb() { $('fb').className = 'fb'; }
  const curM = () => (sec.mode === 'find' ? mk(R.a, R.b) : R.p.M);

  function panel() {
    const M = curM();
    $('mat').innerHTML = LA.matHTML(M);
    let leg = '<span class="c1">left column</span> = where (1,0) lands<br><span class="c2">right column</span> = where (0,1) lands';
    if (sec.mode === 'find') { const d = det(M); leg += `<br>det = ${n(d)} → area × ${n(Math.abs(d))}${d < 0 ? ' (flipped)' : ''}`; }
    $('legend').innerHTML = leg;
  }

  function ui() {
    const s = sp(sec), N = sec.tiers.length, p = R.p;
    // section tabs
    $('tabs').innerHTML = '';
    SECTIONS.forEach((x) => { const b = document.createElement('button'); b.textContent = x.title; if (x === sec) b.className = 'on'; b.onclick = () => goSection(x); $('tabs').appendChild(b); });
    $('blurb').textContent = sec.blurb;
    // steps
    $('steps').innerHTML = '';
    for (let i = 0; i <= N; i++) {
      const b = document.createElement('button');
      b.textContent = i < N ? i + 1 : '∞';
      b.title = i < N ? sec.tiers[i][0] : 'Endless mix';
      b.className = ((s.solved[i] || 0) >= 2 && i < N ? 'done ' : '') + (i === s.tier ? 'cur' : '');
      b.onclick = () => goTier(i);
      $('steps').appendChild(b);
    }
    $('stepname').textContent = s.tier < N ? `Step ${s.tier + 1} of ${N}: ${sec.tiers[s.tier][0]}` : 'Endless mix: every kind of problem';
    $('streak').textContent = `Streak ${s.streak} · best ${s.best}`;
    // question
    let ask = '';
    if (sec.mode === 'land') ask = `The gray input <b>${fmt(p.v)}</b> goes into the machine. Drag the <span class="gold">gold dot</span> to where it comes out.`;
    if (sec.mode === 'find') ask = 'Build a machine that fits both clues.';
    if (sec.mode === 'words') ask = 'In plain words, what does this machine do?';
    if (sec.mode === 'undo') ask = `Which input lands on the star <b>${fmt(p.t)}</b>?`;
    $('ask').innerHTML = ask;
    panel();
    // extra box
    if (sec.mode === 'find') $('extra').innerHTML = `<div class="box">${p.clues.map(([u, img], i) => `Clue ${i + 1}: ${fmt(u)} → ${fmt(img)}`).join('<br>')}<small>Drag the <span class="c1">orange</span> tip (where (1,0) lands) and the <span class="c2">teal</span> tip (where (0,1) lands).</small></div>`;
    else if (sec.mode === 'undo') { const M = p.M; $('extra').innerHTML = `<div class="box">${LA.eqLine(M.a[0], M.b[0], p.t[0])}<br>${LA.eqLine(M.a[1], M.b[1], p.t[1])}<small>Each equation reads one <b>row</b>. The picture reads <b>columns</b>: x·<span class="c1">${fmt(M.a)}</span> + y·<span class="c2">${fmt(M.b)}</span> = ${fmt(p.t)}.</small></div>`; }
    else $('extra').innerHTML = '';
    // choices
    $('choices').innerHTML = '';
    if (sec.mode === 'words') p.choices.forEach((c) => {
      const b = document.createElement('button'); b.textContent = c;
      if (R.won && c === p.desc) b.className = 'right'; else if (R.wrong.has(c)) b.className = 'wrong';
      b.onclick = () => choose(c); $('choices').appendChild(b);
    });
    // buttons
    $('btns').innerHTML = '';
    if (!R.won) {
      if (sec.mode === 'land' || sec.mode === 'find') btn('Check', sec.mode === 'land' ? checkLand : checkFind, true);
      if (sec.mode === 'find' && (s.tier >= 4)) btn('Not enough clues', notEnough);
      if (sec.mode === 'undo') btn('No input can do it', impossible);
      btn('Hint', hint);
      if (sec.mode === 'land' && (R.tries >= 2 || R.hint >= 2) && !R.showPath) btn('Show me the walk', showWalk);
    } else {
      if (sec.mode === 'words') btn('Watch again', play);
      btn(R.levelUp ? 'Next step →' : 'Next problem →', next, true);
    }
  }

  // ---------- drawing ----------
  function render() {
    const p = R.p; let s = LA.grid(), ro = '';
    if (sec.mode === 'land') {
      if (R.showWarp) s += LA.warped(p.M);
      s += LA.square(p.M, 'rgba(31,29,26,.06)');
      s += LA.arrow([0, 0], p.M.a, C.c1) + LA.arrow([0, 0], p.M.b, C.c2);
      s += LA.ring(p.v, C.gray) + LA.dot(p.v, 0.1, C.gray);
      if (R.showPath) { const m = scale(p.v[0], p.M.a); s += LA.line([0, 0], m, C.c1, 5, 'stroke-dasharray="1 7"') + LA.line(m, p.ans, C.c2, 5, 'stroke-dasharray="1 7"'); }
      s += LA.handle(R.pt, C.gold);
      ro = `input ${fmt(p.v)} · gold dot ${fmt(R.pt)}`;
    }
    if (sec.mode === 'find') {
      const B = mk(R.a, R.b);
      s += LA.warped(B) + LA.square(B, 'rgba(201,151,0,.18)');
      p.clues.forEach(([, img]) => (s += LA.ring(img, C.ink, 0.22)));
      s += LA.arrow([0, 0], R.a, C.c1) + LA.arrow([0, 0], R.b, C.c2) + LA.handle(R.a, C.c1) + LA.handle(R.b, C.c2);
      ro = `orange tip ${fmt(R.a)} · teal tip ${fmt(R.b)}`;
    }
    if (sec.mode === 'words') {
      const t = R.t, M = mk([1 + (p.M.a[0] - 1) * t, p.M.a[1] * t], [p.M.b[0] * t, 1 + (p.M.b[1] - 1) * t]);
      s += LA.warped(M) + LA.square(M, 'rgba(201,151,0,.22)', C.ink) + LA.arrow([0, 0], M.a, C.c1) + LA.arrow([0, 0], M.b, C.c2);
      ro = R.won ? 'Watch where the two arrows take the square.' : 'Predict first. The animation plays after you answer.';
    }
    if (sec.mode === 'undo') {
      const out = app(p.M, R.in);
      s += LA.arrow([0, 0], p.M.a, C.c1, 0.45) + LA.arrow([0, 0], p.M.b, C.c2, 0.45);
      R.trail.forEach((q) => (s += LA.dot(q, 0.12, C.gold, 'opacity=".45"')));
      s += LA.star(p.t);
      s += LA.line(R.in, out, C.gray, 1.5, 'stroke-dasharray="3 4"');
      s += LA.handle(R.in, C.gray) + LA.dot(out, 0.26, C.gold, 'stroke="#fff" stroke-width="2" vector-effect="non-scaling-stroke"');
      ro = `input ${fmt(R.in)} → output <span class="gold">${fmt(out)}</span>`;
    }
    $('g').innerHTML = s; $('readout').innerHTML = ro;
  }

  LA.board($('board'), $('g'), {
    handles: () => (sec.mode === 'land' ? [['pt', R.pt]] : sec.mode === 'find' ? [['a', R.a], ['b', R.b]] : sec.mode === 'undo' ? [['in', R.in]] : []),
    onMove: (k, q) => {
      R[k] = q;
      if (sec.mode === 'undo') { const o = app(R.p.M, q); if (!R.trail.some((t) => same(t, o))) R.trail.push(o); }
      if (sec.mode === 'find') panel();
      render();
    },
    onDrop: () => { if (sec.mode === 'undo') checkUndo(); },
  });

  // ---------- scoring ----------
  function win(text) {
    if (R.won) return;
    const s = sp(sec), N = sec.tiers.length;
    R.won = true;
    s.solved[s.tier] = (s.solved[s.tier] || 0) + 1;
    s.streak++; s.best = Math.max(s.best, s.streak);
    if (s.tier < N && s.solved[s.tier] === 2) R.levelUp = true;
    save();
    const up = R.levelUp ? `<br><br><b>Step complete!</b> Next up: ${s.tier + 1 < N ? sec.tiers[s.tier + 1][0] : 'the endless mix'}.` : '';
    showFb('good', '<b>Nice.</b> ' + text + up);
    ui(); render();
  }
  function wrong(html) { const s = sp(sec); s.streak = 0; save(); R.tries++; showFb('bad', html); ui(); }
  function next() {
    const s = sp(sec);
    if (R.levelUp) s.tier = Math.min(s.tier + 1, sec.tiers.length);
    save(); newProblem(); window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // ---------- hints ----------
  function hints() {
    const p = R.p;
    if (sec.mode === 'land') {
      const { M, v } = p;
      return [
        `Where does (1,0) land? That's the ${O} arrow ${fmt(M.a)}. And (0,1)? The ${TL} arrow ${fmt(M.b)}. How many of each does ${fmt(v)} ask for?`,
        `${n(v[0])} ${O} arrows take you to ${fmt(scale(v[0], M.a))}. From there, add ${n(v[1])} ${TL} arrows. Where do you end up?`,
      ];
    }
    if (sec.mode === 'find') {
      const [[u, iu], [w, iw]] = p.clues;
      const byKind = {
        cols: ['The clue inputs are (1,0) and (0,1) themselves. So where must each arrow tip go?'],
        sub: ['One clue tells you an arrow directly. Which one?', (() => { const c = u[0] !== 0 && u[1] !== 0 ? u : w; return `${fmt(c)} is ${recipe(c)}. So where it lands is that same mix of the ${O} and ${TL} arrows. You know one arrow already. What's left over?`; })()],
        half: ['One clue input is twice a basic step. If twice the step lands there, where does the step itself land?', 'Halve the landing spot: half of (4, 2) is (2, 1).'],
        mix: [`Try adding or subtracting the two clue inputs. What do you get, and where must it land?`, `${fmt(u)} + ${fmt(w)} = ${fmt(add(u, w))}, and it lands at ${fmt(add(iu, iw))}. Can you get to (1,0) or (0,1) by combining the clues like that?`],
        enough: ['Do the two clue inputs point in different directions? If one is just a multiple of the other, does the second clue tell you anything new?'],
      };
      return byKind[p.kind];
    }
    if (sec.mode === 'words') return ['Where does (1,0) go? That\'s the left column. Picture that arrow.', 'Now where does (0,1) go? Picture both arrows and the square they make.'];
    if (sec.mode === 'undo') {
      if (p.kind === 'one') return [`Drag the gray dot one step right. How does the gold dot move? (That's the ${O} arrow.)`, 'Look at the equations. Is there one you can solve first, or a way to combine them?'];
      return ['Drag around and watch the gold trail. Where can outputs go at all?', `Every output is x·${fmt(p.M.a)} + y·${fmt(p.M.b)}. Both arrows lie along one line. Is the star on it?`];
    }
  }
  function hint() {
    const h = hints();
    showFb('hint', h[Math.min(R.hint, h.length - 1)]);
    R.hint++;
    if (sec.mode === 'land' && R.hint >= 2) { R.showWarp = true; render(); $('fb').innerHTML += '<br><small>I\'ve drawn the moved grid. Count along its lines.</small>'; }
    ui();
  }

  // ---------- LAND checks ----------
  function showWalk() {
    const { M, v } = R.p; R.showPath = true; render();
    showFb('hint', `Follow the dotted walk: ${n(v[0])} × ${O} ${fmt(M.a)}, then ${n(v[1])} × ${TL} ${fmt(M.b)}. Where does the walk end? Drag there and check.`); ui();
  }
  function checkLand() {
    const p = R.p;
    if (same(R.pt, p.ans)) { R.showPath = true; win(`${fmt(p.v)} → ${fmt(p.ans)}: that's ${n(p.v[0])} ${O} arrows plus ${n(p.v[1])} ${TL} arrows. ${landQ[p.kind](p)}`); return; }
    if (same(R.pt, rowsApp(p.M, p.v))) return wrong(`You landed on ${fmt(R.pt)}. That's what you get by reading the <b>rows</b> as arrows. In the picture, which shows where (1,0) goes: the top row or the <span class="c1">left column</span>?`);
    if (same(R.pt, p.v)) return wrong('That\'s where the input started. The machine moves it. Where does it come out?');
    if (isZero(R.pt)) { showFb('hint', 'Tap or drag the gold dot to where you think the input lands.'); return; }
    wrong(`Not quite. How many ${O} arrows does ${fmt(p.v)} call for, and how many ${TL} ones?`);
  }

  // ---------- FIND checks ----------
  function checkFind() {
    const p = R.p, B = mk(R.a, R.b);
    const bad = p.clues.find(([u, img]) => !same(app(B, u), img));
    if (!bad) {
      if (p.dep) { showFb('hint', 'That machine does fit both clues. Is it the only one? Try moving an arrow tip so both clues still fit. If you can, what does that say about the clues?'); return; }
      const d = det(B);
      return win(`Columns ${fmt(B.a)} and ${fmt(B.b)}. The determinant is ${n(d)}: ${d === 0 ? 'the plane flattens onto a line' : `the square's area gets multiplied by ${Math.abs(d)}${d < 0 ? ' and it flips over' : ''}`}. Check yourself: where does (1,1) land under this machine?`);
    }
    const [u, img] = bad;
    wrong(`Your machine sends ${fmt(u)} to ${fmt(app(B, u))}, but the clue says ${fmt(img)}. ${fmt(u)} is ${recipe(u)}. Which arrow is off?`);
  }
  function notEnough() {
    const p = R.p;
    if (p.dep) {
      const [[u], [w]] = p.clues; const i = u[0] ? 0 : 1; const k = w[i] / u[i];
      return win(`${fmt(w)} is just ${n(k)} × ${fmt(u)}, so the second clue repeats the first. You know where that one direction goes, but nothing about any other. Infinitely many machines fit, which is the same "infinitely many answers" you'll meet in Undo it.`);
    }
    wrong('There are enough clues here: the two inputs point in different directions, so together they pin down both arrows. Try combining them.');
  }

  // ---------- SAY IT checks ----------
  function choose(c) {
    if (R.won) return;
    const p = R.p;
    if (c === p.desc) { win(`(1,0) → ${fmt(p.M.a)} and (0,1) → ${fmt(p.M.b)}. Watch the two arrows carry the square along. Could you have said it in your own words?`); play(); return; }
    R.wrong.add(c);
    if (c === p.trap) wrong('That\'s what the <b>rows</b> would do. Which way does the left column, the orange arrow, point?');
    else wrong('Not quite. Where does (1,0) go under this matrix, and does that description send it there?');
  }
  function play() {
    const id = ++R.anim, t0 = performance.now();
    const step = (now) => { if (id !== R.anim) return; const u = Math.min(1, (now - t0) / 1500); R.t = u < 0.5 ? 2 * u * u : 1 - Math.pow(-2 * u + 2, 2) / 2; render(); if (u < 1) requestAnimationFrame(step); };
    R.t = 0; requestAnimationFrame(step);
  }

  // ---------- UNDO checks ----------
  function checkUndo() {
    const p = R.p;
    if (R.won || !same(app(p.M, R.in), p.t)) return;
    if (p.kind === 'one') return win(`${fmt(R.in)} → ${fmt(p.t)}. Could any other input land there? The determinant is ${n(det(p.M))}, not 0. What does that say about the plane?`);
    R.found.add(R.in.join(','));
    if (R.found.size >= 2) {
      const ex = p.sols.slice(0, 3).map(fmt).join(', ');
      return win(`Infinitely many inputs work, like ${ex}… Adding ${fmt(p.nullv)} to any input never moves the output. Why? What is ${n(p.nullv[0])}·${O} + ${n(p.nullv[1])}·${TL}?`);
    }
    showFb('hint', `Found one: ${fmt(R.in)} lands on the star. Is it the <b>only</b> input that works? Find a different one.`);
  }
  function impossible() {
    const p = R.p;
    if (p.kind === 'none') return win(`No input works. Both arrows lie on the line through ${fmt(p.u)}, so every output does too, and the star is off that line. Look at the equations: can you see why they can't both be true?`);
    wrong('Are you sure? Some input does reach the star. Watch where the gold trail is able to go.');
  }

  // read-only hook for automated tests
  window.ILA_DEBUG = { get R() { return R; }, SECTIONS, describe };

  newProblem();
})();

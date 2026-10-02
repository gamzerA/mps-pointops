import { encodeMorton3D, generateDemoPoints, buildBVH2D, queryRadius } from './lib/index.js';
import { mountMortonFormula, mountOperatorDerivation } from './formula-animation.js?rev=20261002z';
import { mountComputeField } from './compute-field.js?rev=20261002h';
import { mountViewRouter } from './view-router.js';
import { mountExecutionStory, mountPhaseStory } from './narratives.js?rev=20261002y';
import { mountCapabilityMap } from './capability-map.js';
import { mountEcosystemPanels } from './ecosystem-panels.js?rev=20261002j';
import { mountMethodsPanels } from './methods-panels.js?rev=20261002b';
import { mountMeasurementsDashboard } from './measurements-dashboard.js?rev=20261002radius';
import { applyPageCopy } from './page-copy.js?rev=20261002x';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const state = {
  data: null, dictionaries: null, lang: 'en',
  formula: null, operatorFormula: null, playground: null, route: null, phases: null,
  capabilities: null, ecosystem: null, measurements: null, methods: null,
  morton: { x: 0, y: 0, z: 0, backgrounds: new WeakMap() },
  pruning: { points: [], tree: null, distribution: 'uniform', qx: .49, qy: .5,
    radius: .095, background: null, frame: 0, dragging: false, result: null },
};
const PAGE_TITLES = {
  en: { compute: 'Compute', execution: 'Execution', ecosystem: 'What runs', methods: 'Methods', measurements: 'Measurements', install: 'Install' },
};
function renderPageTitle(view = 'compute') {
  document.title = `${PAGE_TITLES[state.lang][view] ?? PAGE_TITLES[state.lang].compute} · mps-pointops`;
}
function t(key, values = {}) {
  const template = state.dictionaries?.[state.lang]?.[key] ?? key;
  return template.replace(/\{([a-zA-Z]+)\}/g, (_, name) => String(values[name] ?? '—'));
}
function number(value, digits = 0) {
  return new Intl.NumberFormat('en-US',
    { minimumFractionDigits: digits, maximumFractionDigits: digits }).format(value);
}
function formattedDecisionPair(left, right) {
  for (let digits = 6; digits <= 15; digits += 1) {
    const leftText = number(left, digits);
    const rightText = number(right, digits);
    if (left === right || leftText !== rightText) return [leftText, rightText];
  }
  return [left.toPrecision(17), right.toPrecision(17)];
}
function element(tag, className = '', content = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== null) node.textContent = String(content);
  return node;
}
function setText(selector, value) {
  const node = $(selector);
  if (node) node.textContent = value;
}
async function loadJson(path) {
  const response = await fetch(path, { cache: 'no-store' });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}
function applyStaticLanguage() {
  document.documentElement.lang = 'en';
  for (const node of $$('[data-i18n]')) node.textContent = t(node.dataset.i18n);
  for (const node of $$('[data-i18n-aria]')) node.setAttribute('aria-label', t(node.dataset.i18nAria));
  applyPageCopy(state.lang);
  document.title = 'mps-pointops · Explore point-cloud computation';
}
function renderMeta() {
  const { meta, verification } = state.data;
  setText('#version-pill', `v${meta.publishedVersion}`);
  setText('#install-command', meta.installation);
  setText('#safe-count', number(verification.testRuns[0].passed));
  setText('#test-summary', `${number(verification.testRuns[0].passed)} Safe / ${number(verification.testRuns[1].passed)} Fast`);
  const list = $('#requirements-list');
  list.replaceChildren();
  const requirements = [
    ['Python', `≥ ${meta.minimumPython}`],
    ['PyTorch', `≥ ${meta.minimumPyTorch}`],
    ['macOS', 'Minimum unconfirmed'],
  ];
  for (const [label, value] of requirements) list.append(element('dt', '', label), element('dd', '', value));
  const links = $('#test-links'); links.replaceChildren();
  for (const run of verification.testRuns) {
    const link = element('a', '', run.id === 'safe' ? 'Safe log ↗' : 'Fast log ↗');
    link.href = `./evidence/${run.logPath}`; link.target = '_blank'; link.rel = 'noopener';
    links.append(link);
  }
  setText('#footer-license', meta.license);
  setText('#footer-snapshot', `v${meta.publishedVersion} source snapshot · ${meta.snapshotDate} · ${meta.mainMergeCommit.slice(0, 8)}`);
}
function renderObservatoryTelemetry(snapshot) {
  if (!snapshot) return;
  const total = Number(snapshot.totalFrames) || 0;
  const frame = Number(snapshot.frameIndex) || 0;
  const count = Number(snapshot.count) || 0;
  const inspected = Number(snapshot.inspected) || 0;
  setText('[data-obs-count]', number(count));
  setText('[data-obs-frame]', `${String(frame).padStart(2, '0')} / ${String(total).padStart(2, '0')}`);
  const progress = $('[data-obs-progress]');
  const progressRoot = progress?.closest('[role="progressbar"]');
  const percent = total > 0 ? Math.min(100, Math.max(0, frame / total * 100)) : 100;
  if (progress) progress.style.width = `${percent}%`;
  if (progressRoot) {
    progressRoot.setAttribute('aria-valuenow', String(Math.round(percent)));
    progressRoot.setAttribute('aria-label', 'Demo progress');
  }
  setText('[data-obs-inspected]', snapshot.mode === 'fps'
    ? `${number(inspected)} distance comparisons`
    : `${number(inspected)} / ${number(count)} candidates inspected`);
  const ruleNotes = {
    fps: 'Maximize minimum distance · lower-index ties',
    knn: 'Ascending distance² · lower-index ties',
    ball: 'First K in input order · distance² < radius²',
  };
  setText('[data-obs-rule-note]', ruleNotes[snapshot.mode] ?? '—');
}
function makeLayer(width, height) {
  if (typeof OffscreenCanvas !== 'undefined') return new OffscreenCanvas(width, height);
  const layer = document.createElement('canvas');
  layer.width = width; layer.height = height;
  return layer;
}

function syncCanvasSize(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(canvas.clientWidth * ratio));
  const height = Math.max(1, Math.round(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width; canvas.height = height;
    return true;
  }
  return false;
}

function mortonGridBackground(canvas) {
  const side = state.data.demo.mortonGridSide;
  const layer = makeLayer(canvas.width, canvas.height);
  const context = layer.getContext('2d');
  const w = layer.width; const h = layer.height;
  context.fillStyle = getComputedStyle(document.body).getPropertyValue('--demo-surface').trim() || '#f4f6f2';
  context.fillRect(0, 0, w, h);
  context.strokeStyle = getComputedStyle(document.body).getPropertyValue('--demo-grid').trim() || '#d9ded8';
  context.lineWidth = Math.max(1, w / 512);
  for (let i = 0; i <= side; i += 1) {
    const p = i / side * w;
    context.beginPath(); context.moveTo(p, 0); context.lineTo(p, h); context.stroke();
    context.beginPath(); context.moveTo(0, p); context.lineTo(w, p); context.stroke();
  }
  const order = [];
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      order.push({ x, y, key: encodeMorton3D(x, y, 0, state.data.demo.mortonBitsPerAxis) });
    }
  }
  order.sort((a, b) => a.key < b.key ? -1 : a.key > b.key ? 1 : 0);
  context.strokeStyle = getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#197458';
  context.lineWidth = Math.max(1.5, w / 240);
  context.beginPath();
  for (const [index, point] of order.entries()) {
    const px = (point.x + 0.5) / side * w;
    const py = (point.y + 0.5) / side * h;
    if (index === 0) context.moveTo(px, py);
    else context.lineTo(px, py);
  }
  context.stroke();
  state.morton.backgrounds.set(canvas, layer);
}

function drawMorton() {
  const side = state.data.demo.mortonGridSide;
  for (const canvas of $$('#morton-canvas')) {
    if (syncCanvasSize(canvas) || !state.morton.backgrounds.has(canvas)) mortonGridBackground(canvas);
    const context = canvas.getContext('2d');
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(state.morton.backgrounds.get(canvas), 0, 0);
    const px = (state.morton.x % side + 0.5) / side * canvas.width;
    const py = (state.morton.y % side + 0.5) / side * canvas.height;
    context.strokeStyle = getComputedStyle(document.body).getPropertyValue('--ink').trim() || '#0f1113';
    context.fillStyle = getComputedStyle(document.body).getPropertyValue('--warm').trim() || '#d37a48';
    context.lineWidth = Math.max(2, canvas.width / 180);
    context.beginPath(); context.arc(px, py, Math.max(7, canvas.width / 48), 0, 2 * Math.PI);
    context.fill(); context.stroke();
  }
}

function bitRow(label, value, bits) {
  const row = element('div', 'bit-row');
  row.append(element('span', 'bit-axis', label),
    element('code', 'bit-value', value.toString(2).padStart(bits, '0')));
  return row;
}

function renderMortonPanel() {
  const { x, y, z } = state.morton;
  const bits = state.data.demo.mortonBitsPerAxis;
  const key = encodeMorton3D(x, y, z, bits);
  setText('#morton-key', key.toString(10));
  $('#morton-x').value = String(x);
  $('#morton-y').value = String(y);
  $('#morton-z').value = String(z);
  const panel = $('#bit-panel');
  panel.replaceChildren(
    bitRow('X', x, bits), bitRow('Y', y, bits), bitRow('Z', z, bits),
    bitRow('M', key, bits * 3),
  );
  setText('#morton-grid-caption', t('mortonGridCaption', {
    side: number(state.data.demo.mortonGridSide),
  }));
  setText('#morton-numerics', t('mortonNumerics', {
    bits: number(bits), totalBits: number(bits * 3),
  }));
  for (const canvas of $$('#morton-canvas')) {
    canvas.setAttribute('aria-label',
      `${t('mortonKeyLabel')}: ${key.toString(10)}; X ${x}, Y ${y}, Z ${z}`);
  }
}

function renderMortonStrip() {
  const side = state.data.demo.mortonGridSide;
  const { x, y, z } = state.morton;
  const xBase = x - x % side; const yBase = y - y % side;
  const points = [];
  for (let index = 0; points.length < 15; index += 1) {
    const candidateX = xBase + (index * 7) % side;
    const candidateY = yBase + (index * 11) % side;
    if (candidateX === x && candidateY === y) continue;
    points.push({ x: candidateX, y: candidateY, z, selected: false });
  }
  points.push({ x, y, z, selected: true });
  points.sort((a, b) => {
    const ak = encodeMorton3D(a.x, a.y, a.z, state.data.demo.mortonBitsPerAxis);
    const bk = encodeMorton3D(b.x, b.y, b.z, state.data.demo.mortonBitsPerAxis);
    return ak < bk ? -1 : ak > bk ? 1 : Number(a.selected) - Number(b.selected);
  });
  const strip = $('#morton-strip'); strip.replaceChildren();
  const rank = points.findIndex(point => point.selected) + 1;
  state.morton.rank = rank;
  setText('#morton-rank', `${number(rank)} / ${number(points.length)}`);
  for (const [index, point] of points.entries()) {
    const pointKey = encodeMorton3D(point.x, point.y, point.z,
      state.data.demo.mortonBitsPerAxis);
    const button = element('button', `strip-point${point.selected ? ' is-selected' : ''}`);
    button.type = 'button';
    button.append(
      element('span', 'strip-point-rank', `#${number(index + 1)}`),
      element('strong', 'strip-point-coordinate',
        t('stripCoordinate', { x: number(point.x), y: number(point.y) })),
      element('code', 'strip-point-key', `M = ${pointKey}`),
    );
    button.title = point.selected ? t('stripSelected') : t('stripPoint', {
      index: number(index + 1), x: number(point.x), y: number(point.y), z: number(point.z),
    });
    button.setAttribute('aria-label', `${button.title}. M = ${pointKey}`);
    button.setAttribute('aria-pressed', String(point.selected));
    button.addEventListener('click', () => {
      updateMorton(point.x, point.y, point.z);
      announceMorton();
      $('#morton-strip .is-selected')?.focus();
    });
    strip.append(button);
  }
  const selected = strip.querySelector('.is-selected');
  if (selected) {
    strip.scrollLeft += selected.getBoundingClientRect().left
      - strip.getBoundingClientRect().left
      - (strip.clientWidth - selected.clientWidth) / 2;
  }
}

function initMortonStrip() {
  const strip = $('#morton-strip');
  let drag = null;
  let suppressClick = false;
  strip.addEventListener('pointerdown', event => {
    if ((event.pointerType !== 'mouse' && event.pointerType !== 'pen')
        || event.button !== 0) return;
    drag = { pointerId: event.pointerId, x: event.clientX,
      startScroll: strip.scrollLeft, moved: false };
  });
  strip.addEventListener('pointermove', event => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    const delta = event.clientX - drag.x;
    if (!drag.moved && Math.abs(delta) < 7) return;
    if (!drag.moved) {
      drag.moved = true;
      strip.setPointerCapture(event.pointerId);
      strip.classList.add('is-dragging');
    }
    event.preventDefault();
    strip.scrollLeft = drag.startScroll - delta;
  });
  const endDrag = event => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (drag.moved) {
      suppressClick = true;
      window.setTimeout(() => { suppressClick = false; }, 0);
      if (strip.hasPointerCapture(event.pointerId)) strip.releasePointerCapture(event.pointerId);
    }
    drag = null;
    strip.classList.remove('is-dragging');
  };
  strip.addEventListener('pointerup', endDrag);
  strip.addEventListener('pointercancel', endDrag);
  strip.addEventListener('click', event => {
    if (!suppressClick || event.detail === 0) return;
    event.preventDefault();
    event.stopPropagation();
  }, true);
  strip.addEventListener('keydown', event => {
    const buttons = [...strip.querySelectorAll('button')];
    if (!buttons.length) return;
    const current = event.target.closest?.('button');
    const selected = strip.querySelector('.is-selected');
    const currentIndex = Math.max(0, buttons.indexOf(current ?? selected));
    let nextIndex;
    if (event.key === 'ArrowLeft') nextIndex = Math.max(0, currentIndex - 1);
    else if (event.key === 'ArrowRight') nextIndex = Math.min(buttons.length - 1, currentIndex + 1);
    else if (event.key === 'Home') nextIndex = 0;
    else if (event.key === 'End') nextIndex = buttons.length - 1;
    else return;
    event.preventDefault();
    buttons[nextIndex].click();
  });
}

function announceMorton() {
  const { x, y, z, rank } = state.morton;
  const key = encodeMorton3D(x, y, z, state.data.demo.mortonBitsPerAxis);
  setText('#morton-announcement', t('mortonAnnouncement', {
    x: number(x), y: number(y), z: number(z), key: key.toString(10), rank: number(rank),
  }));
}

function updateMorton(x, y, z) {
  const bits = state.data.demo.mortonBitsPerAxis;
  const maximum = Number((1n << BigInt(bits)) - 1n);
  const values = [x, y, z].map(value => Math.max(0, Math.min(maximum, Math.trunc(value))));
  if (values[0] === state.morton.x && values[1] === state.morton.y
    && values[2] === state.morton.z && state.morton.rank != null) return;
  [state.morton.x, state.morton.y, state.morton.z] = values;
  state.formula?.update({
    x: values[0], y: values[1], z: values[2], bits,
    key: encodeMorton3D(values[0], values[1], values[2], bits),
    language: state.lang,
  });
  drawMorton(); renderMortonPanel(); renderMortonStrip();
}

function initMorton() {
  const bits = state.data.demo.mortonBitsPerAxis;
  const high = Number(1n << BigInt(bits - 1));
  state.formula = mountMortonFormula($('#morton-formula-animation'), {
    x: high + 3, y: high + 5, z: high + 1, bits, language: state.lang,
    translate: (key, language) => state.dictionaries?.[language]?.[key] ?? key,
  });
  state.operatorFormula = mountOperatorDerivation($('#operator-formula-animation'), {
    language: state.lang,
    translate: (key, language) => state.dictionaries?.[language]?.[key] ?? key,
  });
  initMortonStrip();
  updateMorton(high + 3, high + 5, high + 1);
  const side = state.data.demo.mortonGridSide;
  for (const canvas of $$('#morton-canvas')) {
    let dragging = false;
    const moveFromPointer = event => {
      const bounds = canvas.getBoundingClientRect();
      const dx = Math.max(0, Math.min(side - 1,
        Math.floor((event.clientX - bounds.left) / bounds.width * side)));
      const dy = Math.max(0, Math.min(side - 1,
        Math.floor((event.clientY - bounds.top) / bounds.height * side)));
      updateMorton(state.morton.x - state.morton.x % side + dx,
        state.morton.y - state.morton.y % side + dy, state.morton.z);
    };
    canvas.addEventListener('pointerdown', event => {
      dragging = true; canvas.setPointerCapture(event.pointerId);
      moveFromPointer(event); canvas.focus();
    });
    canvas.addEventListener('pointermove', event => {
      if (dragging) moveFromPointer(event);
    });
    const stopDrag = () => { if (dragging) announceMorton(); dragging = false; };
    canvas.addEventListener('pointerup', stopDrag);
    canvas.addEventListener('pointercancel', stopDrag);
    canvas.addEventListener('keydown', event => {
      const deltas = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
      if (!deltas[event.key]) return;
      event.preventDefault();
      const [dx, dy] = deltas[event.key];
      updateMorton(state.morton.x + dx, state.morton.y + dy, state.morton.z);
      announceMorton();
    });
  }
  for (const axis of ['x', 'y', 'z']) {
    const field = $(`#morton-${axis}`);
    field.max = String(Number((1n << BigInt(bits)) - 1n));
    const updateFromInput = () => {
      if (field.value.trim() === '') return;
      const input = Number(field.value);
      if (!Number.isSafeInteger(input)) return;
      updateMorton(axis === 'x' ? input : state.morton.x,
        axis === 'y' ? input : state.morton.y,
        axis === 'z' ? input : state.morton.z);
      announceMorton();
    };
    field.addEventListener('input', updateFromInput);
    field.addEventListener('change', () => {
      if (field.value.trim() === '') {
        field.value = String(state.morton[axis]);
        return;
      }
      updateFromInput();
    });
  }
}

function rebuildPruning() {
  const demo = state.data.demo;
  state.pruning.points = generateDemoPoints({
    count: demo.pruningPointCount, seed: demo.seed, distribution: state.pruning.distribution,
  });
  state.pruning.tree = buildBVH2D(state.pruning.points, {
    brickSize: demo.pruningBrickSize, mortonBits: demo.mortonBitsPerAxis,
  });
  state.pruning.background = null;
  drawPruning();
}

function pruningBackground() {
  const canvas = $('#pruning-canvas');
  const layer = makeLayer(canvas.width, canvas.height);
  const ctx = layer.getContext('2d');
  const css = getComputedStyle(document.body);
  ctx.fillStyle = css.getPropertyValue('--demo-surface').trim() || '#f4f6f2';
  ctx.fillRect(0, 0, layer.width, layer.height);
  ctx.strokeStyle = css.getPropertyValue('--demo-grid').trim() || '#d9ded8';
  ctx.lineWidth = Math.max(1, layer.width / 960);
  for (let i = 1; i < 10; i += 1) {
    ctx.beginPath(); ctx.moveTo(i / 10 * layer.width, 0);
    ctx.lineTo(i / 10 * layer.width, layer.height); ctx.stroke();
  }
  for (let i = 1; i < 5; i += 1) {
    ctx.beginPath(); ctx.moveTo(0, i / 5 * layer.height);
    ctx.lineTo(layer.width, i / 5 * layer.height); ctx.stroke();
  }
  ctx.fillStyle = css.getPropertyValue('--point-muted').trim() || '#7d8d81';
  const radius = Math.max(1.1, layer.width / 600);
  for (const point of state.pruning.points) {
    ctx.beginPath(); ctx.arc(point.x * layer.width, point.y * layer.height,
      radius, 0, Math.PI * 2); ctx.fill();
  }
  state.pruning.background = layer;
}

function renderPruningCounters(result) {
  const counters = [
    ['counterVisited', result.visitedNodes],
    ['counterPruned', result.prunedNodes],
    ['counterBricks', result.openedBricks],
    ['counterDistances', result.distanceCalculations],
    ['counterHits', result.hitIndices.length],
  ];
  const holder = $('#pruning-counters'); holder.replaceChildren();
  for (const [key, value] of counters) {
    const block = element('div', 'counter');
    block.append(element('span', 'counter-label', t(key)),
      element('strong', 'counter-value mono', number(value)));
    holder.append(block);
  }
  setText('#pruning-fixture-caption', t('pruningFixture', {
    count: number(state.pruning.points.length),
    brick: number(state.data.demo.pruningBrickSize),
    distribution: t({ uniform: 'distributionUniform', clustered: 'distributionClustered',
      coincident: 'distributionCoincident' }[state.pruning.distribution]),
  }));
  setText('#pruning-model-note', t('pruningModelNote', {
    brick: number(state.data.demo.pruningBrickSize),
    sourceBrick: number(state.data.demo.sourceBvhBrickSize),
  }));
  setText('#radius-output', number(state.pruning.radius, 3));
  $('#pruning-canvas').setAttribute('aria-label', t('pruningAnnouncement', {
    visited: number(result.visitedNodes), pruned: number(result.prunedNodes),
    distances: number(result.distanceCalculations), hits: number(result.hitIndices.length),
  }));
  for (const button of $$('[data-distribution]')) {
    button.setAttribute('aria-pressed', String(button.dataset.distribution === state.pruning.distribution));
  }
}

function renderPruningDecision(result) {
  const entry = result.inspectedAabbs.find(item => item.pruned)
    ?? result.inspectedAabbs.find(item => !item.pruned);
  if (!entry) {
    setText('#pruning-decision', '—');
    setText('#pruning-decision-note', t('decisionNone'));
    return null;
  }
  const radiusSquared = state.pruning.radius * state.pruning.radius;
  const [lowerBound, radiusBound] = formattedDecisionPair(entry.minimumDistanceSquared, radiusSquared);
  setText('#pruning-decision',
    `Dmin² ${lowerBound} ${entry.pruned ? '>' : '≤'} r² ${radiusBound}`);
  setText('#pruning-decision-note', `${t(entry.pruned ? 'decisionPruned' : 'decisionVisited', {
    node: number(entry.id),
  })} ${t('decisionChecks', {
    checked: number(result.distanceCalculations), total: number(state.pruning.points.length),
  })}`);
  return entry;
}

function announcePruning() {
  const result = state.pruning.result;
  if (!result) return;
  setText('#pruning-announcement', t('pruningAnnouncement', {
    visited: number(result.visitedNodes), pruned: number(result.prunedNodes),
    distances: number(result.distanceCalculations), hits: number(result.hitIndices.length),
  }));
}

function drawPruning() {
  const canvas = $('#pruning-canvas');
  if (syncCanvasSize(canvas) || !state.pruning.background) pruningBackground();
  const { points, qx, qy, radius, tree } = state.pruning;
  const result = queryRadius(tree.root, points, qx, qy, radius);
  state.pruning.result = result;
  const selectedAabb = renderPruningDecision(result);
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(state.pruning.background, 0, 0);
  const css = getComputedStyle(document.body);
  ctx.save();
  ctx.strokeStyle = css.getPropertyValue('--pruned-outline').trim() || '#c68359';
  ctx.globalAlpha = 0.65;
  ctx.lineWidth = Math.max(1, canvas.width / 960);
  for (const entry of result.inspectedAabbs) {
    if (!entry.pruned) continue;
    const { minX, minY, maxX, maxY } = entry.bounds;
    ctx.strokeRect(minX * canvas.width, minY * canvas.height,
      (maxX - minX) * canvas.width, (maxY - minY) * canvas.height);
  }
  ctx.restore();
  if (selectedAabb) {
    const { minX, minY, maxX, maxY } = selectedAabb.bounds;
    ctx.save();
    ctx.strokeStyle = selectedAabb.pruned
      ? css.getPropertyValue('--pruned-outline').trim() : css.getPropertyValue('--accent').trim();
    ctx.lineWidth = Math.max(2.5, canvas.width / 320);
    const width = (maxX - minX) * canvas.width;
    const height = (maxY - minY) * canvas.height;
    if (width < 2 && height < 2) {
      const cx = minX * canvas.width; const cy = minY * canvas.height;
      const size = Math.max(7, canvas.width / 80);
      ctx.strokeRect(cx - size / 2, cy - size / 2, size, size);
    } else {
      ctx.strokeRect(minX * canvas.width, minY * canvas.height, width, height);
    }
    ctx.restore();
  }
  ctx.strokeStyle = css.getPropertyValue('--accent').trim() || '#197458';
  ctx.fillStyle = css.getPropertyValue('--point-hit').trim() || '#145e48';
  ctx.lineWidth = Math.max(1.5, canvas.width / 420);
  for (const index of result.hitIndices) {
    const point = points[index];
    ctx.beginPath(); ctx.arc(point.x * canvas.width, point.y * canvas.height,
      Math.max(2, canvas.width / 300), 0, Math.PI * 2); ctx.fill();
  }
  ctx.beginPath();
  // Normalized Cartesian coordinates map to a rectangular canvas, so a
  // geometric circle appears as an ellipse in screen pixels.
  ctx.ellipse(qx * canvas.width, qy * canvas.height,
    radius * canvas.width, radius * canvas.height, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = css.getPropertyValue('--warm').trim() || '#d37a48';
  ctx.beginPath(); ctx.arc(qx * canvas.width, qy * canvas.height,
    Math.max(4, canvas.width / 180), 0, Math.PI * 2); ctx.fill();
  renderPruningCounters(result);
}

function queuePruningDraw() {
  if (state.pruning.frame) return;
  state.pruning.frame = requestAnimationFrame(() => {
    state.pruning.frame = 0;
    drawPruning();
  });
}

function initPruning() {
  rebuildPruning();
  const canvas = $('#pruning-canvas');
  const moveFromPointer = event => {
    const bounds = canvas.getBoundingClientRect();
    state.pruning.qx = Math.max(0, Math.min(1, (event.clientX - bounds.left) / bounds.width));
    state.pruning.qy = Math.max(0, Math.min(1, (event.clientY - bounds.top) / bounds.height));
    queuePruningDraw();
  };
  canvas.addEventListener('pointerdown', event => {
    state.pruning.dragging = true; canvas.setPointerCapture(event.pointerId);
    moveFromPointer(event); canvas.focus();
  });
  canvas.addEventListener('pointermove', event => {
    if (state.pruning.dragging || event.pointerType === 'mouse') moveFromPointer(event);
  });
  const stopDrag = () => { state.pruning.dragging = false; announcePruning(); };
  canvas.addEventListener('pointerup', stopDrag);
  canvas.addEventListener('pointercancel', stopDrag);
  canvas.addEventListener('keydown', event => {
    const deltas = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
    if (!deltas[event.key]) return;
    event.preventDefault();
    const [dx, dy] = deltas[event.key];
    const step = event.shiftKey ? 0.05 : 0.01;
    state.pruning.qx = Math.max(0, Math.min(1, state.pruning.qx + dx * step));
    state.pruning.qy = Math.max(0, Math.min(1, state.pruning.qy + dy * step));
    drawPruning(); announcePruning();
  });
  $('#radius-slider').addEventListener('input', event => {
    state.pruning.radius = Number(event.target.value);
    queuePruningDraw();
  });
  $('#radius-slider').addEventListener('change', announcePruning);
  const presets = {
    uniform: { qx: 0.49, qy: 0.5, radius: 0.095 },
    clustered: { qx: 0.72, qy: 0.30, radius: 0.065 },
    coincident: { qx: 0.5, qy: 0.5, radius: 0.095 },
  };
  for (const button of $$('[data-distribution]')) {
    button.addEventListener('click', () => {
      const name = button.dataset.distribution;
      state.pruning.distribution = name;
      Object.assign(state.pruning, presets[name]);
      $('#radius-slider').value = String(state.pruning.radius);
      rebuildPruning(); announcePruning();
    });
  }
}


function rerender() {
  applyStaticLanguage();
  renderPageTitle(state.viewRouter?.currentView);
  renderMeta();
  state.playground?.setLanguage?.(state.lang);
  renderObservatoryTelemetry(state.playground?.getState?.());
  state.route?.setLanguage(state.lang);
  state.phases?.setLanguage(state.lang);
  state.capabilities?.setLanguage(state.lang);
  state.ecosystem?.setLanguage(state.lang);
  state.measurements?.setLanguage(state.lang);
  state.methods?.setLanguage(state.lang);
  state.formula?.update({ language: state.lang });
  state.operatorFormula?.update({ language: state.lang });
  renderMortonPanel(); renderMortonStrip(); drawMorton();
  if (state.pruning.tree) { drawPruning(); announcePruning(); }
}
function setupCopy() {
  for (const button of $$('[data-copy]')) {
    button.addEventListener('click', async () => {
      const target = document.getElementById(button.dataset.copy);
      try { await navigator.clipboard.writeText(target.textContent); button.textContent = 'Copied'; }
      catch { button.textContent = 'Copy failed'; }
      window.setTimeout(() => { button.textContent = 'Copy'; }, 1600);
    });
  }
}
async function main() {
  try {
    const [data, capabilityData, en] = await Promise.all([
      loadJson('./data/benchmarks.json'),
      loadJson('./data/capabilities.json'),
      loadJson('./data/i18n/en.json'),
    ]);
    state.data = data;
    state.dictionaries = { en };
    setupCopy();
    state.playground = mountComputeField({
      root: $('[data-compute-field]'),
      canvas: $('[data-compute-field-canvas]'),
      autoPlay: true,
      onFrame: renderObservatoryTelemetry,
    });
    state.route = mountExecutionStory($('[data-route-story]'), { language: state.lang });
    state.phases = mountPhaseStory($('[data-phase-story]'), { language: state.lang });
    state.capabilities = mountCapabilityMap($('#capability-map'), capabilityData, { language: state.lang });
    state.ecosystem = mountEcosystemPanels($('#content'), {
      language: state.lang, capabilities: state.capabilities, phases: state.phases,
    });
    state.measurements = mountMeasurementsDashboard($('#content'), data, { language: state.lang });
    const redrawViewCanvases = () => {
      state.morton.backgrounds = new WeakMap();
      state.pruning.background = null;
      drawMorton(); queuePruningDraw();
    };
    initMorton();
    initPruning();
    state.methods = mountMethodsPanels($('#methods'), {
      language: state.lang,
      onPanelChange: () => requestAnimationFrame(redrawViewCanvases),
    });
    rerender();
    window.addEventListener('resize', redrawViewCanvases);
    state.viewRouter = mountViewRouter({
      main: $('#content'),
      nav: $('.primary-nav'),
      onViewChange: ({ view }) => {
        renderPageTitle(view);
        requestAnimationFrame(redrawViewCanvases);
      },
    });
  } catch (error) {
    const message = state.dictionaries?.[state.lang]?.dataError
      ?? 'Could not load local data. Open this page through the documented local server.';
    const alert = element('p', 'data-error', message);
    alert.setAttribute('role', 'alert');
    $('#content').prepend(alert);
    console.error(error);
  }
}
main();

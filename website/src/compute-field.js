import { computePlaygroundSteps } from './point-playground.js';

// A viewport-proportional 2D domain keeps a radius circular on every screen.
// This is a teaching field, not a Metal execution trace or benchmark.
export const FIELD_WIDTH = 1.45;
export const FIELD_HEIGHT = 1;
export const MAX_FIELD_POINTS = 1800;
const DEFAULT_COUNT = 1200;
const SEED = 0x43505532;
const MODES = new Set(['fps', 'knn', 'ball']);
const CLUSTERS = Object.freeze([
  { x: 0.13, y: 0.70, sx: 0.075, sy: 0.105 },
  { x: 0.33, y: 0.29, sx: 0.105, sy: 0.085 },
  { x: 0.57, y: 0.73, sx: 0.085, sy: 0.12 },
  { x: 0.78, y: 0.43, sx: 0.12, sy: 0.105 },
  { x: 1.08, y: 0.73, sx: 0.105, sy: 0.09 },
  { x: 1.29, y: 0.25, sx: 0.09, sy: 0.11 },
]);

const COPY = {
  ko: {
    play: '재생', pause: '일시정지', step: '한 단계', reset: '처음으로',
    samples: '추출 점 수', neighbors: '이웃 수', selected: '선택', rejected: '제외',
    ready: '시작 전', inspected: '검사', checks: '거리 비교', result: '선택 인덱스',
    model: '고정된 2D 설명 모형 · Metal 실행이나 벤치마크 아님',
    fps: '기존 선택점까지의 최소 거리가 가장 큰 점',
    knn: '질의점과의 거리순 · 동률은 낮은 인덱스',
    ball: '입력 순서 검사 · 거리² < 반지름²의 첫 K개',
    canvas: '고밀도 2D 점군 연산 설명. FPS는 점을 클릭해 시작점을 바꾸고, 나머지는 질의점을 옮깁니다. 방향키도 사용할 수 있습니다.',
  },
  en: {
    play: 'Play', pause: 'Pause', step: 'One step', reset: 'Restart',
    samples: 'Sample count', neighbors: 'Neighbor count', selected: 'selected', rejected: 'excluded',
    ready: 'Ready', inspected: 'Scanned', checks: 'Distance checks', result: 'Selected indices',
    model: 'Fixed 2D teaching model · not Metal execution or a benchmark',
    fps: 'Largest minimum distance to the selected set',
    knn: 'Query distance order · lower index breaks ties',
    ball: 'Input order · first K with distance² < radius²',
    canvas: 'Dense 2D point cloud explanation. Click a point to choose the FPS start, or move the query in other modes. Arrow keys also work.',
  },
};

const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
const distanceSquared = (a, b) => (a.x - b.x) ** 2 + (a.y - b.y) ** 2;

/** Deterministic, prefix-stable clusters. Every rendered point is an algorithm input. */
export function createComputePoints({ count = DEFAULT_COUNT, seed = SEED, width = FIELD_WIDTH } = {}) {
  if (!Number.isSafeInteger(count) || count < 0 || count > MAX_FIELD_POINTS) {
    throw new RangeError(`count must be an integer from 0 to ${MAX_FIELD_POINTS}`);
  }
  if (!Number.isSafeInteger(seed) || seed < 0 || seed >= 2 ** 32) {
    throw new RangeError('seed must be a 32-bit unsigned integer');
  }
  if (!Number.isFinite(width) || width <= 0 || width > 4) {
    throw new RangeError('width must be positive and at most 4');
  }
  const xRatio = width / FIELD_WIDTH;
  let state = seed;
  const random = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 2 ** 32;
  };
  const points = [];
  for (let i = 0; i < count; i += 1) {
    const cluster = i % CLUSTERS.length;
    const center = CLUSTERS[cluster];
    // Every 13th point crosses between lobes; every 29th is a true outlier.
    // These are still input points, never decorative network vertices.
    let x; let y;
    if (i % 29 === 0) {
      x = random() * width;
      y = random() * FIELD_HEIGHT;
    } else if (i % 13 === 0) {
      const next = CLUSTERS[(cluster + 2) % CLUSTERS.length];
      const t = random();
      x = (center.x + (next.x - center.x) * t + (random() - 0.5) * 0.06) * xRatio;
      y = center.y + (next.y - center.y) * t + (random() - 0.5) * 0.06;
    } else {
      // Sum of uniforms gives a compact, reproducible bell-shaped lobe.
      x = (center.x + (random() + random() + random() + random() - 2) * center.sx) * xRatio;
      y = center.y + (random() + random() + random() + random() - 2) * center.sy;
    }
    points.push({
      x: clamp(x, 0, width), y: clamp(y, 0, FIELD_HEIGHT),
      cluster, family: i % 29 === 0 ? 'neutral' : cluster % 2 === 0 ? 'cyan' : 'coral',
      luminosity: 0.35 + random() * 0.65,
    });
  }
  return points;
}

/** Uniform pixel scale makes the radius and Euclidean distances honest. */
export function layoutComputeField(width, height) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new RangeError('canvas width and height must be positive finite numbers');
  }
  const padX = width < 520 ? 14 : 28;
  const padY = height < 400 ? 18 : 28;
  const availableWidth = Math.max(1, width - padX * 2);
  const availableHeight = Math.max(1, height - padY * 2);
  const worldWidth = clamp(availableWidth / availableHeight, 0.72, 2.65);
  const scale = Math.min(availableWidth / worldWidth, availableHeight / FIELD_HEIGHT);
  return {
    x: (width - worldWidth * scale) / 2,
    y: (height - FIELD_HEIGHT * scale) / 2,
    width: worldWidth * scale,
    height: FIELD_HEIGHT * scale,
    worldWidth, worldHeight: FIELD_HEIGHT,
    scale, canvasWidth: width, canvasHeight: height,
  };
}

export function projectComputePoint(layout, point) {
  return { x: layout.x + point.x * layout.scale, y: layout.y + (layout.worldHeight - point.y) * layout.scale };
}

export function unprojectComputePoint(layout, point) {
  return {
    x: clamp((point.x - layout.x) / layout.scale, 0, layout.worldWidth),
    y: clamp(layout.worldHeight - (point.y - layout.y) / layout.scale, 0, layout.worldHeight),
  };
}

/** Exact selection comes from the existing teaching model; only long Ball scans are condensed for playback. */
export function computeFieldTrace({ mode = 'fps', points, query = { x: 0.74, y: 0.52 }, k = 8, radius = 0.2, seedIndex = 0 } = {}) {
  const exact = computePlaygroundSteps({ mode, points, query, k, radius, seedIndex });
  if (mode !== 'ball' || exact.frames.length <= 84) return exact;
  const stride = Math.max(1, Math.ceil((exact.frames.length - 1) / 64));
  const frames = exact.frames.filter((frame, index) => (
    index === 0 || index === exact.frames.length - 1 || frame.accepted || index % stride === 0
  ));
  return { ...exact, frames };
}

/** Every line is an actual tested distance or nearest-sample assignment. */
export function computeFieldLinks({ mode, points, frame, query, maxLinks = 320 }) {
  if (!MODES.has(mode)) throw new RangeError('mode must be fps, knn, or ball');
  if (!Number.isSafeInteger(maxLinks) || maxLinks < 0) throw new RangeError('maxLinks must be nonnegative');
  if (!frame || !Array.isArray(frame.selected)) throw new TypeError('frame must have selected indices');
  if (maxLinks === 0) return [];
  const links = [];
  const selected = new Set(frame.selected);
  if (mode === 'fps') {
    if (!frame.selected.length) return links;
    const candidateBudget = Math.max(0, maxLinks - 1);
    const step = Math.max(1, Math.ceil(points.length / Math.max(1, candidateBudget)));
    for (let i = 0; i < points.length && links.length < candidateBudget; i += step) {
      if (selected.has(i)) continue;
      let nearest = frame.selected[0];
      let best = distanceSquared(points[i], points[nearest]);
      for (const index of frame.selected.slice(1)) {
        const d2 = distanceSquared(points[i], points[index]);
        if (d2 < best || (d2 === best && index < nearest)) { nearest = index; best = d2; }
      }
      links.push({ from: points[i], to: points[nearest], index: i, targetIndex: nearest, distanceSquared: best, kind: 'assignment' });
    }
    if (frame.focusIndex !== null && frame.nearestIndex !== null && frame.nearestIndex !== undefined) {
      links.push({
        from: points[frame.focusIndex], to: points[frame.nearestIndex],
        index: frame.focusIndex, targetIndex: frame.nearestIndex,
        distanceSquared: distanceSquared(points[frame.focusIndex], points[frame.nearestIndex]), kind: 'chosen',
      });
    }
    return links;
  }

  const inspected = mode === 'knn' ? frame.inspected : Math.min(frame.inspected, points.length);
  const trailBudget = Math.max(0, maxLinks - Math.min(maxLinks, frame.selected.length));
  const start = mode === 'ball' ? Math.max(0, inspected - trailBudget) : 0;
  const step = mode === 'knn' ? Math.max(1, Math.ceil(inspected / Math.max(1, trailBudget))) : 1;
  for (let i = start; i < inspected && links.length < trailBudget; i += step) {
    if (selected.has(i)) continue;
    links.push({ from: query, to: points[i], index: i, targetIndex: null, distanceSquared: distanceSquared(query, points[i]), kind: 'scanned' });
  }
  for (const index of frame.selected) {
    if (links.length >= maxLinks) break;
    links.push({ from: query, to: points[index], index, targetIndex: null, distanceSquared: distanceSquared(query, points[index]), kind: 'selected' });
  }
  return links;
}

function strokeLine(context, a, b, color, width, alpha) {
  context.globalAlpha = alpha;
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  context.moveTo(a.x, a.y);
  context.lineTo(b.x, b.y);
  context.stroke();
  context.globalAlpha = 1;
}

function ring(context, x, y, radius, color, width = 1) {
  context.strokeStyle = color;
  context.lineWidth = width;
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.stroke();
}

/**
 * Mount into a root with [data-compute-field-canvas] and the existing
 * [data-playground-*] controls/outputs. onFrame(snapshot) feeds an optional HUD.
 * Returns destroy(), setMode(), setLanguage(), refreshLocale(), getState().
 */
export function mountComputeField({ root, canvas = root?.querySelector('[data-compute-field-canvas]'), autoPlay = true, onFrame } = {}) {
  if (!root || !canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('mountComputeField requires a root and canvas');
  }
  const context = canvas.getContext('2d');
  if (!context) throw new Error('2D canvas context is unavailable');
  const rootStyle = typeof getComputedStyle === 'function' ? getComputedStyle(root) : null;
  const color = (token, fallback) => rootStyle?.getPropertyValue(token).trim() || fallback;
  const palette = {
    coral: color('--obs-coral', '#ff9289'),
    cyan: color('--obs-cyan', '#81dbdd'),
    amber: color('--obs-amber', '#eac888'),
  };
  const find = (name) => root.querySelector(`[data-playground-${name}]`);
  const countInput = find('count');
  const kInput = find('k');
  const radiusInput = find('radius');
  const playButton = find('play');
  const stepButton = find('step');
  const resetButton = find('reset');
  const stageOutput = find('stage');
  const resultOutput = find('result');
  const queryOutput = find('query-output');
  const countOutput = find('count-output');
  const kOutput = find('k-output');
  const radiusOutput = find('radius-output');
  const kLabel = find('k-label');
  const formulaOutput = find('formula');
  const modeButtons = [...root.querySelectorAll('[data-playground-mode]')];
  const motion = typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : null;

  // Existing small-fixture markup becomes useful for the dense field immediately.
  if (countInput && Number(countInput.max) < 1000) {
    countInput.min = '240'; countInput.max = String(MAX_FIELD_POINTS);
    countInput.step = '120'; countInput.value = String(DEFAULT_COUNT);
  }
  let count = clamp(Math.round(Number(countInput?.value) || DEFAULT_COUNT), 1, MAX_FIELD_POINTS);
  let k = clamp(Math.round(Number(kInput?.value) || 8), 1, 32);
  let radius = clamp(Number(radiusInput?.value) || 0.2, 0, 1.2);
  let mode = MODES.has(root.dataset.mode) ? root.dataset.mode : 'fps';
  let points = createComputePoints({ count });
  let currentWorldWidth = FIELD_WIDTH;
  let query = { x: 0.74, y: 0.52 };
  let seedIndex = 0;
  let trace = computeFieldTrace({ mode, points, query, k, radius, seedIndex });
  let frameIndex = motion?.matches ? trace.frames.length - 1 : 0;
  let requestedPlay = Boolean(autoPlay && !motion?.matches);
  let visible = true;
  let pointerDragging = false;
  let destroyed = false;
  let timer = null;
  let animationFrame = null;
  let layout = null;
  let resizeObserver = null;
  let intersectionObserver = null;
  let language = document.documentElement.lang?.toLowerCase().startsWith('en') ? 'en' : 'ko';
  const removers = [];
  const listen = (target, name, fn, options) => {
    if (!target) return;
    target.addEventListener(name, fn, options);
    removers.push(() => target.removeEventListener(name, fn, options));
  };
  const frame = () => trace.frames[frameIndex];
  const canPlay = () => requestedPlay && visible && !document.hidden && !motion?.matches;

  function stopClock() {
    if (timer !== null) { clearTimeout(timer); timer = null; }
    if (animationFrame !== null) { cancelAnimationFrame(animationFrame); animationFrame = null; }
  }

  function getState() {
    const active = frame();
    return {
      mode, count, k, radius, query: { ...query }, seedIndex, frameIndex,
      totalFrames: trace.frames.length - 1, indices: [...trace.indices],
      selected: [...active.selected], inspected: active.inspected,
      accepted: active.accepted, focusIndex: active.focusIndex, metric: active.metric,
      history: trace.frames.slice(Math.max(0, frameIndex - 31), frameIndex + 1)
        .map((item) => ({
          selectedCount: item.selected.length, inspected: item.inspected,
          accepted: item.accepted, focusIndex: item.focusIndex, metric: item.metric,
        })),
      model: 'fixed-2d-teaching',
    };
  }

  function updateText() {
    const words = COPY[language];
    const active = frame();
    const progress = `${frameIndex}/${trace.frames.length - 1}`;
    if (countOutput) countOutput.textContent = count.toLocaleString(language === 'ko' ? 'ko-KR' : 'en-US');
    if (kOutput) kOutput.textContent = String(k);
    if (radiusOutput) radiusOutput.textContent = radius.toFixed(3);
    if (kLabel) kLabel.textContent = mode === 'fps' ? words.samples : words.neighbors;
    if (queryOutput) queryOutput.textContent = mode === 'fps' ? `#${seedIndex}` : `(${query.x.toFixed(2)}, ${query.y.toFixed(2)})`;
    if (radiusInput) radiusInput.disabled = mode !== 'ball';
    if (playButton) {
      playButton.textContent = canPlay() ? words.pause : words.play;
      playButton.setAttribute('aria-pressed', String(canPlay()));
      playButton.setAttribute('aria-label', playButton.textContent);
      playButton.disabled = Boolean(motion?.matches);
    }
    if (stepButton) { stepButton.textContent = words.step; stepButton.setAttribute('aria-label', words.step); }
    if (resetButton) { resetButton.textContent = words.reset; resetButton.setAttribute('aria-label', words.reset); }
    root.dataset.mode = mode;
    for (const button of modeButtons) button.setAttribute('aria-pressed', String(button.dataset.playgroundMode === mode));
    if (formulaOutput) formulaOutput.textContent = {
      fps: 'argmaxᵢ minₛ∈S ‖pᵢ − pₛ‖²',
      knn: 'sortᵢ (‖pᵢ − q‖², i) → first K',
      ball: '‖pᵢ − q‖² < r² → first K in input order',
    }[mode];
    const focus = active.focusIndex === null ? words.ready
      : `#${active.focusIndex} ${active.accepted ? words.selected : words.rejected}`;
    if (stageOutput) stageOutput.textContent = `${progress} · ${focus} · ${words[mode]}`;
    const effort = mode === 'fps' ? `${words.checks}: ${active.inspected}` : `${words.inspected}: ${active.inspected}/${count}`;
    if (resultOutput) {
      resultOutput.textContent = `${words.result}: [${active.selected.join(', ')}] · ${effort}. ${words.model}.`;
      resultOutput.setAttribute('aria-live', canPlay() ? 'off' : 'polite');
    }
    canvas.setAttribute('aria-label', `${words.canvas} ${words.model}`);
    onFrame?.(getState());
  }

  function draw(progress = 1) {
    if (!layout || destroyed) return;
    const active = frame();
    const width = layout.canvasWidth;
    const height = layout.canvasHeight;
    const accent = { fps: palette.coral, knn: palette.cyan, ball: palette.amber }[mode];
    context.clearRect(0, 0, width, height);
    context.fillStyle = '#050a0c';
    context.fillRect(0, 0, width, height);

    context.save();
    context.beginPath();
    context.rect(layout.x, layout.y, layout.width, layout.height);
    context.clip();
    // Keep a few coordinate guides; the point density carries the field.
    context.strokeStyle = 'rgba(123, 157, 163, 0.06)';
    context.lineWidth = 1;
    for (let x = 0.2; x < layout.worldWidth; x += 0.2) {
      const at = layout.x + x * layout.scale;
      context.beginPath(); context.moveTo(at, layout.y); context.lineTo(at, layout.y + layout.height); context.stroke();
    }
    for (let y = 0.2; y < layout.worldHeight; y += 0.2) {
      const at = layout.y + y * layout.scale;
      context.beginPath(); context.moveTo(layout.x, at); context.lineTo(layout.x + layout.width, at); context.stroke();
    }
    // Cluster haze is anchored to the actual point-generation centers.
    for (let i = 0; i < CLUSTERS.length; i += 1) {
      const cluster = CLUSTERS[i];
      const xRatio = layout.worldWidth / FIELD_WIDTH;
      const center = projectComputePoint(layout, { x: cluster.x * xRatio, y: cluster.y });
      const radiusPx = Math.max(cluster.sx * xRatio, cluster.sy) * layout.scale * 2.5;
      const gradient = context.createRadialGradient(center.x, center.y, 0, center.x, center.y, radiusPx);
      const haze = i % 2 === 0 ? '66, 137, 144' : '145, 74, 77';
      gradient.addColorStop(0, `rgba(${haze}, 0.18)`);
      gradient.addColorStop(1, `rgba(${haze}, 0)`);
      context.fillStyle = gradient;
      context.fillRect(center.x - radiusPx, center.y - radiusPx, radiusPx * 2, radiusPx * 2);
    }

    const links = computeFieldLinks({ mode, points, frame: active, query, maxLinks: mode === 'fps' ? 72 : 48 });
    for (const link of links) {
      const from = projectComputePoint(layout, link.from);
      const to = projectComputePoint(layout, link.to);
      const isStrong = link.kind === 'selected' || link.kind === 'chosen';
      strokeLine(context, from, to, isStrong ? accent : '#81aab1',
        isStrong ? 1.35 : 0.65, isStrong ? 0.62 : mode === 'fps' ? 0.075 : 0.055);
    }

    const selected = new Set(active.selected);
    const inspected = active.inspected;
    for (let i = 0; i < points.length; i += 1) {
      const at = projectComputePoint(layout, points[i]);
      const selectedPoint = selected.has(i);
      const wasInspected = mode !== 'ball' || i < inspected;
      context.globalAlpha = selectedPoint ? 1 : wasInspected ? 0.42 + points[i].luminosity * 0.48 : 0.19 + points[i].luminosity * 0.24;
      const size = selectedPoint ? 3.2 : 0.95 + points[i].luminosity * 0.9;
      context.fillStyle = selectedPoint ? '#f5f3e9' : {
        cyan: palette.cyan, coral: palette.coral, neutral: '#acb2ad',
      }[points[i].family];
      context.fillRect(at.x - size / 2, at.y - size / 2, size, size);
    }
    context.globalAlpha = 1;
    for (const index of active.selected) {
      const at = projectComputePoint(layout, points[index]);
      ring(context, at.x, at.y, 5.5, accent, 1.2);
    }

    if (mode !== 'fps') {
      const at = projectComputePoint(layout, query);
      if (mode === 'ball') {
        context.fillStyle = 'rgba(242, 203, 128, 0.045)';
        context.beginPath(); context.arc(at.x, at.y, radius * layout.scale, 0, Math.PI * 2); context.fill();
        ring(context, at.x, at.y, radius * layout.scale, accent, 1.25);
      }
      strokeLine(context, { x: at.x - 10, y: at.y }, { x: at.x + 10, y: at.y }, accent, 1.3, 0.9);
      strokeLine(context, { x: at.x, y: at.y - 10 }, { x: at.x, y: at.y + 10 }, accent, 1.3, 0.9);
      ring(context, at.x, at.y, 4, accent, 1.2);
    }
    if (active.focusIndex !== null) {
      const at = projectComputePoint(layout, points[active.focusIndex]);
      ring(context, at.x, at.y, 9 + (1 - progress) * 9, accent, 1.25);
    }
    context.restore();

    const x0 = layout.x; const y0 = layout.y;
    const x1 = x0 + layout.width; const y1 = layout.y + layout.height;
    if (width >= 600) {
      context.fillStyle = '#8ba4aa';
      context.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
      context.textAlign = 'left';
      context.fillText(`2D INPUT FIELD  /  ${count.toLocaleString('en-US')} POINTS`, x0 + 16, y0 + 24);
      context.textAlign = 'right';
      context.fillText(`${mode.toUpperCase()}  /  FIXED MODEL`, x1 - 16, y1 - 16);
    }
  }

  function render(progress = 1) { updateText(); draw(progress); }

  function animateFocus(duration) {
    if (!canPlay() || duration <= 0) return;
    const start = performance.now();
    const tick = (now) => {
      animationFrame = null;
      if (!canPlay()) return;
      const t = clamp((now - start) / duration, 0, 1);
      draw(1 - (1 - t) ** 3);
      if (t < 1) animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);
  }

  function schedule() {
    if (!canPlay() || timer !== null) return;
    const atEnd = frameIndex >= trace.frames.length - 1;
    const delay = atEnd ? 1350 : { fps: 760, knn: 480, ball: 105 }[mode];
    timer = setTimeout(() => {
      timer = null;
      if (!canPlay()) return;
      if (animationFrame !== null) { cancelAnimationFrame(animationFrame); animationFrame = null; }
      frameIndex = atEnd ? 0 : frameIndex + 1;
      render(atEnd ? 1 : 0);
      if (!atEnd) animateFocus(Math.min(250, delay * 0.7));
      schedule();
    }, delay);
  }

  function recompute({ reset = true } = {}) {
    stopClock();
    trace = computeFieldTrace({ mode, points, query, k, radius, seedIndex });
    frameIndex = reset && canPlay() ? 0 : trace.frames.length - 1;
    render();
    schedule();
  }

  function setMode(next) {
    if (!MODES.has(next) || next === mode) return;
    mode = next;
    recompute();
  }

  function resize() {
    if (destroyed) return;
    const bounds = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width));
    const height = Math.max(1, Math.round(bounds.height));
    const dpr = clamp(window.devicePixelRatio || 1, 1, 2);
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth; canvas.height = pixelHeight;
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    const nextLayout = layoutComputeField(width, height);
    const changedDomain = !layout || Math.abs(layout.worldWidth - nextLayout.worldWidth) > 0.005;
    layout = nextLayout;
    if (changedDomain) {
      const fraction = query.x / currentWorldWidth;
      query = { ...query, x: clamp(fraction * layout.worldWidth, 0, layout.worldWidth) };
      currentWorldWidth = layout.worldWidth;
      points = createComputePoints({ count, width: currentWorldWidth });
      trace = computeFieldTrace({ mode, points, query, k, radius, seedIndex });
      frameIndex = Math.min(frameIndex, trace.frames.length - 1);
      updateText();
    }
    draw();
  }

  function setPointer(event) {
    if (!layout) return;
    const bounds = canvas.getBoundingClientRect();
    const world = unprojectComputePoint(layout, { x: event.clientX - bounds.left, y: event.clientY - bounds.top });
    if (mode === 'fps') {
      let best = 0; let bestDistance = Infinity;
      for (let i = 0; i < points.length; i += 1) {
        const d2 = distanceSquared(points[i], world);
        if (d2 < bestDistance) { best = i; bestDistance = d2; }
      }
      if (best === seedIndex) return;
      seedIndex = best;
    } else {
      if (world.x === query.x && world.y === query.y) return;
      query = world;
    }
    recompute();
  }

  for (const button of modeButtons) listen(button, 'click', () => setMode(button.dataset.playgroundMode));
  listen(countInput, 'input', () => {
    const next = clamp(Math.round(Number(countInput.value) || 1), 1, MAX_FIELD_POINTS);
    if (next === count) return;
    count = next; seedIndex = Math.min(seedIndex, count - 1);
    points = createComputePoints({ count, width: currentWorldWidth });
    recompute();
  });
  listen(kInput, 'input', () => {
    const next = clamp(Math.round(Number(kInput.value) || 1), 1, 32);
    if (next === k) return;
    k = next; recompute();
  });
  listen(radiusInput, 'input', () => {
    const next = clamp(Number(radiusInput.value), 0, 1.2);
    if (!Number.isFinite(next) || next === radius) return;
    radius = next; recompute();
  });
  listen(playButton, 'click', () => {
    if (motion?.matches) return;
    requestedPlay = !requestedPlay;
    if (requestedPlay && frameIndex === trace.frames.length - 1) frameIndex = 0;
    stopClock(); render(); schedule();
  });
  listen(stepButton, 'click', () => {
    requestedPlay = false; stopClock();
    frameIndex = (frameIndex + 1) % trace.frames.length;
    render();
  });
  listen(resetButton, 'click', () => { stopClock(); frameIndex = 0; render(); schedule(); });
  if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
  listen(canvas, 'pointerdown', (event) => {
    pointerDragging = true;
    canvas.setPointerCapture?.(event.pointerId);
    setPointer(event);
  });
  listen(canvas, 'pointermove', (event) => { if (pointerDragging) setPointer(event); });
  const releasePointer = () => { pointerDragging = false; };
  listen(canvas, 'pointerup', releasePointer);
  listen(canvas, 'pointercancel', releasePointer);
  listen(canvas, 'keydown', (event) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    if (mode === 'fps') {
      const direction = event.key === 'ArrowLeft' || event.key === 'ArrowDown' ? -1 : 1;
      seedIndex = (seedIndex + direction + count) % count;
    } else {
      const delta = event.shiftKey ? 0.1 : 0.025;
      query = {
        x: clamp(query.x + (event.key === 'ArrowRight' ? delta : event.key === 'ArrowLeft' ? -delta : 0), 0, layout?.worldWidth || FIELD_WIDTH),
        y: clamp(query.y + (event.key === 'ArrowUp' ? delta : event.key === 'ArrowDown' ? -delta : 0), 0, FIELD_HEIGHT),
      };
    }
    recompute();
  });
  listen(document, 'visibilitychange', () => { stopClock(); render(); schedule(); });
  if (motion) {
    const onMotion = () => {
      if (motion.matches) { requestedPlay = false; frameIndex = trace.frames.length - 1; }
      stopClock(); render(); schedule();
    };
    if (motion.addEventListener) listen(motion, 'change', onMotion);
    else {
      motion.addListener(onMotion);
      removers.push(() => motion.removeListener(onMotion));
    }
  }
  if (typeof ResizeObserver !== 'undefined') {
    resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(canvas);
  } else listen(window, 'resize', resize);
  if (typeof IntersectionObserver !== 'undefined') {
    intersectionObserver = new IntersectionObserver((entries) => {
      visible = entries[0]?.isIntersecting ?? true;
      stopClock(); render(); schedule();
    }, { threshold: 0.08 });
    intersectionObserver.observe(canvas);
  }
  resize(); render(); schedule();

  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      stopClock(); resizeObserver?.disconnect(); intersectionObserver?.disconnect();
      for (const remove of removers) remove();
    },
    setMode,
    setLanguage(next) {
      if (!['ko', 'en'].includes(next)) throw new RangeError('language must be ko or en');
      language = next; render();
    },
    refreshLocale() {
      language = document.documentElement.lang?.toLowerCase().startsWith('en') ? 'en' : 'ko';
      render();
    },
    getState,
  };
}

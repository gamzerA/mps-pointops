import { generateDemoPoints } from './lib/demo-points.js';

// This is a deterministic 2D teaching model. It illustrates selection rules;
// its JavaScript distances and timings are not a Metal kernel or a parity oracle.
const MAX_DEMO_POINTS = 160;
const FIXTURE_SEED = 0x4d505332;
const DEFAULT_QUERY = Object.freeze({ x: 0.52, y: 0.48 });
const MODES = new Set(['fps', 'knn', 'ball']);

const COPY = {
  ko: {
    play: '재생', pause: '일시정지', step: '한 단계', reset: '처음으로',
    samples: '추출 점 수', neighbors: '이웃 수', canvasTitle: '2D 점군 연산 설명',
    drag: '캔버스를 클릭하거나 드래그해 질의점을 옮기세요. 방향키로도 이동할 수 있습니다.',
    seed: '캔버스의 점을 클릭해 FPS 시작점을 바꾸세요. 방향키로 시작 인덱스를 바꿀 수 있습니다.',
    result: '선택 인덱스', inspected: '검사', distanceChecks: '거리 비교', of: '/', points: '점',
    teaching: '고정된 2D 설명 모형 · Metal 실행이나 벤치마크 아님',
    ready: '시작 전', selected: '선택', rejected: '제외',
    fpsStart: '첫 점', fpsNext: '기존 선택점까지 최소 거리²가 가장 큰 점',
    knn: '거리² 오름차순, 동률은 낮은 인덱스 우선',
    ball: '입력 순서로 검사. 거리² < 반지름²인 첫 K개',
    ballLimit: 'K개를 찾으면 조기 종료',
  },
  en: {
    play: 'Play', pause: 'Pause', step: 'One step', reset: 'Restart',
    samples: 'Sample count', neighbors: 'Neighbor count', canvasTitle: '2D point-cloud operation explanation',
    drag: 'Click or drag the canvas to move the query. Arrow keys also move it.',
    seed: 'Click a point to change the FPS start. Arrow keys change the start index.',
    result: 'Selected indices', inspected: 'Scanned', distanceChecks: 'Distance comparisons', of: '/', points: 'points',
    teaching: 'Fixed 2D teaching model · not a Metal execution or benchmark',
    ready: 'Ready', selected: 'Selected', rejected: 'Excluded',
    fpsStart: 'Initial point', fpsNext: 'Largest minimum squared distance to the selected set',
    knn: 'Ascending squared distance; lower index breaks ties',
    ball: 'Scan input order; take the first K with squared distance < radius²',
    ballLimit: 'Stop once K have been found',
  },
};

function squaredDistance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return dx * dx + dy * dy;
}

function clamp(value, lower, upper) {
  return Math.min(upper, Math.max(lower, value));
}

export function createPlaygroundPoints({ count = 48 } = {}) {
  if (!Number.isSafeInteger(count) || count < 0 || count > MAX_DEMO_POINTS) {
    throw new RangeError(`count must be an integer from 0 to ${MAX_DEMO_POINTS}`);
  }
  // Prefix stability means changing N never shuffles the points already shown.
  return generateDemoPoints({ count, seed: FIXTURE_SEED, distribution: 'clustered' });
}

/**
 * Return every explanatory frame plus the final original-input indices.
 * FPS starts at seedIndex and breaks equal max-min distances by lower index.
 * kNN orders by (squared distance, index). Ball Query scans original order,
 * uses strict d² < r², and stops after the first K hits.
 */
export function computePlaygroundSteps({
  mode = 'knn', points, query = DEFAULT_QUERY, k = 8, radius = 0.2, seedIndex = 0,
} = {}) {
  if (!MODES.has(mode)) throw new RangeError('mode must be fps, knn, or ball');
  if (!Array.isArray(points) || points.some((p) => !p || !Number.isFinite(p.x) || !Number.isFinite(p.y))) {
    throw new TypeError('points must be an array of finite {x, y} coordinates');
  }
  if (!query || !Number.isFinite(query.x) || !Number.isFinite(query.y)) {
    throw new TypeError('query must have finite x and y coordinates');
  }
  if (!Number.isSafeInteger(k) || k < 0) throw new RangeError('k must be a nonnegative integer');
  if (!Number.isFinite(radius) || radius < 0 || !Number.isFinite(radius * radius)) {
    throw new RangeError('radius must be nonnegative with a finite square');
  }
  if (!Number.isSafeInteger(seedIndex) || (points.length && (seedIndex < 0 || seedIndex >= points.length))) {
    throw new RangeError('seedIndex must be a valid point index');
  }

  const selected = [];
  const frames = [{ selected: [], focusIndex: null, accepted: null, inspected: 0, metric: null, nearestIndex: null }];
  const limit = Math.min(k, points.length);

  if (mode === 'fps') {
    const chosen = new Set();
    let distanceComparisons = 0;
    for (let rank = 0; rank < limit; rank += 1) {
      let bestIndex = rank === 0 ? seedIndex : -1;
      let bestMetric = -Infinity;
      let nearestIndex = null;
      if (rank > 0) {
        for (let i = 0; i < points.length; i += 1) {
          if (chosen.has(i)) continue;
          let minimum = Infinity;
          let nearest = null;
          for (const index of selected) {
            distanceComparisons += 1;
            const distance = squaredDistance(points[i], points[index]);
            if (distance < minimum || (distance === minimum && (nearest === null || index < nearest))) {
              minimum = distance;
              nearest = index;
            }
          }
          if (minimum > bestMetric || (minimum === bestMetric && (bestIndex === -1 || i < bestIndex))) {
            bestMetric = minimum;
            bestIndex = i;
            nearestIndex = nearest;
          }
        }
      }
      chosen.add(bestIndex);
      selected.push(bestIndex);
      frames.push({
        selected: [...selected], focusIndex: bestIndex, accepted: true,
        inspected: distanceComparisons,
        metric: rank === 0 ? null : bestMetric, nearestIndex,
      });
    }
  } else if (mode === 'knn') {
    const sorted = points.map((point, index) => ({ index, distance: squaredDistance(point, query) }))
      .sort((a, b) => a.distance - b.distance || a.index - b.index);
    for (const candidate of sorted.slice(0, limit)) {
      selected.push(candidate.index);
      frames.push({
        selected: [...selected], focusIndex: candidate.index, accepted: true,
        inspected: points.length, metric: candidate.distance, nearestIndex: null,
      });
    }
  } else {
    const radiusSquared = radius * radius;
    for (let i = 0; i < points.length && selected.length < limit; i += 1) {
      const distance = squaredDistance(points[i], query);
      const accepted = distance < radiusSquared;
      if (accepted) selected.push(i);
      frames.push({
        selected: [...selected], focusIndex: i, accepted,
        inspected: i + 1, metric: distance, nearestIndex: null,
      });
    }
  }
  return { mode, indices: [...selected], frames, inspected: frames.at(-1).inspected };
}

function getCSSColor(root, variable, fallback) {
  const value = getComputedStyle(root).getPropertyValue(variable).trim();
  return value || fallback;
}

function drawCircle(context, x, y, radius, fill, stroke, width = 1) {
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  if (fill) { context.fillStyle = fill; context.fill(); }
  if (stroke) { context.strokeStyle = stroke; context.lineWidth = width; context.stroke(); }
}

/** One data unit is the same number of CSS pixels on both axes. */
export function layoutPlaygroundPlot(width, height) {
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new RangeError('canvas width and height must be positive and finite');
  }
  const compact = width < 520;
  const left = compact ? 34 : 49;
  const top = 23;
  const bottom = compact ? 39 : 46;
  const right = compact ? 17 : 28;
  const availableWidth = Math.max(1, width - left - right);
  const availableHeight = Math.max(1, height - top - bottom);
  const side = Math.min(availableWidth, availableHeight);
  return {
    x: left + (availableWidth - side) / 2,
    y: top + (availableHeight - side) / 2,
    width: side,
    height: side,
    canvasWidth: width,
    canvasHeight: height,
  };
}

/**
 * Hooks beneath root: [data-playground-mode="fps|knn|ball"],
 * [data-playground-count], [data-playground-k], [data-playground-radius],
 * [data-playground-play], [data-playground-step], [data-playground-reset],
 * [data-playground-stage], [data-playground-result] and optional output/label
 * hooks for the three controls. Return destroy() to remove every listener.
 */
export function mountPointPlayground({ root, canvas = root?.querySelector('[data-playground-canvas]'), autoPlay = true } = {}) {
  if (!root || !canvas || typeof canvas.getContext !== 'function') {
    throw new TypeError('mountPointPlayground requires a root and a canvas');
  }
  const context = canvas.getContext('2d');
  if (!context) throw new Error('2D canvas context is unavailable');
  const find = (name) => root.querySelector(`[data-playground-${name}]`);
  const countInput = find('count');
  const kInput = find('k');
  const radiusInput = find('radius');
  const playButton = find('play');
  const stepButton = find('step');
  const resetButton = find('reset');
  const stageOutput = find('stage');
  const resultOutput = find('result');
  const modeButtons = [...root.querySelectorAll('[data-playground-mode]')];
  const countOutput = find('count-output');
  const kOutput = find('k-output');
  const radiusOutput = find('radius-output');
  const kLabel = find('k-label');
  const queryOutput = find('query-output');
  const formulaOutput = find('formula');
  const match = typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : null;

  let mode = MODES.has(root.dataset.mode) ? root.dataset.mode : 'fps';
  let count = clamp(Math.round(Number(countInput?.value) || 48), 1, MAX_DEMO_POINTS);
  let k = clamp(Math.round(Number(kInput?.value) || 8), 1, 32);
  const initialRadius = Number(radiusInput?.value);
  let radius = clamp(Number.isFinite(initialRadius) ? initialRadius : 0.2, 0, 1.2);
  let points = createPlaygroundPoints({ count });
  let query = { ...DEFAULT_QUERY };
  let seedIndex = 0;
  let calculation = computePlaygroundSteps({ mode, points, query, k, radius, seedIndex });
  let frameIndex = match?.matches ? calculation.frames.length - 1 : 0;
  let requestedPlay = Boolean(autoPlay && !match?.matches);
  let visible = true;
  let pointerDragging = false;
  let destroyed = false;
  let timer = null;
  let animationFrame = null;
  let plot = null;
  let resizeObserver = null;
  let intersectionObserver = null;
  const listeners = [];
  let language = document.documentElement.lang?.toLowerCase().startsWith('en') ? 'en' : 'ko';

  const listen = (target, event, handler, options) => {
    if (!target) return;
    target.addEventListener(event, handler, options);
    listeners.push(() => target.removeEventListener(event, handler, options));
  };
  const dictionary = () => COPY[language];
  const canPlay = () => requestedPlay && visible && !document.hidden && !match?.matches;

  function stopClock() {
    if (timer !== null) { clearTimeout(timer); timer = null; }
    if (animationFrame !== null) { cancelAnimationFrame(animationFrame); animationFrame = null; }
  }

  function currentFrame() { return calculation.frames[frameIndex]; }

  function updateText() {
    const words = dictionary();
    const frame = currentFrame();
    const total = calculation.frames.length - 1;
    if (countOutput) countOutput.textContent = String(count);
    if (kOutput) kOutput.textContent = String(k);
    if (radiusOutput) radiusOutput.textContent = radius.toFixed(3);
    if (kLabel) kLabel.textContent = mode === 'fps' ? words.samples : words.neighbors;
    if (queryOutput) queryOutput.textContent = mode === 'fps' ? `#${seedIndex}` : `(${query.x.toFixed(2)}, ${query.y.toFixed(2)})`;
    if (radiusInput) radiusInput.disabled = mode !== 'ball';
    if (playButton) {
      playButton.textContent = canPlay() ? words.pause : words.play;
      playButton.setAttribute('aria-pressed', String(canPlay()));
      playButton.setAttribute('aria-label', playButton.textContent);
    }
    if (stepButton) { stepButton.textContent = words.step; stepButton.setAttribute('aria-label', words.step); }
    if (resetButton) { resetButton.textContent = words.reset; resetButton.setAttribute('aria-label', words.reset); }
    root.dataset.mode = mode;
    for (const button of modeButtons) {
      button.setAttribute('aria-pressed', String(button.dataset.playgroundMode === mode));
    }
    if (formulaOutput) {
      formulaOutput.textContent = {
        fps: 'argmaxᵢ minₛ∈S ‖pᵢ − pₛ‖²',
        knn: 'sortᵢ (‖pᵢ − q‖², i) → first K',
        ball: '‖pᵢ − q‖² < r² → first K in input order',
      }[mode];
    }
    if (stageOutput) {
      const explanation = mode === 'fps'
        ? frameIndex === 1 ? words.fpsStart : words.fpsNext
        : mode === 'knn' ? words.knn : words.ball;
      const focus = frame.focusIndex === null ? words.ready
        : `#${frame.focusIndex} ${frame.accepted ? words.selected : words.rejected}`;
      stageOutput.textContent = `${frameIndex}/${total} · ${focus} · ${explanation}`;
    }
    if (resultOutput) {
      const effort = mode === 'fps'
        ? `${words.distanceChecks}: ${frame.inspected}`
        : `${words.inspected} ${frame.inspected}${words.of}${count} ${words.points}`;
      resultOutput.textContent = `${words.result}: [${frame.selected.join(', ')}] · ${effort}. ${words.teaching}.`;
      resultOutput.setAttribute('aria-live', canPlay() ? 'off' : 'polite');
    }
    canvas.setAttribute('aria-label', `${words.canvasTitle}: ${mode.toUpperCase()}. ${mode === 'fps' ? words.seed : words.drag}`);
  }

  function resize() {
    if (destroyed) return;
    const bounds = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width));
    const height = Math.max(1, Math.round(bounds.height));
    const dpr = Math.min(3, Math.max(1, window.devicePixelRatio || 1));
    const pixelWidth = Math.round(width * dpr);
    const pixelHeight = Math.round(height * dpr);
    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    plot = layoutPlaygroundPlot(width, height);
    draw(1);
  }

  function toCanvas(point) {
    return { x: plot.x + point.x * plot.width, y: plot.y + (1 - point.y) * plot.height };
  }

  function draw(progress = 1) {
    if (!plot || destroyed) return;
    const { canvasWidth: width, canvasHeight: height } = plot;
    const frame = currentFrame();
    const accent = getCSSColor(root, `--playground-${mode}`, { fps: '#8b5a2b', knn: '#236a9b', ball: '#2d8067' }[mode]);
    const ink = getCSSColor(root, '--playground-ink', '#193541');
    const faint = getCSSColor(root, '--playground-faint', '#aebfc3');
    const surface = getCSSColor(root, '--playground-surface', '#f8fbfa');
    context.clearRect(0, 0, width, height);
    context.fillStyle = surface;
    context.fillRect(0, 0, width, height);

    // Axes are measurement cues, not a decorative background lattice.
    context.strokeStyle = faint;
    context.lineWidth = 1;
    context.strokeRect(plot.x, plot.y, plot.width, plot.height);
    context.fillStyle = ink;
    context.font = '11px ui-monospace, SFMono-Regular, Menlo, monospace';
    context.textAlign = 'center';
    context.textBaseline = 'top';
    for (const value of [0, 0.5, 1]) {
      const x = plot.x + value * plot.width;
      context.beginPath(); context.moveTo(x, plot.y + plot.height); context.lineTo(x, plot.y + plot.height + 5); context.stroke();
      context.fillText(value.toFixed(1), x, plot.y + plot.height + 9);
    }
    context.textAlign = 'right';
    context.textBaseline = 'middle';
    for (const value of [0, 0.5, 1]) {
      const y = plot.y + (1 - value) * plot.height;
      context.beginPath(); context.moveTo(plot.x - 5, y); context.lineTo(plot.x, y); context.stroke();
      context.fillText(value.toFixed(1), plot.x - 9, y);
    }

    context.save();
    context.beginPath(); context.rect(plot.x, plot.y, plot.width, plot.height); context.clip();
    const center = toCanvas(query);
    if (mode === 'ball') {
      drawCircle(context, center.x, center.y, radius * plot.width,
        'rgba(45, 128, 103, 0.07)', accent, 1.25);
    }
    if (mode !== 'fps') {
      context.strokeStyle = '#b5652f';
      context.lineWidth = 1.1;
      context.beginPath(); context.moveTo(center.x - 9, center.y); context.lineTo(center.x + 9, center.y);
      context.moveTo(center.x, center.y - 9); context.lineTo(center.x, center.y + 9); context.stroke();
      drawCircle(context, center.x, center.y, 3.5, surface, '#b5652f', 1.5);
    }

    const focus = frame.focusIndex;
    const selectedSet = new Set(frame.selected);
    // Lines are only drawn for a relation the algorithm really uses.
    if (mode === 'knn') {
      for (const index of frame.selected) {
        if (index === focus) continue;
        const point = toCanvas(points[index]);
        context.beginPath(); context.moveTo(center.x, center.y); context.lineTo(point.x, point.y);
        context.strokeStyle = accent; context.globalAlpha = 0.29; context.lineWidth = 1; context.stroke(); context.globalAlpha = 1;
      }
    }
    if (focus !== null) {
      const target = toCanvas(points[focus]);
      const source = mode === 'fps'
        ? (frame.nearestIndex === null ? null : toCanvas(points[frame.nearestIndex]))
        : center;
      if (source) {
        context.beginPath(); context.moveTo(source.x, source.y);
        context.lineTo(source.x + (target.x - source.x) * progress,
          source.y + (target.y - source.y) * progress);
        context.strokeStyle = accent; context.globalAlpha = frame.accepted ? 0.78 : 0.38;
        context.lineWidth = frame.accepted ? 1.8 : 1.1; context.stroke(); context.globalAlpha = 1;
      }
    }

    for (let index = 0; index < points.length; index += 1) {
      const at = toCanvas(points[index]);
      const selected = selectedSet.has(index);
      drawCircle(context, at.x, at.y, selected ? 5.1 : 3.2,
        selected ? accent : surface, selected ? accent : faint, selected ? 1.4 : 1.15);
    }
    if (focus !== null) {
      const at = toCanvas(points[focus]);
      drawCircle(context, at.x, at.y, 10 + (1 - progress) * 6, null,
        frame.accepted ? accent : ink, 1.25);
    }
    context.restore();
  }

  function render(progress = 1) { updateText(); draw(progress); }

  function animateFocus(duration) {
    if (!canPlay() || duration <= 0) { draw(1); return; }
    const start = performance.now();
    const tick = (now) => {
      animationFrame = null;
      if (!canPlay()) { draw(1); return; }
      const ratio = clamp((now - start) / duration, 0, 1);
      draw(1 - (1 - ratio) ** 3);
      if (ratio < 1) animationFrame = requestAnimationFrame(tick);
    };
    animationFrame = requestAnimationFrame(tick);
  }

  function schedule() {
    if (!canPlay() || timer !== null) return;
    const atEnd = frameIndex >= calculation.frames.length - 1;
    const delay = atEnd ? 1150 : mode === 'ball' ? 125 : 540;
    timer = setTimeout(() => {
      timer = null;
      if (!canPlay()) return;
      if (animationFrame !== null) { cancelAnimationFrame(animationFrame); animationFrame = null; }
      frameIndex = atEnd ? 0 : frameIndex + 1;
      render(atEnd ? 1 : 0);
      if (!atEnd) animateFocus(Math.min(240, delay * 0.72));
      schedule();
    }, delay);
  }

  function recompute({ reset = false } = {}) {
    stopClock();
    calculation = computePlaygroundSteps({ mode, points, query, k, radius, seedIndex });
    frameIndex = reset && canPlay() ? 0 : calculation.frames.length - 1;
    render();
    schedule();
  }

  function setMode(next) {
    if (!MODES.has(next) || next === mode) return;
    mode = next;
    recompute({ reset: true });
  }

  function setQueryFromPointer(event) {
    if (!plot) return;
    const bounds = canvas.getBoundingClientRect();
    const x = clamp((event.clientX - bounds.left - plot.x) / plot.width, 0, 1);
    const y = clamp(1 - (event.clientY - bounds.top - plot.y) / plot.height, 0, 1);
    if (mode === 'fps') {
      let best = 0;
      let bestDistance = Infinity;
      for (let index = 0; index < points.length; index += 1) {
        const distance = squaredDistance(points[index], { x, y });
        if (distance < bestDistance) { best = index; bestDistance = distance; }
      }
      if (best === seedIndex) return;
      seedIndex = best;
    } else {
      if (query.x === x && query.y === y) return;
      query = { x, y };
    }
    recompute({ reset: true });
  }

  for (const button of modeButtons) listen(button, 'click', () => setMode(button.dataset.playgroundMode));
  listen(countInput, 'input', () => {
    const next = clamp(Math.round(Number(countInput.value) || 1), 1, MAX_DEMO_POINTS);
    if (next === count) return;
    count = next;
    seedIndex = Math.min(seedIndex, count - 1);
    points = createPlaygroundPoints({ count });
    recompute({ reset: true });
  });
  listen(kInput, 'input', () => {
    const next = clamp(Math.round(Number(kInput.value) || 1), 1, 32);
    if (next === k) return;
    k = next;
    recompute({ reset: true });
  });
  listen(radiusInput, 'input', () => {
    const next = clamp(Number(radiusInput.value), 0, 1.2);
    if (!Number.isFinite(next) || next === radius) return;
    radius = next;
    recompute({ reset: true });
  });
  listen(playButton, 'click', () => {
    if (match?.matches) {
      frameIndex = calculation.frames.length - 1;
      render();
      return;
    }
    requestedPlay = !requestedPlay;
    if (requestedPlay && frameIndex === calculation.frames.length - 1) frameIndex = 0;
    stopClock(); render(); schedule();
  });
  listen(stepButton, 'click', () => {
    requestedPlay = false;
    stopClock();
    frameIndex = (frameIndex + 1) % calculation.frames.length;
    render();
  });
  listen(resetButton, 'click', () => {
    stopClock(); frameIndex = 0; render(); schedule();
  });
  if (!canvas.hasAttribute('tabindex')) canvas.tabIndex = 0;
  listen(canvas, 'pointerdown', (event) => {
    pointerDragging = true;
    canvas.setPointerCapture?.(event.pointerId);
    setQueryFromPointer(event);
  });
  listen(canvas, 'pointermove', (event) => {
    if (pointerDragging) setQueryFromPointer(event);
  });
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
        x: clamp(query.x + (event.key === 'ArrowRight' ? delta : event.key === 'ArrowLeft' ? -delta : 0), 0, 1),
        y: clamp(query.y + (event.key === 'ArrowUp' ? delta : event.key === 'ArrowDown' ? -delta : 0), 0, 1),
      };
    }
    recompute({ reset: true });
  });
  listen(document, 'visibilitychange', () => {
    stopClock(); render(); schedule();
  });
  if (match) {
    const onMotion = () => {
      if (match.matches) {
        requestedPlay = false;
        frameIndex = calculation.frames.length - 1;
      }
      stopClock(); render(); schedule();
    };
    if (match.addEventListener) listen(match, 'change', onMotion);
    else {
      match.addListener(onMotion);
      listeners.push(() => match.removeListener(onMotion));
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
  resize();
  render();
  schedule();

  return {
    destroy() {
      if (destroyed) return;
      destroyed = true;
      stopClock();
      resizeObserver?.disconnect();
      intersectionObserver?.disconnect();
      for (const remove of listeners) remove();
    },
    setMode,
    setLanguage(nextLanguage) {
      if (!['ko', 'en'].includes(nextLanguage)) throw new RangeError('language must be ko or en');
      language = nextLanguage;
      render();
    },
    refreshLocale() {
      language = document.documentElement.lang?.toLowerCase().startsWith('en') ? 'en' : 'ko';
      render();
    },
    getState() {
      return {
        mode, count, k, radius, query: { ...query }, seedIndex, frameIndex,
        indices: [...calculation.indices], frame: { ...currentFrame(), selected: [...currentFrame().selected] },
      };
    },
  };
}

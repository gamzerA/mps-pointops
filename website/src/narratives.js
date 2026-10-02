const ROUTE = [
  {
    label: { ko: '00 / 차단', en: '00 / BLOCKED' },
    title: { ko: 'CUDA 전용 호출', en: 'CUDA-only operator' },
    detail: { ko: 'Apple Silicon에는 CUDA 런타임이 없어 이 호출이 여기서 멈춥니다.', en: 'An Apple Silicon Mac has no CUDA runtime, so this call stops here.' },
    stepHint: { ko: '왜 Mac에서는 그대로 실행되지 않나', en: 'Why this call cannot run unchanged on a Mac' },
    nodeHint: { ko: 'Apple Silicon에서 중단', en: 'Unavailable on Apple Silicon' },
  },
  {
    label: { ko: '01 / 입력', en: '01 / INPUT' },
    title: { ko: 'PyTorch', en: 'PyTorch' },
    detail: { ko: '모델의 텐서와 연산 요청은 PyTorch 안에서 시작합니다.', en: 'The model begins with PyTorch tensors and an operator call.' },
    stepHint: { ko: '모델은 같은 텐서 연산을 요청합니다', en: 'The model still requests a tensor operation' },
    nodeHint: { ko: '텐서 · 모델 호출', en: 'Tensor · model call' },
  },
  {
    label: { ko: '02 / 연결', en: '02 / ADAPT' },
    title: { ko: '호환 계층', en: 'Compatibility layer' },
    detail: { ko: '지원되는 기존 점군 API 호출을 mps-pointops의 공개 진입점으로 연결합니다.', en: 'Supported existing point-cloud calls enter the published mps-pointops interface.' },
    stepHint: { ko: '지원되는 API를 MPS 경로와 연결합니다', en: 'Connect supported APIs to the MPS path' },
    nodeHint: { ko: '지원 서명 · 장치 연결', en: 'Supported signature · device bridge' },
  },
  {
    label: { ko: '03 / 분기', en: '03 / ROUTE' },
    title: { ko: 'MPS 디스패치', en: 'MPS dispatch' },
    detail: { ko: 'MPS 텐서의 장치와 입력 계약을 확인하고 지원되는 네이티브 경로를 선택합니다.', en: 'Input and device contracts select a supported native path for MPS tensors.' },
    stepHint: { ko: '장치와 입력 계약을 확인합니다', en: 'Check the device and input contract' },
    nodeHint: { ko: '입력 계약 · 경로 선택', en: 'Input contract · route selection' },
  },
  {
    label: { ko: '04 / 실행', en: '04 / EXECUTE' },
    title: { ko: 'Metal 커널', en: 'Metal kernel' },
    detail: { ko: '해당 연산의 거리·선택·집계를 Apple GPU에서 계산합니다.', en: 'The operator computes distances, selections, or reductions on the Apple GPU.' },
    stepHint: { ko: 'GPU에서 점과 이웃을 계산합니다', en: 'Compute point and neighborhood operations on GPU' },
    nodeHint: { ko: 'GPU 계산', en: 'GPU computation' },
  },
  {
    label: { ko: '05 / 출력', en: '05 / OUTPUT' },
    title: { ko: '결과 텐서', en: 'Result tensor' },
    detail: { ko: '선택 인덱스와 거리 또는 피처가 MPS 텐서로 다음 레이어에 전달됩니다.', en: 'Indices and distances or features reach the next layer as MPS tensors.' },
    stepHint: { ko: '인덱스와 거리를 다음 레이어로 전달합니다', en: 'Pass the result to the next layer' },
    nodeHint: { ko: '다음 레이어로 전달', en: 'Pass to the next layer' },
  },
];

const PHASES = [
  {
    label: { ko: '단계 01', en: 'PHASE 01' },
    name: { ko: '점과 이웃', en: 'Points and neighbors' },
    detail: { ko: 'FPS · kNN · Ball Query', en: 'FPS · kNN · Ball Query' },
    status: { ko: '배포된 핵심 연산', en: 'Core operators in the published package' },
    buttonTitle: { ko: '샘플링과 이웃', en: 'Sampling and neighbors' },
    stepHint: { ko: '점 → 대표점과 이웃', en: 'Points → samples and neighborhoods' },
  },
  {
    label: { ko: '단계 02', en: 'PHASE 02' },
    name: { ko: '점과 피처', en: 'Points and features' },
    detail: { ko: 'three_nn · three_interpolate · 특징 공간 kNN', en: 'three_nn · three_interpolate · feature-space kNN' },
    status: { ko: '배포된 실험 연산과 한정된 모델 검증', en: 'Shipped experimental operators and bounded model validation' },
    buttonTitle: { ko: '피처 전파', en: 'Feature propagation' },
    stepHint: { ko: '점 + 피처 → 전파', en: 'Points + features → propagation' },
  },
  {
    label: { ko: '단계 03', en: 'PHASE 03' },
    name: { ko: '그래프와 격자', en: 'Graph and grid' },
    detail: { ko: 'PyG 일부 경로 · voxel ID · compact pooling', en: 'Bounded PyG paths · voxel IDs · compact pooling' },
    status: { ko: '검증 범위가 한정된 배포 기능', en: 'Published functions with bounded validation' },
    buttonTitle: { ko: '그래프와 격자', en: 'Graph and grid' },
    stepHint: { ko: '점 → 그래프와 복셀 ID', en: 'Points → graphs and voxel IDs' },
  },
  {
    label: { ko: '단계 04', en: 'PHASE 04' },
    name: { ko: '기하 관계', en: 'Geometry and scale' },
    detail: { ko: 'Chamfer 부분집합 · 대규모 BVH 연구', en: 'Chamfer subset · large-scale BVH research' },
    status: { ko: '배포된 손실 부분집합과 제한된 선택형 BVH 경로', en: 'Shipped loss subset and bounded opt-in BVH path' },
    buttonTitle: { ko: '기하와 대규모 탐색', en: 'Geometry and scale' },
    stepHint: { ko: '거리 손실과 공간 인덱스', en: 'Geometry loss and spatial indexing' },
  },
  {
    label: { ko: '단계 05 · 실험', en: 'PHASE 05 · EXPERIMENTAL' },
    name: { ko: '희소 3D 계산', en: 'Sparse 3D computation' },
    detail: { ko: '복셀 rulebook → sparse convolution', en: 'Voxel rulebook → sparse convolution' },
    status: { ko: '비공개 실험 API · 제한된 Metal 희소 합성곱', en: 'Private experimental API · bounded Metal sparse convolution' },
    buttonTitle: { ko: '희소 3D', en: 'Sparse 3D' },
    stepHint: { ko: '작은 사례의 순방향·1차 역전파 검증', en: 'Tiny-fixture forward and first-gradient checks' },
  },
];

const COPY = {
  ko: { routeGroup: '실행 단계 선택', phaseGroup: '개발 단계 선택' },
  en: { routeGroup: 'Choose an execution step', phaseGroup: 'Choose a development phase' },
};

function tr(value, language) { return value[language]; }
function normalizeLanguage(language) { return language === 'en' ? 'en' : 'ko'; }

const POINTS = [
  [95, 172], [130, 122], [154, 190], [180, 147], [214, 98], [236, 178],
  [270, 133], [304, 201], [333, 116], [361, 167], [395, 95], [425, 150],
  [451, 210], [485, 126], [520, 183], [553, 99], [579, 163], [612, 123],
];
const COMPACT_POINTS = POINTS.map(([x, y]) => [
  Math.round(29 + (x - 95) * 300 / 517),
  Math.round(35 + (y - 95) * 170 / 115),
]);

function useCompactPhaseScene() {
  if (typeof window.matchMedia === 'function') return window.matchMedia('(max-width: 560px)').matches;
  return Number.isFinite(window.innerWidth) && window.innerWidth <= 560;
}

function svgElement(name, attributes = {}) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, String(value));
  return node;
}

function phaseAccessibleLabel(phase, language) {
  return `${tr(PHASES[phase].name, language)}: ${tr(PHASES[phase].detail, language)}. ${tr(PHASES[phase].status, language)}`;
}

function renderPhase(svg, phase, language, compact = false) {
  // Desktop crops unused margins. Narrow screens reflow the same conceptual
  // point scene into a taller coordinate system instead of scaling a panorama.
  svg.setAttribute('viewBox', compact ? '0 0 360 240' : '52 50 596 200');
  svg.setAttribute('aria-label', phaseAccessibleLabel(phase, language));
  const scene = svgElement('g', { class: 'phase-diagram' });
  svg.replaceChildren(scene);
  const frame = svgElement('rect', {
    x: compact ? 0 : 1, y: compact ? 0 : 1,
    width: compact ? 360 : 698, height: compact ? 240 : 298,
    rx: 4, class: 'phase-frame',
  });
  scene.append(frame);
  if (phase >= 2) {
    const grid = compact
      ? { x0: 30, x1: 330, dx: 50, y0: 30, y1: 210, dy: 45 }
      : { x0: 62, x1: 650, dx: 84, y0: 55, y1: 250, dy: 60 };
    for (let x = grid.x0; x <= grid.x1; x += grid.dx) scene.append(svgElement('line', { x1: x, x2: x, y1: grid.y0, y2: grid.y1, class: 'phase-grid' }));
    for (let y = grid.y0; y <= grid.y1; y += grid.dy) scene.append(svgElement('line', { x1: grid.x0, x2: grid.x1, y1: y, y2: y, class: 'phase-grid' }));
  }
  const points = compact ? COMPACT_POINTS : POINTS;
  if (phase >= 2) {
    for (let i = 0; i < points.length - 2; i += 3) {
      const a = points[i]; const b = points[i + 2];
      scene.append(svgElement('line', { x1: a[0], y1: a[1], x2: b[0], y2: b[1], class: 'phase-edge' }));
      if (phase === 2) {
        scene.append(svgElement('circle', {
          cx: a[0], cy: a[1], r: compact ? 3 : 2.5,
          class: `phase-edge-packet phase-packet-${i / 3}`,
        }));
      }
    }
  }
  if (phase === 3) {
    const bounds = compact
      ? [[26, 30, 136, 106], [182, 27, 151, 109], [90, 144, 184, 73]]
      : [[84, 83, 170, 133], [270, 83, 180, 141], [468, 78, 145, 148]];
    bounds.forEach(([x, y, w, h], i) => {
      scene.append(svgElement('rect', {
        x, y, width: w, height: h, class: `phase-bvh phase-bound-${i}`,
      }));
    });
  }
  if (phase === 4) {
    const cells = compact
      ? Array.from({ length: 15 }, (_, i) => [32 + (i % 5) * 61, 40 + Math.floor(i / 5) * 53])
      : [[66, 99], [150, 99], [234, 99], [318, 99], [402, 99], [486, 99], [570, 99], [108, 159], [192, 159], [276, 159], [360, 159], [444, 159], [528, 159]];
    cells.forEach(([x, y], i) => {
      scene.append(svgElement('rect', {
        x, y, width: compact ? 48 : 68, height: compact ? 40 : 52,
        class: `phase-voxel phase-voxel-row-${Math.floor(i / 5)}`,
      }));
    });
  }
  if (phase === 0) {
    scene.append(svgElement('circle', { cx: compact ? 180 : 355, cy: compact ? 120 : 150, r: compact ? 92 : 98, class: 'phase-radius' }));
  }
  points.forEach(([x, y], i) => {
    if (phase === 4 && i % 2) return;
    const selected = i === 4 || i === 10 || i === 15;
    const className = selected ? 'phase-point-selected' : 'phase-point';
    scene.append(svgElement('circle', {
      cx: x, cy: y, r: phase === 4 ? 5 : compact ? 5 : 4,
      class: phase === 0 && selected
        ? `${className} phase-sample-${i === 4 ? 0 : i === 10 ? 1 : 2}` : className,
    }));
    if (phase === 1 && i % 3 === 0) {
      const rise = 16 + i % 4 * 4;
      scene.append(svgElement('line', { x1: x, y1: y - 8, x2: x, y2: y - 8 - rise, class: 'phase-feature' }));
      scene.append(svgElement('circle', {
        cx: x, cy: y - 8, r: compact ? 3 : 2.5,
        class: `phase-feature-packet phase-packet-${i / 3}`,
      }));
    }
  });
  if (phase === 3) {
    scene.append(svgElement('circle', { cx: compact ? 180 : 350, cy: compact ? 120 : 156, r: 8, class: 'phase-query' }));
  }
}

function observeSteps(root, selector, onSelect) {
  if (!('IntersectionObserver' in window)) return () => {};
  const observer = new IntersectionObserver(entries => {
    for (const entry of entries) {
      if (entry.isIntersecting && !root.dataset.manualSelection) onSelect(Number(entry.target.dataset[selector]));
    }
  }, { rootMargin: '-34% 0px -44% 0px', threshold: 0 });
  root.querySelectorAll(`[data-${selector.replace(/[A-Z]/g, c => `-${c.toLowerCase()}`)}]`).forEach(node => observer.observe(node));
  return () => observer.disconnect();
}

export function mountExecutionStory(root, { language: initialLanguage = 'ko' } = {}) {
  if (!root) return { setLanguage() {}, select() {}, disconnect() {} };
  const steps = [...root.querySelectorAll('[data-route-step]')];
  const nodes = [...root.querySelectorAll('[data-route-node]')];
  const title = root.querySelector('[data-route-title]');
  const detail = root.querySelector('[data-route-detail]');
  const group = root.querySelector('.story-steps');
  const readout = root.querySelector('.story-readout');
  readout?.setAttribute('aria-live', 'polite');
  readout?.setAttribute('aria-atomic', 'true');
  const hoverMedia = typeof window.matchMedia === 'function'
    ? window.matchMedia('(hover: hover) and (pointer: fine)') : null;
  const removers = [];
  const listen = (element, eventName, handler) => {
    element.addEventListener(eventName, handler);
    removers.push(() => element.removeEventListener?.(eventName, handler));
  };
  let language = normalizeLanguage(initialLanguage);
  let index = 0;
  let committedIndex = 0;
  let hoveredElement = null;
  let previewTimer = null;
  let restoreTimer = null;
  const cancelTimer = timer => { if (timer !== null) window.clearTimeout(timer); };
  const cancelPending = () => {
    cancelTimer(previewTimer);
    cancelTimer(restoreTimer);
    previewTimer = null;
    restoreTimer = null;
  };
  const display = (next, { announce = true, force = false } = {}) => {
    const selectedIndex = Math.max(0, Math.min(ROUTE.length - 1, next));
    if (selectedIndex === index && !force) {
      if (announce) readout?.setAttribute('aria-live', 'polite');
      return;
    }
    index = selectedIndex;
    // Pointer scrubbing is visual feedback; only deliberate activation is announced.
    readout?.setAttribute('aria-live', announce ? 'polite' : 'off');
    root.dataset.routeActive = String(index);
    steps.forEach((step, i) => step.setAttribute('aria-current', i === index ? 'step' : 'false'));
    nodes.forEach(node => {
      const order = Number(node.dataset.routeNode);
      node.dataset.state = index === 0 ? (order === 0 ? 'active' : 'future')
        : order === 0 ? 'hidden' : order === index ? 'active' : order < index ? 'done' : 'future';
    });
    title.textContent = tr(ROUTE[index].title, language);
    detail.textContent = tr(ROUTE[index].detail, language);
  };
  const select = next => {
    cancelPending();
    hoveredElement = null;
    committedIndex = Math.max(0, Math.min(ROUTE.length - 1, next));
    display(committedIndex);
  };
  const setLanguage = nextLanguage => {
    language = normalizeLanguage(nextLanguage);
    group?.setAttribute('aria-label', COPY[language].routeGroup);
    steps.forEach((step, i) => {
      const item = ROUTE[i];
      step.querySelector('span').textContent = tr(item.label, language);
      step.querySelector('strong').textContent = tr(item.title, language);
      step.querySelector('small').textContent = tr(item.stepHint, language);
      step.setAttribute('aria-label', `${tr(item.title, language)}. ${tr(item.stepHint, language)}`);
    });
    nodes.forEach((node, i) => {
      const item = ROUTE[i];
      node.querySelector('strong').textContent = tr(item.title, language);
      node.querySelector('small').textContent = tr(item.nodeHint, language);
    });
    display(index, { force: true, announce: index === committedIndex });
  };
  const keyboardFocusActive = () => steps.includes(document.activeElement)
    && document.activeElement.matches?.(':focus-visible');
  const canPreview = event => event.pointerType === 'mouse'
    && (!hoverMedia || hoverMedia.matches) && !keyboardFocusActive();
  const hoverSelect = (element, i) => {
    listen(element, 'pointerenter', event => {
      if (!canPreview(event)) return;
      cancelPending();
      hoveredElement = element;
      previewTimer = window.setTimeout(() => {
        previewTimer = null;
        if (hoveredElement === element && !keyboardFocusActive()) display(i, { announce: false });
      }, 80);
    });
    listen(element, 'pointerleave', event => {
      if (!canPreview(event) || hoveredElement !== element) return;
      cancelTimer(previewTimer);
      previewTimer = null;
      hoveredElement = null;
      restoreTimer = window.setTimeout(() => {
        restoreTimer = null;
        if (hoveredElement === null) display(committedIndex, { announce: false });
      }, 100);
    });
  };
  steps.forEach((step, i) => {
    listen(step, 'click', () => select(i));
    listen(step, 'focus', () => {
      cancelPending();
      hoveredElement = null;
      display(committedIndex, { announce: false });
    });
    hoverSelect(step, i);
  });
  nodes.forEach(node => hoverSelect(node, Number(node.dataset.routeNode)));
  setLanguage(language);
  return {
    setLanguage, select,
    disconnect() { cancelPending(); hoveredElement = null; removers.forEach(remove => remove()); },
  };
}

export function mountPhaseStory(root, { language: initialLanguage = 'ko' } = {}) {
  if (!root) return { setLanguage() {}, select() {}, disconnect() {} };
  const svg = root.querySelector('[data-phase-scene]');
  const steps = [...root.querySelectorAll('[data-phase-step]')];
  const name = root.querySelector('[data-phase-name]');
  const detail = root.querySelector('[data-phase-detail]');
  const status = root.querySelector('[data-phase-status]');
  const group = root.querySelector('.story-steps');
  const content = root.closest?.('#content');
  const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)');
  let language = normalizeLanguage(initialLanguage);
  let currentPhase = 0;
  let compactScene = null;
  let inViewport = typeof IntersectionObserver !== 'function';
  let firstRevealPending = true;
  let revealTimer = null;
  const cancelReveal = () => {
    if (revealTimer !== null) window.clearTimeout(revealTimer);
    revealTimer = null;
    root.dataset.phaseReveal = 'off';
  };
  const reveal = () => {
    cancelReveal();
    root.dataset.phaseReveal = 'on';
    // The planned scaffold has the longest stagger (14 × 35ms + 420ms).
    revealTimer = window.setTimeout(() => {
      revealTimer = null;
      root.dataset.phaseReveal = 'off';
    }, 950);
  };
  const sceneIsVisible = () => (!content || (
    content.dataset.currentView === 'ecosystem' && content.dataset.ecosystemPanel === 'phases'
  )) && (root.getClientRects?.().length ?? 1) > 0
    && document.visibilityState !== 'hidden' && inViewport && !reducedMotion?.matches;
  const syncMotion = () => {
    const running = sceneIsVisible();
    const wasRunning = root.dataset.phaseMotion === 'running';
    root.dataset.phaseMotion = running ? 'running' : 'paused';
    if (!running) cancelReveal();
    else if (!wasRunning && firstRevealPending) {
      firstRevealPending = false;
      reveal();
    }
  };
  const updateCurrentText = () => {
    root.dataset.phaseActive = String(currentPhase);
    steps.forEach((step, i) => step.setAttribute('aria-current', i === currentPhase ? 'step' : 'false'));
    name.textContent = tr(PHASES[currentPhase].name, language);
    detail.textContent = tr(PHASES[currentPhase].detail, language);
    status.textContent = tr(PHASES[currentPhase].status, language);
    svg.setAttribute('aria-label', phaseAccessibleLabel(currentPhase, language));
  };
  const select = index => {
    const phase = Math.max(0, Math.min(PHASES.length - 1, index));
    if (phase === currentPhase && compactScene !== null) return;
    cancelReveal();
    currentPhase = phase;
    updateCurrentText();
    compactScene = useCompactPhaseScene();
    renderPhase(svg, phase, language, compactScene);
    if (root.dataset.phaseMotion === 'running') reveal();
  };
  const onResize = () => {
    const nextCompact = useCompactPhaseScene();
    if (nextCompact !== compactScene) {
      cancelReveal();
      compactScene = nextCompact;
      renderPhase(svg, currentPhase, language, compactScene);
    }
    syncMotion();
  };
  const setLanguage = nextLanguage => {
    language = normalizeLanguage(nextLanguage);
    group?.setAttribute('aria-label', COPY[language].phaseGroup);
    steps.forEach((step, i) => {
      const item = PHASES[i];
      step.querySelector('span').textContent = tr(item.label, language);
      step.querySelector('strong').textContent = tr(item.buttonTitle, language);
      step.querySelector('small').textContent = tr(item.stepHint, language);
      step.setAttribute('aria-label', `${tr(item.buttonTitle, language)}. ${tr(item.stepHint, language)}. ${tr(item.status, language)}`);
    });
    updateCurrentText();
    if (compactScene === null) {
      compactScene = useCompactPhaseScene();
      renderPhase(svg, currentPhase, language, compactScene);
    }
  };
  const stepHandlers = steps.map((step, i) => {
    const handler = () => select(i);
    step.addEventListener('click', handler);
    return () => step.removeEventListener?.('click', handler);
  });
  const viewportObserver = typeof IntersectionObserver === 'function'
    ? new IntersectionObserver(entries => {
      inViewport = entries.some(entry => entry.isIntersecting);
      syncMotion();
    }, { threshold: .05 }) : null;
  viewportObserver?.observe(svg);
  const attributeObserver = typeof MutationObserver === 'function'
    ? new MutationObserver(syncMotion) : null;
  if (content) attributeObserver?.observe(content, {
    attributes: true, attributeFilter: ['data-current-view', 'data-ecosystem-panel'],
  });
  attributeObserver?.observe(root, { attributes: true, attributeFilter: ['hidden'] });
  document.addEventListener?.('visibilitychange', syncMotion);
  reducedMotion?.addEventListener?.('change', syncMotion);
  window.addEventListener?.('resize', onResize);
  setLanguage(language);
  syncMotion();
  return {
    setLanguage, select,
    disconnect() {
      cancelReveal();
      root.dataset.phaseMotion = 'paused';
      viewportObserver?.disconnect();
      attributeObserver?.disconnect();
      document.removeEventListener?.('visibilitychange', syncMotion);
      reducedMotion?.removeEventListener?.('change', syncMotion);
      window.removeEventListener?.('resize', onResize);
      stepHandlers.forEach(remove => remove());
    },
  };
}

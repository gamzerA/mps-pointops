// A compact view over recorded fixtures. Every bar is a query time from one
// benchmark record; build times and measurements from other fixtures stay apart.

const DEFAULT_RECORD = 'spatial-knn-uniform-independent-1m-q65536';
const COMMIT = /^[0-9a-f]{40}$/;
const RAW_PATH = /^bench\/results\/[A-Za-z0-9_./-]+\.json$/;

const COPY = {
  ko: {
    title: '측정과 검증',
    ciLabel: '기록된 검사',
    statusNote: 'Safe/Fast는 M5 Pro 연구 소스의 별도 실행입니다. PyPI 패키지 검증 수치로 합치지 않습니다.',
    chartTitle: '한 조건 안의 실측 비교',
    fixtureLabel: '실측 기록',
    selectedRecord: '선택한 실측 기록',
    detailsTitle: '측정 조건과 근거',
    close: '닫기',
    proofTitle: '출처와 아직 남은 범위',
    releaseLabel: '공개 패키지',
    doiLink: '버전 DOI ↗',
    measurementSourceLabel: '현재 측정 소스',
    testSourceLabel: 'Safe/Fast 테스트 소스',
    boundary: '물리 GPU 메모리 피크는 아직 계측되지 않았습니다. BVH 경로는 공개 v0.8.0 wheel에 없습니다.',
    boundaryM1: 'M1 8GB의 100만 점 실측은 미배포 소스의 opt-in BVH와 scan을 비교합니다. Auto는 여섯 조건 모두 scan을 선택했습니다. 실제 GPU 메모리 피크는 아직 계측되지 않았습니다.',
    queryTime: '동일 기록의 쿼리 시간',
    queryNote: '막대 길이는 선택한 한 기록 안에서만 비교합니다. 빌드 시간은 포함하지 않습니다.',
    queryNoteM1: '사전 적재한 MPS 입력의 동기화된 쿼리 중앙값입니다. 인덱스 빌드·전송·컴파일은 제외합니다. Auto는 M1에서 scan입니다.',
    fixture: '입력 조건',
    provenance: '미배포 소스 실측',
    prototype: '연구 시제품 실측',
    listUnreleased: '미배포 소스',
    listPrototype: '시제품',
    auto: 'Auto 선택',
    detailButton: '조건과 원시 근거 보기',
    rawLink: '원시 JSON ↗',
    sourceLink: '측정 커밋 ↗',
    testLogSafe: 'Safe 로그 ↗',
    testLogFast: 'Fast 로그 ↗',
    passMeta: (run) => `건너뜀 ${run.skipped} · 예상 실패 ${run.xfailed}`,
    pending: '검증 대기',
    pendingM1: 'M1 8GB 공간 탐색',
    pendingMemory: '물리 GPU 메모리 피크',
    memorySummary: (tensor, driver) => `M1의 별도 메모리 계측: 텐서 할당기 피크 ${tensor} MiB · 드라이버 표본 최고 ${driver} MiB. 실제 GPU 피크가 아닙니다.`,
    operation: { knn: 'kNN', radius: '반경 검색' },
    distribution: { uniform: '균일', coincident: '동일 좌표' },
    conditions: '측정 환경',
    statistic: '통계 방식',
    repetitions: '반복',
    inputResidence: '입력 위치',
    querySource: '쿼리 구성',
    compilation: '셰이더 컴파일 포함',
    transfer: '호스트→장치 전송 포함',
    parity: '동일 인덱스 슬롯 불일치',
    distanceParity: '거리 제곱 비트 불일치',
    orderParity: '입력 순서 위반',
    stackFallback: 'BVH 스택 대체 처리 행',
    mathMode: 'MPS 수학 모드',
    fallback: 'MPS CPU fallback',
    safeMath: 'Safe math',
    fastMath: 'Fast math',
    build: '빌드 시간 (차트에서 제외)',
    caution: '해석 범위',
    yes: '예', no: '아니요',
    sourceRecord: '원시 기록',
    recordedOn: '기록된 기기',
    unknown: '기록 없음',
  },
  en: {
    title: 'Measurements & validation',
    ciLabel: 'recorded checks',
    statusNote: 'Safe and Fast are separate M5 Pro research-source runs, not PyPI package pass counts.',
    chartTitle: 'Measured within one fixture',
    fixtureLabel: 'Recorded fixtures',
    selectedRecord: 'Selected record',
    detailsTitle: 'Conditions and evidence',
    close: 'Close',
    proofTitle: 'Sources and open limits',
    releaseLabel: 'Published package',
    doiLink: 'Version DOI ↗',
    measurementSourceLabel: 'Selected measurement source',
    testSourceLabel: 'Safe/Fast test source',
    boundary: 'Physical GPU memory peak remains unmeasured. BVH is absent from the published v0.8.0 wheel.',
    boundaryM1: 'M1 8 GB measurements compare opt-in BVH with scan in unreleased source at 1M points. Auto selected scan in all six fixtures. Physical GPU memory peak remains unmeasured.',
    queryTime: 'Query time in one record',
    queryNote: 'Bar lengths compare only within this record. Build time is excluded.',
    queryNoteM1: 'Synchronized query medians on preloaded MPS inputs. Build, transfer, and compilation are excluded. Auto selected scan on M1; a 3 px mark reveals subpixel bars.',
    fixture: 'Fixture',
    provenance: 'Unreleased source measurement',
    prototype: 'Research prototype measurement',
    listUnreleased: 'Unreleased',
    listPrototype: 'Prototype',
    auto: 'Auto selected',
    detailButton: 'Inspect conditions and raw evidence',
    rawLink: 'Raw JSON ↗',
    sourceLink: 'Measurement commit ↗',
    testLogSafe: 'Safe log ↗',
    testLogFast: 'Fast log ↗',
    passMeta: (run) => `${run.skipped} skipped · ${run.xfailed} expected failure`,
    pending: 'pending validation',
    pendingM1: 'M1 8GB spatial search',
    pendingMemory: 'physical GPU memory peak',
    memorySummary: (tensor, driver) => `Separate M1 memory run: ${tensor} MiB tensor allocator peak · ${driver} MiB sampled driver high. Neither is a physical GPU peak.`,
    operation: { knn: 'kNN', radius: 'radius query' },
    distribution: { uniform: 'uniform', coincident: 'coincident', 'cluster-sparse': 'cluster + sparse', collapsed: 'collapsed' },
    conditions: 'Environment',
    statistic: 'Statistic',
    repetitions: 'Repetitions',
    inputResidence: 'Input residence',
    querySource: 'Query source',
    compilation: 'Shader compilation included',
    transfer: 'Host-to-device transfer included',
    parity: 'Mismatched index slots',
    distanceParity: 'Squared-distance bit mismatches',
    orderParity: 'Input-order violations',
    stackFallback: 'BVH stack fallback rows',
    mathMode: 'MPS math mode',
    fallback: 'MPS CPU fallback',
    safeMath: 'Safe math',
    fastMath: 'Fast math',
    build: 'Build time (excluded from chart)',
    caution: 'Interpretation limit',
    yes: 'Yes', no: 'No',
    sourceRecord: 'Raw record',
    recordedOn: 'Recorded hardware',
    unknown: 'Not recorded',
  },
};

function languageOf(value) { return value === 'en' ? 'en' : 'ko'; }
function formatNumber(value, language, digits = 0) {
  return new Intl.NumberFormat(language === 'ko' ? 'ko-KR' : 'en-US', {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  }).format(value);
}
function formatMs(value, language) { return `${formatNumber(value, language, value < 10 ? 3 : 2)} ms`; }
function element(tag, className = '', content) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (content !== undefined) result.textContent = String(content);
  return result;
}
function safeSourcePath(path) { return RAW_PATH.test(path ?? '') && !path.split('/').includes('..'); }
function safeLogPath(path) { return /^docs\/evidence\/[A-Za-z0-9_./-]+\.log$/.test(path ?? '') && !path.split('/').includes('..'); }
function setText(root, selector, value) {
  const target = root.querySelector(selector);
  if (target) target.textContent = String(value);
}

export function measuredComparison(data, requestedId = DEFAULT_RECORD) {
  if (!data || !Array.isArray(data.benchmarks)) throw new TypeError('benchmark records are required');
  const records = data.benchmarks.filter(record => record.status === 'measured'
    && record.conditions && Array.isArray(record.baselines)
    && record.baselines.length && Number.isFinite(record.ours?.queryMs));
  if (!records.length) throw new TypeError('no measured comparison records');
  const record = records.find(candidate => candidate.id === requestedId)
    ?? records.find(candidate => candidate.id === DEFAULT_RECORD) ?? records[0];
  const series = [
    ...record.baselines.map(item => ({ id: item.id, name: item.name, queryMs: item.queryMs, kind: 'baseline' })),
    { id: record.ours.id, name: record.ours.name, queryMs: record.ours.queryMs, kind: 'ours' },
    ...(record.auto ? [{ id: 'auto', name: `Auto · ${record.auto.selectedBackend}`, queryMs: record.auto.queryMs, kind: 'auto' }] : []),
  ];
  if (series.some(item => !Number.isFinite(item.queryMs) || item.queryMs <= 0)) {
    throw new TypeError(`invalid query timing in ${record.id}`);
  }
  return { record, records, series, maxMs: Math.max(...series.map(item => item.queryMs)) };
}

function optionText(record, language) {
  const t = COPY[language];
  const condition = record.conditions;
  const operation = t.operation[record.operation] ?? record.operation;
  const distribution = t.distribution[record.distribution] ?? record.distribution;
  const radius = Number.isFinite(condition.radius) ? ` · r ${formatNumber(condition.radius, language)}` : '';
  return `${operation} · ${distribution}${radius} · N ${formatNumber(condition.n, language)} · Q ${formatNumber(condition.q, language)}`;
}

function detailFact(list, label, value) {
  if (value === null || value === undefined || value === '') return;
  const pair = element('div', 'measurement-detail-fact');
  pair.append(element('dt', '', label), element('dd', '', value));
  list.append(pair);
}

function renderDetail(root, model, language) {
  const { record } = model;
  const t = COPY[language];
  const condition = record.conditions;
  const body = root.querySelector('#measurement-detail-content');
  const list = element('dl', 'measurement-detail-facts');
  detailFact(list, t.fixture, optionText(record, language));
  detailFact(list, t.recordedOn, `${condition.hardware} · ${condition.memoryGiB} GiB`);
  detailFact(list, t.conditions, `macOS ${condition.macos} · PyTorch ${condition.pytorch} · ${condition.dtype}`);
  detailFact(list, t.statistic, condition.statistic);
  detailFact(list, t.repetitions, String(condition.repetitions));
  detailFact(list, t.inputResidence, condition.inputResidence);
  detailFact(list, t.querySource, condition.querySource);
  if (condition.mpsFastMath !== undefined) {
    detailFact(list, t.mathMode, condition.mpsFastMath ? t.fastMath : t.safeMath);
  }
  if (condition.mpsFallback !== undefined) {
    detailFact(list, t.fallback, condition.mpsFallback ? t.yes : t.no);
  }
  detailFact(list, t.compilation, condition.shaderCompilationIncluded ? t.yes : t.no);
  if (condition.hostToDeviceTransferIncluded !== undefined) {
    detailFact(list, t.transfer, condition.hostToDeviceTransferIncluded ? t.yes : t.no);
  }
  detailFact(list, t.parity, String(record.parity?.mismatchedIndexSlots ?? t.unknown));
  if (record.parity?.mismatchedSquaredDistanceBits !== undefined) {
    detailFact(list, t.distanceParity, String(record.parity.mismatchedSquaredDistanceBits));
  }
  if (record.parity?.originalIndexOrderViolations !== undefined) {
    detailFact(list, t.orderParity, String(record.parity.originalIndexOrderViolations));
  }
  if (record.parity?.bvhStackFallbackRows !== undefined) {
    detailFact(list, t.stackFallback, String(record.parity.bvhStackFallbackRows));
  }
  if (record.ours.buildMs !== undefined || record.baselines.some(item => item.buildMs !== undefined)) {
    const builds = [...record.baselines, record.ours].filter(item => item.buildMs !== undefined)
      .map(item => `${item.name} ${formatMs(item.buildMs, language)}`).join(' · ');
    detailFact(list, t.build, builds);
  }
  const links = element('div', 'measurement-detail-links');
  if (safeSourcePath(condition.sourcePath)) {
    const raw = element('a', '', t.rawLink);
    raw.href = `./evidence/${condition.sourcePath}`;
    raw.target = '_blank'; raw.rel = 'noopener noreferrer';
    links.append(raw);
  }
  if (COMMIT.test(condition.sourceCommit ?? '')) {
    const commit = element('a', '', t.sourceLink);
    commit.href = `https://github.com/gamzerA/mps-pointops/commit/${condition.sourceCommit}`;
    commit.target = '_blank'; commit.rel = 'noopener noreferrer';
    links.append(commit);
  }
  const note = element('p', 'measurement-detail-note', record.scope === 'research-prototype'
    ? (language === 'ko' ? '연구 시제품의 기록입니다. 공개 패키지의 성능으로 해석하지 마세요.' : 'This is a research prototype record, not published-package performance.')
    : (language === 'ko' ? '미배포 공개 소스의 기록입니다. BVH는 v0.8.0 wheel에 없습니다.' : 'This records unreleased public source. BVH is absent from the v0.8.0 wheel.'));
  body.replaceChildren(list, links, note);
}

function renderChart(root, model, language, openDetails) {
  const { record, series, maxMs } = model;
  const t = COPY[language];
  const target = root.querySelector('#measurement-dashboard');
  const chart = element('figure', 'measurement-figure');
  const heading = element('figcaption', 'measurement-figure-caption');
  heading.append(element('span', 'measurement-scope', record.scope === 'research-prototype' ? t.prototype : t.provenance),
    element('strong', '', t.queryTime));
  const context = element('p', 'measurement-fixture-context',
    `${record.conditions.hardware} · ${optionText(record, language)} · ${record.conditions.dtype} · ${record.conditions.mpsFastMath ? t.fastMath : t.safeMath}`);
  const bars = element('div', 'measurement-bars');
  bars.setAttribute('role', 'list');
  bars.setAttribute('aria-label', t.queryTime);
  for (const item of series) {
    const row = element('div', `measurement-bar-row measurement-bar-${item.kind}`);
    if (item.queryMs / maxMs < .004) row.classList.add('measurement-bar-subpixel');
    row.setAttribute('role', 'listitem');
    row.setAttribute('aria-label', `${item.name}: ${formatMs(item.queryMs, language)}`);
    const label = element('div', 'measurement-bar-label');
    label.append(element('span', '', item.name), element('strong', '', formatMs(item.queryMs, language)));
    const track = element('div', 'measurement-bar-track');
    track.setAttribute('aria-hidden', 'true');
    const fill = element('span', 'measurement-bar-fill');
    fill.style.setProperty('--bar-scale', String(item.queryMs / maxMs));
    track.append(fill);
    row.append(label, track);
    bars.append(row);
  }
  const note = element('p', 'measurement-chart-note',
    record.conditions.hardware === 'Apple M1' ? t.queryNoteM1 : t.queryNote);
  const footer = element('div', 'measurement-chart-footer');
  const raw = element('a', 'measurement-raw-link', t.rawLink);
  if (safeSourcePath(record.conditions.sourcePath)) {
    raw.href = `./evidence/${record.conditions.sourcePath}`;
    raw.target = '_blank'; raw.rel = 'noopener noreferrer';
    footer.append(raw);
  }
  const details = element('button', 'measurement-detail-button', t.detailButton);
  details.type = 'button';
  details.addEventListener('click', openDetails);
  footer.append(details);
  chart.append(heading, context, bars, note, footer);
  target.replaceChildren(chart);
}

/** Mount the one-screen view from the verified benchmark and test records. */
export function mountMeasurementsDashboard(root, data, { language: initialLanguage = 'ko' } = {}) {
  if (!root?.querySelector?.('#measurement-record-list') || !root.querySelector('#measurement-dashboard')
      || !root.querySelector('#measurement-hardware-list')) {
    throw new TypeError('measurement dashboard markup is required');
  }
  let language = languageOf(initialLanguage);
  let selectedId = DEFAULT_RECORD;
  const recordList = root.querySelector('#measurement-record-list');
  const hardwareList = root.querySelector('#measurement-hardware-list');
  const hardware = [...new Set(data.benchmarks.filter(record => record.status === 'measured')
    .map(record => record.conditions?.hardware).filter(Boolean))];
  let selectedHardware = data.benchmarks.find(record => record.id === selectedId)?.conditions?.hardware
    ?? hardware[0];
  const dialog = root.querySelector('#measurement-details');
  const closeButton = root.querySelector('[data-measure-close]');
  const safe = data.verification?.testRuns?.find(run => run.id === 'safe');
  const fast = data.verification?.testRuns?.find(run => run.id === 'fast');
  if (!safe || !fast || !data.meta || !dialog || !closeButton) {
    throw new TypeError('measurement verification and detail markup are required');
  }

  const openDetails = () => {
    if (typeof dialog.showModal === 'function') dialog.showModal();
    else dialog.setAttribute('open', '');
  };
  const closeDetails = () => {
    if (typeof dialog.close === 'function') dialog.close();
    else dialog.removeAttribute('open');
  };
  closeButton.addEventListener('click', closeDetails);
  dialog.addEventListener('click', event => { if (event.target === dialog) closeDetails(); });

  function renderHardwareList() {
    const buttons = hardware.map(name => {
      const button = element('button', 'measurement-hardware-button', name);
      button.type = 'button';
      button.dataset.hardware = name;
      button.setAttribute('aria-pressed', String(name === selectedHardware));
      button.setAttribute('aria-controls', 'measurement-record-list');
      return button;
    });
    hardwareList.replaceChildren(...buttons);
  }

  function renderRecordList(records) {
    const t = COPY[language];
    if (recordList.dataset.language === language && recordList.dataset.hardware === selectedHardware
        && recordList.children.length === records.length) {
      for (const button of recordList.children) {
        button.setAttribute('aria-pressed', String(button.dataset.recordId === selectedId));
      }
      return;
    }
    const buttons = [];
    const displayOrder = [
      ...records.filter(record => record.id === DEFAULT_RECORD),
      ...records.filter(record => record.id !== DEFAULT_RECORD && record.scope !== 'research-prototype'),
      ...records.filter(record => record.scope === 'research-prototype'),
    ];
    for (const record of displayOrder) {
      const button = element('button', 'measurement-record');
      button.type = 'button';
      button.dataset.recordId = record.id;
      button.setAttribute('aria-controls', 'measurement-dashboard');
      button.setAttribute('aria-pressed', String(record.id === selectedId));
      const operation = t.operation[record.operation] ?? record.operation;
      const distribution = t.distribution[record.distribution] ?? record.distribution;
      const radius = Number.isFinite(record.conditions.radius)
        ? ` · r ${formatNumber(record.conditions.radius, language)}` : '';
      const title = element('strong', 'measurement-record-title', `${operation} · ${distribution}${radius}`);
      const counts = element('span', 'measurement-record-counts',
        `N ${formatNumber(record.conditions.n, language)} · Q ${formatNumber(record.conditions.q, language)}`);
      const scope = element('span', 'measurement-record-scope',
        record.scope === 'research-prototype' ? t.listPrototype : t.listUnreleased);
      button.append(title, counts, scope);
      buttons.push(button);
    }
    recordList.replaceChildren(...buttons);
    recordList.dataset.language = language;
    recordList.dataset.hardware = selectedHardware;
    recordList.classList.toggle('is-six-records', records.length > 4);
  }

  function renderStatus() {
    const t = COPY[language];
    setText(root, '#safe-count', formatNumber(safe.passed, language));
    setText(root, '[data-measure-fast-count]', formatNumber(fast.passed, language));
    setText(root, '[data-measure-safe-meta]', t.passMeta(safe));
    setText(root, '[data-measure-fast-meta]', t.passMeta(fast));
    setText(root, '[data-measure-ci-count]', `${data.verification.ci.passed}/${data.verification.ci.total}`);
    setText(root, '#test-summary', `${formatNumber(safe.passed, language)} Safe · ${formatNumber(fast.passed, language)} Fast`);
    const links = root.querySelector('#test-links');
    links.replaceChildren();
    for (const run of [safe, fast]) {
      if (!safeLogPath(run.logPath)) continue;
      const link = element('a', '', run.id === 'safe' ? t.testLogSafe : t.testLogFast);
      link.href = `./evidence/${run.logPath}`;
      link.target = '_blank'; link.rel = 'noopener noreferrer';
      links.append(link);
    }
    const pending = data.support.filter(item => item.status === 'pending')
      .map(item => item.id === 'm1-8gb-spatial' ? t.pendingM1
        : item.id === 'physical-gpu-peak' ? t.pendingMemory : item.id);
    setText(root, '[data-measure-pending]', `${t.pending}: ${pending.join(' · ')}`);
    setText(root, '[data-measure-version]', `v${data.meta.publishedVersion}`);
    const testSource = root.querySelector('[data-measure-test-source]');
    if (COMMIT.test(safe.sourceCommit ?? '')) {
      testSource.href = `https://github.com/gamzerA/mps-pointops/commit/${safe.sourceCommit}`;
      testSource.textContent = safe.sourceCommit.slice(0, 8);
    }
    root.querySelector('#measurement-validation').setAttribute('aria-label', language === 'ko' ? '검증 기록' : 'Validation records');
  }

  function render() {
    const t = COPY[language];
    for (const node of root.querySelectorAll('[data-measure-copy]')) {
      const copy = t[node.dataset.measureCopy];
      if (typeof copy === 'string') node.textContent = copy;
    }
    const model = measuredComparison(data, selectedId);
    selectedId = model.record.id;
    selectedHardware = model.record.conditions.hardware;
    renderHardwareList();
    renderRecordList(model.records.filter(record => record.conditions.hardware === selectedHardware));
    setText(root, '#measurement-active-record', `${t.selectedRecord}: ${optionText(model.record, language)}`);
    renderStatus();
    renderChart(root, model, language, openDetails);
    renderDetail(root, model, language);
    const source = root.querySelector('[data-measure-source]');
    if (COMMIT.test(model.record.conditions.sourceCommit ?? '')) {
      source.href = `https://github.com/gamzerA/mps-pointops/commit/${model.record.conditions.sourceCommit}`;
      source.textContent = model.record.conditions.sourceCommit.slice(0, 8);
    }
    setText(root, '[data-measure-scope]', model.record.scope === 'research-prototype' ? t.prototype : t.provenance);
    setText(root, '[data-measure-copy="boundary"]', selectedHardware === 'Apple M1' ? t.boundaryM1 : t.boundary);
    const memory = root.querySelector('#measurement-memory');
    const evidence = data.memoryEvidence;
    memory.hidden = selectedHardware !== 'Apple M1' || !evidence || !safeSourcePath(evidence.sourcePath);
    if (!memory.hidden) {
      setText(memory, '[data-measure-memory-summary]',
        t.memorySummary(formatNumber(evidence.allocatorTensorPeakBytes / 1048576, language, 2),
          formatNumber(evidence.sampledDriverHighBytes / 1048576, language, 2)));
      memory.querySelector('[data-measure-memory-link]').href = `./evidence/${evidence.sourcePath}`;
    }
  }

  hardwareList.addEventListener('click', event => {
    const button = event.target.closest('button[data-hardware]');
    if (!button || !hardwareList.contains(button) || button.dataset.hardware === selectedHardware) return;
    selectedHardware = button.dataset.hardware;
    selectedId = data.benchmarks.find(record => record.status === 'measured'
      && record.conditions.hardware === selectedHardware)?.id;
    render();
    Array.from(hardwareList.children).find(item => item.dataset.hardware === selectedHardware)
      ?.focus({ preventScroll: true });
  });

  recordList.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-record-id]');
    if (!button || !recordList.contains(button)) return;
    selectedId = button.dataset.recordId;
    render();
    Array.from(recordList.children).find(item => item.dataset.recordId === selectedId)?.focus({ preventScroll: true });
  });
  render();
  return {
    setLanguage(value) { language = languageOf(value); render(); },
    select(id) { if (!data.benchmarks.some(record => record.id === id && record.status === 'measured')) return false; selectedId = id; render(); return true; },
    getSelectedId() { return selectedId; },
    destroy() { closeDetails(); },
  };
}

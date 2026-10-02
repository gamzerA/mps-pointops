// Every bar in this explorer comes from one recorded fixture. We never join
// timings from different raw files to calculate a cross-device speedup.

const COPY = {
  ko: {
    hardware: '기기', operation: '연산', pointCount: '참조 점 수 N', all: '전체',
    fps: 'FPS', knn: 'kNN', 'ball-query': 'Ball Query / radius',
    resultCount: (count) => `조건에 맞는 실측 기록 ${count}건`,
    empty: '이 조건의 저장된 실측값은 없습니다.',
    emptyDetail: '측정하지 않은 기기·연산·크기 조합에 값을 채워 넣지 않았습니다.',
    clearSize: '모든 크기 보기',
    released: '공개된 코드의 실측 스냅샷',
    unreleased: '미배포 소스 실측',
    comparison: '같은 입력과 측정 범위 안에서 비교',
    method: '측정 조건과 출처',
    fixture: '입력', environment: '환경', timing: '계측',
    parity: '검증', source: '원시 JSON', sourceCode: '기록된 소스',
    noSourceCommit: '원시 기록에 소스 커밋 없음 · SHA-256 파일 해시 기록',
    measured: 'ms · 각 기록 안에서만 막대 길이 비교',
    note: '해석 범위', repetitions: '반복', warmups: '워밍업',
    math: 'Math', fallback: 'MPS fallback',
    versionNote: '현재 PyPI는 0.8.0입니다. BVH 경로는 미배포 소스 측정입니다.',
    indexMismatch: '인덱스 불일치', exact: 'CPU/MPS 결과 일치',
    indexedSource: '원시 JSON에는 각 실행의 세부 샘플과 소스 기록이 있습니다.',
  },
  en: {
    hardware: 'Hardware', operation: 'Operation', pointCount: 'Reference points N', all: 'All',
    fps: 'FPS', knn: 'kNN', 'ball-query': 'Ball Query / radius',
    resultCount: (count) => `${count} measured record${count === 1 ? '' : 's'} for this selection`,
    empty: 'No saved measurement for this combination.',
    emptyDetail: 'Unmeasured hardware, operation, and size combinations are left blank.',
    clearSize: 'Show all sizes',
    released: 'Measurement of released code',
    unreleased: 'Unreleased source measurement',
    comparison: 'Compared only within the same fixture and timing scope',
    method: 'Method and provenance',
    fixture: 'Fixture', environment: 'Environment', timing: 'Timing',
    parity: 'Parity', source: 'Raw JSON', sourceCode: 'Recorded source',
    noSourceCommit: 'No source commit in raw record · SHA-256 file hashes recorded',
    measured: 'ms · bar scales reset for each record',
    note: 'Interpretation limit', repetitions: 'repeats', warmups: 'warmups',
    math: 'Math', fallback: 'MPS fallback',
    versionNote: 'The current PyPI package is 0.8.0. The BVH path is measured from unreleased source.',
    indexMismatch: 'index mismatches', exact: 'CPU/MPS outputs equal',
    indexedSource: 'The raw JSON contains sample times and source records.',
  },
};

const OPERATIONS = ['fps', 'knn', 'ball-query'];

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}

function addTerm(list, label, value) {
  if (value === null || value === undefined || value === '') return;
  const group = node('div', 'be-fact');
  group.append(node('dt', 'be-fact-label', label), node('dd', 'be-fact-value', value));
  list.append(group);
}

function fmt(value, language, digits = 0) {
  return new Intl.NumberFormat(language === 'ko' ? 'ko-KR' : 'en-US', {
    minimumFractionDigits: digits, maximumFractionDigits: digits,
  }).format(value);
}

function fmtMs(value, language) {
  return `${fmt(value, language, value < 10 ? 3 : 2)} ms`;
}

function safeRawPath(path) {
  return typeof path === 'string'
    && /^bench\/results\/[A-Za-z0-9_./-]+\.json$/.test(path)
    && !path.split('/').includes('..');
}

function fixtureSummary(record, language) {
  const f = record.fixture;
  const pieces = [record.api, record.distribution, f.dtype, `B=${f.batch}`, `N=${fmt(record.n, language)}`];
  if (record.q !== null) pieces.push(`Q=${fmt(record.q, language)}`);
  if (f.sampleCount !== null && f.sampleCount !== undefined) pieces.push(`S=${fmt(f.sampleCount, language)}`);
  if (f.sampleRatio !== null && f.sampleRatio !== undefined) pieces.push(`ratio=${f.sampleRatio}`);
  if (f.k !== null && f.k !== undefined) pieces.push(`K=${f.k}`);
  if (f.radius !== null && f.radius !== undefined) pieces.push(`r=${f.radius}`);
  return pieces.join(' · ');
}

function paritySummary(record, language) {
  const t = COPY[language];
  const p = record.parity;
  if (p.exactCpuMps !== undefined) return `${t.exact}: ${p.exactCpuMps ? 'yes' : 'no'}`;
  if (p.indexMismatches !== undefined) return `${t.indexMismatch}: ${p.indexMismatches}`;
  if (p.mismatchedIndexSlots !== undefined) return `${t.indexMismatch}: ${p.mismatchedIndexSlots}`;
  return p.detail ?? '—';
}

function makeSelect(filterName, labelText, values, current, formatValue, onChange) {
  const label = node('label', 'be-filter');
  const caption = node('span', 'be-filter-label', labelText);
  const select = node('select', 'be-select');
  select.dataset.filter = filterName;
  for (const value of values) {
    const option = node('option', '', formatValue(value));
    option.value = String(value);
    select.append(option);
  }
  select.value = String(current);
  select.addEventListener('change', () => onChange(select.value));
  label.append(caption, select);
  return label;
}

function renderRecord(record, language) {
  const t = COPY[language];
  const card = node('article', 'be-record');
  const heading = node('div', 'be-record-heading');
  const headingText = node('div', 'be-record-heading-text');
  headingText.append(
    node('span', 'be-record-kicker', `${record.hardware} / ${record.api}`),
    node('h3', 'be-record-title', `${t[record.operation]} · ${fmt(record.n, language)} points`),
  );
  const badge = node('span', `be-status be-status-${record.availability}`,
    record.availability === 'unreleased-source' ? t.unreleased : t.released);
  heading.append(headingText, badge);
  card.append(heading);

  const fixture = node('p', 'be-fixture', fixtureSummary(record, language));
  card.append(fixture);

  const chart = node('div', 'be-chart');
  chart.setAttribute('role', 'list');
  chart.setAttribute('aria-label', t.comparison);
  const max = Math.max(...record.series.map(series => series.milliseconds));
  for (const series of record.series) {
    const row = node('div', `be-row be-row-${series.kind}`);
    row.setAttribute('role', 'listitem');
    row.setAttribute('aria-label', `${series.label}: ${fmtMs(series.milliseconds, language)}`);
    const line = node('div', 'be-row-line');
    line.append(node('span', 'be-row-name', series.label),
      node('strong', 'be-row-number', fmtMs(series.milliseconds, language)));
    const track = node('div', 'be-track');
    const fill = node('span', 'be-fill');
    fill.style.width = `${Math.max(1.5, series.milliseconds / max * 100)}%`;
    track.append(fill);
    row.append(line, track);
    chart.append(row);
  }
  card.append(chart, node('p', 'be-chart-caption', t.measured));

  const details = node('details', 'be-details');
  details.append(node('summary', 'be-details-summary', t.method));
  const facts = node('dl', 'be-facts');
  addTerm(facts, t.fixture, fixtureSummary(record, language));
  addTerm(facts, t.environment,
    `${record.environment.macos} · PyTorch ${record.environment.pytorch} · ${record.environment.memoryGiB} GiB`);
  addTerm(facts, t.math, record.environment.math);
  addTerm(facts, t.fallback, record.environment.fallback);
  addTerm(facts, t.timing, record.method.timing);
  addTerm(facts, t.repetitions, record.method.repetitions);
  addTerm(facts, t.warmups, record.method.warmups);
  addTerm(facts, t.comparison, record.method.comparisonScope);
  addTerm(facts, t.parity, paritySummary(record, language));
  addTerm(facts, t.note, record.caution);
  details.append(facts);

  const sources = node('div', 'be-sources');
  if (safeRawPath(record.source.rawPath)) {
    const raw = node('a', 'be-source-link', t.source);
    raw.href = `./evidence/${record.source.rawPath}`;
    raw.target = '_blank'; raw.rel = 'noopener';
    sources.append(raw);
  }
  const commit = record.source.sourceCommit ?? record.source.archiveCommit;
  if (/^[a-f0-9]{40}$/.test(commit)) {
    const link = node('a', 'be-source-link', `${t.sourceCode} ${commit.slice(0, 8)}`);
    link.href = `https://github.com/gamzerA/mps-pointops/tree/${commit}`;
    link.target = '_blank'; link.rel = 'noopener';
    sources.append(link);
  }
  if (!record.source.sourceCommit) sources.append(node('span', 'be-source-note', t.noSourceCommit));
  details.append(sources, node('p', 'be-source-note', t.indexedSource));
  card.append(details);
  return card;
}

export function filterBenchmarkRecords(data, filters = {}) {
  if (!data || !Array.isArray(data.records)) throw new TypeError('benchmark data must contain records');
  const { hardware = 'all', operation = 'all', n = 'all' } = filters;
  return data.records.filter(record =>
    (hardware === 'all' || record.hardware === hardware)
    && (operation === 'all' || record.operation === operation)
    && (n === 'all' || record.n === Number(n)));
}

export function mountBenchmarkExplorer(root, data, options = {}) {
  if (!(root instanceof Element)) throw new TypeError('root must be an Element');
  if (!data || data.schemaVersion !== 1 || !Array.isArray(data.records)) {
    throw new TypeError('unsupported benchmark data');
  }
  let language = options.language === 'en' ? 'en' : 'ko';
  const state = {
    hardware: options.hardware ?? 'Apple M5 Pro',
    operation: options.operation ?? 'all',
    n: options.n ?? '100000',
  };
  const hardwareValues = ['all', ...new Set(data.records.map(record => record.hardware))];
  const nValues = ['all', ...new Set(data.records.map(record => record.n))].sort((a, b) =>
    a === 'all' ? -1 : b === 'all' ? 1 : a - b);
  const operationValues = ['all', ...OPERATIONS];

  function render() {
    const t = COPY[language];
    const container = node('div', 'be-explorer');
    const filters = node('div', 'be-filters');
    filters.append(
      makeSelect('hardware', t.hardware, hardwareValues, state.hardware,
        value => value === 'all' ? t.all : value,
        value => { state.hardware = value; render(); root.querySelector('[data-filter="hardware"]')?.focus(); }),
      makeSelect('operation', t.operation, operationValues, state.operation,
        value => value === 'all' ? t.all : t[value],
        value => { state.operation = value; render(); root.querySelector('[data-filter="operation"]')?.focus(); }),
      makeSelect('n', t.pointCount, nValues, state.n,
        value => value === 'all' ? t.all : fmt(Number(value), language),
        value => { state.n = value; render(); root.querySelector('[data-filter="n"]')?.focus(); }),
    );
    const found = filterBenchmarkRecords(data, state);
    const count = node('p', 'be-count', t.resultCount(found.length));
    count.setAttribute('role', 'status');
    const records = node('div', 'be-records');
    if (found.length === 0) {
      const empty = node('div', 'be-empty');
      empty.append(node('strong', '', t.empty), node('p', '', t.emptyDetail));
      if (state.n !== 'all') {
        const clear = node('button', 'be-clear', t.clearSize);
        clear.type = 'button';
        clear.addEventListener('click', () => { state.n = 'all'; render(); });
        empty.append(clear);
      }
      records.append(empty);
    } else {
      records.append(...found.map(record => renderRecord(record, language)));
    }
    container.append(filters, count, records, node('p', 'be-version-note', t.versionNote));
    root.replaceChildren(container);
  }

  render();
  return {
    setLanguage(value) { language = value === 'en' ? 'en' : 'ko'; render(); },
    setSelection(filters) {
      if (filters.hardware !== undefined) state.hardware = filters.hardware;
      if (filters.operation !== undefined) state.operation = filters.operation;
      if (filters.n !== undefined) state.n = String(filters.n);
      render();
    },
    getSelection() { return { ...state }; },
    destroy() { root.replaceChildren(); },
  };
}

// An evidence-bound view of tested ecosystem surfaces. Release availability and
// fixture validation are separate dimensions; neither means universal support.
const COPY = {
  ko: {
    group: '프레임워크 또는 모델 선택',
    source: '기준: PyPI {version} / 병합 소스 {commit}',
    release: '패키지 상태',
    validation: '검증 범위',
    functions: '관련 연산',
    fixture: '실제로 확인한 사례',
    forward: '순방향',
    backward: '역전파',
    hardware: '검증 기기',
    hardwareNone: '실기기 검증 기록 없음',
    limitations: '남은 범위',
    evidence: '근거 열기',
    released: '배포된 부분집합',
    experimental: '실험적 부분집합',
    planned: '계획 단계',
    'model-fixture': '모델 사례 통과',
    'operator-fixture': '연산 사례 통과',
    unverified: '검증 기록 없음',
    validated: '검증됨',
    partial: '일부 검증',
    selected: '{name} 선택됨',
  },
  en: {
    group: 'Choose a framework or model',
    source: 'Scope: PyPI {version} / merged source {commit}',
    release: 'Package stage',
    validation: 'Validation scope',
    functions: 'Relevant operations',
    fixture: 'Tested fixture',
    forward: 'Forward',
    backward: 'Backward',
    hardware: 'Tested hardware',
    hardwareNone: 'No physical hardware validation recorded',
    limitations: 'Open limits',
    evidence: 'Open evidence',
    released: 'Shipped subset',
    experimental: 'Experimental subset',
    planned: 'Planned',
    'model-fixture': 'Model fixture passed',
    'operator-fixture': 'Operator fixtures passed',
    unverified: 'No validation record',
    validated: 'Validated',
    partial: 'Partly validated',
    selected: '{name} selected',
  },
};

const RELEASE_STATES = new Set(['released', 'experimental', 'planned']);
const VALIDATION_STATES = new Set(['model-fixture', 'operator-fixture', 'unverified']);
const DIRECTION_STATES = new Set(['validated', 'partial', 'unverified']);
const EVIDENCE_PATH = /^(?:README\.md|docs\/[A-Za-z0-9_./-]+\.md)$/;

function localized(value, language) {
  return value[language] ?? value.ko;
}

function node(tag, className, content) {
  const result = document.createElement(tag);
  if (className) result.className = className;
  if (content !== undefined) result.textContent = String(content);
  return result;
}

function fact(label, value, status) {
  const row = node('div', 'capability-fact');
  const name = node('dt', 'capability-fact-label', label);
  const result = node('dd', 'capability-fact-value', value);
  if (status) result.dataset.status = status;
  row.append(name, result);
  return row;
}

function evidenceUrl(data, path) {
  if (!EVIDENCE_PATH.test(path) || path.split('/').includes('..')) {
    throw new RangeError(`Invalid capability evidence path: ${path}`);
  }
  return `https://github.com/${data.source.repository}/blob/${data.source.mainCommit}/${path}`;
}

/** Validate the claims schema before putting any content into the page. */
export function validateCapabilities(data) {
  if (data?.schemaVersion !== 1 || data.source?.repository !== 'gamzerA/mps-pointops'
      || !/^[a-f0-9]{40}$/.test(data.source?.mainCommit ?? '')
      || !/^\d+\.\d+\.\d+$/.test(data.source?.publishedVersion ?? '')
      || !Array.isArray(data.items) || data.items.length === 0) {
    throw new TypeError('Invalid capability data or provenance');
  }
  const ids = new Set();
  for (const item of data.items) {
    if (!/^[a-z0-9-]+$/.test(item.id ?? '') || ids.has(item.id)
        || typeof item.name !== 'string' || !item.name
        || !RELEASE_STATES.has(item.releaseStatus)
        || !VALIDATION_STATES.has(item.validationStatus)
        || !Array.isArray(item.functions) || !Array.isArray(item.hardware)
        || !Array.isArray(item.evidence) || !item.evidence.length) {
      throw new TypeError(`Invalid capability: ${item?.id ?? 'unknown'}`);
    }
    ids.add(item.id);
    for (const key of ['releaseScope', 'fixture', 'limitations']) {
      if (!item[key] || !['ko', 'en'].every(language => typeof item[key][language] === 'string'
          && item[key][language].trim())) {
        throw new TypeError(`Missing ${key} translation: ${item.id}`);
      }
    }
    for (const key of ['forward', 'backward']) {
      if (!DIRECTION_STATES.has(item[key]?.status)
          || !['ko', 'en'].every(language => typeof item[key][language] === 'string'
            && item[key][language].trim())) {
        throw new TypeError(`Invalid ${key} contract: ${item.id}`);
      }
    }
    if (item.releaseStatus === 'planned' && item.validationStatus !== 'unverified') {
      throw new TypeError(`Planned surface cannot claim a passing fixture: ${item.id}`);
    }
    for (const record of item.evidence) {
      if (!record.label || !['ko', 'en'].every(language => record.label[language]?.trim())
          || !EVIDENCE_PATH.test(record.path ?? '') || record.path.split('/').includes('..')) {
        throw new TypeError(`Invalid evidence for ${item.id}`);
      }
    }
  }
  return data;
}

/**
 * Mount a keyboard-accessible list/detail explorer. The root can be any empty
 * container. `setLanguage('ko'|'en')` updates the entire component in place.
 */
export function mountCapabilityMap(root, rawData, { language = 'ko' } = {}) {
  if (!root || typeof root.replaceChildren !== 'function') {
    throw new TypeError('Capability map requires a container element');
  }
  const data = validateCapabilities(rawData);
  let currentLanguage = language === 'en' ? 'en' : 'ko';
  let selectedId = data.items[0].id;
  const list = node('div', 'capability-list');
  list.setAttribute('role', 'group');
  const detail = node('article', 'capability-detail');
  const announcement = node('p', 'sr-only capability-announcement');
  announcement.setAttribute('role', 'status');
  announcement.setAttribute('aria-live', 'polite');
  const source = node('p', 'capability-source');
  const buttons = new Map();

  root.classList.add('capability-map');
  root.replaceChildren(source, list, detail, announcement);

  function renderDetail() {
    const item = data.items.find(candidate => candidate.id === selectedId);
    const c = COPY[currentLanguage];
    detail.dataset.releaseStatus = item.releaseStatus;
    detail.dataset.validationStatus = item.validationStatus;
    const header = node('div', 'capability-detail-header');
    const title = node('h3', 'capability-title', item.name);
    title.id = `capability-title-${item.id}`;
    detail.setAttribute('aria-labelledby', title.id);
    const status = node('div', 'capability-status');
    status.append(
      fact(c.release, c[item.releaseStatus], item.releaseStatus),
      fact(c.validation, c[item.validationStatus], item.validationStatus),
    );
    header.append(title, status);

    const intro = node('p', 'capability-release-scope', localized(item.releaseScope, currentLanguage));
    const functions = node('div', 'capability-functions');
    functions.append(node('span', 'capability-label', c.functions));
    const functionList = node('ul', 'capability-function-list');
    for (const name of item.functions) functionList.append(node('li', '', name));
    functions.append(functionList);

    const fixture = node('section', 'capability-fixture');
    fixture.append(node('h4', 'capability-label', c.fixture),
      node('p', '', localized(item.fixture, currentLanguage)));

    const facts = node('dl', 'capability-facts');
    facts.append(
      fact(c.forward, `${c[item.forward.status]} · ${localized(item.forward, currentLanguage)}`, item.forward.status),
      fact(c.backward, `${c[item.backward.status]} · ${localized(item.backward, currentLanguage)}`, item.backward.status),
      fact(c.hardware, item.hardware.length
        ? item.hardware.map(device => typeof device === 'string'
          ? device : localized(device, currentLanguage)).join(' · ')
        : c.hardwareNone),
    );

    const limitations = node('section', 'capability-limitations');
    limitations.append(node('h4', 'capability-label', c.limitations),
      node('p', '', localized(item.limitations, currentLanguage)));

    const links = node('nav', 'capability-evidence');
    links.setAttribute('aria-label', c.evidence);
    for (const record of item.evidence) {
      const anchor = node('a', 'capability-evidence-link', localized(record.label, currentLanguage));
      anchor.href = evidenceUrl(data, record.path);
      anchor.target = '_blank';
      anchor.rel = 'noopener noreferrer';
      links.append(anchor);
    }
    detail.replaceChildren(header, intro, functions, fixture, facts, limitations, links);
    for (const [id, button] of buttons) {
      const active = id === selectedId;
      button.setAttribute('aria-pressed', String(active));
      button.classList.toggle('is-active', active);
    }
  }

  for (const item of data.items) {
    const button = node('button', 'capability-option');
    button.type = 'button';
    button.dataset.capability = item.id;
    button.dataset.releaseStatus = item.releaseStatus;
    button.dataset.validationStatus = item.validationStatus;
    button.addEventListener('click', () => {
      selectedId = item.id;
      renderDetail();
      announcement.textContent = COPY[currentLanguage].selected.replace('{name}', item.name);
    });
    buttons.set(item.id, button);
    list.append(button);
  }

  function setLanguage(nextLanguage) {
    currentLanguage = nextLanguage === 'en' ? 'en' : 'ko';
    const c = COPY[currentLanguage];
    list.setAttribute('aria-label', c.group);
    source.textContent = c.source
      .replace('{version}', data.source.publishedVersion)
      .replace('{commit}', data.source.mainCommit.slice(0, 8));
    for (const item of data.items) {
      const button = buttons.get(item.id);
      button.replaceChildren(
        node('span', 'capability-option-name', item.name),
        node('span', 'capability-option-status', c[item.releaseStatus]),
        node('span', 'capability-option-validation', c[item.validationStatus]),
      );
    }
    renderDetail();
  }

  setLanguage(currentLanguage);
  return {
    setLanguage,
    select(id) {
      if (!buttons.has(id)) return false;
      selectedId = id;
      renderDetail();
      return true;
    },
    getSelectedId() { return selectedId; },
  };
}

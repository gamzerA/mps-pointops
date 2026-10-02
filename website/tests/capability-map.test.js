import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { mountCapabilityMap, validateCapabilities } from '../src/capability-map.js';

const data = JSON.parse(readFileSync(new URL('../data/capabilities.json', import.meta.url), 'utf8'));
const benchmarks = JSON.parse(readFileSync(new URL('../data/benchmarks.json', import.meta.url), 'utf8'));

class FakeElement {
  constructor(tagName) {
    this.tagName = tagName;
    this.children = [];
    this.attributes = {};
    this.dataset = {};
    this.listeners = {};
    this.ownText = '';
    this.classes = new Set();
    this.classList = {
      add: name => this.classes.add(name),
      toggle: (name, force) => {
        if (force) this.classes.add(name);
        else this.classes.delete(name);
      },
    };
  }
  set className(value) { this.classes = new Set(value.split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classes].join(' '); }
  set textContent(value) { this.ownText = String(value); this.children = []; }
  get textContent() { return this.ownText + this.children.map(child => child.textContent).join(''); }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  getAttribute(key) { return this.attributes[key]; }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.ownText = ''; this.children = [...children]; }
  addEventListener(name, listener) { this.listeners[name] = listener; }
  click() { this.listeners.click?.(); }
}

function descendants(root) {
  return [root, ...root.children.flatMap(descendants)];
}

test('capability data pins the merged source separately from the published package', () => {
  assert.equal(validateCapabilities(data), data);
  assert.equal(data.source.mainCommit, benchmarks.meta.mainMergeCommit);
  assert.equal(data.source.publishedVersion, benchmarks.meta.publishedVersion);
  assert.equal(data.items.length, 6);
  assert.deepEqual(data.items.map(item => item.id),
    ['pointnet2', 'pytorch3d', 'pyg', 'dgcnn', 'pointcept', 'sparse-3d']);
  assert.equal(data.items.find(item => item.id === 'pointcept').releaseStatus, 'experimental');
  assert.equal(data.items.find(item => item.id === 'sparse-3d').validationStatus, 'model-fixture');
  for (const item of data.items) {
    assert.ok(item.releaseScope.ko && item.releaseScope.en);
    assert.ok(item.fixture.ko && item.fixture.en);
    assert.ok(item.limitations.ko && item.limitations.en);
    assert.ok(item.evidence.every(record => record.path.startsWith('docs/')));
  }
});

test('invalid status, missing translation, and path traversal are rejected', () => {
  const badStatus = structuredClone(data);
  badStatus.items[0].validationStatus = 'universally-supported';
  assert.throws(() => validateCapabilities(badStatus), /Invalid capability/);
  const missingEnglish = structuredClone(data);
  missingEnglish.items[0].limitations.en = '';
  assert.throws(() => validateCapabilities(missingEnglish), /translation/);
  const traversal = structuredClone(data);
  traversal.items[0].evidence[0].path = 'docs/../secrets.md';
  assert.throws(() => validateCapabilities(traversal), /Invalid evidence/);
});

test('selection and KO/EN switch update details without conflating release and validation', () => {
  const previousDocument = globalThis.document;
  globalThis.document = { createElement: tag => new FakeElement(tag) };
  try {
    const root = new FakeElement('div');
    const map = mountCapabilityMap(root, data);
    assert.equal(map.getSelectedId(), 'pointnet2');
    assert.ok(root.children[0].textContent.includes(`PyPI ${data.source.publishedVersion} / 병합 소스 ${data.source.mainCommit.slice(0, 8)}`));
    const buttons = descendants(root).filter(element => element.className.includes('capability-option ')
      || element.className === 'capability-option');
    assert.equal(buttons.length, 6);
    const sparseButton = buttons.find(button => button.dataset.capability === 'sparse-3d');
    sparseButton.click();
    assert.equal(map.getSelectedId(), 'sparse-3d');
    assert.equal(sparseButton.getAttribute('aria-pressed'), 'true');
    assert.match(root.children[2].textContent, /실험적 부분집합/);
    assert.match(root.children[2].textContent, /모델 사례 통과/);
    assert.match(root.children[2].textContent, /비공개/);
    map.setLanguage('en');
    assert.match(root.children[2].textContent, /Experimental subset/);
    assert.match(root.children[2].textContent, /Model fixture passed/);
    assert.ok(root.children[0].textContent.includes(`merged source ${data.source.mainCommit.slice(0, 8)}`));
    assert.equal(map.select('missing'), false);
    assert.equal(map.select('pyg'), true);
    assert.match(root.children[2].textContent, /bounded operator suite/);
    map.setLanguage('ko');
    assert.match(root.children[2].textContent, /한정된 연산자 스위트/);
    const evidence = descendants(root).find(element => element.tagName === 'a');
    assert.ok(evidence.href.startsWith(`https://github.com/gamzerA/mps-pointops/blob/${data.source.mainCommit}/docs/`));
    assert.equal(evidence.rel, 'noopener noreferrer');
  } finally {
    if (previousDocument === undefined) delete globalThis.document;
    else globalThis.document = previousDocument;
  }
});

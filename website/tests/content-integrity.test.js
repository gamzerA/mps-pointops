import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, readFileSync } from 'node:fs';

const read = (path) => readFileSync(new URL(path, import.meta.url), 'utf8');
const english = JSON.parse(read('../data/i18n/en.json'));
const html = read('../index.html');
const app = read('../src/app.js');
const formula = read('../src/formula-animation.js');
const data = JSON.parse(read('../data/benchmarks.json'));

test('the English dictionary has nonempty values', () => {
  assert.ok(Object.keys(english).length > 0);
  for (const key of Object.keys(english)) {
    assert.equal(typeof english[key], 'string', `EN ${key}`);
    assert.ok(english[key].trim(), `EN ${key} is empty`);
  }
});

test('the page exposes English only', () => {
  assert.match(html, /<html\s+lang="en"/);
  assert.doesNotMatch(html, /class="language-switch"|data-lang="ko"/);
  assert.doesNotMatch(html, /[가-힣]/, 'static HTML includes Korean fallback copy');
});

test('every HTML translation reference exists in the English dictionary', () => {
  const references = [...html.matchAll(/\bdata-i18n(?:-aria)?="([^"]+)"/g)]
    .map((match) => match[1]);
  assert.ok(references.length > 0, 'no data-i18n references found');
  for (const key of references) {
    assert.ok(Object.hasOwn(english, key), `missing EN key ${key}`);
  }
});

test('literal application and formula translation references exist in the English dictionary', () => {
  const references = [...app.matchAll(/\bt\(\s*['"]([^'"]+)['"]/g)]
    .map((match) => match[1]);
  references.push(...[...formula.matchAll(/\bcopy\(\s*['"]([^'"]+)['"]/g)]
    .map((match) => match[1]));
  references.push(...[...formula.matchAll(/\bstate\.translate\(\s*['"]([^'"]+)['"]/g)]
    .map((match) => match[1]));
  assert.ok(references.length > 0, 'no application translation references found');
  for (const key of references) {
    assert.ok(Object.hasOwn(english, key), `missing EN key ${key}`);
  }
});

test('runtime markup and styles reference only local assets', () => {
  const links = [...html.matchAll(/\bsrc="([^"]+)"/g)]
    .map((match) => match[1]);
  links.push(...[...html.matchAll(/<link\b[^>]*\bhref="([^"]+)"/g)]
    .map((match) => match[1]));
  assert.ok(links.length > 0);
  for (const link of links) {
    assert.ok(link.startsWith('./') || link.startsWith('#'),
      `nonlocal runtime asset ${link}`);
  }
  // Outbound source and DOI anchors are citations, not loaded runtime assets.
  for (const [, destination] of html.matchAll(/<a\b[^>]*\bhref="(https?:\/\/[^\"]+)"/g)) {
    assert.match(destination, /^https:\/\/(?:github\.com\/gamzerA\/mps-pointops|doi\.org\/10\.5281\/zenodo\.)/);
  }
  for (const path of ['../src/styles.css', '../src/formula-animation.css',
    '../src/noscript.css']) {
    const css = read(path);
    assert.doesNotMatch(css, /@import|url\(\s*['"]?https?:/i,
      `${path} imports an external asset`);
  }
  assert.doesNotMatch(app + formula, /\b(?:fetch|import)\(\s*['"]https?:/i);
  assert.match(html, /connect-src 'self'/);
});

test('every displayed raw evidence link resolves to a bundled local file', () => {
  const paths = [
    ...data.benchmarks.map(record => record.conditions.sourcePath),
    ...data.verification.testRuns.map(run => run.logPath),
    data.memoryEvidence.sourcePath,
  ];
  for (const path of paths) {
    assert.ok(!path.split('/').includes('..'));
    assert.ok(existsSync(new URL(`../evidence/${path}`, import.meta.url)),
      `missing local evidence file: ${path}`);
  }
});

test('the recorded CI snapshot has a reviewable PR checks destination', () => {
  const ci = data.verification.ci;
  assert.match(ci.headCommit, /^[0-9a-f]{40}$/);
  assert.equal(ci.sourceUrl,
    `https://github.com/gamzerA/mps-pointops/pull/${ci.pullRequest}/checks`);
  assert.match(ci.recordedAtUtc, /^\d{4}-\d\d-\d\dT/);
});

test('benchmark data follows a bounded measured/pending/unsupported schema', () => {
  assert.equal(data.schemaVersion, 1);
  assert.match(data.meta.publishedVersion, /^\d+\.\d+\.\d+$/);
  assert.match(data.meta.mainMergeCommit, /^[0-9a-f]{40}$/);
  assert.match(data.meta.developmentSourceCommit, /^[0-9a-f]{40}$/);
  assert.equal(data.meta.minimumMacOS.status, 'pending');
  assert.equal(data.demo.mortonBitsPerAxis, 21);
  assert.equal(data.demo.sourceBvhBrickSize, 128);

  const ids = new Set();
  for (const record of data.benchmarks) {
    assert.ok(!ids.has(record.id), `duplicate benchmark id ${record.id}`);
    ids.add(record.id);
    assert.equal(record.status, 'measured', `${record.id} needs recorded samples`);
    assert.ok(['research-prototype', 'unreleased-public-source'].includes(record.scope));
    assert.ok(Array.isArray(record.baselines) && record.baselines.length > 0,
      `${record.id} must allow multiple baseline entries`);
    assert.match(record.conditions.sourceCommit, /^[0-9a-f]{40}$/);
    assert.match(record.conditions.measurementDateUtc,
      /^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d/);
    assert.match(record.conditions.sourcePath, /^bench\/results\/[\w./-]+\.json$/);
    assert.ok(!record.conditions.sourcePath.split('/').includes('..'));
    for (const field of ['hardware', 'dtype', 'statistic', 'benchmarkScript']) {
      assert.ok(record.conditions[field], `${record.id} lacks ${field}`);
    }
    for (const field of ['n', 'q', 'k', 'repetitions']) {
      assert.ok(Number.isInteger(record.conditions[field]) && record.conditions[field] > 0,
        `${record.id} has invalid ${field}`);
    }
    assert.equal(record.conditions.mpsFallback, false);
    assert.equal(record.conditions.mpsFastMath, false);
    assert.equal(record.conditions.shaderCompilationIncluded, false);
    for (const path of [...record.baselines, record.ours]) {
      assert.ok(path.id && path.name, `${record.id} path needs id and name`);
      assert.ok(Number.isFinite(path.queryMs) && path.queryMs >= 0,
        `${record.id} has invalid query time`);
      if (record.scope === 'research-prototype') {
        assert.ok(Number.isFinite(path.buildMs) && path.buildMs >= 0,
          `${record.id} has invalid build time`);
      }
    }
    if (record.auto) {
      assert.ok(['scan', 'bvh'].includes(record.auto.selectedBackend));
      assert.ok(Number.isFinite(record.auto.queryMs) && record.auto.queryMs >= 0);
    }
  }

  assert.ok(data.benchmarks.some((record) => record.scope === 'research-prototype'));
  assert.ok(data.benchmarks.some((record) => record.scope === 'unreleased-public-source'));
  for (const entry of data.support) {
    assert.ok(['measured', 'pending', 'unsupported'].includes(entry.status));
  }
  assert.equal(data.support.find((entry) => entry.id === 'm1-8gb-spatial').status,
    'measured');
  assert.equal(data.support.find((entry) => entry.id === 'physical-gpu-peak').status,
    'pending');
  assert.equal(data.memoryEvidence.physicalGpuPeakStatus, 'pending');
  assert.equal(data.memoryEvidence.hardware, 'Apple M1');
  assert.equal(data.memoryEvidence.sourceCommit,
    'ef07a9337b40065c18a6c950bda4240c789f31f1');
  assert.equal(data.support.find((entry) => entry.id === 'spatial-index-pypi-080').status,
    'unsupported');
});

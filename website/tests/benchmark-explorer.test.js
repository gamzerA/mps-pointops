import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { filterBenchmarkRecords } from '../src/benchmark-explorer.js';

const base = new URL('../', import.meta.url);
const data = JSON.parse(readFileSync(new URL('data/benchmark-explorer.json', base), 'utf8'));
const rawCache = new Map();

function raw(path) {
  assert.match(path, /^bench\/results\/[A-Za-z0-9_./-]+\.json$/);
  assert.ok(!path.split('/').includes('..'));
  if (!rawCache.has(path)) {
    rawCache.set(path, JSON.parse(readFileSync(new URL(`evidence/${path}`, base), 'utf8')));
  }
  return rawCache.get(path);
}

function series(record, id) {
  const result = record.series.find(item => item.id === id);
  assert.ok(result, `${record.id}: missing ${id}`);
  return result.milliseconds;
}

test('every displayed card is measured and links a bundled raw file', () => {
  assert.equal(data.schemaVersion, 1);
  assert.equal(data.publishedPackageVersion, '1.0.0');
  assert.equal(new Set(data.records.map(record => record.id)).size, data.records.length);
  assert.ok(data.records.some(record => record.hardware === 'Apple M1'));
  assert.ok(data.records.every(record => record.availability === 'released-code-snapshot'));
  assert.ok(data.records.some(record => record.availability === 'released-code-snapshot'));
  for (const record of data.records) {
    assert.ok(['fps', 'knn', 'ball-query'].includes(record.operation));
    assert.ok(Number.isInteger(record.n) && record.n > 0);
    assert.ok(record.method.repetitions > 0);
    assert.ok(record.method.timing);
    assert.ok(record.method.comparisonScope);
    assert.ok(record.caution);
    assert.ok(record.series.length >= 2);
    assert.ok(record.series.every(item => Number.isFinite(item.milliseconds)
      && item.milliseconds >= 0));
    assert.ok(record.series.every(item => item.device === 'CPU' || item.device === 'MPS'));
    assert.ok(record.series.some(item => item.kind === 'ours'));
    assert.match(record.source.archiveCommit, /^[a-f0-9]{40}$/);
    if (record.source.sourceCommit !== null) {
      assert.match(record.source.sourceCommit, /^[a-f0-9]{40}$/);
    }
    raw(record.source.rawPath);
  }
});

test('20k and 100k dense source-snapshot values map to the same raw fixture', () => {
  for (const record of data.records.filter(item => item.id.startsWith('m5-dense-'))) {
    const source = raw(record.source.rawPath);
    assert.equal(source.env.chip, record.hardware);
    assert.equal(source.args.repeat, String(record.method.repetitions));
    const rawOp = record.operation === 'ball-query' ? 'ball_query' : record.operation;
    const rows = source.rows.filter(row => row.op === rawOp && row.n_points === record.n);
    const metal = rows.find(row => row.impl === 'mps-pointops Metal (MPS)');
    const torchMps = rows.find(row => row.device === 'mps' && row !== metal);
    assert.equal(series(record, 'metal'), metal.median_ms);
    assert.equal(series(record, 'torch-mps'), torchMps.median_ms);
    if (record.operation === 'fps') {
      const fpsample = rows.find(row => row.impl === 'fpsample vanilla (CPU)');
      assert.equal(series(record, 'fpsample'), fpsample.median_ms);
    } else {
      assert.ok(!record.series.some(item => item.label.includes('cKDTree')),
        'SciPy tree build + query must not be compared to MPS query-only timing');
    }
  }
});

test('flat public rows retain paired CPU/MPS timings and parity', () => {
  for (const record of data.records.filter(item => item.id.startsWith('m5-flat-'))) {
    const source = raw(record.source.rawPath);
    assert.equal(source.source_dirty, false);
    assert.equal(source.source_commit, record.source.sourceCommit);
    const rawOp = record.operation === 'ball-query' ? 'radius' : record.operation;
    const rawCase = source.cases.find(item => item.n === record.n);
    assert.ok(rawCase);
    assert.equal(series(record, 'metal'), rawCase.timings.mps[rawOp].median_ms);
    assert.equal(series(record, 'cpu'), rawCase.timings.cpu[rawOp].median_ms);
    assert.equal(record.parity.exactCpuMps, rawCase.exact_cpu_mps_parity[rawOp]);
  }
});

test('M1 FPS pairs use one clean source and matching B=1 inputs', () => {
  for (const record of data.records.filter(item => item.id.startsWith('m1-fps-'))) {
    const source = raw(record.source.rawPath);
    assert.equal(source.environment.chip, 'Apple M1');
    assert.equal(source.environment.dirty, false);
    assert.equal(source.environment.commit, record.source.sourceCommit);
    const rawCase = source.results.find(item => item.N === record.n);
    assert.equal(rawCase.B, 1);
    assert.equal(series(record, 'multigroup'), rawCase.multigroup_median_ms);
    assert.equal(series(record, 'single'), rawCase.single_median_ms);
    assert.equal(record.parity.indexMismatches, rawCase.index_mismatches);
  }
});

test('historical BVH rows match exact steady-query medians and backend choice', () => {
  function median(values) {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle]
      : (sorted[middle - 1] + sorted[middle]) / 2;
  }
  for (const record of data.records.filter(item => item.api === 'spatial-index')) {
    const source = raw(record.source.rawPath);
    assert.equal(source.source_dirty, false);
    assert.equal(source.source_commit, record.source.sourceCommit);
    assert.equal(source.hardware, record.hardware);
    assert.equal(record.method.buildIncluded, false);
    assert.equal(series(record, 'bvh'), median(source.bvh_steady_query_ms));
    assert.equal(series(record, 'scan'), median(source.scan_steady_query_ms));
    assert.equal(series(record, 'auto'), median(source.auto_steady_query_ms));
    const chosen = source.auto_selected_backend ?? source.adaptive_decision.selected_backend;
    assert.ok(record.series.some(item => item.id === 'auto' && item.label.endsWith(chosen)));
  }
});

test('sparse selection leaves unmeasured combinations blank', () => {
  assert.equal(filterBenchmarkRecords(data, {
    hardware: 'Apple M1', operation: 'knn', n: '1000000',
  }).length, 0);
  assert.equal(filterBenchmarkRecords(data, {
    hardware: 'Apple M5 Pro', operation: 'fps', n: '100000',
  }).length, 1);
  assert.equal(filterBenchmarkRecords(data, {
    hardware: 'Apple M5 Pro', operation: 'knn', n: '1000000',
  }).length, 2);
  assert.equal(filterBenchmarkRecords(data, {
    hardware: 'Apple M5 Pro', operation: 'ball-query', n: '1000000',
  }).length, 1);
});

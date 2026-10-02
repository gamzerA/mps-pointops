import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { measuredComparison } from '../src/measurements-dashboard.js';

const data = JSON.parse(readFileSync(fileURLToPath(new URL('../data/benchmarks.json', import.meta.url)), 'utf8'));

test('the default comparison uses query times from one measured unreleased fixture', () => {
  const { record, series, maxMs } = measuredComparison(data);
  assert.equal(record.id, 'spatial-knn-uniform-independent-1m-q65536');
  assert.equal(record.scope, 'unreleased-public-source');
  assert.equal(record.conditions.n, 1_000_000);
  assert.equal(record.conditions.q, 65_536);
  assert.deepEqual(series.map(item => [item.id, item.queryMs]), [
    ['native-scan', 883.099167],
    ['bvh', 164.0595],
    ['auto', 166.254959],
  ]);
  assert.equal(maxMs, 883.099167);
  assert.equal(record.auto.selectedBackend, 'bvh');
});

test('changing fixture can show BVH losing while Auto selects scan', () => {
  const { record, series } = measuredComparison(data, 'spatial-knn-coincident-1m-q65536');
  assert.equal(record.distribution, 'coincident');
  assert.equal(record.auto.selectedBackend, 'scan');
  assert.equal(series.find(item => item.id === 'bvh').queryMs, 1293.181375);
  assert.equal(series.find(item => item.id === 'native-scan').queryMs, 863.551083);
  assert.equal(series.find(item => item.id === 'auto').queryMs, 865.1105);
});

test('research prototype keeps build time separate from its query comparison', () => {
  const { record, series } = measuredComparison(data, 'grid-radius-uniform-1m-q1024');
  assert.equal(record.scope, 'research-prototype');
  assert.equal(record.conditions.statistic, 'sum-of-separate-stage-medians');
  assert.deepEqual(series.map(item => item.queryMs), [3.817417, 2.589]);
  assert.equal(record.baselines[0].buildMs, 252.358042);
  assert.equal(record.ours.buildMs, 4.080417);
  assert.ok(series.every(item => !Object.hasOwn(item, 'buildMs')));
});

test('dashboard evidence remains tied to separate source and test records', () => {
  const { records } = measuredComparison(data);
  assert.equal(records.length, 10);
  assert.ok(records.every(record => record.status === 'measured'));
  assert.ok(records.every(record => record.scope !== 'published-package'));
  assert.ok(records.every(record => record.conditions.sourcePath.startsWith('bench/results/')));
  assert.notEqual(records[0].conditions.sourceCommit, records[1].conditions.sourceCommit);
  assert.equal(data.verification.testRuns.find(run => run.id === 'safe').sourceCommit, records[1].conditions.sourceCommit);
  assert.equal(data.meta.publishedVersion, '0.8.0');
  assert.ok(data.support.some(item => item.id === 'm1-8gb-spatial' && item.status === 'measured'));
  assert.ok(data.support.some(item => item.id === 'physical-gpu-peak' && item.status === 'pending'));
});

test('M1 evidence keeps opt-in BVH timing separate from scan-only Auto', () => {
  const m1 = data.benchmarks.filter(record => record.conditions.hardware === 'Apple M1');
  assert.equal(m1.length, 6);
  assert.ok(m1.every(record => record.scope === 'unreleased-public-source'
    && record.conditions.sourceCommit === 'ef07a9337b40065c18a6c950bda4240c789f31f1'
    && record.conditions.n === 1_000_000 && record.conditions.q === 65_536
    && record.conditions.repetitions === 3 && record.auto.selectedBackend === 'scan'
    && record.parity.mismatchedIndexSlots === 0));
  const uniformKnn = measuredComparison(data, 'm1-spatial-knn-uniform-1m-q65536');
  assert.deepEqual(uniformKnn.series.map(item => [item.id, item.queryMs]), [
    ['native-scan', 4762.5115],
    ['bvh', 784.613041],
    ['auto', 4832.069083],
  ]);
  const collapsedKnn = measuredComparison(data, 'm1-spatial-knn-collapsed-1m-q65536');
  assert.ok(collapsedKnn.record.ours.queryMs > collapsedKnn.record.baselines[0].queryMs);
  const uniformRadius = measuredComparison(data, 'm1-spatial-radius-uniform-1m-q65536');
  assert.equal(uniformRadius.record.ours.queryMs, 159.030917);
  assert.equal(uniformRadius.record.baselines[0].queryMs, 11461.166042);
  assert.ok(m1.filter(record => record.operation === 'radius').every(record =>
    record.parity.mismatchedSquaredDistanceBits === 0
    && record.parity.originalIndexOrderViolations === 0));
});

test('an invalid fixture cannot quietly become a measured bar', () => {
  const changed = structuredClone(data);
  changed.benchmarks.find(record => record.id === 'spatial-knn-uniform-independent-1m-q65536')
    .auto.queryMs = Number.NaN;
  assert.throws(() => measuredComparison(changed), /invalid query timing/);
  assert.equal(measuredComparison(data, 'unknown').record.id, 'spatial-knn-uniform-independent-1m-q65536');
});

import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

const data = JSON.parse(readFileSync(new URL('../data/benchmarks.json', import.meta.url)));
// The page is portable. Its upstream source checkout is only needed for this
// local provenance audit, never for browser rendering or remote fetching.
const sourceRoot = resolve(process.env.MPS_POINTOPS_SOURCE_ROOT
  ?? '/private/tmp/mps-pointops-v090-work');
const m1RawRoot = resolve(process.env.MPS_POINTOPS_M1_RESULTS
  ?? '/private/tmp/mps-v090-m1-results');
const haveSource = existsSync(join(sourceRoot, 'bench', 'results'));

test('physical M1 Chamfer parity callout matches the archived upstream matrices',
  { skip: !haveSource && 'Set MPS_POINTOPS_SOURCE_ROOT to the local source checkout' },
  () => {
    const archive = join(sourceRoot, 'docs', 'evidence', 'chamfer-v090-m1-upstream');
    const read = name => JSON.parse(readFileSync(join(archive, name), 'utf8'));
    const files = ['chamfer-safe.json', 'chamfer-fast.json',
      'chamfer-extended-safe.json', 'chamfer-extended-fast.json'];
    const records = files.map(read);
    assert.deepEqual(records.map(record => record.summary.mps.cases), [480, 480, 252, 252]);
    assert.ok(records.every(record => record.summary.mps.failed_elements === 0
      && (record.summary.mps.failed_checks ?? record.summary.mps.failed_tensors) === 0
      && record.summary.mps.max_abs <= 1.91e-6
      && record.environment.mps_fallback === '0'));
    const html = readFileSync(new URL('../index.html', import.meta.url), 'utf8');
    assert.match(html, /Physical M1 Chamfer.*480 base \+ 252 normals\/Pointclouds cases, 0 failures, max \|Δ\| 1\.91e-6/);
    assert.match(html, /docs\/evidence\/chamfer-v090-m1-upstream\/README\.md/);
  });

function median(samples) {
  assert.ok(Array.isArray(samples) && samples.length > 0);
  assert.ok(samples.every((value) => Number.isFinite(value) && value >= 0));
  const sorted = [...samples].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle]
    : (sorted[middle - 1] + sorted[middle]) / 2;
}

function requireSafeRelativePath(path) {
  assert.match(path, /^[A-Za-z0-9_./-]+$/);
  assert.ok(!path.split('/').includes('..'), `unsafe path ${path}`);
}

function rawFor(path) {
  requireSafeRelativePath(path);
  return JSON.parse(readFileSync(sourceRecordPath(path), 'utf8'));
}

function sourceRecordPath(path) {
  requireSafeRelativePath(path);
  const inCheckout = join(sourceRoot, path);
  if (existsSync(inCheckout)) return inCheckout;
  if (path.startsWith('bench/results/spatial-v090-m1/')) {
    const captured = join(m1RawRoot, basename(path));
    if (existsSync(captured)) return captured;
  }
  throw new Error(`source record not found: ${path}`);
}

function verifySourceHashes(raw) {
  for (const [path, expected] of Object.entries(raw.source_sha256)) {
    requireSafeRelativePath(path);
    assert.match(expected, /^[0-9a-f]{64}$/);
    // Read the exact recorded commit. The current checkout may contain later
    // edits to the research scripts, so hashing HEAD would be misleading.
    const blob = execFileSync('git',
      ['cat-file', 'blob', `${raw.source_commit}:${path}`],
      { cwd: sourceRoot, maxBuffer: 32 * 1024 * 1024 });
    assert.equal(createHash('sha256').update(blob).digest('hex'), expected,
      `${raw.source_commit}:${path} differs from the recorded hash`);
  }
}

test('each displayed benchmark matches its raw record and recorded source',
  { skip: !haveSource && 'Set MPS_POINTOPS_SOURCE_ROOT to the local source checkout' },
  () => {
    for (const record of data.benchmarks) {
      const conditions = record.conditions;
      const raw = rawFor(conditions.sourcePath);
      assert.equal(raw.source_dirty, false, `${record.id}: source was dirty`);
      assert.equal(raw.source_commit, conditions.sourceCommit);
      assert.equal(raw.hardware, conditions.hardware);
      assert.equal(raw.macos, conditions.macos);
      assert.equal(raw.torch, conditions.pytorch);
      assert.equal(raw.mps_fast_math, '0');
      assert.equal(raw.mps_fallback, '0');
      assert.equal(raw.fixture.points, conditions.n);
      assert.equal(raw.fixture.queries, conditions.q);
      assert.equal(raw.repeats ?? raw.cpu_ckdtree_build_ms.length,
        conditions.repetitions);
      assert.ok(Object.hasOwn(raw.source_sha256, conditions.benchmarkScript));

      if (record.scope === 'research-prototype') {
        assert.equal(raw.date_utc, conditions.measurementDateUtc);
        assert.equal(raw.python, conditions.python);
        assert.equal(raw.scipy, conditions.scipy);
        assert.equal(raw.fixture.kind, 'uniform synthetic float32');
        assert.equal(raw.fixture.limit, conditions.k);
        assert.equal(raw.fixture.radius, conditions.radius);
        assert.equal(raw.fixture.cell_size, conditions.cellSize);
        assert.equal(conditions.statistic, 'sum-of-separate-stage-medians');
        assert.equal(median(raw.cpu_ckdtree_build_ms), record.baselines[0].buildMs);
        assert.equal(median(raw.cpu_ckdtree_query_ms), record.baselines[0].queryMs);
        assert.equal(median(raw.mps_morton_build_ms), record.ours.buildMs);
        assert.equal(median(raw.mps_morton_query_ms), record.ours.queryMs);
        assert.equal(raw.mismatched_slots_vs_ckdtree,
          record.parity.mismatchedIndexSlots);
        assert.equal(raw.native_brute_sample_queries,
          record.parity.nativeFullScanSampleQueries);
        assert.equal(raw.native_brute_sample_equal,
          record.parity.nativeFullScanSampleEqual);
      } else {
        assert.equal(raw.utc, conditions.measurementDateUtc);
        assert.equal(raw.physical_ram_bytes / (1024 ** 3), conditions.memoryGiB);
        assert.equal(conditions.statistic, 'median-of-synchronized-query-samples');
        assert.equal(raw.fixture.k ?? raw.fixture.limit, conditions.k);
        assert.equal(raw.fixture.distribution,
          record.distribution === 'coincident' ? 'collapsed' : record.distribution);
        if (record.operation === 'radius') {
          assert.equal(raw.fixture.radius_float32, conditions.radius);
          assert.equal(raw.auto_selected_backend, record.auto.selectedBackend);
          assert.equal(raw.original_index_order_violations,
            record.parity.originalIndexOrderViolations);
          assert.equal(raw.mismatch.bvh_steady_vs_scan_indices,
            record.parity.mismatchedIndexSlots);
          assert.equal(raw.mismatch.bvh_steady_vs_scan_squared_distance_bits,
            record.parity.mismatchedSquaredDistanceBits);
          if (record.parity.bvhStackFallbackRows !== undefined) {
            assert.equal(raw.bvh_stack_fallback.rows,
              record.parity.bvhStackFallbackRows);
            assert.equal(raw.bvh_stack_fallback.query_count, conditions.q);
          }
        } else {
          assert.equal(raw.adaptive_decision.selected_backend,
            record.auto.selectedBackend);
          assert.equal(raw.index_mismatch.bvh_vs_scan_indices,
            record.parity.mismatchedIndexSlots);
          assert.equal(raw.euclidean_distance_max_abs_error.bvh_vs_scan_max_abs,
            record.parity.maxAbsoluteDistanceError);
        }
        assert.equal(median(raw.scan_steady_query_ms), record.baselines[0].queryMs);
        assert.equal(median(raw.bvh_steady_query_ms), record.ours.queryMs);
        assert.equal(median(raw.auto_steady_query_ms), record.auto.queryMs);
      }
      verifySourceHashes(raw);
    }
  });

test('displayed Safe and Fast counts match the saved pytest summaries',
  { skip: !haveSource && 'Set MPS_POINTOPS_SOURCE_ROOT to the local source checkout' },
  () => {
    for (const run of data.verification.testRuns) {
      const log = readFileSync(join(sourceRoot, run.logPath), 'utf8');
      const summary = log.match(/(\d+) passed, (\d+) skipped, (\d+) xfailed in [\d.]+s/);
      assert.ok(summary, `${run.id}: pytest summary not found`);
      assert.equal(Number(summary[1]), run.passed);
      assert.equal(Number(summary[2]), run.skipped);
      assert.equal(Number(summary[3]), run.xfailed);
      assert.equal(run.sourceCommit, data.meta.developmentSourceCommit);
    }
  });

test('bundled browser evidence exactly matches the audited source records',
  { skip: !haveSource && 'Set MPS_POINTOPS_SOURCE_ROOT to the local source checkout' },
  () => {
    const paths = [
      ...data.benchmarks.map(record => record.conditions.sourcePath),
      ...data.verification.testRuns.map(run => run.logPath),
      data.memoryEvidence.sourcePath,
    ];
    for (const path of paths) {
      requireSafeRelativePath(path);
      const source = readFileSync(path.endsWith('.json') ? sourceRecordPath(path) : join(sourceRoot, path));
      const bundled = readFileSync(new URL(`../evidence/${path}`, import.meta.url));
      assert.deepEqual(bundled, source, `bundled evidence differs: ${path}`);
    }
  });

test('M1 memory figures are allocator and sampled-driver evidence, not physical GPU peak',
  { skip: !haveSource && 'Set MPS_POINTOPS_SOURCE_ROOT to the local source checkout' },
  () => {
    const evidence = data.memoryEvidence;
    const raw = rawFor(evidence.sourcePath);
    assert.equal(raw.source_dirty, false);
    assert.equal(raw.source_commit, evidence.sourceCommit);
    assert.equal(raw.hardware, evidence.hardware);
    assert.equal(raw.fixture.points, evidence.n);
    assert.equal(raw.fixture.queries, evidence.q);
    assert.equal(raw.allocator_high_water_bytes.tensor_bytes, evidence.allocatorTensorPeakBytes);
    assert.equal(raw.sampled_high_water_bytes.driver_bytes, evidence.sampledDriverHighBytes);
    assert.equal(raw.allocator_high_water_is_total_gpu_peak, false);
    assert.equal(raw.sampled_driver_high_water_is_true_driver_peak, false);
    assert.equal(evidence.physicalGpuPeakStatus, 'pending');
    verifySourceHashes(raw);
  });

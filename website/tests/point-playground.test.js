import test from 'node:test';
import assert from 'node:assert/strict';
import {
  computePlaygroundSteps,
  createPlaygroundPoints,
  layoutPlaygroundPlot,
} from '../src/point-playground.js';

test('one deterministic fixture is stable across the operation tabs and point counts', () => {
  const small = createPlaygroundPoints({ count: 24 });
  const large = createPlaygroundPoints({ count: 96 });
  assert.deepEqual(small, large.slice(0, 24));
  assert.deepEqual(small, createPlaygroundPoints({ count: 24 }));
  assert.notEqual(small[0].x, small[1].x);
});

test('wide, tablet, and mobile canvases preserve Euclidean circle geometry', () => {
  for (const [width, height] of [[1440, 375], [768, 450], [390, 420], [320, 250]]) {
    const plot = layoutPlaygroundPlot(width, height);
    assert.equal(plot.width, plot.height, `${width}×${height} must have one pixel scale`);
    assert.ok(plot.x >= 0 && plot.y >= 0);
    assert.ok(plot.x + plot.width <= width);
    assert.ok(plot.y + plot.height <= height);
    const query = { x: 0.52, y: 0.48 };
    const radius = 0.2;
    const cx = plot.x + query.x * plot.width;
    const cy = plot.y + (1 - query.y) * plot.height;
    const right = plot.x + (query.x + radius) * plot.width;
    const up = plot.y + (1 - query.y - radius) * plot.height;
    assert.ok(Math.abs((right - cx) - (cy - up)) < 1e-10,
      `${width}×${height} should draw equal horizontal and vertical radii`);
    assert.ok(Math.abs((cx - plot.x) / plot.width - query.x) < 1e-12);
    assert.ok(Math.abs(1 - (cy - plot.y) / plot.height - query.y) < 1e-12);
  }
  assert.throws(() => layoutPlaygroundPlot(0, 300), RangeError);
});

test('FPS uses maximum minimum squared distance, with lower-index ties', () => {
  const points = [
    { x: 0, y: 0 }, { x: 1, y: 0 },
    { x: 1, y: 1 }, { x: 0, y: 1 },
  ];
  const trace = computePlaygroundSteps({ mode: 'fps', points, k: 3, seedIndex: 0 });
  assert.deepEqual(trace.indices, [0, 2, 1]);
  assert.deepEqual(trace.frames.map((frame) => frame.selected),
    [[], [0], [0, 2], [0, 2, 1]]);
  assert.equal(trace.frames[2].metric, 2);
  assert.equal(trace.frames[3].nearestIndex, 0);
  assert.equal(trace.inspected, 7); // 3 then 2 × 2 actual distance comparisons.
  assert.deepEqual(computePlaygroundSteps({ mode: 'fps', points, k: 2, seedIndex: 3 }).indices,
    [3, 1]);
});

test('kNN sorts by squared distance then original index, including exact ties', () => {
  const points = [
    { x: 0, y: 0 }, { x: 1, y: 0 },
    { x: 0, y: 1 }, { x: 1, y: 1 },
  ];
  const trace = computePlaygroundSteps({
    mode: 'knn', points, query: { x: 0.5, y: 0.5 }, k: 3,
  });
  assert.deepEqual(trace.indices, [0, 1, 2]);
  assert.equal(trace.frames[1].metric, 0.5);
  assert.equal(trace.inspected, points.length);
  assert.deepEqual(computePlaygroundSteps({
    mode: 'knn', points, query: { x: 0, y: 0.2 }, k: 2,
  }).indices, [0, 2]);
});

test('Ball Query excludes the strict radius boundary and stops at the first K input-order hits', () => {
  const points = [
    { x: 1, y: 0 },     // exactly on the boundary: excluded
    { x: 0, y: 0 },     // coincident: accepted
    { x: 0.5, y: 0 },   // accepted
    { x: 0.1, y: 0 },   // nearer, but after the first K: never inspected
  ];
  const trace = computePlaygroundSteps({
    mode: 'ball', points, query: { x: 0, y: 0 }, radius: 1, k: 2,
  });
  assert.deepEqual(trace.indices, [1, 2]);
  assert.equal(trace.inspected, 3);
  assert.deepEqual(trace.frames.slice(1).map((frame) => frame.accepted), [false, true, true]);
  assert.deepEqual(computePlaygroundSteps({
    mode: 'ball', points, query: { x: 0, y: 0 }, radius: 0, k: 2,
  }).indices, []);
});

test('selection is bounded by available points and zero K produces no scan', () => {
  const points = [{ x: 0, y: 0 }, { x: 0.1, y: 0 }];
  for (const mode of ['fps', 'knn', 'ball']) {
    const result = computePlaygroundSteps({ mode, points, k: 0, radius: 1 });
    assert.deepEqual(result.indices, []);
    assert.equal(result.frames.length, 1);
  }
  assert.deepEqual(computePlaygroundSteps({ mode: 'fps', points, k: 9 }).indices, [0, 1]);
  assert.deepEqual(computePlaygroundSteps({ mode: 'knn', points, k: 9 }).indices, [1, 0]);
  assert.deepEqual(computePlaygroundSteps({ mode: 'ball', points, k: 9, radius: 1 }).indices, [0, 1]);
});

test('bad teaching-model inputs fail explicitly', () => {
  assert.throws(() => createPlaygroundPoints({ count: 161 }), RangeError);
  assert.throws(() => computePlaygroundSteps({ mode: 'random', points: [] }), RangeError);
  assert.throws(() => computePlaygroundSteps({ mode: 'fps', points: [{ x: NaN, y: 0 }] }), TypeError);
  assert.throws(() => computePlaygroundSteps({ mode: 'ball', points: [], radius: -1 }), RangeError);
  assert.throws(() => computePlaygroundSteps({ mode: 'knn', points: [], k: 0.5 }), RangeError);
});

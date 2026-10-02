import test from 'node:test';
import assert from 'node:assert/strict';
import {
  createComputePoints, layoutComputeField, projectComputePoint, unprojectComputePoint,
  computeFieldTrace, computeFieldLinks, MAX_FIELD_POINTS,
} from '../src/compute-field.js';

test('dense fixture is deterministic, prefix stable, bounded, and has two cluster families', () => {
  const short = createComputePoints({ count: 240, width: 2.3 });
  const long = createComputePoints({ count: MAX_FIELD_POINTS, width: 2.3 });
  assert.deepEqual(short, long.slice(0, short.length));
  assert.deepEqual(short, createComputePoints({ count: 240, width: 2.3 }));
  assert.ok(short.some((point) => point.family === 'cyan'));
  assert.ok(short.some((point) => point.family === 'coral'));
  assert.ok(short.some((point) => point.family === 'neutral'));
  assert.ok(long.every((point) => point.x >= 0 && point.x <= 2.3 && point.y >= 0 && point.y <= 1));
  assert.throws(() => createComputePoints({ count: MAX_FIELD_POINTS + 1 }), RangeError);
  assert.throws(() => createComputePoints({ width: 0 }), RangeError);
});

test('wide and narrow canvases fill the viewport and retain a circular radius', () => {
  for (const [width, height] of [[1100, 475], [720, 500], [320, 400]]) {
    const layout = layoutComputeField(width, height);
    assert.ok(layout.width <= width && layout.height <= height);
    assert.ok(layout.width >= width * 0.7, `${width}x${height}: plot should occupy the stage`);
    assert.ok(layout.height >= height * 0.7, `${width}x${height}: plot should occupy the stage`);
    const center = { x: layout.worldWidth / 2, y: 0.5 };
    const radius = 0.2;
    const at = projectComputePoint(layout, center);
    const right = projectComputePoint(layout, { x: center.x + radius, y: center.y });
    const up = projectComputePoint(layout, { x: center.x, y: center.y + radius });
    assert.ok(Math.abs((right.x - at.x) - (at.y - up.y)) < 1e-10);
    assert.ok(Math.abs(unprojectComputePoint(layout, at).x - center.x) < 1e-10);
    assert.ok(Math.abs(unprojectComputePoint(layout, at).y - center.y) < 1e-10);
  }
});

test('FPS assignment rays connect each candidate to its true nearest selected sample', () => {
  const points = [
    { x: 0, y: 0 }, { x: 1, y: 0 }, { x: 0, y: 1 },
    { x: 0.2, y: 0.1 }, { x: 0.8, y: 0.1 },
  ];
  const trace = computeFieldTrace({ mode: 'fps', points, k: 2, seedIndex: 0 });
  assert.deepEqual(trace.indices, [0, 1]);
  const links = computeFieldLinks({ mode: 'fps', points, frame: trace.frames[2], query: { x: 0, y: 0 } });
  const assignments = links.filter((link) => link.kind === 'assignment');
  assert.equal(assignments.find((link) => link.index === 3).targetIndex, 0);
  assert.equal(assignments.find((link) => link.index === 4).targetIndex, 1);
  assert.ok(links.some((link) => link.kind === 'chosen' && link.index === 1 && link.targetIndex === 0));
  for (const link of links) {
    assert.equal(link.distanceSquared, (link.from.x - link.to.x) ** 2 + (link.from.y - link.to.y) ** 2);
  }
});

test('kNN rays represent measured query distances and selected rays match the true nearest indices', () => {
  const points = [
    { x: 1, y: 0 }, { x: 0.1, y: 0 }, { x: 0.2, y: 0 }, { x: 0.9, y: 0 },
  ];
  const query = { x: 0, y: 0 };
  const trace = computeFieldTrace({ mode: 'knn', points, query, k: 2 });
  assert.deepEqual(trace.indices, [1, 2]);
  const links = computeFieldLinks({ mode: 'knn', points, frame: trace.frames.at(-1), query, maxLinks: 4 });
  assert.deepEqual(links.filter((link) => link.kind === 'selected').map((link) => link.index), [1, 2]);
  assert.ok(links.every((link) => link.from === query));
  assert.ok(links.length <= 4);
});

test('Ball Query retains strict boundary, first-K order, and only scanned distance rays', () => {
  const points = [
    { x: 1, y: 0 }, { x: 0.8, y: 0 }, { x: 0.1, y: 0 },
    { x: 0.2, y: 0 }, { x: 0.01, y: 0 },
  ];
  const query = { x: 0, y: 0 };
  const trace = computeFieldTrace({ mode: 'ball', points, query, k: 2, radius: 1 });
  assert.deepEqual(trace.indices, [1, 2]);
  assert.equal(trace.inspected, 3);
  const frame = trace.frames.at(-1);
  const links = computeFieldLinks({ mode: 'ball', points, frame, query, maxLinks: 6 });
  assert.ok(links.every((link) => link.index < frame.inspected));
  assert.deepEqual(links.filter((link) => link.kind === 'selected').map((link) => link.index), [1, 2]);
  assert.ok(links.some((link) => link.kind === 'scanned' && link.index === 0));
  assert.ok(!links.some((link) => link.index === 3 || link.index === 4));
});

test('long Ball Query playback is bounded while retaining every accepted frame and exact final inspection', () => {
  const points = createComputePoints({ count: MAX_FIELD_POINTS });
  const trace = computeFieldTrace({
    mode: 'ball', points, query: { x: 0.6, y: 0.5 }, k: 16, radius: 0.08,
  });
  assert.ok(trace.frames.length <= 82);
  assert.equal(trace.frames.at(-1).inspected, trace.inspected);
  assert.equal(trace.frames.at(-1).selected.length, trace.indices.length);
  assert.ok(trace.frames.filter((frame) => frame.accepted).length === trace.indices.length);
  assert.ok(trace.frames.every((frame, index, frames) => index === 0 || frame.inspected > frames[index - 1].inspected));
});

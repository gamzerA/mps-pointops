import test from 'node:test';
import assert from 'node:assert/strict';
import {
  buildBVH2D,
  bruteForceRadius,
  generateDemoPoints,
  minAabbDistanceSquared,
  queryRadius,
  sortPointsByMorton2D,
} from '../src/lib/index.js';

test('fixture generation is deterministic and distinguishes distributions', () => {
  const defaults = generateDemoPoints();
  assert.equal(defaults.length, 2000);
  assert.deepEqual(defaults, generateDemoPoints());
  assert.notDeepEqual(defaults, generateDemoPoints({ seed: 7 }));
  const clustered = generateDemoPoints({ distribution: 'clustered', count: 128 });
  assert.deepEqual(clustered,
    generateDemoPoints({ distribution: 'clustered', count: 128 }));
  assert.notDeepEqual(clustered,
    generateDemoPoints({ distribution: 'uniform', count: 128 }));
  for (const point of [...defaults, ...clustered]) {
    assert.ok(point.x >= 0 && point.x <= 1);
    assert.ok(point.y >= 0 && point.y <= 1);
  }
});

test('Morton order is stable for equal codes and uses BigInt comparisons', () => {
  const points = [{ x: 1, y: 1 }, { x: 0, y: 0 }, { x: 1, y: 1 }];
  const sorted = sortPointsByMorton2D(points);
  assert.deepEqual(sorted.map((point) => point.index), [1, 0, 2]);
  assert.equal(typeof sorted[0].morton, 'bigint');
  assert.ok(sorted[2].morton > 2n ** 53n);
});

test('balanced bricks cover every original point with actual-point bounds', () => {
  const points = generateDemoPoints({ count: 257, seed: 53 });
  const bvh = buildBVH2D(points);
  assert.equal(bvh.brickCount, Math.ceil(257 / 16));
  assert.equal(bvh.nodeCount, 2 * bvh.brickCount - 1);
  assert.equal(bvh.sortedPoints.length, points.length);
  const covered = [];
  const leafDepths = [];
  function inspect(node) {
    if (node.leaf) {
      leafDepths.push(node.depth);
      assert.ok(node.pointIndices.length >= 1);
      assert.ok(node.pointIndices.length <= 16);
      for (const index of node.pointIndices) {
        covered.push(index);
        const { x, y } = points[index];
        assert.ok(x >= node.bounds.minX && x <= node.bounds.maxX);
        assert.ok(y >= node.bounds.minY && y <= node.bounds.maxY);
      }
      return;
    }
    inspect(node.left);
    inspect(node.right);
  }
  inspect(bvh.root);
  assert.deepEqual(covered.sort((a, b) => a - b),
    Array.from({ length: points.length }, (_, i) => i));
  assert.ok(Math.max(...leafDepths) - Math.min(...leafDepths) <= 1);
});

test('BVH radius result equals full scan for varied fixtures and queries', () => {
  for (const distribution of ['uniform', 'clustered']) {
    const points = generateDemoPoints({ count: 257, seed: 913, distribution });
    const { root } = buildBVH2D(points);
    const queries = [[0, 0], [0.5, 0.5], [1, 1], [0.23, 0.22], [2, 2]];
    for (const [qx, qy] of queries) {
      for (const radius of [0, 0.01, 0.05, 0.1, 0.3, 2]) {
        const result = queryRadius(root, points, qx, qy, radius);
        assert.deepEqual(result.hitIndices,
          bruteForceRadius(points, qx, qy, radius),
          `${distribution} q=(${qx},${qy}) r=${radius}`);
        assert.equal(result.inspectedAabbs.length, result.visitedNodes);
        assert.ok(result.prunedNodes <= result.visitedNodes);
        assert.ok(result.openedBricks <= Math.ceil(points.length / 16));
        assert.ok(result.distanceCalculations <= points.length);
      }
    }
  }
});

test('strict radius boundary and duplicate coordinates preserve original indices', () => {
  const points = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: -1, y: 0 },
    { x: 0.5, y: 0 },
    { x: 0, y: 0 },
  ];
  const { root } = buildBVH2D(points, { brickSize: 1 });
  assert.deepEqual(queryRadius(root, points, 0, 0, 1).hitIndices, [0, 3, 4]);
  assert.deepEqual(queryRadius(root, points, 0, 0, 0).hitIndices, []);
  const boundaryPoint = [{ x: 1, y: 0 }];
  const boundaryRoot = buildBVH2D(boundaryPoint, { brickSize: 1 }).root;
  const equality = queryRadius(boundaryRoot, boundaryPoint, 0, 0, 1);
  assert.equal(minAabbDistanceSquared(boundaryRoot.bounds, 0, 0), 1);
  assert.equal(equality.prunedNodes, 0); // Dmin² == r² is not pruned.
  assert.equal(equality.openedBricks, 1);
  assert.deepEqual(equality.hitIndices, []); // Point d² == r² is excluded.
});

test('counts report only nodes actually visited and point distances evaluated', () => {
  const points = generateDemoPoints({ count: 100, seed: 4 });
  const { root } = buildBVH2D(points);
  const far = queryRadius(root, points, 2, 2, 0.01);
  assert.equal(far.visitedNodes, 1);
  assert.equal(far.prunedNodes, 1);
  assert.equal(far.openedBricks, 0);
  assert.equal(far.distanceCalculations, 0);
  assert.equal(far.inspectedAabbs[0].pruned, true);
  const all = queryRadius(root, points, 0.5, 0.5, 2);
  assert.equal(all.hitIndices.length, points.length);
  assert.equal(all.distanceCalculations, points.length);
  assert.equal(all.openedBricks, Math.ceil(points.length / 16));
});

test('all-coincident preset remains exact and exposes its full-scan worst case', () => {
  const points = generateDemoPoints({ count: 2000, distribution: 'coincident' });
  const { root } = buildBVH2D(points);
  const result = queryRadius(root, points, 0.5, 0.5, 0.095);
  assert.deepEqual(result.hitIndices,
    bruteForceRadius(points, 0.5, 0.5, 0.095));
  assert.equal(result.hitIndices.length, points.length);
  assert.equal(result.prunedNodes, 0);
  assert.equal(result.distanceCalculations, points.length);
  assert.equal(result.openedBricks, Math.ceil(points.length / 16));
});

test('empty BVH and invalid inputs are explicit', () => {
  const empty = buildBVH2D([]);
  assert.equal(empty.root, null);
  assert.deepEqual(queryRadius(empty.root, [], 0, 0, 1).hitIndices, []);
  assert.throws(() => buildBVH2D([{ x: NaN, y: 0 }]), TypeError);
  assert.throws(() => buildBVH2D([{ x: 0, y: 0 }], { brickSize: 0 }), RangeError);
  assert.throws(() => queryRadius(empty.root, [], 0, 0, -1), RangeError);
  assert.throws(() => queryRadius(empty.root, [], Infinity, 0, 1), TypeError);
});

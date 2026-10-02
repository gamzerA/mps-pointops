import { encodeMorton3D } from './morton.js';

function validatePoints(points) {
  if (!Array.isArray(points)) {
    throw new TypeError('points must be an array');
  }
  for (let i = 0; i < points.length; i += 1) {
    const point = points[i];
    if (point === null || typeof point !== 'object' ||
        !Number.isFinite(point.x) || !Number.isFinite(point.y)) {
      throw new TypeError(`points[${i}] must have finite x and y`);
    }
  }
}

function validateMortonBits(mortonBits) {
  if (!Number.isInteger(mortonBits) || mortonBits < 1 || mortonBits > 21) {
    throw new RangeError('mortonBits must be an integer from 1 to 21');
  }
}

/** The 2D demo puts x/y in the x/y lanes of the exact 3D Morton encoder. */
export function sortPointsByMorton2D(points, mortonBits = 21) {
  validatePoints(points);
  validateMortonBits(mortonBits);
  if (points.length === 0) return [];

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const { x, y } of points) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }
  const spanX = maxX - minX;
  const spanY = maxY - minY;
  if (!Number.isFinite(spanX) || !Number.isFinite(spanY)) {
    throw new RangeError('point coordinate span must be finite');
  }
  const maximumCell = 2 ** mortonBits - 1;
  const quantize = (value, minimum, span) => {
    if (span === 0) return 0;
    return Math.min(maximumCell, Math.max(0,
      Math.floor(((value - minimum) / span) * maximumCell)));
  };

  return points.map(({ x, y }, index) => ({
    index,
    x,
    y,
    morton: encodeMorton3D(
      quantize(x, minX, spanX), quantize(y, minY, spanY), 0, mortonBits,
    ),
  })).sort((a, b) => {
    if (a.morton < b.morton) return -1;
    if (a.morton > b.morton) return 1;
    return a.index - b.index;
  });
}

function actualBounds(sortedPoints, first, last) {
  const bounds = { minX: Infinity, minY: Infinity, maxX: -Infinity, maxY: -Infinity };
  for (let i = first; i < last; i += 1) {
    const point = sortedPoints[i];
    bounds.minX = Math.min(bounds.minX, point.x);
    bounds.minY = Math.min(bounds.minY, point.y);
    bounds.maxX = Math.max(bounds.maxX, point.x);
    bounds.maxY = Math.max(bounds.maxY, point.y);
  }
  return bounds;
}

function unionBounds(a, b) {
  return {
    minX: Math.min(a.minX, b.minX),
    minY: Math.min(a.minY, b.minY),
    maxX: Math.max(a.maxX, b.maxX),
    maxY: Math.max(a.maxY, b.maxY),
  };
}

/** Balanced binary BVH of Morton-sorted bricks, with bounds from actual points. */
export function buildBVH2D(points, { brickSize = 16, mortonBits = 21 } = {}) {
  validatePoints(points);
  if (!Number.isSafeInteger(brickSize) || brickSize < 1) {
    throw new RangeError('brickSize must be a positive integer');
  }
  const sortedPoints = sortPointsByMorton2D(points, mortonBits);
  const brickCount = Math.ceil(sortedPoints.length / brickSize);
  let nextId = 0;

  function build(firstBrick, lastBrick, depth) {
    const id = nextId++;
    if (lastBrick - firstBrick === 1) {
      const start = firstBrick * brickSize;
      const end = Math.min(start + brickSize, sortedPoints.length);
      return {
        id, depth, leaf: true, brickIndex: firstBrick, start, end,
        bounds: actualBounds(sortedPoints, start, end),
        pointIndices: sortedPoints.slice(start, end).map((point) => point.index),
      };
    }
    const middle = firstBrick + Math.floor((lastBrick - firstBrick) / 2);
    const left = build(firstBrick, middle, depth + 1);
    const right = build(middle, lastBrick, depth + 1);
    return {
      id, depth, leaf: false,
      start: left.start, end: right.end,
      bounds: unionBounds(left.bounds, right.bounds),
      left, right,
    };
  }

  return {
    root: brickCount === 0 ? null : build(0, brickCount, 0),
    sortedPoints,
    brickSize,
    mortonBits,
    brickCount,
    nodeCount: nextId,
  };
}

export function minAabbDistanceSquared(bounds, qx, qy) {
  const dx = qx < bounds.minX ? bounds.minX - qx
    : qx > bounds.maxX ? qx - bounds.maxX : 0;
  const dy = qy < bounds.minY ? bounds.minY - qy
    : qy > bounds.maxY ? qy - bounds.maxY : 0;
  return dx * dx + dy * dy;
}

function validateQuery(points, qx, qy, radius) {
  if (!Array.isArray(points)) throw new TypeError('points must be an array');
  if (!Number.isFinite(qx) || !Number.isFinite(qy)) {
    throw new TypeError('query coordinates must be finite');
  }
  if (!Number.isFinite(radius) || radius < 0 || !Number.isFinite(radius * radius)) {
    throw new RangeError('radius must be nonnegative with a finite square');
  }
}

/** Strict point test d² < r²; conservative AABB prune only when Dmin² > r².
 * visitedNodes counts examined AABBs, including pruned ones. The point array
 * must remain unchanged after buildBVH2D: node bounds refer to those values.
 */
export function queryRadius(root, points, qx, qy, radius) {
  validateQuery(points, qx, qy, radius);
  const result = {
    visitedNodes: 0,
    prunedNodes: 0,
    openedBricks: 0,
    distanceCalculations: 0,
    hitIndices: [],
    inspectedAabbs: [],
  };
  if (root === null) return result;
  const radiusSquared = radius * radius;
  const stack = [root];
  while (stack.length > 0) {
    const node = stack.pop();
    result.visitedNodes += 1;
    const minimumDistanceSquared = minAabbDistanceSquared(node.bounds, qx, qy);
    const pruned = minimumDistanceSquared > radiusSquared;
    result.inspectedAabbs.push({
      id: node.id,
      depth: node.depth,
      bounds: node.bounds,
      leaf: node.leaf,
      pruned,
      minimumDistanceSquared,
    });
    if (pruned) {
      result.prunedNodes += 1;
      continue;
    }
    if (node.leaf) {
      result.openedBricks += 1;
      for (const index of node.pointIndices) {
        const point = points[index];
        const dx = point.x - qx;
        const dy = point.y - qy;
        const squaredDistance = dx * dx + dy * dy;
        result.distanceCalculations += 1;
        if (squaredDistance < radiusSquared) result.hitIndices.push(index);
      }
    } else {
      stack.push(node.right, node.left);
    }
  }
  result.hitIndices.sort((a, b) => a - b);
  return result;
}

export function bruteForceRadius(points, qx, qy, radius) {
  validateQuery(points, qx, qy, radius);
  const radiusSquared = radius * radius;
  const hits = [];
  for (let index = 0; index < points.length; index += 1) {
    const dx = points[index].x - qx;
    const dy = points[index].y - qy;
    if (dx * dx + dy * dy < radiusSquared) hits.push(index);
  }
  return hits;
}

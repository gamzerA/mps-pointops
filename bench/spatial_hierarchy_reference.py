"""Exact-arithmetic CPU reference for a Morton-ordered AABB hierarchy.

This is research code, not a public operator or a performance baseline. Morton
quantization changes only traversal order. Every box is the min/max of the
*stored points*, and all pruning comparisons use exact rational arithmetic
over the finite binary floating-point inputs. This makes the pruning argument
independent of float32 rounding, FMA contraction, and flush-to-zero behaviour.
An eventual Metal implementation needs its own conservative bound proof.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction

Point = tuple[float, float, float]
ExactPoint = tuple[Fraction, Fraction, Fraction]


def _point(value: Sequence[float]) -> Point:
    if len(value) != 3:
        raise ValueError("coordinates must have exactly three components")
    point = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(component) for component in point):
        raise ValueError("coordinates must be finite")
    return point


def _exact(point: Point) -> ExactPoint:
    return tuple(Fraction.from_float(component) for component in point)  # type: ignore[return-value]


def _distance_sq(a: ExactPoint, b: ExactPoint) -> Fraction:
    return sum(((a[axis] - b[axis]) ** 2 for axis in range(3)), Fraction(0))


def brute_radius_exact(
    points: Sequence[Sequence[float]], query: Sequence[float], radius: float,
    limit: int | None = None,
) -> list[int]:
    """Strict-radius oracle, retaining the first original indices."""
    q = _exact(_point(query))
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit == 0 or radius == 0:
        return []
    radius_sq = Fraction.from_float(float(radius)) ** 2
    found: list[int] = []
    for index, point in enumerate(points):
        if _distance_sq(q, _exact(_point(point))) < radius_sq:
            found.append(index)
            if len(found) == limit:
                break
    return found


def brute_knn_exact(
    points: Sequence[Sequence[float]], query: Sequence[float], k: int,
) -> list[int]:
    """Exact-distance oracle with the smallest original index on ties."""
    q = _exact(_point(query))
    if k < 0:
        raise ValueError("k must be nonnegative")
    return sorted(range(len(points)), key=lambda i: (
        _distance_sq(q, _exact(_point(points[i]))), i,
    ))[:k]


@dataclass(frozen=True)
class _Node:
    lo: Point
    hi: Point
    min_index: int
    left: int | None = None
    right: int | None = None
    indices: tuple[int, ...] = ()


class HierarchicalMortonReference:
    """Balanced hierarchy over Morton-sorted point bricks.

    The hierarchy is bounded by ``max_node_visits``. When that bound is hit,
    the method visibly falls back to an exact full scan. This balanced tree is
    a correctness prototype; it is not a Karras-style optimized LBVH build.
    """

    def __init__(
        self, points: Iterable[Sequence[float]], *, leaf_size: int = 64,
        bits_per_axis: int = 10, max_node_visits: int = 4096,
    ) -> None:
        if leaf_size < 1 or not 1 <= bits_per_axis <= 21 or max_node_visits < 1:
            raise ValueError("leaf_size and max_node_visits must be positive; bits_per_axis must be 1..21")
        self.points = tuple(_point(point) for point in points)
        self.leaf_size = leaf_size
        self.bits_per_axis = bits_per_axis
        self.max_node_visits = max_node_visits
        self.radius_fallbacks = 0
        self.knn_fallbacks = 0
        self.last_radius_nodes = 0
        self.last_knn_nodes = 0
        self._exact_points: dict[int, ExactPoint] = {}
        self.nodes: list[_Node] = []
        if not self.points:
            self.root: int | None = None
            return
        self._lo = tuple(min(p[axis] for p in self.points) for axis in range(3))
        self._hi = tuple(max(p[axis] for p in self.points) for axis in range(3))
        order = sorted(range(len(self.points)), key=lambda i: (self._morton_key(self.points[i]), i))
        leaves = [self._leaf(tuple(order[start:start + leaf_size]))
                  for start in range(0, len(order), leaf_size)]
        self.root = self._build(leaves, 0, len(leaves))

    def _morton_key(self, point: Point) -> int:
        quantized = []
        maximum = (1 << self.bits_per_axis) - 1
        for axis in range(3):
            lo = Fraction.from_float(self._lo[axis])
            hi = Fraction.from_float(self._hi[axis])
            if lo == hi:
                quantized.append(0)
            else:
                value = Fraction.from_float(point[axis])
                quantized.append(max(0, min(maximum, ((value - lo) * maximum) // (hi - lo))))
        key = 0
        for bit in range(self.bits_per_axis):
            for axis in range(3):
                key |= ((quantized[axis] >> bit) & 1) << (3 * bit + axis)
        return key

    def _leaf(self, indices: tuple[int, ...]) -> int:
        node = _Node(
            tuple(min(self.points[i][axis] for i in indices) for axis in range(3)),
            tuple(max(self.points[i][axis] for i in indices) for axis in range(3)),
            min(indices), indices=indices,
        )
        self.nodes.append(node)
        return len(self.nodes) - 1

    def _build(self, leaves: list[int], start: int, end: int) -> int:
        if end - start == 1:
            return leaves[start]
        middle = (start + end) // 2
        left, right = self._build(leaves, start, middle), self._build(leaves, middle, end)
        a, b = self.nodes[left], self.nodes[right]
        self.nodes.append(_Node(
            tuple(min(a.lo[axis], b.lo[axis]) for axis in range(3)),
            tuple(max(a.hi[axis], b.hi[axis]) for axis in range(3)),
            min(a.min_index, b.min_index), left=left, right=right,
        ))
        return len(self.nodes) - 1

    def _point_exact(self, index: int) -> ExactPoint:
        if index not in self._exact_points:
            self._exact_points[index] = _exact(self.points[index])
        return self._exact_points[index]

    @staticmethod
    def _box_lower_sq(q: ExactPoint, node: _Node) -> Fraction:
        """Exact minimum squared distance from q to the inclusive AABB."""
        result = Fraction(0)
        for axis in range(3):
            lo, hi = Fraction.from_float(node.lo[axis]), Fraction.from_float(node.hi[axis])
            if q[axis] < lo:
                result += (lo - q[axis]) ** 2
            elif q[axis] > hi:
                result += (q[axis] - hi) ** 2
        return result

    def radius(self, query: Sequence[float], radius: float, limit: int | None = None) -> list[int]:
        q = _exact(_point(query))
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative")
        self.last_radius_nodes = 0
        if self.root is None or limit == 0 or radius == 0:
            return []
        radius_sq = Fraction.from_float(float(radius)) ** 2
        queue: list[tuple[Fraction, int, int]] = [(self._box_lower_sq(q, self.nodes[self.root]),
                                                   self.nodes[self.root].min_index, self.root)]
        selected: list[int] = []  # Negative indices form a bounded max-index heap.
        while queue:
            if self.last_radius_nodes == self.max_node_visits:
                self.radius_fallbacks += 1
                return brute_radius_exact(self.points, query, radius, limit)
            bound, _, node_id = heapq.heappop(queue)
            self.last_radius_nodes += 1
            if bound >= radius_sq:
                continue
            node = self.nodes[node_id]
            if limit is not None and len(selected) == limit and node.min_index >= -selected[0]:
                continue
            if node.indices:
                for index in node.indices:
                    if limit is not None and len(selected) == limit and index >= -selected[0]:
                        continue
                    if _distance_sq(q, self._point_exact(index)) < radius_sq:
                        if limit is None:
                            selected.append(index)
                        elif len(selected) < limit:
                            heapq.heappush(selected, -index)
                        else:
                            heapq.heapreplace(selected, -index)
            else:
                assert node.left is not None and node.right is not None
                for child_id in (node.left, node.right):
                    child = self.nodes[child_id]
                    heapq.heappush(queue, (self._box_lower_sq(q, child), child.min_index, child_id))
        return sorted(selected if limit is None else (-index for index in selected))

    def knn(self, query: Sequence[float], k: int) -> list[int]:
        q = _exact(_point(query))
        if k < 0:
            raise ValueError("k must be nonnegative")
        self.last_knn_nodes = 0
        if self.root is None or k == 0:
            return []
        k = min(k, len(self.points))
        queue: list[tuple[Fraction, int, int]] = [(self._box_lower_sq(q, self.nodes[self.root]),
                                                   self.nodes[self.root].min_index, self.root)]
        best: list[tuple[Fraction, int]] = []  # (-distance², -original_index)
        while queue:
            if self.last_knn_nodes == self.max_node_visits:
                self.knn_fallbacks += 1
                return brute_knn_exact(self.points, query, k)
            bound, _, node_id = heapq.heappop(queue)
            self.last_knn_nodes += 1
            if len(best) == k and bound > -best[0][0]:
                continue  # Equality cannot be pruned: a smaller index may win.
            node = self.nodes[node_id]
            if node.indices:
                for index in node.indices:
                    candidate = (-_distance_sq(q, self._point_exact(index)), -index)
                    if len(best) < k:
                        heapq.heappush(best, candidate)
                    elif candidate > best[0]:
                        heapq.heapreplace(best, candidate)
            else:
                assert node.left is not None and node.right is not None
                for child_id in (node.left, node.right):
                    child = self.nodes[child_id]
                    heapq.heappush(queue, (self._box_lower_sq(q, child), child.min_index, child_id))
        return [index for _, index in sorted((-distance, -index) for distance, index in best)]

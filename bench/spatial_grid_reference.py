"""CPU-only uniform-grid research reference for the v0.9 spatial index.

This is deliberately outside the public package. It defines exact geometric
selection and original-index ordering for finite 3D points, and exposes when
the grid falls back to a full scan. The eventual Metal path must be checked
against both this index and the package's existing brute-force operators.
All distances here use Python binary64; near float32 boundaries need their
own numerical-contract tests before any public API uses this algorithm.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence

Point = tuple[float, float, float]
Cell = tuple[int, int, int]


def _point(value: Sequence[float]) -> Point:
    if len(value) != 3:
        raise ValueError("points must have three coordinates")
    result = (float(value[0]), float(value[1]), float(value[2]))
    if not all(math.isfinite(coordinate) for coordinate in result):
        raise ValueError("point coordinates must be finite")
    return result


def _squared_distance(a: Point, b: Point) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def brute_radius(
    points: Sequence[Sequence[float]], query: Sequence[float], radius: float,
    limit: int | None = None,
) -> list[int]:
    """Return the first in-input-order points at strict distance < radius."""
    q = _point(query)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be finite and nonnegative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    if limit == 0:
        return []
    radius_sq = radius * radius
    result: list[int] = []
    for index, value in enumerate(points):
        if _squared_distance(q, _point(value)) < radius_sq:
            result.append(index)
            if limit is not None and len(result) == limit:
                break
    return result


def brute_knn(points: Sequence[Sequence[float]], query: Sequence[float], k: int) -> list[int]:
    """Sort by squared distance, then original index on an exact tie."""
    q = _point(query)
    if k < 0:
        raise ValueError("k must be nonnegative")
    return sorted(range(len(points)), key=lambda i: (_squared_distance(q, _point(points[i])), i))[:k]


class UniformGridReference:
    """Sparse-cell CPU reference with conservative query coverage.

    `max_radius_cells` and `max_knn_shells` bound pathological sparse scans;
    crossing either bound uses the brute-force oracle. The counts make such
    fallback observable in the benchmark instead of hiding it.
    """

    def __init__(
        self, points: Iterable[Sequence[float]], cell_size: float, *,
        max_radius_cells: int = 4096, max_knn_shells: int = 16,
    ) -> None:
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be finite and positive")
        if max_radius_cells < 1 or max_knn_shells < 0:
            raise ValueError("cell scan limits must be nonnegative, with radius limit positive")
        self.points = tuple(_point(point) for point in points)
        self.cell_size = float(cell_size)
        self.max_radius_cells = max_radius_cells
        self.max_knn_shells = max_knn_shells
        self.radius_fallbacks = 0
        self.knn_fallbacks = 0
        self.cells: dict[Cell, list[int]] = defaultdict(list)
        if self.points:
            self.origin = tuple(min(point[axis] for point in self.points) for axis in range(3))
        else:
            self.origin = (0.0, 0.0, 0.0)
        for index, point in enumerate(self.points):
            self.cells[self._cell(point)].append(index)
        self.min_cell = tuple(min((cell[axis] for cell in self.cells), default=0) for axis in range(3))
        self.max_cell = tuple(max((cell[axis] for cell in self.cells), default=0) for axis in range(3))

    def _cell(self, point: Point) -> Cell:
        return tuple(math.floor((point[axis] - self.origin[axis]) / self.cell_size) for axis in range(3))  # type: ignore[return-value]

    def radius(self, query: Sequence[float], radius: float, limit: int | None = None) -> list[int]:
        q = _point(query)
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        if limit is not None and limit < 0:
            raise ValueError("limit must be nonnegative")
        if limit == 0 or radius == 0 or not self.points:
            return []
        # One extra cell on either side conservatively covers quantization
        # roundoff when a point lies on a cell boundary. This is a research
        # reference; the Metal key policy needs a separately proved bound.
        lo = self._cell(tuple(q[axis] - radius for axis in range(3)))
        hi = self._cell(tuple(q[axis] + radius for axis in range(3)))
        spans = [range(lo[axis] - 1, hi[axis] + 2) for axis in range(3)]
        candidate_cells = math.prod(len(span) for span in spans)
        if candidate_cells > self.max_radius_cells:
            self.radius_fallbacks += 1
            return brute_radius(self.points, q, radius, limit)
        candidate_indices: list[int] = []
        for cx in spans[0]:
            for cy in spans[1]:
                for cz in spans[2]:
                    candidate_indices.extend(self.cells.get((cx, cy, cz), ()))
        radius_sq = radius * radius
        result: list[int] = []
        for index in sorted(candidate_indices):
            if _squared_distance(q, self.points[index]) < radius_sq:
                result.append(index)
                if limit is not None and len(result) == limit:
                    break
        return result

    def knn(self, query: Sequence[float], k: int) -> list[int]:
        q = _point(query)
        if k < 0:
            raise ValueError("k must be nonnegative")
        if k == 0 or not self.points:
            return []
        k = min(k, len(self.points))
        center = self._cell(q)
        best: list[tuple[float, int]] = []  # max heap represented by negatives
        for shell in range(self.max_knn_shells + 1):
            for dx in range(-shell, shell + 1):
                for dy in range(-shell, shell + 1):
                    for dz in range(-shell, shell + 1):
                        if max(abs(dx), abs(dy), abs(dz)) != shell:
                            continue
                        cell = (center[0] + dx, center[1] + dy, center[2] + dz)
                        for index in self.cells.get(cell, ()):
                            pair = (-_squared_distance(q, self.points[index]), -index)
                            if len(best) < k:
                                heapq.heappush(best, pair)
                            elif pair > best[0]:
                                heapq.heapreplace(best, pair)

            # The shell cube now contains the entire index if its extents
            # enclose every occupied cell. This is exact regardless of float
            # lower-bound arithmetic.
            if all(center[axis] - shell <= self.min_cell[axis]
                   and center[axis] + shell >= self.max_cell[axis] for axis in range(3)):
                return [index for _, index in sorted((-distance, -index) for distance, index in best)]
            if len(best) == k and shell >= 1:
                # Unvisited cells are outside the visited cube. Shrink it by
                # one cell to account for boundary-key rounding. A strict
                # inequality keeps equal-distance, lower-index candidates.
                lower_bound = min(
                    min(q[axis] - (self.origin[axis] + (center[axis] - shell + 1) * self.cell_size),
                        (self.origin[axis] + (center[axis] + shell) * self.cell_size) - q[axis])
                    for axis in range(3)
                )
                if lower_bound > 0 and -best[0][0] < lower_bound * lower_bound:
                    return [index for _, index in sorted((-distance, -index) for distance, index in best)]
        self.knn_fallbacks += 1
        return brute_knn(self.points, q, k)

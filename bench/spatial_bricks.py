"""Experimental Morton-sorted point bricks with AABB-pruned kNN.

This is a research prototype, outside the public package API. It uses the
existing Metal Morton-key stage, stable MPS sort, and Metal brick build/query
stages. It agrees with the native full-scan kernel on the tested fixtures.
The floating-point safety margin needed for a general exact-pruning proof,
including FTZ and boundary rounding, is not yet established. Ties retained
by the traversal are ordered by smallest original index.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from bench.spatial_keys import morton_keys_f32

_SOURCE = Path(__file__).with_suffix(".metal")
_BRICK_SIZE = 128
_GROUP_SIZE = 128
_MAX_K = 32


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE.read_text())


@dataclass(frozen=True)
class MortonBrickIndex:
    """Single-cloud float32 research index; neither a public API nor LBVH.

    ``points`` is borrowed without a copy. Callers must not mutate it after
    build, because the sorted order and brick bounds would become stale.
    """

    points: Tensor
    sorted_indices: Tensor
    brick_bounds: Tensor
    cell_size: float

    @classmethod
    def build(cls, points: Tensor, origin: Tensor, cell_size: float) -> "MortonBrickIndex":
        keys, invalid = morton_keys_f32(points, origin, cell_size)
        if bool(invalid.any().item()):
            raise ValueError("reference points must be finite and fit the 21-bit Morton domain")
        permutation = torch.argsort(keys, stable=True).long()
        brick_count = (len(points) + _BRICK_SIZE - 1) // _BRICK_SIZE
        bounds = torch.empty((brick_count, 6), dtype=torch.float32, device=points.device)
        if brick_count:
            groups = (brick_count + _GROUP_SIZE - 1) // _GROUP_SIZE
            _library().spatial_brick_bounds_f32(
                points.contiguous(), permutation.contiguous(), bounds,
                len(points), brick_count,
                threads=[groups * _GROUP_SIZE, 1, 1],
                group_size=[_GROUP_SIZE, 1, 1],
            )
        return cls(points, permutation, bounds, cell_size)

    def knn(self, query: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Return squared float32 distance and original index, padded inf/-1.

        Rows are ordered by (squared distance, original index), and the
        strict AABB comparison is intended to retain ties. Boundary pruning
        has no general floating-point proof yet. Supports finite float32 XYZ
        and at most 32 neighbors in this prototype.
        """
        if query.device != self.points.device or query.dtype != torch.float32 or query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query must be MPS float32 [Q, 3]")
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= _MAX_K:
            raise ValueError(f"k must be an integer in [0, {_MAX_K}]")
        distances = torch.empty((len(query), k), dtype=torch.float32, device=query.device)
        indices = torch.empty((len(query), k), dtype=torch.int64, device=query.device)
        if len(query) and k:
            groups = (len(query) + _GROUP_SIZE - 1) // _GROUP_SIZE
            _library().spatial_brick_knn_f32(
                query.contiguous(), self.points.contiguous(), self.sorted_indices,
                self.brick_bounds, distances, indices,
                len(query), len(self.points), len(self.brick_bounds), k,
                threads=[groups * _GROUP_SIZE, 1, 1],
                group_size=[_GROUP_SIZE, 1, 1],
            )
        return distances, indices

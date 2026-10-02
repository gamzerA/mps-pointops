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
import os

import torch
from torch import Tensor

from bench.spatial_keys import morton_keys_f32

_SOURCE = Path(__file__).with_suffix(".metal")
_BRICK_SIZE = 128
_GROUP_SIZE = 128
_MAX_K = 32
_MAX_ABS_BITS = 0x5D800000  # IEEE binary32 2**60
# PyTorch reads this process-level switch while initializing its Metal shader
# compiler. If it is absent or Fast, use the complete brick scan. This is a
# conservative *research* policy, not a claim about all compiler versions.
_PRUNE_AABBS = os.environ.get("PYTORCH_MPS_FAST_MATH") == "0"


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE.read_text())


def _validate_f32_domain(value: Tensor, name: str) -> None:
    """Reject nonfinite, subnormal, and overflow-prone values by raw bits.

    Integer reinterpretation keeps the check meaningful in Fast Math, where
    a float comparison with a subnormal operand could itself be flushed.
    """
    if not value.numel():
        return
    magnitude = value.contiguous().view(torch.int32).bitwise_and(0x7FFFFFFF)
    bad = ((magnitude >= 0x7F800000) | (magnitude > _MAX_ABS_BITS)
           | ((magnitude != 0) & (magnitude < 0x00800000)))
    if bool(bad.any().item()):
        raise ValueError(f"{name} must be finite normal-or-zero float32 with |coordinate| <= 2**60; subnormal coordinates are unsupported")


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
    pruning_enabled: bool
    point_version: int

    @classmethod
    def build(cls, points: Tensor, origin: Tensor, cell_size: float) -> "MortonBrickIndex":
        if points.dtype == torch.float32 and points.device.type == "mps":
            _validate_f32_domain(points, "reference points")
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
        return cls(points, permutation, bounds, cell_size, _PRUNE_AABBS, points._version)

    def _search(self, query: Tensor, k: int, *, audit: bool) -> tuple[Tensor, Tensor, Tensor]:
        """Run kNN; audit counts pruned bricks and would-be winning points."""
        if query.device != self.points.device or query.dtype != torch.float32 or query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query must be MPS float32 [Q, 3]")
        if self.points._version != self.point_version:
            raise ValueError("reference tensor changed after spatial index construction")
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= _MAX_K:
            raise ValueError(f"k must be an integer in [0, {_MAX_K}]")
        if len(query) >= 2**32:
            raise ValueError("query count must be below 2**32")
        _validate_f32_domain(query, "queries")
        distances = torch.empty((len(query), k), dtype=torch.float32, device=query.device)
        indices = torch.empty((len(query), k), dtype=torch.int64, device=query.device)
        audit_counts = torch.zeros((len(query), 2), dtype=torch.int32, device=query.device)
        if len(query) and k:
            groups = (len(query) + _GROUP_SIZE - 1) // _GROUP_SIZE
            _library().spatial_brick_knn_f32(
                query.contiguous(), self.points.contiguous(), self.sorted_indices,
                self.brick_bounds, distances, indices, audit_counts,
                len(query), len(self.points), len(self.brick_bounds), k,
                int(self.pruning_enabled), int(audit),
                threads=[groups * _GROUP_SIZE, 1, 1],
                group_size=[_GROUP_SIZE, 1, 1],
            )
        return distances, indices, audit_counts

    def knn(self, query: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Return squared distances and indices with inf/-1 padding.

        Rows use (rounded squared distance, original index) order. Safe mode
        prunes using the conditional monotonicity proof in the spatial report;
        Fast mode scans every brick.
        """
        distances, indices, _ = self._search(query, k, audit=False)
        return distances, indices

    def audit_pruning(self, query: Tensor, k: int) -> Tensor:
        """Debug-only [pruned bricks, would-be winning points] per query."""
        _, _, counts = self._search(query, k, audit=True)
        return counts

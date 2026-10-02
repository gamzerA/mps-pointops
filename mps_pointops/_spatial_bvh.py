"""Private Safe-Math-only two-level BVH for exact float32 Metal search.

Morton keys determine point order only; AABBs use original stored coordinates.
The conditional Safe-Math pruning argument is documented in the v0.9 spatial
report. Fast Math is rejected until an independent conservative bound exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
import math
import os

import torch
from torch import Tensor

from ._spatial_keys import morton_keys_f32
from ._ball_query_mps import _checked_radius_and_k

_BRICK_SIZE = 128
_MICRO_LEAVES = 64
_GROUP_SIZE = 128
_MAX_K = 32
_MAX_ABS_BITS = 0x5D800000  # float32 encoding of 2**60


@lru_cache(maxsize=1)
def _library():
    source = resources.files(__package__).joinpath("kernels", "spatial_bvh.metal").read_text()
    return torch.mps.compile_shader(source)


def _validate_coordinate_domain(tensor: Tensor, name: str) -> None:
    if tensor.numel() == 0:
        return
    bits = tensor.contiguous().view(torch.int32).bitwise_and(0x7FFFFFFF)
    bad = ((bits >= 0x7F800000) | (bits > _MAX_ABS_BITS)
           | ((bits != 0) & (bits < 0x00800000)))
    if bool(bad.any().item()):
        raise ValueError(f"{name} must be finite normal-or-zero float32 with abs <= 2**60")


def _padded_leaf_count(n: int) -> int:
    return 1 << (max(_MICRO_LEAVES, (n + _BRICK_SIZE - 1) // _BRICK_SIZE) - 1).bit_length()


@dataclass(frozen=True)
class MortonTwoLevelBVH:
    """One cloud, float32 XYZ, K<=32, with bounded DFS and explicit fallback.

    The reference tensor is borrowed. Its tracked version must not change
    after construction; external raw-buffer mutation is outside the contract.
    """

    points: Tensor
    sorted_indices: Tensor
    bounds: Tensor
    min_index: Tensor
    padded_leaves: int
    point_version: int

    @classmethod
    def build(cls, points: Tensor, origin: Tensor, cell_size: float) -> "MortonTwoLevelBVH":
        if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
            raise RuntimeError("two-level BVH pruning requires PYTORCH_MPS_FAST_MATH=0 before Python starts")
        if points.device.type != "mps" or points.dtype != torch.float32 or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must be MPS float32 [N, 3]")
        if origin.device != points.device or origin.dtype != torch.float32 or origin.shape != (3,):
            raise ValueError("origin must be MPS float32 [3]")
        if not math.isfinite(cell_size) or cell_size <= 0:
            raise ValueError("cell_size must be positive and finite")
        if len(points) > 1_000_000:
            raise ValueError("prototype supports at most 1,000,000 points")
        _validate_coordinate_domain(points, "reference points")
        keys, invalid = morton_keys_f32(points, origin, cell_size)
        if bool(invalid.any().item()):
            raise ValueError("reference points must fit the Morton domain")
        sorted_indices = torch.argsort(keys, stable=True).long().contiguous()
        padded = _padded_leaf_count(len(points))
        bounds = torch.empty((2 * padded, 6), dtype=torch.float32, device=points.device)
        min_index = torch.empty((2 * padded,), dtype=torch.int32, device=points.device)
        microtrees = padded // _MICRO_LEAVES
        _library().bvh_build_micro_f32(
            points.contiguous(), sorted_indices, bounds, min_index,
            len(points), padded,
            threads=[microtrees * _MICRO_LEAVES, 1, 1],
            group_size=[_MICRO_LEAVES, 1, 1],
        )
        if microtrees > 1:
            _library().bvh_build_macro_f32(
                bounds, min_index, microtrees,
                threads=[_GROUP_SIZE, 1, 1], group_size=[_GROUP_SIZE, 1, 1],
            )
        return cls(points, sorted_indices, bounds, min_index, padded, points._version)

    def knn(self, query: Tensor, k: int, *, audit: bool = False,
            parallel_microtrees: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
            raise RuntimeError("two-level BVH query requires Safe Math")
        if query.device != self.points.device or query.dtype != torch.float32 or query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query must be MPS float32 [Q, 3]")
        if self.points._version != self.point_version:
            raise ValueError("reference tensor was changed after BVH construction")
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= _MAX_K:
            raise ValueError(f"k must be in [0, {_MAX_K}]")
        if len(query) >= 2**32:
            raise ValueError("prototype supports fewer than 2**32 queries")
        if parallel_microtrees and (len(query) > 256 or audit):
            raise ValueError("parallel microtrees support Q<=256 and no full prune audit")
        _validate_coordinate_domain(query, "query points")
        distances = torch.empty((len(query), k), dtype=torch.float32, device=query.device)
        indices = torch.empty((len(query), k), dtype=torch.int64, device=query.device)
        stats = torch.zeros((len(query), 5), dtype=torch.int32, device=query.device)
        if len(query) and k:
            groups = (len(query) + _GROUP_SIZE - 1) // _GROUP_SIZE
            if parallel_microtrees:
                microtrees = self.padded_leaves // _MICRO_LEAVES
                task_count = len(query) * microtrees
                task_groups = (task_count + _GROUP_SIZE - 1) // _GROUP_SIZE
                seed_threshold = torch.empty((len(query),), dtype=torch.float32, device=query.device)
                local_dist = torch.empty((task_count, k), dtype=torch.float32, device=query.device)
                local_idx = torch.empty((task_count, k), dtype=torch.int32, device=query.device)
                local_count = torch.empty((task_count,), dtype=torch.int32, device=query.device)
                local_stats = torch.empty((task_count, 4), dtype=torch.int32, device=query.device)
                _library().bvh_seed_f32(
                    query.contiguous(), self.points.contiguous(), self.sorted_indices,
                    self.bounds, self.min_index, seed_threshold,
                    len(query), len(self.points), self.padded_leaves, k,
                    threads=[groups * _GROUP_SIZE, 1, 1], group_size=[_GROUP_SIZE, 1, 1],
                )
                _library().bvh_search_micro_f32(
                    query.contiguous(), self.points.contiguous(), self.sorted_indices,
                    self.bounds, self.min_index, seed_threshold,
                    local_dist, local_idx, local_count, local_stats,
                    len(query), len(self.points), self.padded_leaves, k,
                    threads=[task_groups * _GROUP_SIZE, 1, 1], group_size=[_GROUP_SIZE, 1, 1],
                )
                _library().bvh_merge_micro_f32(
                    local_dist, local_idx, local_count, local_stats,
                    distances, indices, stats, len(query), microtrees, k,
                    threads=[groups * _GROUP_SIZE, 1, 1], group_size=[_GROUP_SIZE, 1, 1],
                )
            else:
                _library().bvh_knn_f32(
                    query.contiguous(), self.points.contiguous(), self.sorted_indices,
                    self.bounds, self.min_index, distances, indices, stats,
                    len(query), len(self.points), self.padded_leaves, k, int(audit),
                    threads=[groups * _GROUP_SIZE, 1, 1],
                    group_size=[_GROUP_SIZE, 1, 1],
                )
        return distances, indices, stats

    def radius(self, query: Tensor, radius: float, limit: int = 32,
               *, audit: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """First ``limit`` strict-radius hits in original point order.

        Returns squared distances, int64 indices, and per-query counters in
        the same order as ``knn``. Missing slots are ``(0, -1)``. This private
        research path supports one float32 cloud and Safe Math only; it does
        not replace the public batched Ball Query operator.

        Stats columns count visited nodes, inspected points, pruned nodes,
        missed qualifying points in a debug-only prune audit, and bounded-stack
        fallbacks. In the normalized small-radius branch, only the exact
        original-index certificate is used for pruning; no floating AABB bound
        is trusted. A stack overflow resets the row and scans all points.
        """
        if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
            raise RuntimeError("two-level BVH query requires Safe Math")
        if query.device != self.points.device or query.dtype != torch.float32 or query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query must be MPS float32 [Q, 3]")
        if self.points._version != self.point_version:
            raise ValueError("reference tensor was changed after BVH construction")
        radius_f32, radius_sq = _checked_radius_and_k(radius, limit)
        if limit > _MAX_K:
            raise ValueError(f"limit must be in [0, {_MAX_K}]")
        if len(query) >= 2**32:
            raise ValueError("prototype supports fewer than 2**32 queries")
        _validate_coordinate_domain(query, "query points")
        distances = torch.empty((len(query), limit), dtype=torch.float32, device=query.device)
        indices = torch.empty((len(query), limit), dtype=torch.int64, device=query.device)
        stats = torch.zeros((len(query), 5), dtype=torch.int32, device=query.device)
        if len(query) and limit:
            groups = (len(query) + _GROUP_SIZE - 1) // _GROUP_SIZE
            _library().bvh_radius_f32(
                query.contiguous(), self.points.contiguous(), self.sorted_indices,
                self.bounds, self.min_index, distances, indices, stats,
                len(query), len(self.points), self.padded_leaves, limit,
                radius_f32, radius_sq, int(audit),
                threads=[groups * _GROUP_SIZE, 1, 1],
                group_size=[_GROUP_SIZE, 1, 1],
            )
        return distances, indices, stats

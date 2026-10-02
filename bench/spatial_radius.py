"""Experimental sorted-Morton radius index on MPS, outside the public API.

The builder uses the Metal key stage plus PyTorch's MPS stable sort. The query
uses a second Metal dispatch. This is not yet a release implementation: it is
single-cloud float32, has an explicit bounded-cell fallback status, and needs
large-scale parity and timing evidence before integration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor

from bench.spatial_keys import morton_keys_f32
from mps_pointops._flat_search_mps import _torch_cluster_radius_sq

_SOURCE = Path(__file__).with_suffix(".metal")
_GROUP_SIZE = 256


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(_SOURCE.read_text())


@dataclass(frozen=True)
class SortedMortonIndex:
    points: Tensor
    origin: Tensor
    cell_size: float
    sorted_keys: Tensor
    sorted_indices: Tensor

    @classmethod
    def build(cls, points: Tensor, origin: Tensor, cell_size: float) -> "SortedMortonIndex":
        keys, invalid = morton_keys_f32(points, origin, cell_size)
        if bool(invalid.any().item()):
            raise ValueError("reference points must be finite and fit the 21-bit Morton domain")
        permutation = torch.argsort(keys, stable=True)
        return cls(points, origin, cell_size, keys[permutation], permutation.long())

    def radius(self, query: Tensor, radius: float, limit: int = 32) -> tuple[Tensor, Tensor]:
        if query.device != self.points.device or query.dtype != torch.float32 or query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query must be MPS float32 [Q, 3]")
        if not math.isfinite(radius) or radius < 0:
            raise ValueError("radius must be finite and nonnegative")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 0 <= limit <= 256:
            raise ValueError("limit must be an integer in [0, 256]")
        output = torch.empty((len(query), limit), dtype=torch.int64, device=query.device)
        status = torch.empty((len(query),), dtype=torch.uint8, device=query.device)
        if len(query):
            groups = (len(query) + _GROUP_SIZE - 1) // _GROUP_SIZE
            _library().spatial_radius_first_k_f32(
                query.contiguous(), self.points.contiguous(), self.sorted_keys,
                self.sorted_indices, self.origin.contiguous(), output, status,
                len(query), len(self.points), limit, float(radius),
                _torch_cluster_radius_sq(float(radius)), float(self.cell_size),
                threads=[groups * _GROUP_SIZE, 1, 1],
                group_size=[_GROUP_SIZE, 1, 1],
            )
        return output, status

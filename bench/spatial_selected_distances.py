"""Independent Metal distance probe for selected kNN pairs."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from torch import Tensor


@lru_cache(maxsize=1)
def _library():
    return torch.mps.compile_shader(Path(__file__).with_suffix(".metal").read_text())


def selected_sqdist_f32(query: Tensor, points: Tensor, indices: Tensor) -> Tensor:
    if (query.device.type != "mps" or points.device != query.device
            or indices.device != query.device or query.dtype != torch.float32
            or points.dtype != torch.float32 or indices.dtype != torch.int64
            or query.ndim != 2 or query.shape[1] != 3
            or points.ndim != 2 or points.shape[1] != 3
            or indices.ndim != 2 or indices.shape[0] != len(query)):
        raise ValueError("expected MPS float32 query/ref [*,3] and int64 indices [Q,K]")
    out = torch.empty(indices.shape, dtype=torch.float32, device=query.device)
    if indices.numel():
        group = 128
        groups = (indices.numel() + group - 1) // group
        _library().spatial_selected_sqdist_f32(
            query.contiguous(), points.contiguous(), indices.contiguous(), out,
            len(query), indices.shape[1], len(points),
            threads=[groups * group, 1, 1], group_size=[group, 1, 1],
        )
    return out

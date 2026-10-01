"""Metal searches over sorted, flattened point-cloud batches.

These private kernels produce rectangular global-index arrays with ``-1``
padding. The public flat API removes padding to build torch_cluster-style
``[query, reference]`` edge indices.
"""

from __future__ import annotations

from functools import cache
from importlib import resources
import math
import struct

import torch
from torch import Tensor

from ._ball_query_mps import _checked_radius_and_k, _require_compile_shader
from .ops import _check_simd_width


_MAX_K = 256
_QUERIES_PER_GROUP = 8
_UINT_MAX = (1 << 32) - 1


@cache
def _library():
    _require_compile_shader()
    source = resources.files(__package__).joinpath("kernels", "flat_search.metal").read_text()
    return torch.mps.compile_shader(source)


def _check_inputs(x: Tensor, y: Tensor, ptr_x: Tensor, ptr_y: Tensor) -> tuple[int, int]:
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != 3 or y.shape[1] != 3:
        raise ValueError("x and y must have shapes (N, 3) and (M, 3)")
    if x.device.type != "mps" or y.device != x.device:
        raise ValueError("x and y must be on the same MPS device")
    if ptr_x.ndim != 1 or ptr_y.ndim != 1 or ptr_x.shape != ptr_y.shape or not ptr_x.numel():
        raise ValueError("ptr_x and ptr_y must have the same nonempty 1-D shape")
    for name, ptr, count in (("ptr_x", ptr_x, x.shape[0]), ("ptr_y", ptr_y, y.shape[0])):
        if ptr.dtype != torch.int64 or ptr.device != x.device:
            raise ValueError(f"{name} must be an int64 tensor on {x.device}")
        if int(ptr[0].item()) != 0 or int(ptr[-1].item()) != count:
            raise ValueError(f"{name} must start at 0 and end at {count}")
        lengths = ptr[1:] - ptr[:-1]
        if bool(torch.any((lengths < 0) | (lengths > _UINT_MAX)).item()):
            raise ValueError(f"{name} must be nondecreasing with per-batch size <= 2**32-1")
    if y.shape[0] > _UINT_MAX:
        raise ValueError("flat query count exceeds the Metal 1-D dispatch limit")
    return int(ptr_x.numel() - 1), int(y.shape[0])


def _torch_cluster_radius_sq(r: float) -> float:
    """The threshold torch_cluster compares against: fl32(r * r), r * r in double.

    torch_cluster takes ``r`` as a double and passes ``r * r`` to a float32
    kernel. PyTorch3D-style Ball Query instead squares the float32-rounded
    radius; the two can differ by one float32 ULP (for example at r = 0.21).
    """
    try:
        return struct.unpack("f", struct.pack("f", r * r))[0]
    except OverflowError:
        return math.inf


def knn_indices(x: Tensor, y: Tensor, ptr_x: Tensor, ptr_y: Tensor, k: int) -> Tensor:
    """Return global x indices, sorted by (squared distance, x index)."""
    batch_count, query_count = _check_inputs(x, y, ptr_x, ptr_y)
    if x.dtype != torch.float32 or y.dtype != torch.float32:
        raise TypeError("flat Metal kNN requires float32 x and y")
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= _MAX_K:
        raise ValueError(f"k must be an integer in [0, {_MAX_K}] for the flat Metal kernel")
    if query_count * k > (1 << 63) - 1:
        raise ValueError("kNN output size exceeds int64 indexing")
    out = torch.empty((query_count, k), dtype=torch.int64, device=x.device)
    if query_count == 0 or k == 0:
        return out
    _check_simd_width()
    group = 32 * _QUERIES_PER_GROUP
    groups = (query_count + _QUERIES_PER_GROUP - 1) // _QUERIES_PER_GROUP
    _library().flat_knn_indices(
        y.contiguous(), x.contiguous(), ptr_y.contiguous(), ptr_x.contiguous(), out,
        query_count, batch_count, k,
        threads=[groups * group, 1, 1], group_size=[group, 1, 1],
    )
    return out


def radius_indices(
    x: Tensor, y: Tensor, ptr_x: Tensor, ptr_y: Tensor, r: float,
    max_num_neighbors: int, ignore_same_index: bool = False,
) -> Tensor:
    """Return first valid global x indices per query, with ``-1`` padding.

    When requested, equal global x/y index numbers are excluded before the
    neighbor cap is applied, matching pyg-lib's radius operator.
    """
    batch_count, query_count = _check_inputs(x, y, ptr_x, ptr_y)
    if x.dtype not in (torch.float32, torch.float16) or y.dtype != x.dtype:
        raise TypeError("flat Metal radius requires matching float32 or float16 x and y")
    radius_f32, _ = _checked_radius_and_k(r, max_num_neighbors)
    radius_sq = _torch_cluster_radius_sq(float(r))
    if query_count * max_num_neighbors > (1 << 63) - 1:
        raise ValueError("radius output size exceeds int64 indexing")
    out = torch.empty((query_count, max_num_neighbors), dtype=torch.int64, device=x.device)
    if query_count == 0 or max_num_neighbors == 0:
        return out
    if radius_f32 == 0.0:
        out.fill_(-1)
        return out
    _check_simd_width()
    kernel = (
        _library().flat_radius_indices_f32
        if x.dtype == torch.float32
        else _library().flat_radius_indices_f16
    )
    group = 32 * _QUERIES_PER_GROUP
    groups = (query_count + _QUERIES_PER_GROUP - 1) // _QUERIES_PER_GROUP
    kernel(
        y.contiguous(), x.contiguous(), ptr_y.contiguous(), ptr_x.contiguous(), out,
        query_count, batch_count, max_num_neighbors, radius_sq, radius_f32,
        int(ignore_same_index),
        threads=[groups * group, 1, 1], group_size=[group, 1, 1],
    )
    return out

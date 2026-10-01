"""Legacy ``torch_cluster.nearest`` style cluster assignment.

For every row of ``x``, return the global index of the nearest row of ``y``
in the same sorted batch. The MPS path uses a dedicated Metal kernel because
the legacy CUDA operator's finite ``1e38`` initial distance and fallback
index zero differ from the project's kNN padding contract.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import torch
from torch import Tensor

from ._ball_query_mps import _require_compile_shader
from .flat import _check_batch, _features, _ptr_from_batch
from .ops import _check_simd_width


_CUDA_INITIAL_DISTANCE = 1e38
_QUERIES_PER_GROUP = 8
_UINT_MAX = (1 << 32) - 1


@cache
def _library():
    _require_compile_shader()
    source = resources.files(__package__).joinpath("kernels", "nearest.metal").read_text()
    return torch.mps.compile_shader(source)


def _input(value: Tensor, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim == 1:
        value = value.view(-1, 1)
    _features(value, name)
    return value


def _paired_ptrs(
    x: Tensor, y: Tensor, batch_x: Tensor | None, batch_y: Tensor | None
) -> tuple[Tensor, Tensor]:
    size_x = _check_batch(batch_x, len(x), x.device, "batch_x")
    size_y = _check_batch(batch_y, len(y), y.device, "batch_y")
    size = max(1, size_x, size_y)
    if size > 1 and (batch_x is None or batch_y is None):
        raise ValueError("Some batch indices occur in 'batch_x' that do not occur in 'batch_y'")
    ptr_x = _ptr_from_batch(batch_x, len(x), x.device, size)
    ptr_y = _ptr_from_batch(batch_y, len(y), y.device, size)
    if not torch.equal(ptr_x[1:] > ptr_x[:-1], ptr_y[1:] > ptr_y[:-1]):
        raise ValueError("Some batch indices occur in 'batch_x' that do not occur in 'batch_y'")
    return ptr_x.contiguous(), ptr_y.contiguous()


def _nearest_cpu(x: Tensor, y: Tensor, ptr_x: Tensor, ptr_y: Tensor) -> Tensor:
    """Deterministic reference for the CUDA threshold and 1024-lane ties."""
    result = torch.zeros(len(x), dtype=torch.long, device=x.device)
    x_offsets = ptr_x.tolist()
    y_offsets = ptr_y.tolist()
    for (x_lo, x_hi), (y_lo, y_hi) in zip(
        zip(x_offsets, x_offsets[1:]), zip(y_offsets, y_offsets[1:])
    ):
        refs = y[y_lo:y_hi]
        for row in range(x_lo, x_hi):
            delta = x[row] - refs
            dist = delta[:, 0] * delta[:, 0]
            for d in range(1, x.shape[1]):
                dist = dist + delta[:, d] * delta[:, d]
            eligible = torch.nonzero(dist < _CUDA_INITIAL_DISTANCE).flatten()
            if eligible.numel():
                tied = eligible[dist[eligible] == torch.min(dist[eligible])]
                lane = tied % 1024
                local = tied[lane == lane.min()].min()
                result[row] = y_lo + local
    return result


def nearest(
    x: Tensor,
    y: Tensor,
    batch_x: Tensor | None = None,
    batch_y: Tensor | None = None,
) -> Tensor:
    """Map each ``x`` row to a global ``y`` index in its sorted batch.

    One-dimensional inputs are treated as one-feature rows. Both batch vectors
    must describe the same nonempty batch IDs, but IDs may have empty gaps.
    Float32 MPS inputs use native Metal; unsupported MPS dtypes fail explicitly.
    A candidate is accepted only when its squared distance is less than 1e38,
    matching the legacy CUDA initialization. If all candidates are rejected,
    the legacy CUDA fallback index zero is returned even for a later batch.
    For equal distances beyond 1024 references, the original CUDA reduction
    chooses the lowest stride-1024 lane before the earliest index in that lane.
    Empty ``x`` and ``y`` together yield an empty result; a batch present in
    only one input raises ValueError. Indices are not differentiable.
    """
    x = _input(x, "x")
    y = _input(y, "y")
    if x.shape[1] != y.shape[1]:
        raise ValueError(f"x and y feature dimensions differ: {x.shape[1]} and {y.shape[1]}")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} and {y.device}")
    if x.dtype != y.dtype:
        raise TypeError(f"x and y must have the same dtype, got {x.dtype} and {y.dtype}")
    if x.device.type not in ("cpu", "mps"):
        raise NotImplementedError("nearest shim supports CPU and native MPS tensors only")
    if x.device.type == "mps" and x.dtype != torch.float32:
        raise TypeError("x and y must be float32 for nearest on MPS")
    ptr_x, ptr_y = _paired_ptrs(x, y, batch_x, batch_y)
    if len(x) == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)
    if x.device.type == "cpu":
        return _nearest_cpu(x, y, ptr_x, ptr_y)
    if len(x) > _UINT_MAX or x.shape[1] >= 2**32:
        raise ValueError("nearest MPS query count and feature dimension must fit in uint32")
    if len(ptr_x) - 1 >= 2**32 or bool(((ptr_y[1:] - ptr_y[:-1]) > _UINT_MAX).any().item()):
        raise ValueError("nearest MPS batch count and per-batch reference count must fit in uint32")
    _check_simd_width()
    out = torch.empty(len(x), dtype=torch.long, device=x.device)
    group = 32 * _QUERIES_PER_GROUP
    groups = (len(x) + _QUERIES_PER_GROUP - 1) // _QUERIES_PER_GROUP
    _library().nearest_flat_f32(
        x.contiguous(), y.contiguous(), ptr_x, ptr_y, out,
        len(x), len(ptr_x) - 1, x.shape[1],
        threads=[groups * group, 1, 1], group_size=[group, 1, 1],
    )
    return out

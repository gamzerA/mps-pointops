"""PointNet++ three-neighbor search and weighted feature propagation.

The public shapes follow ``pointnet2_ops.pointnet2_utils``. This module is an
independent implementation; no PointNet++ source code is included.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ._ball_query_mps import _require_compile_shader

_GROUP_SIZE = 256
_MAX_INDEX = (1 << 31) - 1
_REFERENCE_PAIRS_PER_CHUNK = 1 << 24


@cache
def _library():
    _require_compile_shader()
    source = resources.files(__package__).joinpath("kernels", "pointnet2.metal").read_text()
    return torch.mps.compile_shader(source)


def _validate_three_nn(unknown: Tensor, known: Tensor) -> tuple[int, int, int]:
    if not isinstance(unknown, Tensor) or not isinstance(known, Tensor):
        raise TypeError("unknown and known must be tensors")
    if unknown.ndim != 3 or unknown.shape[-1] != 3:
        raise ValueError("unknown must have shape (B, N, 3)")
    if known.ndim != 3 or known.shape[-1] != 3:
        raise ValueError("known must have shape (B, M, 3)")
    if unknown.shape[0] != known.shape[0]:
        raise ValueError("unknown and known must have the same batch size")
    if unknown.device != known.device:
        raise ValueError("unknown and known must be on the same device")
    if unknown.dtype != torch.float32 or known.dtype != torch.float32:
        raise TypeError("unknown and known must be float32")
    batch, queries, _ = unknown.shape
    points = known.shape[1]
    if batch and queries and points < 3:
        raise ValueError("three_nn needs at least 3 known points for nonempty queries")
    if points > _MAX_INDEX:
        raise ValueError("three_nn supports at most 2**31-1 known points")
    if batch * queries > (1 << 32) - 1:
        raise ValueError("three_nn query count exceeds the native index range")
    return batch, queries, points


def _three_nn_reference(unknown: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
    """Chunked PyTorch reference with stable input-index tie order."""
    batch, queries, _ = unknown.shape
    points = known.shape[1]
    distances = torch.empty((batch, queries, 3), dtype=torch.float32, device=unknown.device)
    indices = torch.empty((batch, queries, 3), dtype=torch.int32, device=unknown.device)
    if not batch or not queries:
        return distances, indices
    step = max(1, _REFERENCE_PAIRS_PER_CHUNK // max(1, batch * points))
    with torch.no_grad():
        for lo in range(0, queries, step):
            delta = unknown[:, lo:lo + step, None] - known[:, None]
            d2 = delta[..., 0] * delta[..., 0]
            d2 = d2 + delta[..., 1] * delta[..., 1]
            d2 = d2 + delta[..., 2] * delta[..., 2]
            order = torch.argsort(d2, dim=-1, stable=True)[..., :3]
            indices[:, lo:lo + step] = order.to(torch.int32)
            distances[:, lo:lo + step] = d2.gather(-1, order).sqrt()
    return distances, indices


def three_nn(unknown: Tensor, known: Tensor) -> tuple[Tensor, Tensor]:
    """Return three nearest known points for every unknown point.

    ``unknown`` is ``(B, N, 3)`` and ``known`` is ``(B, M, 3)`` float32, with
    ``M >= 3`` whenever ``B*N > 0``. Returns Euclidean distances ``(B, N, 3)``
    in float32 and known indices ``(B, N, 3)`` in int32. Equal squared
    distances prefer the smaller known index. As in PointNet++, neighbor
    selection and returned distances have no coordinate gradient. Coordinates
    should be finite; roundoff can change near ties between devices.

    CPU tensors use the pure PyTorch reference. MPS tensors stay on MPS and
    use one native Metal kernel. Empty query or batch dimensions return empty
    outputs without launching a kernel.
    """
    batch, queries, points = _validate_three_nn(unknown, known)
    if unknown.device.type != "mps":
        return _three_nn_reference(unknown, known)
    distances = torch.empty((batch, queries, 3), dtype=torch.float32, device=unknown.device)
    indices = torch.empty((batch, queries, 3), dtype=torch.int32, device=unknown.device)
    count = batch * queries
    if count:
        _library().three_nn_f32(
            unknown.contiguous(), known.contiguous(), distances, indices,
            queries, points, threads=count, group_size=min(count, _GROUP_SIZE),
        )
    return distances, indices


def _validate_interpolate(features: Tensor, indices: Tensor, weights: Tensor) -> tuple[int, int, int, int]:
    if not all(isinstance(t, Tensor) for t in (features, indices, weights)):
        raise TypeError("features, indices, and weights must be tensors")
    if features.ndim != 3:
        raise ValueError("features must have shape (B, C, M)")
    if indices.ndim != 3 or indices.shape[-1] != 3:
        raise ValueError("indices must have shape (B, N, 3)")
    if weights.shape != indices.shape:
        raise ValueError("weights must have the same (B, N, 3) shape as indices")
    if features.shape[0] != indices.shape[0]:
        raise ValueError("features and indices must have the same batch size")
    if features.device != indices.device or features.device != weights.device:
        raise ValueError("features, indices, and weights must be on the same device")
    if features.dtype != torch.float32 or weights.dtype != torch.float32:
        raise TypeError("features and weights must be float32")
    if indices.dtype not in (torch.int32, torch.int64):
        raise TypeError("indices must be int32 or int64")
    batch, channels, points = features.shape
    queries = indices.shape[1]
    if batch and queries and points == 0:
        raise ValueError("nonempty interpolation requires at least one source point")
    if points > _MAX_INDEX:
        raise ValueError("three_interpolate supports at most 2**31-1 source points")
    if batch * channels * queries > (1 << 32) - 1:
        raise ValueError("three_interpolate output exceeds the native index range")
    # Indices supplied by callers need a bounds check before native buffer
    # access. On MPS this scalar check synchronizes once with the CPU.
    if indices.numel() and torch.any((indices < 0) | (indices >= points)).item():
        raise IndexError(f"indices must lie in [0, {points})")
    return batch, channels, points, queries


def _three_interpolate_reference(features: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
    """PyTorch gather reference; only source features receive gradients."""
    batch, channels, _ = features.shape
    queries = indices.shape[1]
    selected = features.gather(
        2, indices.to(torch.long).reshape(batch, 1, queries * 3).expand(-1, channels, -1)
    ).reshape(batch, channels, queries, 3)
    return (selected * weights.detach().unsqueeze(1)).sum(dim=-1)


class _ThreeInterpolate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
        batch, channels, points = features.shape
        queries = indices.shape[1]
        output = torch.empty((batch, channels, queries), dtype=torch.float32, device=features.device)
        count = batch * channels * queries
        if count:
            _library().three_interpolate_f32(
                features.contiguous(), indices, weights, output,
                channels, points, queries,
                threads=count, group_size=min(count, _GROUP_SIZE),
            )
        ctx.save_for_backward(indices, weights)
        ctx.points = points
        ctx.channels = channels
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor | None):
        if grad_output is None or not ctx.needs_input_grad[0]:
            return None, None, None
        indices, weights = ctx.saved_tensors
        batch, queries, _ = indices.shape
        channels = ctx.channels
        points = ctx.points
        grad_features = torch.zeros((batch, channels, points), device=indices.device, dtype=torch.float32)
        count = batch * channels * queries
        if count:
            _library().three_interpolate_backward_f32(
                grad_output.contiguous(), indices, weights, grad_features,
                channels, points, queries,
                threads=count, group_size=min(count, _GROUP_SIZE),
            )
        return grad_features, None, None


def three_interpolate(features: Tensor, indices: Tensor, weights: Tensor) -> Tensor:
    """Interpolate ``(B,C,M)`` features at ``(B,N,3)`` source indices.

    ``weights`` are caller-provided ``(B,N,3)`` float32 values, not computed
    from distances here. The output is ``(B,C,N)`` float32:

    ``out[b,c,n] = sum(t=0..2, features[b,c,indices[b,n,t]] * weights[b,n,t])``.

    Backward accumulates ``grad_out[b,c,n] * weights[b,n,t]`` at the selected
    source feature positions, including repeated indices. Indices and weights
    are not differentiated, matching the original PointNet++ operation. The
    MPS accumulation uses integer CAS on float32 bits; floating-point addition
    order (and therefore low bits) can vary across runs. Index validation
    requires one device-to-host scalar synchronization on MPS.
    """
    _validate_interpolate(features, indices, weights)
    if features.device.type != "mps":
        return _three_interpolate_reference(features, indices, weights)
    index32 = indices.contiguous() if indices.dtype == torch.int32 else indices.to(torch.int32)
    return _ThreeInterpolate.apply(features, index32, weights.contiguous())


__all__ = ["three_nn", "three_interpolate"]

"""Bidirectional L1 or squared-L2 Chamfer loss on CPU and PyTorch MPS.

MPS uses a native Metal nearest-neighbor search; other devices use a tiled
PyTorch search. L1 uses a custom first-order backward to match PyTorch3D's
subgradient at equal coordinates; bidirectional loss accumulates both searches.
"""

from __future__ import annotations

import math
from functools import lru_cache
from importlib.resources import files
from typing import Literal

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

_PAIRS_PER_TILE = 1 << 20
_QUERIES_PER_GROUP = 8
_GROUP_SIZE = 256
PointReduction = Literal["mean", "sum", "max"] | None
BatchReduction = Literal["mean", "sum"] | None


def _lengths(points: Tensor, value: Tensor | None, name: str) -> Tensor:
    batch, maximum, _ = points.shape
    if value is None:
        return torch.full((batch,), maximum, dtype=torch.long, device=points.device)
    if value.shape != (batch,) or value.dtype != torch.long or value.device != points.device:
        raise ValueError(f"{name} must be an int64 tensor of shape ({batch},) on {points.device}")
    # Invalid lengths must fail instead of silently selecting padded coordinates.
    # Reading the result synchronizes MPS with the host.
    if bool(((value < 0) | (value > maximum)).any().item()):
        raise ValueError(f"{name} must lie in [0, {maximum}]")
    return value


def _validate_clouds(
    x: Tensor, y: Tensor, x_lengths: Tensor | None, y_lengths: Tensor | None
) -> tuple[Tensor, Tensor]:
    if x.ndim != 3 or y.ndim != 3 or x.shape[-1] != 3 or y.shape[-1] != 3:
        raise ValueError("x and y must have shapes (B, P, 3) and (B, Q, 3)")
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have the same batch size")
    if x.device != y.device or x.dtype != y.dtype:
        raise ValueError("x and y must have the same device and dtype")
    if x.dtype not in (torch.float32, torch.float64) or (x.device.type == "mps" and x.dtype != torch.float32):
        raise TypeError("Chamfer inputs must be float32 (or float64 outside MPS)")

    lx = _lengths(x, x_lengths, "x_lengths")
    ly = _lengths(y, y_lengths, "y_lengths")
    # A nearest point does not exist for a nonempty cloud paired with an empty
    # one. Require both sides to have at least one point in every batch item.
    if bool(((lx == 0) | (ly == 0)).any().item()):
        raise ValueError("each x and y cloud must contain at least one valid point")
    for points, lengths, name in ((x, lx, "x"), (y, ly, "y")):
        valid = torch.arange(points.shape[1], device=points.device)[None] < lengths[:, None]
        if bool((valid & ~torch.isfinite(points).all(dim=-1)).any().item()):
            raise ValueError(f"valid {name} coordinates must be finite")
    return lx, ly


@lru_cache(maxsize=1)
def _metal_library():
    from ._ball_query_mps import _check_simd_width

    _check_simd_width()
    source = files(__package__).joinpath("kernels", "chamfer_nn.metal").read_text()
    return torch.mps.compile_shader(source)


@torch.no_grad()
def _nearest_mps(
    query: Tensor, ref: Tensor, query_lengths: Tensor, ref_lengths: Tensor,
    norm: int = 2,
) -> tuple[Tensor, Tensor]:
    """Return distance and first-minimum index from native Metal."""
    batch, query_count, _ = query.shape
    ref_count = ref.shape[1]
    if ref_count >= 2**32:
        raise ValueError("native Chamfer requires fewer than 2**32 reference points")
    distances = torch.empty((batch, query_count), device=query.device, dtype=torch.float32)
    indices = torch.empty((batch, query_count), device=query.device, dtype=torch.int64)
    if distances.numel():
        group_count = (batch * query_count + _QUERIES_PER_GROUP - 1) // _QUERIES_PER_GROUP
        kernel = (
            _metal_library().chamfer_nearest_l1_f32
            if norm == 1 else _metal_library().chamfer_nearest_f32
        )
        kernel(
            query.contiguous(), ref.contiguous(), query_lengths.contiguous(),
            ref_lengths.contiguous(), distances, indices, batch, query_count,
            ref_count, threads=[group_count * _GROUP_SIZE, 1, 1],
            group_size=[_GROUP_SIZE, 1, 1],
        )
    return distances, indices


@torch.no_grad()
def _nearest_indices(
    query: Tensor, ref: Tensor, query_lengths: Tensor, ref_lengths: Tensor,
    norm: int = 2,
) -> Tensor:
    """Find the first minimum under the requested coordinate-wise distance."""
    if query.device.type == "mps":
        return _nearest_mps(query, ref, query_lengths, ref_lengths, norm)[1]
    batch, query_count, _ = query.shape
    ref_count = ref.shape[1]
    if batch == 0 or query_count == 0:
        return torch.empty((batch, query_count), dtype=torch.long, device=query.device)

    query_step = max(1, min(query_count, math.isqrt(max(1, _PAIRS_PER_TILE // batch))))
    ref_step = max(1, _PAIRS_PER_TILE // (batch * query_step))
    chunks: list[Tensor] = []
    for q0 in range(0, query_count, query_step):
        q = query[:, q0 : q0 + query_step, :]
        shape = (batch, q.shape[1])
        best_dist = torch.full(shape, float("inf"), dtype=query.dtype, device=query.device)
        best_index = torch.zeros(shape, dtype=torch.long, device=query.device)
        for r0 in range(0, ref_count, ref_step):
            r = ref[:, r0 : r0 + ref_step, :]
            delta = q[:, :, None, :] - r[:, None, :, :]
            if norm == 1:
                dist = delta[..., 0].abs()
                dist = dist + delta[..., 1].abs()
                dist = dist + delta[..., 2].abs()
            else:
                dist = delta[..., 0] * delta[..., 0]
                dist = dist + delta[..., 1] * delta[..., 1]
                dist = dist + delta[..., 2] * delta[..., 2]
            ref_index = torch.arange(r0, r0 + r.shape[1], device=ref.device)
            dist = dist.masked_fill(ref_index[None, None, :] >= ref_lengths[:, None, None], float("inf"))
            tile_dist, local_index = dist.min(dim=-1)
            # Tiles are visited in increasing index order; keeping the old
            # winner on equal distances implements the lowest-index tie rule.
            take = tile_dist < best_dist
            best_dist = torch.where(take, tile_dist, best_dist)
            best_index = torch.where(take, local_index + r0, best_index)
        chunks.append(best_index)
    return torch.cat(chunks, dim=1)


class _MetalNearestSquared(torch.autograd.Function):
    """Metal forward, PyTorch native scatter-add backward for selected pairs."""

    @staticmethod
    def forward(
        ctx, query: Tensor, ref: Tensor, query_lengths: Tensor, ref_lengths: Tensor
    ) -> Tensor:
        distances, indices = _nearest_mps(query, ref, query_lengths, ref_lengths)
        ctx.save_for_backward(query, ref, indices)
        return distances

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_distances: Tensor | None):
        if grad_distances is None:
            return None, None, None, None
        query, ref, indices = ctx.saved_tensors
        if not indices.numel():
            return torch.zeros_like(query), torch.zeros_like(ref), None, None

        valid = indices >= 0
        safe_indices = indices.clamp_min(0)
        selected = ref.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        safe_query = torch.where(valid.unsqueeze(-1), query, torch.zeros_like(query))
        delta = safe_query - selected
        # grad_distances already includes each point mean, batch mean, and
        # optional batch weight from the surrounding PyTorch reductions.
        scale = 2 * torch.where(valid, grad_distances, torch.zeros_like(grad_distances))
        query_grad = delta * scale.unsqueeze(-1)
        ref_grad = torch.zeros_like(ref, memory_format=torch.contiguous_format)
        ref_grad.scatter_add_(
            1, safe_indices.unsqueeze(-1).expand(-1, -1, 3), -query_grad
        )
        return query_grad, ref_grad, None, None


class _NearestL1(torch.autograd.Function):
    """L1 nearest distance with PyTorch3D's selected-pair subgradient."""

    @staticmethod
    def forward(
        ctx, query: Tensor, ref: Tensor, query_lengths: Tensor, ref_lengths: Tensor
    ) -> Tensor:
        if query.device.type == "mps":
            distances, indices = _nearest_mps(query, ref, query_lengths, ref_lengths, norm=1)
        else:
            indices = _nearest_indices(query, ref, query_lengths, ref_lengths, norm=1)
            distances = _selected_l1_distance(query, ref, indices, query_lengths)
        ctx.save_for_backward(query, ref, indices, query_lengths)
        return distances

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_distances: Tensor | None):
        if grad_distances is None:
            return None, None, None, None
        query, ref, indices, query_lengths = ctx.saved_tensors
        if not indices.numel():
            return torch.zeros_like(query), torch.zeros_like(ref), None, None

        valid = (indices >= 0) & (
            torch.arange(query.shape[1], device=query.device)[None] < query_lengths[:, None]
        )
        safe_indices = indices.clamp_min(0)
        selected = ref.gather(1, safe_indices.unsqueeze(-1).expand(-1, -1, 3))
        # The pinned PyTorch3D CPU/CUDA kNN backward chooses -1 at equality.
        # torch.abs backward chooses zero there, so it cannot provide parity.
        sign = torch.where(query > selected, 1.0, -1.0)
        scale = torch.where(valid, grad_distances, torch.zeros_like(grad_distances))
        query_grad = sign * scale.unsqueeze(-1)
        ref_grad = torch.zeros_like(ref, memory_format=torch.contiguous_format)
        ref_grad.scatter_add_(
            1, safe_indices.unsqueeze(-1).expand(-1, -1, 3), -query_grad
        )
        return query_grad, ref_grad, None, None


def _selected_squared_distance(query: Tensor, ref: Tensor, index: Tensor, lengths: Tensor) -> Tensor:
    valid = torch.arange(query.shape[1], device=query.device)[None] < lengths[:, None]
    safe_query = torch.where(valid.unsqueeze(-1), query, torch.zeros_like(query))
    near = ref.gather(1, index.clamp_min(0).unsqueeze(-1).expand(-1, -1, 3))
    delta = safe_query - near
    dist = delta[..., 0] * delta[..., 0]
    dist = dist + delta[..., 1] * delta[..., 1]
    dist = dist + delta[..., 2] * delta[..., 2]
    return torch.where(valid, dist, torch.zeros_like(dist))


def _selected_l1_distance(query: Tensor, ref: Tensor, index: Tensor, lengths: Tensor) -> Tensor:
    valid = torch.arange(query.shape[1], device=query.device)[None] < lengths[:, None]
    safe_query = torch.where(valid.unsqueeze(-1), query, torch.zeros_like(query))
    near = ref.gather(1, index.clamp_min(0).unsqueeze(-1).expand(-1, -1, 3))
    delta = safe_query - near
    dist = delta[..., 0].abs()
    dist = dist + delta[..., 1].abs()
    dist = dist + delta[..., 2].abs()
    return torch.where(valid, dist, torch.zeros_like(dist))


def _reduce_points(dist: Tensor, lengths: Tensor, mode: PointReduction) -> Tensor:
    if mode is None:
        return dist
    if mode == "max":
        return dist.max(dim=1).values if dist.shape[1] else dist.new_zeros((dist.shape[0],))
    result = dist.sum(dim=1)
    return result / lengths.clamp(min=1) if mode == "mean" else result


def _zero_weight_anchor(points: Tensor, lengths: Tensor, weights: Tensor) -> Tensor:
    """Graph-connected zeros with upstream's observed (B, B) broadcast shape."""
    valid = torch.arange(points.shape[1], device=points.device)[None] < lengths[:, None]
    selected = torch.where(valid.unsqueeze(-1), points, torch.zeros_like(points))
    per_cloud = selected.sum(dim=(1, 2))
    return (weights[:, None] * per_cloud[None, :]) * 0


def chamfer_distance(
    x: Tensor,
    y: Tensor,
    x_lengths: Tensor | None = None,
    y_lengths: Tensor | None = None,
    x_normals: Tensor | None = None,
    y_normals: Tensor | None = None,
    weights: Tensor | None = None,
    batch_reduction: BatchReduction = "mean",
    point_reduction: PointReduction = "mean",
    norm: int = 2,
    single_directional: bool = False,
    abs_cosine: bool = True,
) -> tuple[Tensor | tuple[Tensor, Tensor], None]:
    """L1 or squared-L2 Chamfer loss with PyTorch3D reduction conventions.

    Returns ``(loss, None)`` for ordinary inputs. For all-zero batch weights,
    the observed PyTorch3D special case instead returns zero normal output
    except for bidirectional ``point_reduction="max"``. Normals are not
    implemented. With
    ``point_reduction=None``, ``loss`` is a pair of padded per-point distance
    tensors (one tensor if ``single_directional=True``) and batch reduction
    must also be None. A valid cloud must have at least one point. MPS supports
    float32; CPU and CUDA additionally support float64. Input validation reads
    device scalars and can synchronize MPS. Nearest indices use the lowest
    reference index on a rounded-distance tie. The tiled search bounds its
    temporary pair storage but still takes O(B * P * Q) time.
    """
    if norm not in (1, 2):
        raise ValueError("Support for 1 or 2 norm.")
    if x_normals is not None or y_normals is not None:
        raise NotImplementedError("normal-distance terms are not supported")
    if batch_reduction not in (None, "mean", "sum"):
        raise ValueError("batch_reduction must be None, 'mean', or 'sum'")
    if point_reduction not in (None, "mean", "sum", "max"):
        raise ValueError("point_reduction must be None, 'mean', 'sum', or 'max'")
    if point_reduction is None and batch_reduction is not None:
        raise ValueError("batch_reduction must be None when point_reduction is None")
    _ = abs_cosine  # relevant only to unsupported normal-distance terms
    lx, ly = _validate_clouds(x, y, x_lengths, y_lengths)
    if weights is not None:
        if weights.shape != (x.shape[0],) or weights.device != x.device or weights.dtype != x.dtype:
            raise ValueError("weights must match the batch size, input device, and input dtype")
        if bool((~torch.isfinite(weights) | (weights < 0)).any().item()):
            raise ValueError("weights must be finite and nonnegative")
        # PyTorch3D's all-zero-weight branch broadcasts a per-cloud zero to
        # (B, B) before applying reductions, including a zero normal result
        # when normals were absent. Preserve that observed API shape and
        # autograd connectivity, while excluding padded coordinates from the
        # zero anchor.
        if bool((weights == 0).all().item()):
            zero_x = _zero_weight_anchor(x, lx, weights)
            if single_directional:
                zero_loss: Tensor | tuple[Tensor, Tensor] = zero_x
                zero_normals: Tensor | tuple[Tensor, Tensor] | None = zero_x
            else:
                zero_y = _zero_weight_anchor(y, ly, weights)
                if point_reduction == "max":
                    zero_loss = torch.maximum(zero_x, zero_y)
                    zero_normals = None
                elif point_reduction is None:
                    zero_loss = (zero_x, zero_y)
                    zero_normals = (zero_x, zero_y)
                else:
                    zero_loss = zero_x + zero_y
                    zero_normals = zero_x + zero_y
            if batch_reduction is not None:
                assert isinstance(zero_loss, Tensor)
                zero_loss = zero_loss.sum()
                if zero_normals is not None:
                    assert isinstance(zero_normals, Tensor)
                    zero_normals = zero_normals.sum()
            return zero_loss, zero_normals

    if norm == 1:
        dx = _NearestL1.apply(x, y, lx, ly)
    elif x.device.type == "mps":
        dx = _MetalNearestSquared.apply(x, y, lx, ly)
    else:
        ix = _nearest_indices(x, y, lx, ly)
        dx = _selected_squared_distance(x, y, ix, lx)
    if weights is not None:
        dx = dx * weights[:, None]
    rx = _reduce_points(dx, lx, point_reduction)
    if single_directional:
        loss: Tensor | tuple[Tensor, Tensor] = rx
    else:
        if norm == 1:
            dy = _NearestL1.apply(y, x, ly, lx)
        elif x.device.type == "mps":
            dy = _MetalNearestSquared.apply(y, x, ly, lx)
        else:
            iy = _nearest_indices(y, x, ly, lx)
            dy = _selected_squared_distance(y, x, iy, ly)
        if weights is not None:
            dy = dy * weights[:, None]
        ry = _reduce_points(dy, ly, point_reduction)
        loss = (rx, ry) if point_reduction is None else (
            torch.maximum(rx, ry) if point_reduction == "max" else rx + ry
        )

    if batch_reduction is not None:
        assert isinstance(loss, Tensor)
        loss = loss.sum()
        if batch_reduction == "mean":
            if weights is None:
                denominator = max(x.shape[0], 1)
            else:
                total_weight = weights.sum()
                denominator = torch.where(total_weight > 0, total_weight, torch.ones_like(total_weight))
            loss = loss / denominator
    return loss, None

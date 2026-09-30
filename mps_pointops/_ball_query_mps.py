# SPDX-License-Identifier: MIT
"""Deterministic radius-neighbor search on Apple GPUs.

The Metal source is original to this project. The public contract follows the
mathematical definition of radius search and the documented first-K convention
used by PyTorch3D; no third-party implementation is copied here.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import math
import struct
from typing import NamedTuple

import torch


_MIN_SUPPORTED_RADIUS_F32 = 2.0**-112


class BallQueryResult(NamedTuple):
    """Squared distances and indices, both shaped ``(B, Q, K)``."""

    distances: torch.Tensor
    indices: torch.Tensor


@lru_cache(maxsize=1)
def _shader_library():
    source = files(__package__).joinpath("kernels", "ball_query.metal").read_text()
    return torch.mps.compile_shader(source)


def _lengths_or_full(
    lengths: torch.Tensor | None, *, batch: int, count: int, device: torch.device, name: str
) -> torch.Tensor:
    if lengths is None:
        return torch.full((batch,), count, device=device, dtype=torch.int64)
    if not isinstance(lengths, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if lengths.shape != (batch,) or lengths.dtype != torch.int64 or lengths.device != device:
        raise ValueError(f"{name} must have shape ({batch},), int64 dtype and device {device}")
    if torch.any((lengths < 0) | (lengths > count)).item():
        raise ValueError(f"{name} values must lie in [0, {count}]")
    return lengths.contiguous()


class _BallQuery(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        queries: torch.Tensor,
        points: torch.Tensor,
        query_lengths: torch.Tensor,
        point_lengths: torch.Tensor,
        radius_sq: float,
        radius_f32: float,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_count, _ = queries.shape
        point_count = points.shape[1]
        indices = torch.empty((batch, query_count, k), device=queries.device, dtype=torch.int64)
        distances = torch.empty((batch, query_count, k), device=queries.device, dtype=torch.float32)

        if indices.numel():
            kernel = (
                _shader_library().ball_query_f32
                if queries.dtype == torch.float32
                else _shader_library().ball_query_f16
            )
            kernel(
                queries,
                points,
                query_lengths,
                point_lengths,
                indices,
                distances,
                batch,
                query_count,
                point_count,
                k,
                radius_sq,
                radius_f32,
                threads=[batch * query_count, 1, 1],
                group_size=[256, 1, 1],
            )

        ctx.save_for_backward(queries, points, indices)
        ctx.mark_non_differentiable(indices)
        return distances, indices

    @staticmethod
    def backward(ctx, grad_distances: torch.Tensor | None, grad_indices: None):
        if grad_distances is None:
            return None, None, None, None, None, None, None

        queries, points, indices = ctx.saved_tensors
        batch, query_count, k = indices.shape
        if not indices.numel() or points.shape[1] == 0:
            return torch.zeros_like(queries), torch.zeros_like(points), None, None, None, None, None

        valid = indices >= 0
        safe_indices = indices.clamp_min(0)
        gathered = points.float().gather(
            1, safe_indices.reshape(batch, -1, 1).expand(-1, -1, 3)
        ).reshape(batch, query_count, k, 3)
        delta = queries.float().unsqueeze(2) - gathered
        # Padded slots may gather a non-finite point 0. Mask before multiplying:
        # IEEE-754 makes 0 * NaN equal NaN, not zero.
        delta = torch.where(valid.unsqueeze(-1), delta, torch.zeros_like(delta))
        valid_grad = torch.where(valid, grad_distances, torch.zeros_like(grad_distances))
        coeff = (2.0 * valid_grad).unsqueeze(-1)
        contributions = coeff * delta

        grad_queries = (
            contributions.sum(dim=2).to(queries.dtype) if ctx.needs_input_grad[0] else None
        )
        grad_points = None
        if ctx.needs_input_grad[1]:
            grad_points = torch.zeros_like(points, dtype=torch.float32)
            grad_points.scatter_add_(
                1,
                safe_indices.reshape(batch, -1, 1).expand(-1, -1, 3),
                -contributions.reshape(batch, -1, 3),
            )
            grad_points = grad_points.to(points.dtype)
        return grad_queries, grad_points, None, None, None, None, None


def ball_query(
    queries: torch.Tensor,
    points: torch.Tensor,
    *,
    radius: float,
    k: int,
    query_lengths: torch.Tensor | None = None,
    point_lengths: torch.Tensor | None = None,
) -> BallQueryResult:
    """Find the first ``k`` points strictly inside each radius.

    ``queries`` and ``points`` are float32 or float16 MPS tensors with shapes ``(B,Q,3)``
    and ``(B,P,3)``. The output is ordered by input point index, not distance.
    Missing neighbors use index ``-1`` and squared distance ``0``. Under the
    supported Safe math mode, non-finite coordinates never match. Gradients of
    valid squared distances propagate to both coordinate tensors; selection
    indices are non-differentiable. ``radius`` is a Python number, not a
    differentiable Tensor.

    Coordinates remain on MPS. Supplying lengths checks their bounds with a
    scalar MPS-to-CPU synchronization. Non-contiguous coordinate tensors are
    copied to contiguous MPS storage before dispatch. The radius is rounded
    to float32 first and must then be zero or at least 2**-112. Smaller
    positive values are rejected because Metal may flush coordinate differences
    that matter to the radius decision. Boundary decisions use float32 arithmetic.
    """
    if not isinstance(queries, torch.Tensor) or not isinstance(points, torch.Tensor):
        raise TypeError("queries and points must be torch.Tensor values")
    if queries.ndim != 3 or points.ndim != 3 or queries.shape[-1] != 3 or points.shape[-1] != 3:
        raise ValueError("queries and points must have shapes (B, Q, 3) and (B, P, 3)")
    if queries.shape[0] != points.shape[0]:
        raise ValueError("queries and points must have the same batch size")
    if queries.device.type != "mps" or points.device != queries.device:
        raise ValueError("queries and points must be on the same MPS device")
    if queries.dtype not in (torch.float32, torch.float16) or points.dtype != queries.dtype:
        raise TypeError("queries and points must have the same float32 or float16 dtype")
    if not isinstance(k, int) or isinstance(k, bool) or k < 0 or k > (1 << 63) - 1:
        raise ValueError("k must be a non-negative integer")
    if not isinstance(radius, (int, float)) or isinstance(radius, bool):
        raise ValueError("radius must be a finite non-negative number")
    try:
        radius = float(radius)
    except OverflowError as exc:
        raise ValueError("radius must be a finite non-negative number") from exc
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be a finite non-negative number")

    try:
        radius_f32 = struct.unpack("f", struct.pack("f", radius))[0]
    except OverflowError as exc:
        raise ValueError("radius must fit in float32") from exc
    if not math.isfinite(radius_f32) or (radius > 0 and radius_f32 < _MIN_SUPPORTED_RADIUS_F32):
        raise ValueError("positive radius must round to a normal float32 value at least 2**-112")

    # This is fl32(fl32(radius) * fl32(radius)), as in PyTorch3D's float
    # radius2 = radius * radius. Two binary32 operands multiply exactly in
    # binary64 (at most 48 significant bits); struct.pack rounds once to f32.
    radius_squared = float(radius_f32) * float(radius_f32)
    try:
        radius_squared = struct.unpack("f", struct.pack("f", radius_squared))[0]
    except OverflowError as exc:
        radius_squared = math.inf

    batch, query_count, _ = queries.shape
    point_count = points.shape[1]
    if batch * query_count > (1 << 32) - 1:
        raise ValueError("B * Q exceeds the Metal 1-D dispatch limit")
    if batch * query_count * 3 > (1 << 63) - 1 or batch * point_count * 3 > (1 << 63) - 1:
        raise ValueError("coordinate element count exceeds int64 indexing")
    if batch * query_count * k > (1 << 63) - 1:
        raise ValueError("output element count exceeds int64 indexing")

    if not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable on this host")
    query_lengths = _lengths_or_full(
        query_lengths, batch=batch, count=query_count, device=queries.device, name="query_lengths"
    )
    point_lengths = _lengths_or_full(
        point_lengths, batch=batch, count=point_count, device=queries.device, name="point_lengths"
    )

    distances, indices = _BallQuery.apply(
        queries.contiguous(),
        points.contiguous(),
        query_lengths,
        point_lengths,
        radius_squared,
        radius_f32,
        k,
    )
    return BallQueryResult(distances=distances, indices=indices)

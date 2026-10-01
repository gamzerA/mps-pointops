"""Compact, batched voxelization and feature downsampling on CPU and MPS.

This is a new mps-pointops API. It does not implement the signature or raw
mixed-radix identifiers of PyG ``voxel_grid`` or ``torch_cluster.grid_cluster``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Sequence

import torch
from torch import Tensor


@dataclass(frozen=True)
class Voxelization:
    """Compact voxels, with both point-to-voxel and voxel-to-point maps.

    Rows are sorted lexicographically by ``(batch, cell_0, ..., cell_D-1)``.
    ``inverse[i]`` is the row of point ``i``. The original point indices in
    row ``v`` are ``point_order[ptr[v]:ptr[v + 1]]``, in input order.
    """

    voxel_coords: Tensor  # (V, D), int64
    batch: Tensor  # (V,), int64
    inverse: Tensor  # (N,), int64
    counts: Tensor  # (V,), int64
    point_order: Tensor  # (N,), int64
    ptr: Tensor  # (V + 1,), int64


@dataclass(frozen=True)
class VoxelDownsample:
    """Mean position and optional mean/sum features for compact voxels."""

    voxels: Voxelization
    pos: Tensor  # (V, D), float32
    features: Tensor | None  # (V, C), float32, when supplied


@cache
def _metal_pool_library():
    from ._ball_query_mps import _require_compile_shader

    _require_compile_shader()
    source = resources.files(__package__).joinpath("kernels", "voxel_pool.metal").read_text()
    return torch.mps.compile_shader(source)


class _FusedVoxelPool(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, pos: Tensor, features: Tensor | None, point_order: Tensor,
        ptr: Tensor, inverse: Tensor, counts: Tensor, mean_features: bool,
    ) -> tuple[Tensor, Tensor | None]:
        rows = counts.numel()
        dimensions = pos.shape[1]
        channels = 0 if features is None else features.shape[1]
        work = rows * (dimensions + channels)
        if work >= 2**32:
            raise ValueError("voxel output size exceeds the Metal 1-D dispatch limit")
        out_pos = torch.empty((rows, dimensions), dtype=pos.dtype, device=pos.device)
        out_features = (
            None if features is None else
            torch.empty((rows, channels), dtype=pos.dtype, device=pos.device)
        )
        # A missing feature tensor has no feature-channel work. Reusing valid
        # buffers avoids passing a zero-byte tensor to the shader launcher.
        _metal_pool_library().voxel_pool_f32(
            pos.contiguous(),
            pos if features is None else features.contiguous(),
            point_order.contiguous(), ptr.contiguous(), out_pos,
            out_pos if out_features is None else out_features,
            rows, dimensions, channels, int(mean_features),
            threads=work, group_size=min(work, 256),
        )
        ctx.save_for_backward(inverse, counts)
        ctx.has_features = features is not None
        ctx.mean_features = mean_features
        return out_pos, out_features

    @staticmethod
    def backward(ctx, grad_pos: Tensor | None, grad_features: Tensor | None):
        inverse, counts = ctx.saved_tensors
        needs_pos, needs_features = ctx.needs_input_grad[:2]
        point_counts = None
        if (needs_pos and grad_pos is not None) or (
            needs_features and ctx.has_features and ctx.mean_features
            and grad_features is not None
        ):
            point_counts = counts.index_select(0, inverse).to(torch.float32).unsqueeze(1)
        pos_grad = (
            grad_pos.index_select(0, inverse) / point_counts
            if needs_pos and grad_pos is not None else None
        )
        features_grad = (
            grad_features.index_select(0, inverse)
            if needs_features and ctx.has_features and grad_features is not None else None
        )
        if features_grad is not None and ctx.mean_features:
            features_grad = features_grad / point_counts
        return pos_grad, features_grad, None, None, None, None, None


def _vector(value: float | Sequence[float] | Tensor, name: str, pos: Tensor) -> Tensor:
    dimensions = pos.shape[1]
    if isinstance(value, Tensor):
        if value.dtype != torch.float32 or value.device != pos.device:
            raise ValueError(f"{name} must be float32 on {pos.device}")
        result = value
    else:
        result = torch.as_tensor(value, dtype=torch.float32, device=pos.device)
    if result.ndim == 0:
        result = result.expand(dimensions)
    if result.shape != (dimensions,):
        raise ValueError(f"{name} must be a scalar or shape ({dimensions},)")
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def voxelize(
    pos: Tensor,
    size: float | Sequence[float] | Tensor,
    batch: Tensor | None = None,
    *,
    start: float | Sequence[float] | Tensor | None = None,
    end: float | Sequence[float] | Tensor | None = None,
) -> Voxelization:
    """Return compact cell rows and bidirectional maps for flat batched points.

    ``pos`` is finite float32 ``(N, D)`` with ``D`` in ``{1, 2, 3}`` on CPU
    or MPS. ``batch`` is an optional nonnegative int64 ``(N,)`` vector on the
    same device; arbitrary batch order and gaps are accepted. ``size`` is a
    positive scalar or ``(D,)`` vector. ``start`` is the grid origin (zero by
    default), and cell ``j`` is ``floor(float32((pos_j-start_j)/size_j))``.
    Negative cell coordinates are allowed. If ``end`` is provided, all points
    must lie in the half-open region ``[start, end)``; it is an input bound,
    not a change to the cell numbering. The empty input returns empty maps.

    The integer assignment has no gradient. Coordinates whose quotient lies
    near a cell boundary can differ between CPU, MPS Safe and MPS Fast due
    to float32 arithmetic; representable boundary cases are tested.
    """
    if not isinstance(pos, Tensor):
        raise TypeError("pos must be a torch.Tensor")
    if pos.ndim != 2 or pos.shape[1] not in (1, 2, 3):
        raise ValueError("pos must have shape (N, D), with D in {1, 2, 3}")
    if pos.dtype != torch.float32:
        raise TypeError("pos must be float32")
    if pos.device.type not in ("cpu", "mps"):
        raise ValueError("pos must be on CPU or MPS")
    if not bool(torch.isfinite(pos).all()):
        raise ValueError("pos must be finite")
    dimensions = pos.shape[1]
    cell_size = _vector(size, "size", pos)
    if not bool((cell_size > 0).all()):
        raise ValueError("size must be positive")
    origin = (
        torch.zeros((dimensions,), dtype=torch.float32, device=pos.device)
        if start is None else _vector(start, "start", pos)
    )
    if end is not None:
        upper = _vector(end, "end", pos)
        if not bool((upper > origin).all()):
            raise ValueError("end must exceed start in every dimension")
        if not bool(((pos >= origin) & (pos < upper)).all()):
            raise ValueError("points must lie in the half-open [start, end) region")
    if batch is None:
        batch = torch.zeros((pos.shape[0],), dtype=torch.int64, device=pos.device)
    elif not isinstance(batch, Tensor):
        raise TypeError("batch must be a torch.Tensor")
    elif batch.shape != (pos.shape[0],) or batch.dtype != torch.int64 or batch.device != pos.device:
        raise ValueError("batch must be int64 (N,) on the same device as pos")
    if not bool((batch >= 0).all()):
        raise ValueError("batch IDs must be nonnegative")

    count = pos.shape[0]
    if count == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=pos.device)
        return Voxelization(
            voxel_coords=empty.reshape(0, 1).expand(0, dimensions),
            batch=empty,
            inverse=empty,
            counts=empty,
            point_order=empty,
            ptr=torch.zeros((1,), dtype=torch.int64, device=pos.device),
        )

    quotient = (pos - origin) / cell_size
    # A large or overflowing quotient cannot be converted safely to int64.
    # The conservative 2**62 limit also keeps floor's signed range clear.
    if not bool((torch.isfinite(quotient) & (quotient.abs() < 2**62)).all()):
        raise ValueError("cell coordinate exceeds the supported int64 range")
    cells = torch.floor(quotient).to(torch.int64)
    keys = torch.cat((batch[:, None], cells), dim=1)
    # torch.unique(dim=0) calls aten::unique_dim, which has no MPS kernel in
    # the oldest supported PyTorch 2.7. Stable column sorts build the same
    # lexicographic order without packing int64 coordinates or CPU fallback.
    point_order = torch.arange(count, dtype=torch.int64, device=pos.device)
    for column in range(dimensions, -1, -1):
        point_order = point_order[
            torch.argsort(keys[point_order, column], stable=True)
        ]
    ordered_keys = keys[point_order]
    begins = torch.cat((
        torch.ones((1,), dtype=torch.bool, device=pos.device),
        (ordered_keys[1:] != ordered_keys[:-1]).any(dim=1),
    ))
    ptr = torch.cat((torch.nonzero(begins).flatten(), point_order.new_tensor([count])))
    counts = ptr[1:] - ptr[:-1]
    rows = ordered_keys[ptr[:-1]]
    sorted_inverse = begins.to(torch.int64).cumsum(dim=0) - 1
    inverse = torch.empty_like(point_order).scatter_(0, point_order, sorted_inverse)
    return Voxelization(
        voxel_coords=rows[:, 1:], batch=rows[:, 0], inverse=inverse,
        counts=counts, point_order=point_order, ptr=ptr,
    )


def voxel_downsample(
    pos: Tensor,
    size: float | Sequence[float] | Tensor,
    batch: Tensor | None = None,
    features: Tensor | None = None,
    *,
    start: float | Sequence[float] | Tensor | None = None,
    end: float | Sequence[float] | Tensor | None = None,
    feature_reduce: str = "mean",
    pool_backend: str = "index_add",
) -> VoxelDownsample:
    """Aggregate mean positions and optional mean/sum float32 features.

    ``inverse`` and the CSR maps in ``voxels`` use the same row order as the
    output tensors. Differentiation treats the integer cell assignment as
    fixed: each position gets ``grad_pos[v]/counts[v]`` and each feature gets
    ``grad_features[v]/counts[v]`` for mean (or ``grad_features[v]`` for sum).
    ``pool_backend="fused_csr"`` opts into the experimental MPS Metal
    reduction; the default ``"index_add"`` preserves the existing path.
    Floating-point accumulation order on the fused path is not promised to
    be bitwise identical to PyTorch ``index_add_``. Integer maps and output
    row order are exact for both backends.
    """
    if feature_reduce not in ("mean", "sum"):
        raise ValueError("feature_reduce must be 'mean' or 'sum'")
    if pool_backend not in ("index_add", "fused_csr"):
        raise ValueError("pool_backend must be 'index_add' or 'fused_csr'")
    voxels = voxelize(pos, size, batch, start=start, end=end)
    if pool_backend == "fused_csr" and pos.device.type != "mps":
        raise ValueError("pool_backend='fused_csr' requires MPS tensors")
    if features is not None:
        if not isinstance(features, Tensor):
            raise TypeError("features must be a torch.Tensor")
        if features.ndim != 2 or features.shape[0] != pos.shape[0] or features.shape[1] < 1:
            raise ValueError("features must have shape (N, C), C >= 1")
        if features.dtype != torch.float32 or features.device != pos.device:
            raise ValueError("features must be float32 on the same device as pos")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must be finite")
    rows = voxels.counts.numel()
    if pos.shape[0] == 0:
        return VoxelDownsample(voxels, pos[:0], None if features is None else features[:0])
    if pool_backend == "fused_csr":
        means, pooled = _FusedVoxelPool.apply(
            pos, features, voxels.point_order, voxels.ptr, voxels.inverse,
            voxels.counts, feature_reduce == "mean",
        )
    else:
        means = pos.new_zeros((rows, pos.shape[1])).index_add_(0, voxels.inverse, pos)
        means = means / voxels.counts.to(pos.dtype).unsqueeze(1)
        if features is None:
            pooled = None
        else:
            pooled = features.new_zeros((rows, features.shape[1])).index_add_(
                0, voxels.inverse, features,
            )
            if feature_reduce == "mean":
                pooled = pooled / voxels.counts.to(features.dtype).unsqueeze(1)
    return VoxelDownsample(voxels, means, pooled)

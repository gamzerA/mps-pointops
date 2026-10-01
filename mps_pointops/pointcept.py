"""Narrow ``pointops`` shim for Pointcept v1.2.1 Point Transformer V1.

This module covers the calls made by ``point_transformer_seg.py`` at the
pinned Pointcept v1.2.1 tag. Coordinates/features use flat point rows, while
offsets are inclusive cumulative counts per batch (for example ``[4, 9]``).
Integer neighbor selection has no coordinate gradient. Grouping propagates
feature and relative-coordinate gradients; interpolation propagates feature
gradients with fixed distance-derived weights. This is not a general Pointcept
or CUDA pointops replacement.
"""

from __future__ import annotations

import torch
from torch import Tensor

from . import ops

_PAD_SQUARED_DISTANCE = 1e10  # upstream CUDA heap's strict distance ceiling
_PAD_DISTANCE = 100_000.0  # sqrt(1e10), the upstream unfilled distance value


def _cpu_knn(query: Tensor, ref: Tensor, take: int) -> tuple[Tensor, Tensor]:
    """Stable CPU selection, including ties at the final neighbor slot."""
    distances = []
    indices = []
    chunk = max(1, 1_000_000 // len(ref))
    for start in range(0, len(query), chunk):
        delta = query[start:start + chunk, None, :] - ref[None, :, :]
        squared = delta[..., 0] * delta[..., 0]
        for dimension in (1, 2):
            squared = squared + delta[..., dimension] * delta[..., dimension]
        order = torch.argsort(squared, dim=1, stable=True)[:, :take]
        distances.append(torch.sqrt(squared.gather(1, order)))
        indices.append(order)
    return torch.cat(distances), torch.cat(indices)


def _points(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must be a tensor of shape (N, 3)")
    if value.dtype != torch.float32 or value.device.type not in ("cpu", "mps"):
        raise ValueError(f"{name} must be float32 on CPU or MPS")


def _offsets(value: Tensor, count: int | None, device: torch.device, name: str) -> list[int]:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional int32/int64 tensor")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}")
    ends = value.tolist()
    if count is not None and ((ends and ends[-1] != count) or (not ends and count != 0)):
        raise ValueError(f"{name} must end at its point count ({count})")
    if any(end < start for start, end in zip((0, *ends[:-1]), ends)):
        raise ValueError(f"{name} must be nondecreasing and nonnegative")
    return [0, *ends]


def _pairs(
    xyz: Tensor, offset: Tensor, new_xyz: Tensor | None, new_offset: Tensor | None,
) -> tuple[Tensor, list[int], list[int]]:
    _points(xyz, "xyz")
    if new_xyz is None and new_offset is None:
        new_xyz, new_offset = xyz, offset
    elif new_xyz is None or new_offset is None:
        raise ValueError("new_xyz and new_offset must be provided together")
    _points(new_xyz, "new_xyz")
    if xyz.device != new_xyz.device:
        raise ValueError("xyz and new_xyz must be on the same device")
    ref_ends = _offsets(offset, len(xyz), xyz.device, "offset")
    query_ends = _offsets(new_offset, len(new_xyz), xyz.device, "new_offset")
    if len(ref_ends) != len(query_ends):
        raise ValueError("offset and new_offset must have the same batch count")
    return new_xyz, ref_ends, query_ends


def farthest_point_sampling(xyz: Tensor, offset: Tensor, new_offset: Tensor) -> Tensor:
    """Return global int32 FPS indices, starting at each batch's first point.

    ``offset`` and ``new_offset`` are same-length cumulative int32/int64
    vectors. Each requested batch count must be at most its input count.
    Equal-distance ties use this package's smaller-index rule; upstream CUDA
    reduction tie order is not promised to match.
    """
    _points(xyz, "xyz")
    source = _offsets(offset, len(xyz), xyz.device, "offset")
    target = _offsets(new_offset, None, xyz.device, "new_offset")
    if len(source) != len(target):
        raise ValueError("offset and new_offset must have the same batch count")
    chunks = []
    for lo, hi, out_lo, out_hi in zip(source, source[1:], target, target[1:]):
        requested = out_hi - out_lo
        if requested > hi - lo:
            raise ValueError("a batch requests more FPS samples than input points")
        if requested:
            local = ops.furthest_point_sample(xyz[lo:hi].unsqueeze(0), requested)[0]
            chunks.append((local + lo).to(torch.int32))
    return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.int32, device=xyz.device)


def knn_query(
    nsample: int, xyz: Tensor, offset: Tensor,
    new_xyz: Tensor | None = None, new_offset: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Return global int32 neighbors and float32 Euclidean distances.

    Results have shape ``(M, nsample)``. A batch with fewer references uses
    ``-1`` indices and ``1e5`` distances for missing slots, including fully
    empty reference batches or candidates with squared distance at least
    ``1e10``. Ties use the smaller global reference index.
    Neighbor assignment and distances are detached from coordinate autograd.
    """
    if isinstance(nsample, bool) or not isinstance(nsample, int) or nsample < 0:
        raise ValueError("nsample must be a nonnegative integer")
    new_xyz, source, target = _pairs(xyz, offset, new_xyz, new_offset)
    if nsample > 256 and xyz.device.type == "mps":
        raise ValueError("MPS kNN supports at most 256 neighbors")
    indices = []
    distances = []
    with torch.no_grad():
        for lo, hi, qlo, qhi in zip(source, source[1:], target, target[1:]):
            queries = qhi - qlo
            if not queries:
                continue
            take = min(nsample, hi - lo)
            if take:
                if xyz.device.type == "cpu":
                    dist, idx = _cpu_knn(new_xyz[qlo:qhi], xyz[lo:hi], take)
                else:
                    distance, selected = ops.knn(
                        new_xyz[qlo:qhi].unsqueeze(0), xyz[lo:hi].unsqueeze(0), take,
                    )
                    dist, idx = distance[0], selected[0]
                idx = (idx + lo).to(torch.int32)
                # The upstream CUDA heap starts at 1e10 and replaces a slot
                # only for a strictly smaller squared distance. Recompute
                # squared distance from the selected coordinates: comparing
                # sqrt distances can round a just-inside candidate to 1e5.
                delta = new_xyz[qlo:qhi, None, :] - xyz[idx.long()]
                squared = delta[..., 0] * delta[..., 0]
                squared = squared + delta[..., 1] * delta[..., 1]
                squared = squared + delta[..., 2] * delta[..., 2]
                valid = squared < _PAD_SQUARED_DISTANCE
                idx = torch.where(valid, idx, -1)
                dist = torch.where(valid, dist, _PAD_DISTANCE)
            else:
                idx = torch.empty((queries, 0), dtype=torch.int32, device=xyz.device)
                dist = torch.empty((queries, 0), dtype=torch.float32, device=xyz.device)
            pad = nsample - take
            if pad:
                idx = torch.cat((idx, torch.full((queries, pad), -1, dtype=torch.int32, device=xyz.device)), dim=1)
                dist = torch.cat((dist, torch.full((queries, pad), _PAD_DISTANCE, dtype=torch.float32, device=xyz.device)), dim=1)
            indices.append(idx)
            distances.append(dist)
    if indices:
        return torch.cat(indices), torch.cat(distances)
    return (
        torch.empty((len(new_xyz), nsample), dtype=torch.int32, device=xyz.device),
        torch.empty((len(new_xyz), nsample), dtype=torch.float32, device=xyz.device),
    )


def grouping(
    idx: Tensor, feat: Tensor, xyz: Tensor,
    new_xyz: Tensor | None = None, with_xyz: bool = False,
) -> Tensor:
    """Gather ``(N,C)`` features to ``(M,K,C)`` with zero-filled ``-1`` slots.

    With ``with_xyz=True``, prepend relative positions to produce
    ``(M,K,3+C)``. Feature and valid relative-position gradients propagate;
    invalid slots contribute zero gradient.
    """
    _points(xyz, "xyz")
    if not isinstance(feat, Tensor) or feat.ndim != 2 or len(feat) != len(xyz):
        raise ValueError("feat must have shape (N, C) matching xyz")
    if feat.dtype != torch.float32 or feat.device != xyz.device:
        raise ValueError("feat must be float32 on the xyz device")
    if not isinstance(idx, Tensor) or idx.ndim != 2 or idx.dtype not in (torch.int32, torch.int64):
        raise ValueError("idx must be a two-dimensional int32/int64 tensor")
    if idx.device != xyz.device:
        raise ValueError("idx must be on the xyz device")
    if bool((idx < -1).any()) or bool((idx >= len(xyz)).any()):
        raise ValueError("idx contains an out-of-range point index")
    valid = idx >= 0
    safe_idx = torch.where(valid, idx.long(), len(xyz))
    padded_feat = torch.cat((feat, feat.new_zeros((1, feat.shape[1]))), dim=0)
    grouped_feat = padded_feat[safe_idx]
    if not with_xyz:
        return grouped_feat
    if new_xyz is None:
        new_xyz = xyz
    _points(new_xyz, "new_xyz")
    if new_xyz.device != xyz.device or len(new_xyz) != len(idx):
        raise ValueError("new_xyz must have one row per idx row on the xyz device")
    padded_xyz = torch.cat((xyz, xyz.new_zeros((1, 3))), dim=0)
    relative = padded_xyz[safe_idx] - new_xyz.unsqueeze(1)
    relative = torch.where(valid.unsqueeze(-1), relative, 0)
    return torch.cat((relative, grouped_feat), dim=-1)


def knn_query_and_group(
    feat: Tensor, xyz: Tensor, offset: Tensor | None = None,
    new_xyz: Tensor | None = None, new_offset: Tensor | None = None,
    idx: Tensor | None = None, nsample: int | None = None, with_xyz: bool = False,
) -> tuple[Tensor, Tensor]:
    """Query kNN if needed, then group features and optional relative xyz."""
    if idx is None:
        if offset is None or nsample is None:
            raise ValueError("offset and nsample are required when idx is absent")
        idx, _ = knn_query(nsample, xyz, offset, new_xyz, new_offset)
    return grouping(idx, feat, xyz, new_xyz, with_xyz), idx


def interpolation(
    xyz: Tensor, new_xyz: Tensor, feat: Tensor,
    offset: Tensor, new_offset: Tensor, k: int = 3,
) -> Tensor:
    """Inverse-distance interpolate ``(N,C)`` features to new points.

    Invalid ``-1`` neighbors carry zero weight. If a query's reference batch
    is empty, its output is zero. Distance weights are fixed for autograd,
    matching the upstream kNN selection's absent coordinate gradient.
    """
    if not isinstance(feat, Tensor) or feat.ndim != 2 or len(feat) != len(xyz):
        raise ValueError("feat must have shape (N, C) matching xyz")
    if feat.dtype != torch.float32 or feat.device != xyz.device:
        raise ValueError("feat must be float32 on the xyz device")
    idx, dist = knn_query(k, xyz, offset, new_xyz, new_offset)
    valid = idx >= 0
    reciprocal = torch.where(valid, 1.0 / (dist + 1e-8), 0.0)
    normalizer = reciprocal.sum(dim=1, keepdim=True)
    weight = reciprocal / normalizer.clamp_min(1e-20)
    padded_feat = torch.cat((feat, feat.new_zeros((1, feat.shape[1]))), dim=0)
    safe_idx = torch.where(valid, idx.long(), len(feat))
    return (padded_feat[safe_idx] * weight.unsqueeze(-1)).sum(dim=1)


__all__ = [
    "farthest_point_sampling", "knn_query", "grouping", "knn_query_and_group",
    "interpolation",
]

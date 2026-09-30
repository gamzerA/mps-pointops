"""Pure-PyTorch reference implementations of point cloud ops.

They run on any device (CPU, MPS, CUDA). They are the baseline the Metal
kernels are measured against and the oracle for correctness tests.
"""

from __future__ import annotations

import torch
from torch import Tensor


def furthest_point_sample(xyz: Tensor, npoint: int, start_idx: int = 0) -> Tensor:
    """Farthest point sampling, like ``pointnet2_ops.furthest_point_sample``.

    Args:
        xyz: (B, N, 3) points.
        npoint: number of points to sample.
        start_idx: first sampled index. pointnet2_ops always starts at 0.

    Returns:
        (B, npoint) int64 indices into ``xyz``.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.empty(B, npoint, dtype=torch.long, device=device)
    min_d2 = torch.full((B, N), float("inf"), dtype=xyz.dtype, device=device)
    farthest = torch.full((B,), start_idx, dtype=torch.long, device=device)
    batch = torch.arange(B, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        d2 = (xyz - centroid).square().sum(-1)
        min_d2 = torch.minimum(min_d2, d2)
        farthest = min_d2.argmax(-1)
    return idx


def knn(query: Tensor, ref: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Brute-force k nearest neighbors, like ``knn_cuda.KNN(k, transpose_mode=True)``.

    Args:
        query: (B, M, 3) query points.
        ref: (B, N, 3) reference points.
        k: number of neighbors.

    Returns:
        dist: (B, M, k) Euclidean distances, ascending.
        idx: (B, M, k) int64 indices into ``ref``.
    """
    d = torch.cdist(query, ref)
    dist, idx = d.topk(k, dim=-1, largest=False, sorted=True)
    return dist, idx


def ball_query(query: Tensor, ref: Tensor, radius: float, K: int) -> tuple[Tensor, Tensor]:
    """Ball query with the PyTorch3D contract.

    Neighbors are the first ``K`` points of ``ref`` in input order whose squared
    distance is strictly less than ``radius**2``. Empty slots get index -1 and
    distance 0. ``radius**2`` is computed in float32 from the float32-rounded
    radius, as PyTorch3D does.

    Distances come from ``torch.cdist``, so they can differ from PyTorch3D's
    sum of squared differences in the last bits.

    Args:
        query: (B, M, 3) query points.
        ref: (B, N, 3) reference points.
        radius: search radius.
        K: maximum number of neighbors.

    Returns:
        dist2: (B, M, K) squared distances.
        idx: (B, M, K) int64 indices into ``ref``, -1 for empty slots.
    """
    B, M, _ = query.shape
    N = ref.shape[1]
    r32 = torch.tensor(radius, dtype=torch.float32)
    r2 = (r32 * r32).item()

    d2 = torch.cdist(query, ref).square()
    order = torch.arange(N, device=ref.device).expand(B, M, N)
    cand = torch.where(d2 < r2, order, N)
    kk = min(K, N)
    first = cand.topk(kk, dim=-1, largest=False, sorted=True).values
    valid = first < N
    idx = torch.where(valid, first, -1)
    dist2 = torch.where(valid, d2.gather(-1, first.clamp(max=N - 1)), 0.0)
    if kk < K:
        idx = torch.nn.functional.pad(idx, (0, K - kk), value=-1)
        dist2 = torch.nn.functional.pad(dist2, (0, K - kk), value=0.0)
    return dist2, idx

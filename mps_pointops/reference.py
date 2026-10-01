"""Pure-PyTorch reference implementations of point cloud ops.

They run on any device (CPU, MPS, CUDA). They are the baseline the Metal
kernels are measured against and the oracle for correctness tests.

Squared distances are computed as ((dx^2 + dy^2) + dz^2), one rounded
elementwise op at a time, like the kernels and PyTorch3D's CPU code. Each op
is a separate tensor op, so nothing is fused into an FMA.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Pairwise distances are computed in query chunks of at most this many
# (query, point) pairs, to bound memory.
_PAIRS_PER_CHUNK = 1 << 24


def _sqdist(a: Tensor, b: Tensor) -> Tensor:
    """((dx^2 + dy^2) + dz^2) between broadcastable (..., 3) tensors."""
    d = a - b
    return (d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1]) + d[..., 2] * d[..., 2]


def furthest_point_sample(
    xyz: Tensor, npoint: int, start_idx: int = 0, skip_near_origin: bool = False
) -> Tensor:
    """Farthest point sampling.

    Args:
        xyz: (B, N, 3) points.
        npoint: number of points to sample.
        start_idx: first sampled index. pointnet2_ops always starts at 0.
        skip_near_origin: never pick points with x^2 + y^2 + z^2 <= 1e-3,
            other than the start index, like pointnet2_ops.

    Returns:
        (B, npoint) int64 indices into ``xyz``. Ties go to the smaller index.
    """
    B, N, _ = xyz.shape
    device = xyz.device
    idx = torch.empty(B, npoint, dtype=torch.long, device=device)
    min_d2 = torch.full((B, N), float("inf"), dtype=xyz.dtype, device=device)
    if skip_near_origin:
        zero = torch.zeros(3, dtype=xyz.dtype, device=device)
        # mag <= 1e-3 in double is mag < fl32(1e-3) for float32 mag.
        skipped = _sqdist(xyz, zero) < torch.tensor(1e-3, dtype=xyz.dtype)
        min_d2[skipped] = -float("inf")
    farthest = torch.full((B,), start_idx, dtype=torch.long, device=device)
    batch = torch.arange(B, device=device)
    for i in range(npoint):
        idx[:, i] = farthest
        centroid = xyz[batch, farthest].unsqueeze(1)
        min_d2 = torch.minimum(min_d2, _sqdist(xyz, centroid))
        farthest = min_d2.argmax(-1)
    return idx


def knn(query: Tensor, ref: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """PyTorch cdist/topk kNN for coordinate or feature vectors.

    This is the plain PyTorch approach (``cdist`` then ``topk``), used as the
    benchmark baseline. ``cdist`` may use a matrix multiply, so distances and
    the order of near ties can differ from the Metal kernel. Tests compare the
    kernel against an exact oracle instead.

    Args:
        query: (B, M, D) query points or features.
        ref: (B, N, D) reference points or features.
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
    radius, and distances as ((dx^2 + dy^2) + dz^2), as PyTorch3D's CPU code
    does.

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
    kk = min(K, N)
    idx = torch.full((B, M, K), -1, dtype=torch.long, device=query.device)
    dist2 = torch.zeros(B, M, K, dtype=query.dtype, device=query.device)
    if kk == 0 or M == 0:
        return dist2, idx

    order = torch.arange(N, device=ref.device)
    step = max(1, _PAIRS_PER_CHUNK // max(1, B * N))
    for lo in range(0, M, step):
        d2 = _sqdist(query[:, lo : lo + step, None, :], ref[:, None, :, :])
        cand = torch.where(d2 < r2, order, N)
        first = cand.topk(kk, dim=-1, largest=False, sorted=True).values
        valid = first < N
        idx[:, lo : lo + step, :kk] = torch.where(valid, first, -1)
        dist2[:, lo : lo + step, :kk] = torch.where(valid, d2.gather(-1, first.clamp(max=N - 1)), 0.0)
    return dist2, idx

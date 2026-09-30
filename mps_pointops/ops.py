"""Point cloud ops backed by Metal kernels on MPS.

Tensors on other devices fall back to the pure PyTorch versions in
``reference``, so the same call works everywhere.
"""

from __future__ import annotations

import math
from functools import cache
from importlib import resources

import torch
from torch import Tensor

from . import reference
from ._ball_query_mps import _checked_radius_and_k, _require_compile_shader, ball_query as _metal_ball_query

# Threads per threadgroup. 1024 is the Apple GPU maximum.
_THREADS = 1024
# Must match QUERIES_PER_GROUP and MAX_K in kernels/knn.metal.
_KNN_QUERIES_PER_GROUP = 8
_KNN_MAX_K = 256


@cache
def _check_simd_width() -> None:
    """The kernels assume 32-wide simdgroups. Check once, on first use."""
    _require_compile_shader()
    lib = torch.mps.compile_shader(
        "kernel void width(device long* out, uint w [[threads_per_simdgroup]]) { out[0] = w; }"
    )
    out = torch.zeros(1, dtype=torch.long, device="mps")
    lib.width(out, threads=1, group_size=1)
    width = int(out.item())
    if width != 32:
        raise RuntimeError(f"mps_pointops kernels need 32-wide simdgroups, this GPU uses {width}")


@cache
def _library(name: str):
    _check_simd_width()
    source = resources.files(__package__).joinpath("kernels", f"{name}.metal").read_text()
    return torch.mps.compile_shader(source)


def furthest_point_sample(
    xyz: Tensor, npoint: int, start_idx: int = 0, skip_near_origin: bool = False
) -> Tensor:
    """Farthest point sampling.

    Same contract as ``reference.furthest_point_sample``: sampling starts at
    ``start_idx`` and ties go to the smaller index. Once every point has been
    sampled, the remaining slots repeat index 0. For a drop-in replacement of
    ``pointnet2_ops`` (int32 output, points near the origin skipped), use
    ``mps_pointops.compat``.

    Args:
        xyz: (B, N, 3) float32 points.
        npoint: number of points to sample.
        start_idx: first sampled index.
        skip_near_origin: never pick points with x^2 + y^2 + z^2 <= 1e-3,
            other than the start index, like pointnet2_ops.

    Returns:
        (B, npoint) int64 indices into ``xyz``.
    """
    if xyz.dim() != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"xyz must have shape (B, N, 3), got {tuple(xyz.shape)}")
    B, N, _ = xyz.shape
    if npoint < 0:
        raise ValueError(f"npoint must be >= 0, got {npoint}")
    if npoint > 0 and N == 0:
        raise ValueError("cannot sample from an empty point cloud")
    if npoint > 0 and not 0 <= start_idx < N:
        raise IndexError(f"start_idx {start_idx} is out of range for {N} points")
    if xyz.device.type != "mps":
        return reference.furthest_point_sample(xyz, npoint, start_idx, skip_near_origin)
    if xyz.dtype != torch.float32:
        raise TypeError(f"xyz must be float32 on MPS, got {xyz.dtype}")

    out = torch.empty(B, npoint, dtype=torch.long, device=xyz.device)
    if B == 0 or npoint == 0:
        return out
    xyz = xyz.contiguous()
    min_d2 = torch.empty(B, N, dtype=torch.float32, device=xyz.device)
    _library("fps").furthest_point_sample(
        xyz, min_d2, out, N, npoint, start_idx, int(skip_near_origin),
        threads=(_THREADS, B), group_size=(_THREADS, 1),
    )
    return out


def knn(query: Tensor, ref: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """Brute-force k nearest neighbors, like ``knn_cuda.KNN(k, transpose_mode=True)``.

    Neighbors are sorted by squared distance, then by index, so ties are
    deterministic. On MPS, ``k`` can be at most 256.

    Args:
        query: (B, M, 3) float32 query points.
        ref: (B, N, 3) float32 reference points.
        k: number of neighbors, at most N.

    Returns:
        dist: (B, M, k) Euclidean distances, ascending.
        idx: (B, M, k) int64 indices into ``ref``.
    """
    if query.dim() != 3 or query.shape[-1] != 3:
        raise ValueError(f"query must have shape (B, M, 3), got {tuple(query.shape)}")
    if ref.dim() != 3 or ref.shape[-1] != 3:
        raise ValueError(f"ref must have shape (B, N, 3), got {tuple(ref.shape)}")
    if query.device != ref.device:
        raise ValueError(f"query and ref are on different devices: {query.device} and {ref.device}")
    if query.shape[0] != ref.shape[0]:
        raise ValueError(f"batch sizes differ: {query.shape[0]} and {ref.shape[0]}")
    B, M, _ = query.shape
    N = ref.shape[1]
    if not 0 <= k <= N:
        raise ValueError(f"k must be in [0, {N}], got {k}")
    if query.device.type != "mps" or ref.device.type != "mps":
        return reference.knn(query, ref, k)
    if query.dtype != torch.float32 or ref.dtype != torch.float32:
        raise TypeError(f"query and ref must be float32 on MPS, got {query.dtype} and {ref.dtype}")
    if k > _KNN_MAX_K:
        raise ValueError(f"k must be at most {_KNN_MAX_K} on MPS, got {k}")

    dist = torch.empty(B, M, k, dtype=torch.float32, device=query.device)
    idx = torch.empty(B, M, k, dtype=torch.long, device=query.device)
    if B == 0 or M == 0 or k == 0:
        return dist, idx
    groups = -(-M // _KNN_QUERIES_PER_GROUP)
    chunks = -(-N // 32)
    stride = round(chunks * 0.6180339887) % chunks
    while math.gcd(stride, chunks) != 1:
        stride += 1
    group = 32 * _KNN_QUERIES_PER_GROUP
    _library("knn").knn(
        query.contiguous(), ref.contiguous(), dist, idx, M, N, k, stride,
        threads=(groups * group, B), group_size=(group, 1),
    )
    return dist, idx


def ball_query(query: Tensor, ref: Tensor, radius: float, K: int) -> tuple[Tensor, Tensor]:
    """First-K radius search with squared distances and -1 padded indices.

    Input and output shapes are (B, M, 3), (B, N, 3) and (B, M, K).
    MPS float32/float16 inputs use the Metal kernel; other devices use the
    PyTorch reference. The MPS kernel supports coordinate gradients.
    """
    # Same radius and K rules on every device, so CPU and MPS accept and
    # reject the same arguments.
    _checked_radius_and_k(radius, K)
    if query.device.type == "mps" or ref.device.type == "mps":
        result = _metal_ball_query(query, ref, radius=radius, k=K)
        return result.distances, result.indices
    if query.dim() != 3 or ref.dim() != 3 or query.shape[-1] != 3 or ref.shape[-1] != 3:
        raise ValueError("query and ref must have shapes (B, M, 3) and (B, N, 3)")
    if query.shape[0] != ref.shape[0]:
        raise ValueError("query and ref must have the same batch size")
    if query.device != ref.device:
        raise ValueError(f"query and ref are on different devices: {query.device} and {ref.device}")
    return reference.ball_query(query, ref, radius, K)

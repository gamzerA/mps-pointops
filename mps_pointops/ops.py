"""Point cloud ops backed by Metal kernels on MPS.

Tensors on other devices fall back to the pure PyTorch versions in
``reference``, so the same call works everywhere.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import torch
from torch import Tensor

from . import reference

# Threads per threadgroup. 1024 is the Apple GPU maximum.
_THREADS = 1024


@cache
def _library(name: str):
    source = resources.files(__package__).joinpath("kernels", f"{name}.metal").read_text()
    return torch.mps.compile_shader(source)


def furthest_point_sample(xyz: Tensor, npoint: int, start_idx: int = 0) -> Tensor:
    """Farthest point sampling, like ``pointnet2_ops.furthest_point_sample``.

    Same contract as ``reference.furthest_point_sample``: sampling starts at
    ``start_idx`` and ties go to the smaller index. Once every point has been
    sampled, the remaining slots repeat index 0.

    Args:
        xyz: (B, N, 3) float32 points.
        npoint: number of points to sample.
        start_idx: first sampled index.

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
        return reference.furthest_point_sample(xyz, npoint, start_idx)
    if xyz.dtype != torch.float32:
        raise TypeError(f"xyz must be float32 on MPS, got {xyz.dtype}")

    out = torch.empty(B, npoint, dtype=torch.long, device=xyz.device)
    if B == 0 or npoint == 0:
        return out
    xyz = xyz.contiguous()
    min_d2 = torch.empty(B, N, dtype=torch.float32, device=xyz.device)
    _library("fps").furthest_point_sample(
        xyz, min_d2, out, N, npoint, start_idx,
        threads=(_THREADS, B), group_size=(_THREADS, 1),
    )
    return out

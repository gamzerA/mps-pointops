"""Farthest point sampling for flat, variable-length point clouds on MPS."""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import torch
from torch import Tensor

from ._ball_query_mps import _require_compile_shader


_THREADS = 1024
_MAX_LOCAL_POINTS = (1 << 32) - 1


@lru_cache(maxsize=1)
def _shader_library():
    _require_compile_shader()
    width_shader = torch.mps.compile_shader(
        "kernel void width(device long* out, uint w [[threads_per_simdgroup]]) { out[0] = w; }"
    )
    width = torch.empty(1, dtype=torch.long, device="mps")
    width_shader.width(width, threads=1, group_size=1)
    actual_width = int(width.item())
    if actual_width != 32:
        raise RuntimeError(
            f"mps_pointops kernels need 32-wide simdgroups, this GPU uses {actual_width}"
        )
    source = files(__package__).joinpath("kernels", "fps_flat.metal").read_text()
    return torch.mps.compile_shader(source)


def fps_flat(x: Tensor, ptr: Tensor, out_ptr: Tensor, starts: Tensor) -> Tensor:
    """Sample flat ``x`` by segment and return global int64 point indices.

    ``ptr[b:b+2]`` bounds input cloud ``b``; ``out_ptr[b:b+2]`` bounds its
    output. ``starts[b]`` is a *local* index in that cloud. A cloud with zero
    requested samples may be empty. As in the existing FPS kernel, requests
    longer than a cloud repeat its local index zero after exhaustion.
    """
    if not all(isinstance(t, Tensor) for t in (x, ptr, out_ptr, starts)):
        raise TypeError("x, ptr, out_ptr, and starts must be torch.Tensor values")
    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError(f"x must have shape (Total_N, 3), got {tuple(x.shape)}")
    if ptr.ndim != 1 or ptr.numel() == 0:
        raise ValueError("ptr must be a nonempty one-dimensional tensor")
    batch = ptr.numel() - 1
    if out_ptr.shape != (batch + 1,) or starts.shape != (batch,):
        raise ValueError("out_ptr must have shape (B+1,) and starts must have shape (B,)")
    if x.device.type != "mps" or any(t.device != x.device for t in (ptr, out_ptr, starts)):
        raise ValueError("x, ptr, out_ptr, and starts must be on the same MPS device")
    if x.dtype != torch.float32:
        raise TypeError(f"x must be float32 on MPS, got {x.dtype}")
    if any(t.dtype != torch.int64 for t in (ptr, out_ptr, starts)):
        raise TypeError("ptr, out_ptr, and starts must be int64")
    if batch > (1 << 32) - 1:
        raise ValueError("B exceeds the Metal grid limit")

    x = x.contiguous()
    ptr = ptr.contiguous()
    out_ptr = out_ptr.contiguous()
    starts = starts.contiguous()
    lengths = ptr[1:] - ptr[:-1]
    counts = out_ptr[1:] - out_ptr[:-1]
    invalid = (
        (ptr[0] != 0)
        | (ptr[-1] != x.shape[0])
        | (out_ptr[0] != 0)
        | torch.any(lengths < 0)
        | torch.any(lengths > _MAX_LOCAL_POINTS)
        | torch.any(counts < 0)
        | torch.any((counts > 0) & ((lengths == 0) | (starts < 0) | (starts >= lengths)))
    )
    # Output shape is data-dependent. Transfer only its total size and the
    # validation flag; coordinates and segment offsets remain on the GPU.
    output_size, is_invalid = torch.stack((out_ptr[-1], invalid.to(torch.int64))).cpu().tolist()
    if is_invalid:
        raise ValueError("invalid ptr, out_ptr, or starts for the input point clouds")

    out = torch.empty(output_size, dtype=torch.int64, device=x.device)
    if batch == 0 or output_size == 0:
        return out
    min_d2 = torch.empty(x.shape[0], dtype=torch.float32, device=x.device)
    _shader_library().fps_flat(
        x,
        ptr,
        out_ptr,
        starts,
        min_d2,
        out,
        threads=(_THREADS, batch),
        group_size=(_THREADS, 1),
    )
    return out

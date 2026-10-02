"""Private Morton-key builder shared by the packaged spatial index and benchmarks."""

from __future__ import annotations

import math
from functools import lru_cache
from importlib import resources

import torch
from torch import Tensor

_GROUP_SIZE = 256


@lru_cache(maxsize=1)
def _library():
    source = resources.files(__package__).joinpath("kernels", "spatial_keys.metal").read_text()
    return torch.mps.compile_shader(source)


def morton_keys_f32(points: Tensor, origin: Tensor, cell_size: float) -> tuple[Tensor, Tensor]:
    """Produce signed int64 keys and uint8 invalid flags for MPS float32 XYZ."""
    if points.device.type != "mps" or points.dtype != torch.float32 or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be MPS float32 [N, 3]")
    if origin.device != points.device or origin.dtype != torch.float32 or origin.shape != (3,):
        raise ValueError("origin must be MPS float32 [3]")
    if not math.isfinite(cell_size) or cell_size <= 0:
        raise ValueError("cell_size must be finite and positive")
    if points.shape[0] >= 2**32:
        raise ValueError("spatial index supports fewer than 2**32 points")
    keys = torch.empty((len(points),), dtype=torch.int64, device=points.device)
    invalid = torch.empty((len(points),), dtype=torch.uint8, device=points.device)
    if len(points):
        groups = (len(points) + _GROUP_SIZE - 1) // _GROUP_SIZE
        _library().spatial_morton_keys_f32(
            points.contiguous(), origin.contiguous(), keys, invalid,
            len(points), float(cell_size),
            threads=[groups * _GROUP_SIZE, 1, 1],
            group_size=[_GROUP_SIZE, 1, 1],
        )
    return keys, invalid


def morton_key_cpu(x: int, y: int, z: int) -> int:
    """Bit-exact integer oracle for the 21-bit key packing."""
    if not all(0 <= value < 2**21 for value in (x, y, z)):
        raise ValueError("each cell coordinate must fit 21 bits")
    key = 0
    for bit in range(21):
        key |= ((x >> bit) & 1) << (3 * bit)
        key |= ((y >> bit) & 1) << (3 * bit + 1)
        key |= ((z >> bit) & 1) << (3 * bit + 2)
    return key

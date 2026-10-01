"""Experimental tensor path for pyg-lib's grid-clustering operator.

PyG 2.8 appends the batch number to spatial coordinates before calling the
``pyg::grid_cluster`` dispatcher. This module implements that operator's
finite float32 arithmetic on CPU and MPS; the legacy ``torch_cluster`` shim
has a separate entry point in :mod:`mps_pointops.grid`.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _grid_vector(value: Tensor, name: str, pos: Tensor) -> Tensor:
    dimensions = pos.shape[1]
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (dimensions,):
        raise ValueError(f"{name} must have shape ({dimensions},)")
    if value.dtype != torch.float32 or value.device != pos.device:
        raise ValueError(f"{name} must be float32 on {pos.device}")
    return value


def grid_cluster_ids(
    pos: Tensor,
    size: Tensor,
    start: Tensor | None = None,
    end: Tensor | None = None,
) -> Tensor:
    """Return pyg-lib 0.7 style mixed-radix voxel IDs.

    ``pos`` is ``(N, D)`` finite float32 on CPU or MPS, with ``D`` in
    ``{2, 3, 4}`` and ``N > 0``. ``size``, ``start`` and ``end`` are float32
    vectors of length ``D`` on the same device. ``size`` must be positive.
    Bounds default to the coordinate-wise extrema. Returned IDs are int64 on
    the input device; they are not compacted cluster labels.

    Nonfinite values, nonpositive sizes, inverted bounds and int64 overflow
    are outside this experimental contract. Gradients are not defined.
    """
    if not isinstance(pos, Tensor):
        raise TypeError("pos must be a torch.Tensor")
    if pos.ndim != 2 or pos.shape[1] not in (2, 3, 4):
        raise ValueError("pos must have shape (N, D), with D in {2, 3, 4}")
    if pos.dtype != torch.float32:
        raise TypeError("pos must be float32")
    if pos.device.type not in ("cpu", "mps"):
        raise ValueError("pos must be on CPU or MPS")
    if pos.shape[0] == 0:
        raise ValueError("pos must contain at least one point")
    size = _grid_vector(size, "size", pos)
    if start is not None:
        start = _grid_vector(start, "start", pos)
    if end is not None:
        end = _grid_vector(end, "end", pos)
    if start is None:
        start = pos.min(dim=0).values
    if end is None:
        end = pos.max(dim=0).values

    # Each float32 quotient is rounded, then truncated toward zero before
    # conversion to int64. Spatial dimensions are least significant; PyG's
    # appended batch dimension therefore keeps equal spatial cells separate.
    coordinates = torch.div(pos - start, size, rounding_mode="trunc").to(torch.long)
    extents = torch.div(end - start, size, rounding_mode="trunc").to(torch.long) + 1
    strides = torch.cat((
        torch.ones((1,), dtype=torch.long, device=pos.device),
        extents[:-1].cumprod(dim=0),
    ))
    return (coordinates * strides).sum(dim=1)

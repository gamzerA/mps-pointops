"""Experimental 3D grid clustering for the ``torch_cluster`` call surface.

The CPU and MPS paths use PyTorch tensor operators. The returned values are
linear voxel identifiers, not compact consecutive cluster labels.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _vector(value: Tensor, name: str, pos: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {tuple(value.shape)}")
    if value.dtype != torch.float32 or value.device != pos.device:
        raise ValueError(f"{name} must be float32 on {pos.device}")
    return value


def grid_cluster(
    pos: Tensor,
    size: Tensor,
    start: Tensor | None = None,
    end: Tensor | None = None,
) -> Tensor:
    """Return ``torch_cluster`` 1.6.3 style voxel IDs for finite 3D points.

    ``pos`` is ``(N, 3)`` float32 on CPU or MPS. ``size`` is a positive
    ``(3,)`` float32 tensor on the same device. Optional ``start`` and ``end``
    have the same vector shape, dtype and device. If omitted, each is the
    dimensionwise minimum or maximum of ``pos``. ``N`` must be nonzero, as in
    the upstream 1.6.3 operator.

    Like upstream, converting each quotient to int64 truncates toward zero.
    A point just below ``start`` can therefore share its voxel with a point
    just above ``start``. The output is an ``(N,)`` int64 tensor on the input
    device. No coordinate or size gradients are defined.

    Nonfinite values, nonpositive sizes, bounds with ``end < start`` and
    mixed-radix products that overflow int64 are outside this initial scope.
    """
    if not isinstance(pos, Tensor):
        raise TypeError("pos must be a torch.Tensor")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"pos must have shape (N, 3), got {tuple(pos.shape)}")
    if pos.dtype != torch.float32:
        raise TypeError("pos must be float32")
    if pos.device.type not in ("cpu", "mps"):
        raise ValueError("pos must be on CPU or MPS")
    size = _vector(size, "size", pos)
    if start is not None:
        start = _vector(start, "start", pos)
    if end is not None:
        end = _vector(end, "end", pos)
    if pos.shape[0] == 0:
        raise ValueError("pos must contain at least one point")
    if start is None:
        start = pos.min(dim=0).values
    if end is None:
        end = pos.max(dim=0).values

    # The upstream CPU operator subtracts, divides, then converts to int64;
    # it does not floor negative quotients. Preserve that operation order.
    voxel = ((pos - start) / size).to(torch.int64)
    extent = ((end - start) / size).to(torch.int64) + 1
    return voxel[:, 0] + voxel[:, 1] * extent[0] + voxel[:, 2] * extent[0] * extent[1]

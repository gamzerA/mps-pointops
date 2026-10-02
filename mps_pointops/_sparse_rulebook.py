"""Experimental CPU coordinate contract and 3D sparse-convolution rulebooks.

This is a deliberately small, independent oracle for future Metal work.  It
does not implement ``spconv`` operators, weight layouts, or a public API.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Sequence

import torch


Triplet = tuple[int, int, int]


@dataclass(frozen=True)
class SparseTensor3D:
    indices: torch.Tensor
    features: torch.Tensor
    spatial_shape: Triplet
    batch_size: int


@dataclass(frozen=True)
class Rulebook3D:
    """Rows in ``pairs`` are ``(kernel_offset, input_row, output_row)``.

    Kernel offsets are flattened in spatial-axis order, last axis fastest.
    Rows are ordered first by kernel offset and then by output row.  This
    ordering is an oracle convention; spconv's internal pair order is not an
    API guarantee.
    """

    output_indices: torch.Tensor
    output_spatial_shape: Triplet
    pairs: torch.Tensor
    kernel_size: Triplet


def _triple(name: str, value: int | Sequence[int], *, minimum: int) -> Triplet:
    if isinstance(value, int) and not isinstance(value, bool):
        values = (value, value, value)
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be an integer or three integers") from exc
    if len(values) != 3 or any(
        not isinstance(item, int) or isinstance(item, bool) or item < minimum
        for item in values
    ):
        raise ValueError(f"{name} must contain three integers >= {minimum}")
    return values  # type: ignore[return-value]


def validate_sparse_tensor_3d(
    indices: torch.Tensor,
    features: torch.Tensor,
    spatial_shape: Sequence[int],
    batch_size: int,
) -> SparseTensor3D:
    """Validate the CPU/float32-or-float64 subset used by this oracle.

    Coordinates are ``[batch, axis0, axis1, axis2]`` and must be unique.
    Axis order must agree with ``spatial_shape`` and convolution parameters.
    No sort or duplicate coalescing occurs; feature row ``i`` belongs to
    coordinate row ``i``.  Empty inputs are valid.
    """

    if not isinstance(indices, torch.Tensor) or not isinstance(features, torch.Tensor):
        raise TypeError("indices and features must be torch tensors")
    if indices.device.type != "cpu" or features.device.type != "cpu":
        raise ValueError("the experimental rulebook oracle requires CPU tensors")
    if indices.ndim != 2 or indices.shape[1] != 4 or indices.dtype != torch.int32:
        raise ValueError("indices must have shape [N, 4] and dtype torch.int32")
    if features.ndim != 2 or features.shape[0] != indices.shape[0] or features.shape[1] < 1:
        raise ValueError("features must have shape [N, C] with C >= 1")
    if features.dtype not in (torch.float32, torch.float64):
        raise ValueError("features must have dtype torch.float32 or torch.float64")
    shape = _triple("spatial_shape", spatial_shape, minimum=1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
        raise ValueError("batch_size must be a nonnegative integer")
    seen: dict[tuple[int, int, int, int], int] = {}
    for row, values in enumerate(indices.tolist()):
        coord = tuple(int(value) for value in values)
        if not 0 <= coord[0] < batch_size or any(
            not 0 <= coord[axis + 1] < shape[axis] for axis in range(3)
        ):
            raise ValueError(f"index row {row} lies outside batch or spatial_shape")
        prior = seen.get(coord)
        if prior is not None:
            raise ValueError(f"duplicate coordinate at rows {prior} and {row}")
        seen[coord] = row
    return SparseTensor3D(indices, features, shape, batch_size)


def _offsets(kernel_size: Triplet):
    return enumerate(product(*(range(size) for size in kernel_size)))


def _pair_tensor(rows: list[tuple[int, int, int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.int64).reshape(-1, 3)


def generate_subm_rulebook(
    sparse: SparseTensor3D,
    *,
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> Rulebook3D:
    """Preserve input coordinates; center an odd kernel at each active output.

    For output ``o`` and offset ``k``, the input coordinate is
    ``p = o + (k - floor(K/2)) * dilation`` on each spatial axis.  We support
    odd kernels only so that the center is unambiguous.  Stride is one and
    padding is implicit in the center; the upstream SubM index generator
    accepts kernel size and dilation, but not explicit padding or stride.
    """

    kernel = _triple("kernel_size", kernel_size, minimum=1)
    dil = _triple("dilation", dilation, minimum=1)
    if any(size % 2 == 0 for size in kernel):
        raise ValueError("the SubM oracle requires odd kernel sizes")
    coords = [tuple(int(v) for v in row) for row in sparse.indices.tolist()]
    input_row = {coord: row for row, coord in enumerate(coords)}
    center = tuple(size // 2 for size in kernel)
    pairs: list[tuple[int, int, int]] = []
    for flat_offset, offset in _offsets(kernel):
        for output_row, coord in enumerate(coords):
            source = (coord[0],) + tuple(
                coord[axis + 1] + (offset[axis] - center[axis]) * dil[axis]
                for axis in range(3)
            )
            row = input_row.get(source)
            if row is not None:
                pairs.append((flat_offset, row, output_row))
    return Rulebook3D(sparse.indices.clone(), sparse.spatial_shape, _pair_tensor(pairs), kernel)


def generate_sparse_conv_rulebook(
    sparse: SparseTensor3D,
    *,
    kernel_size: int | Sequence[int],
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
) -> Rulebook3D:
    """Build active outputs for ordinary 3D sparse cross-correlation.

    Per axis, ``p = o * stride - padding + k * dilation``.  The dense output
    spatial size is ``floor((S + 2P - D(K-1) - 1)/T) + 1``.  All reachable
    output coordinates are sorted lexicographically in batch-first order.
    """

    kernel = _triple("kernel_size", kernel_size, minimum=1)
    steps = _triple("stride", stride, minimum=1)
    pads = _triple("padding", padding, minimum=0)
    dil = _triple("dilation", dilation, minimum=1)
    out_shape = tuple(
        (sparse.spatial_shape[i] + 2 * pads[i] - dil[i] * (kernel[i] - 1) - 1)
        // steps[i]
        + 1
        for i in range(3)
    )
    if any(size < 1 for size in out_shape):
        raise ValueError("convolution output spatial shape must be positive")

    input_coords = [tuple(int(v) for v in row) for row in sparse.indices.tolist()]
    input_row = {coord: row for row, coord in enumerate(input_coords)}
    reachable: set[tuple[int, int, int, int]] = set()
    for coord in input_coords:
        for _, offset in _offsets(kernel):
            numerators = tuple(
                coord[axis + 1] + pads[axis] - offset[axis] * dil[axis]
                for axis in range(3)
            )
            if any(numerators[axis] % steps[axis] for axis in range(3)):
                continue
            output = tuple(numerators[axis] // steps[axis] for axis in range(3))
            if all(0 <= output[axis] < out_shape[axis] for axis in range(3)):
                if any(value > 2**31 - 1 for value in output):
                    raise ValueError("active output coordinate exceeds int32 range")
                reachable.add((coord[0], *output))
    outputs = sorted(reachable)
    pairs: list[tuple[int, int, int]] = []
    for flat_offset, offset in _offsets(kernel):
        for output_row, output in enumerate(outputs):
            source = (output[0],) + tuple(
                output[axis + 1] * steps[axis]
                - pads[axis]
                + offset[axis] * dil[axis]
                for axis in range(3)
            )
            row = input_row.get(source)
            if row is not None:
                pairs.append((flat_offset, row, output_row))
    output_indices = torch.tensor(outputs, dtype=torch.int32).reshape(-1, 4)
    return Rulebook3D(output_indices, out_shape, _pair_tensor(pairs), kernel)

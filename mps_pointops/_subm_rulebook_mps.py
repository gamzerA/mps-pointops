"""Experimental MPS construction of a SubM sparse rulebook.

Four stable device sorts make a lexicographic index over the full int32
coordinate tuple. PyTorch 2.7 uses a small Metal gather before each sort to
avoid its MPS int32 index_select boundary bug. Each Metal lookup then uses
O(log N) binary-search comparisons instead of scanning N rows.
Total work also includes the MPS sorts, whose algorithm and complexity depend
on the PyTorch backend, and linear pair compaction. The logical count remains
on the MPS device.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Sequence

import torch
from torch import Tensor

from ._ball_query_mps import _require_compile_shader
from ._sparse_rulebook import _triple


_GROUP_SIZE = 256
# The lookup and compaction kernels use a uint grid position. Leave one
# position for output_ptr[N] when the kernel volume is one. Each slot also
# needs several temporary/padded tensors, so the device allocator may reject
# large requests well before this dispatch limit.
_MAX_SLOTS = 2**32 - 2

# PyTorch 2.7 MPS index_select/gather can corrupt nearby large int32 values.
# Its argsort is correct when the source values arrive intact. Use direct
# Metal reads for sorted coordinate values on that version only.
_TORCH_MAJOR_MINOR = tuple(
    int(part) for part in torch.__version__.split("+", 1)[0].split(".")[:2]
)
_METAL_GATHER_I32 = _TORCH_MAJOR_MINOR < (2, 8)


@dataclass(frozen=True)
class SubmRulebookMPS:
    """Padded GPU storage and a GPU-resident logical pair count.

    ``pairs[:pair_count]`` is ordered by offset then output row. Each row is
    ``(offset, input_row, output_row)``. The padded suffix is all -1.
    ``output_ptr/sources/offsets`` are output CSR, ordered by output row then
    offset, and can be passed directly to the private Metal SubM forward.
    ``pair_count`` is a one-element MPS int64 tensor; reading its value on CPU
    synchronizes, so callers should normally use the CSR without readback.
    """

    output_indices: Tensor
    pairs: Tensor
    pair_count: Tensor
    output_ptr: Tensor
    output_sources: Tensor
    output_offsets: Tensor
    kernel_size: tuple[int, int, int]


@lru_cache(maxsize=1)
def _library():
    _require_compile_shader()
    source = resources.files(__package__).joinpath("kernels", "subm_rulebook.metal").read_text()
    return torch.mps.compile_shader(source)


def generate_subm_rulebook_mps(
    indices: Tensor,
    *,
    kernel_size: int | Sequence[int],
    dilation: int | Sequence[int] = 1,
) -> SubmRulebookMPS:
    """Create a private SubM rulebook entirely on the MPS device.

    Input must be unique, valid MPS int32 coordinates ``[N,4]`` with columns
    ``[batch, axis0, axis1, axis2]``. Device-side coordinate validation and
    duplicate rejection are not provided. Callers can prevalidate on CPU with
    ``validate_sparse_tensor_3d`` before transferring, or otherwise guarantee
    this precondition. Input row order becomes output coordinate row order.

    Odd kernels and positive dilations follow ``generate_subm_rulebook``.
    The one-dimensional Metal dispatch supports at most 2**32-2
    (kernel offset, output row) slots. Device memory can impose a lower
    practical limit. Coordinates use all four signed int32 fields as the
    search key; target coordinates outside int32 cannot match any input.
    """
    if not isinstance(indices, Tensor):
        raise TypeError("indices must be a torch.Tensor")
    if indices.device.type != "mps" or indices.dtype != torch.int32 or indices.ndim != 2 or indices.shape[1] != 4:
        raise ValueError("indices must be MPS int32 [N,4]")
    kernel = _triple("kernel_size", kernel_size, minimum=1)
    dil = _triple("dilation", dilation, minimum=1)
    if any(size % 2 == 0 for size in kernel):
        raise ValueError("the SubM rulebook requires odd kernel sizes")
    if any(value > 2**31 - 1 for value in (*kernel, *dil)):
        raise ValueError("kernel_size and dilation must fit Metal int32 parameters")
    rows = indices.shape[0]
    volume = kernel[0] * kernel[1] * kernel[2]
    slots = rows * volume
    if volume > _MAX_SLOTS or slots > _MAX_SLOTS:
        raise ValueError(f"MPS rulebook exceeds the Metal 1-D dispatch limit of {_MAX_SLOTS} slots")

    device = indices.device
    # CPU oracle output_indices is a clone. Preserve the same snapshot even
    # when the caller supplies an already-contiguous MPS tensor and mutates it
    # immediately after this asynchronous call.
    stable_indices = indices.clone(memory_format=torch.contiguous_format)
    pairs = torch.full((slots, 3), -1, dtype=torch.int64, device=device)
    output_sources = torch.full((slots,), -1, dtype=torch.int64, device=device)
    output_offsets = torch.full((slots,), -1, dtype=torch.int64, device=device)
    output_ptr = torch.empty((rows + 1,), dtype=torch.int64, device=device)
    if slots == 0:
        output_ptr.zero_()
        return SubmRulebookMPS(
            stable_indices, pairs, torch.zeros((1,), dtype=torch.int64, device=device),
            output_ptr, output_sources, output_offsets, kernel,
        )

    # Stable sorts from the last coordinate field to the first produce a
    # lexicographic row permutation. PyTorch 2.7's index_select can corrupt
    # large int32 coordinates, so Metal gathers those values directly before
    # sorting on that version. No coordinate values move to the host.
    # Sorting only row IDs leaves output order and source row IDs unchanged.
    sorted_rows = torch.arange(rows, dtype=torch.int64, device=device)
    for axis in (3, 2, 1, 0):
        if _METAL_GATHER_I32:
            axis_values = torch.empty((rows,), dtype=torch.int32, device=device)
            _library().subm_rulebook_gather_axis_i32(
                stable_indices, sorted_rows, axis_values, rows, axis,
                threads=rows, group_size=min(rows, _GROUP_SIZE),
            )
        else:
            axis_values = stable_indices.index_select(0, sorted_rows)[:, axis]
        sorted_rows = sorted_rows.index_select(
            0, torch.argsort(axis_values, stable=True)
        )

    dense_sources = torch.empty((slots,), dtype=torch.int64, device=device)
    valid_offset = torch.empty((slots,), dtype=torch.int32, device=device)
    _library().subm_rulebook_lookup_i32(
        stable_indices, sorted_rows, dense_sources, valid_offset,
        rows, *kernel, *dil,
        threads=slots, group_size=min(slots, _GROUP_SIZE),
    )
    offset_prefix = torch.cumsum(valid_offset, dim=0, dtype=torch.int64)
    valid_output = valid_offset.view(volume, rows).transpose(0, 1).contiguous().view(slots)
    output_prefix = torch.cumsum(valid_output, dim=0, dtype=torch.int64)
    _library().subm_rulebook_compact_i64(
        dense_sources, offset_prefix, output_prefix, pairs,
        output_ptr, output_sources, output_offsets, rows, volume,
        threads=max(slots, rows + 1), group_size=min(max(slots, rows + 1), _GROUP_SIZE),
    )
    return SubmRulebookMPS(
        stable_indices, pairs, offset_prefix[-1:], output_ptr,
        output_sources, output_offsets, kernel,
    )

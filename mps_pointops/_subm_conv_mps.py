"""Private, experimental SubMConv3d forward using a CPU rulebook and Metal.

This is not an ``spconv`` adapter. Coordinates and the deterministic rulebook
are built on CPU; only the feature/weight reduction executes on MPS. Backward
is rejected explicitly until a tested gradient path exists.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Sequence

import torch
from torch import Tensor

from ._ball_query_mps import _require_compile_shader
from ._sparse_rulebook import (
    SparseTensor3D,
    generate_subm_rulebook,
    validate_sparse_tensor_3d,
)


@lru_cache(maxsize=1)
def _library():
    _require_compile_shader()
    source = resources.files(__package__).joinpath("kernels", "subm_conv.metal").read_text()
    return torch.mps.compile_shader(source)


def _output_csr(pairs: Tensor, output_count: int) -> tuple[Tensor, Tensor, Tensor]:
    """Sort rulebook rows by output, then kernel offset, and build output CSR."""
    rows = sorted(pairs.tolist(), key=lambda pair: (pair[2], pair[0]))
    counts = [0] * output_count
    for _, _, output in rows:
        counts[output] += 1
    ptr = [0]
    for count in counts:
        ptr.append(ptr[-1] + count)
    return (
        torch.tensor(ptr, dtype=torch.int64),
        torch.tensor([row[1] for row in rows], dtype=torch.int64),
        torch.tensor([row[0] for row in rows], dtype=torch.int64),
    )


def subm_conv3d_forward_mps(
    indices: Tensor,
    features: Tensor,
    weights: Tensor,
    spatial_shape: Sequence[int],
    batch_size: int,
    *,
    dilation: int | Sequence[int] = 1,
    bias: Tensor | None = None,
) -> Tensor:
    """Calculate a small SubM cross-correlation without an autograd path.

    ``indices`` are unique CPU int32 ``[N,4]`` rows ``[batch,a0,a1,a2]``.
    ``features`` and ``weights`` are contiguous-or-copyable MPS float32 tensors
    ``[N,Cin]`` and ``[Cout,Cin,K0,K1,K2]``. Odd spatial kernel sizes are
    required. Optional MPS float32 ``bias`` is ``[Cout]``. Output is MPS
    float32 ``[N,Cout]`` in the input coordinate row order. This weight layout
    is PyTorch Conv3d's layout, not an asserted ``spconv`` layout.

    The CPU rulebook uses offset-major ordering. We convert it to output CSR
    with offset ascending, then each output/channel has one Metal writer. The
    kernel applies ``fma(feature, weight, accumulator)`` in that order. CPU and
    GPU convolution algorithms may round differently, so bitwise equivalence
    to dense PyTorch is not promised. Gradients are rejected, not dropped.
    """
    if not isinstance(features, Tensor) or not isinstance(weights, Tensor):
        raise TypeError("features and weights must be torch.Tensor values")
    if (
        features.ndim != 2 or features.shape[1] < 1
        or weights.ndim != 5 or weights.shape[0] < 1
        or weights.shape[1] != features.shape[1]
    ):
        raise ValueError("expected features [N,Cin] and weights [Cout,Cin,K0,K1,K2]")
    if features.dtype != torch.float32 or weights.dtype != torch.float32:
        raise TypeError("features and weights must be float32")
    if features.device.type != "mps" or weights.device != features.device:
        raise ValueError("features and weights must be on the same MPS device")
    if bias is not None and (
        not isinstance(bias, Tensor) or bias.shape != (weights.shape[0],)
        or bias.dtype != torch.float32 or bias.device != features.device
    ):
        raise ValueError("bias must be MPS float32 with shape [Cout]")
    if features.requires_grad or weights.requires_grad or (
        bias is not None and bias.requires_grad
    ):
        raise RuntimeError("experimental SubM Metal forward does not support autograd")

    # Allocate a shape-only CPU feature fixture: no MPS feature transfer or
    # synchronization occurs while validating coordinates.
    coordinate_fixture = torch.empty(features.shape, dtype=torch.float32, device="cpu")
    sparse: SparseTensor3D = validate_sparse_tensor_3d(
        indices, coordinate_fixture, spatial_shape, batch_size
    )
    kernel_size = tuple(int(size) for size in weights.shape[2:])
    rulebook = generate_subm_rulebook(
        sparse, kernel_size=kernel_size, dilation=dilation
    )
    row_count = features.shape[0]
    out_channels = weights.shape[0]
    work = row_count * out_channels
    if work >= 2**32:
        raise ValueError("output scalar count exceeds the Metal 1-D dispatch limit")
    output = torch.empty((row_count, out_channels), dtype=torch.float32, device=features.device)
    if not work:
        return output

    ptr, sources, offsets = _output_csr(rulebook.pairs, row_count)
    ptr = ptr.to(features.device)
    sources = sources.to(features.device)
    offsets = offsets.to(features.device)
    # A scalar bias flag allows a valid nonempty buffer even when bias is None.
    bias_buffer = output if bias is None else bias.contiguous()
    _library().subm_conv3d_f32(
        features.contiguous(), weights.contiguous(), ptr, sources, offsets,
        bias_buffer, output, row_count, features.shape[1], out_channels,
        kernel_size[0] * kernel_size[1] * kernel_size[2], int(bias is not None),
        threads=work, group_size=min(work, 256),
    )
    return output

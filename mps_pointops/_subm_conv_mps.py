"""Private, experimental SubMConv3d using a CPU rulebook and Metal.

This is not an ``spconv`` adapter. Coordinates and the deterministic rulebook
are built on CPU; the forward reduction executes on MPS. First-order backward
uses native PyTorch MPS gathers, matrix products, and indexed accumulation.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

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


class _SubmConv3d(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        features: Tensor,
        weights: Tensor,
        bias: Tensor | None,
        ptr: Tensor,
        sources: Tensor,
        offsets: Tensor,
        pair_sources: Tensor,
        pair_outputs: Tensor,
        offset_ptr: tuple[int, ...],
    ) -> Tensor:
        row_count, in_channels = features.shape
        out_channels = weights.shape[0]
        volume = weights.shape[2] * weights.shape[3] * weights.shape[4]
        work = row_count * out_channels
        output = torch.empty((row_count, out_channels), dtype=torch.float32, device=features.device)
        if work:
            # A scalar bias flag permits a valid buffer even when bias is None.
            bias_buffer = output if bias is None else bias.contiguous()
            _library().subm_conv3d_f32(
                features.contiguous(), weights.contiguous(), ptr, sources, offsets,
                bias_buffer, output, row_count, in_channels, out_channels,
                volume, int(bias is not None), threads=work, group_size=min(work, 256),
            )
        ctx.save_for_backward(features, weights, pair_sources, pair_outputs)
        ctx.offset_ptr = offset_ptr
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        features, weights, pair_sources, pair_outputs = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_features = torch.zeros_like(features) if ctx.needs_input_grad[0] else None
        # zeros_like preserves unusual input strides; a spatial transpose can
        # then make view(Cout, Cin, -1) invalid. A logical, contiguous gradient
        # is accepted by autograd and maps back through the caller's view.
        grad_weights = (
            torch.zeros(weights.shape, dtype=weights.dtype, device=weights.device)
            if ctx.needs_input_grad[1] else None
        )
        grad_bias = (
            grad_output.sum(dim=0)
            if ctx.has_bias and ctx.needs_input_grad[2] else None
        )

        # For pair (k, i, o), Y[o, cout] += X[i, cin] * W[cout, cin, k].
        # Grouping by k makes each dW slice a matrix product. Repeated i rows
        # are accumulated by MPS index_add_; its float reduction order is not
        # specified, so comparisons use a tolerance rather than bit parity.
        cout, cin = weights.shape[:2]
        weight_flat = weights.contiguous().view(cout, cin, -1)
        grad_weight_flat = (
            grad_weights.view(cout, cin, -1) if grad_weights is not None else None
        )
        for offset, (begin, end) in enumerate(zip(ctx.offset_ptr, ctx.offset_ptr[1:])):
            if begin == end:
                continue
            source_rows = pair_sources[begin:end]
            output_rows = pair_outputs[begin:end]
            selected_grad = grad_output.index_select(0, output_rows)
            if grad_features is not None:
                grad_features.index_add_(
                    0, source_rows, selected_grad @ weight_flat[:, :, offset]
                )
            if grad_weight_flat is not None:
                selected_features = features.index_select(0, source_rows)
                grad_weight_flat[:, :, offset].copy_(
                    selected_grad.transpose(0, 1) @ selected_features
                )
        return (
            grad_features, grad_weights, grad_bias,
            None, None, None, None, None, None,
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
    """Calculate a small SubM cross-correlation with first-order gradients.

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
    to dense PyTorch is not promised. First-order gradients use native MPS
    operations; higher derivatives are unsupported. Indexed feature-gradient
    accumulation may have a different floating-point reduction order.
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
    ptr, sources, offsets = _output_csr(rulebook.pairs, row_count)
    ptr = ptr.to(features.device)
    sources = sources.to(features.device)
    offsets = offsets.to(features.device)
    # Avoid extra pair transfers in inference and bias-only backward.
    if torch.is_grad_enabled() and (features.requires_grad or weights.requires_grad):
        # The original rulebook is sorted by kernel offset, then output row.
        pairs = rulebook.pairs
        volume = kernel_size[0] * kernel_size[1] * kernel_size[2]
        counts = torch.bincount(pairs[:, 0], minlength=volume).tolist()
        offset_ptr = [0]
        for count in counts:
            offset_ptr.append(offset_ptr[-1] + int(count))
        pair_sources = pairs[:, 1].contiguous().to(features.device)
        pair_outputs = pairs[:, 2].contiguous().to(features.device)
    else:
        offset_ptr = []
        pair_sources = torch.empty(0, dtype=torch.int64, device=features.device)
        pair_outputs = torch.empty(0, dtype=torch.int64, device=features.device)
    return _SubmConv3d.apply(
        features, weights, bias, ptr, sources, offsets,
        pair_sources, pair_outputs, tuple(offset_ptr),
    )

"""Private, experimental SubMConv3d with CPU or MPS rulebook construction.

This is not an ``spconv`` adapter. Coordinates and the deterministic rulebook
are validated on CPU. The default MPS rulebook path keeps pair construction,
forward, and first-order backward on device without reading pair counts back
to the host. The CPU rulebook remains available as a comparison baseline.
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
from ._subm_rulebook_mps import generate_subm_rulebook_mps


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
        offset_ptr: tuple[int, ...] | Tensor,
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
        ctx.save_for_backward(
            features, weights, pair_sources, pair_outputs, ptr, sources, offsets
        )
        ctx.offset_ptr = offset_ptr
        ctx.gpu_rulebook = isinstance(offset_ptr, Tensor)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        features, weights, pair_sources, pair_outputs, ptr, sources, offsets = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        # A contiguous logical gradient is accepted by autograd even for a
        # caller's noncontiguous feature view, and Metal writes it linearly.
        grad_features = (
            torch.zeros(features.shape, dtype=features.dtype, device=features.device)
            if ctx.needs_input_grad[0] else None
        )
        # A spatially transposed weight view can make zeros_like noncontiguous;
        # keep the logical weight gradient contiguous for the Metal writer.
        grad_weights = (
            torch.zeros(weights.shape, dtype=weights.dtype, device=weights.device)
            if ctx.needs_input_grad[1] else None
        )
        grad_bias = (
            grad_output.sum(dim=0)
            if ctx.has_bias and ctx.needs_input_grad[2] else None
        )

        if ctx.gpu_rulebook:
            row_count, in_channels = features.shape
            out_channels = weights.shape[0]
            volume = weights.shape[2] * weights.shape[3] * weights.shape[4]
            feature_work = row_count * in_channels
            weight_work = weights.numel()
            if grad_features is not None and feature_work:
                _library().subm_conv3d_grad_features_f32(
                    grad_output, weights.contiguous(), ptr, sources, offsets,
                    grad_features, row_count, in_channels, out_channels, volume,
                    threads=feature_work, group_size=min(feature_work, 256),
                )
            if grad_weights is not None and row_count:
                # pair_sources holds the padded offset-major (k, i, o) rows.
                # The GPU offset_ptr bounds each valid segment exactly.
                _library().subm_conv3d_grad_weights_f32(
                    features.contiguous(), grad_output, pair_sources,
                    ctx.offset_ptr, grad_weights, in_channels, out_channels, volume,
                    threads=weight_work, group_size=min(weight_work, 256),
                )
            return (
                grad_features, grad_weights, grad_bias,
                None, None, None, None, None, None,
            )

        # CPU baseline: for pair (k, i, o), Y[o, cout] += X[i, cin] * W[cout, cin, k].
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
    rulebook_backend: str = "mps",
) -> Tensor:
    """Calculate a small SubM cross-correlation with first-order gradients.

    ``indices`` are unique CPU int32 ``[N,4]`` rows ``[batch,a0,a1,a2]``.
    ``features`` and ``weights`` are contiguous-or-copyable MPS float32 tensors
    ``[N,Cin]`` and ``[Cout,Cin,K0,K1,K2]``. Odd spatial kernel sizes are
    required. Optional MPS float32 ``bias`` is ``[Cout]``. Output is MPS
    float32 ``[N,Cout]`` in the input coordinate row order. This weight layout
    is PyTorch Conv3d's layout, not an asserted ``spconv`` layout.

    CPU coordinates are validated for bounds and uniqueness. By default,
    deterministic MPS rulebook construction produces output CSR and the pair
    ranges used by Metal forward/backward without a pair-count host readback.
    ``rulebook_backend="cpu"`` retains the CPU rulebook and PyTorch MPS
    backward as a comparison baseline. Each output/channel has one Metal
    writer, applying ``fma(feature, weight, accumulator)``. CPU and
    GPU convolution algorithms may round differently, so bitwise equivalence
    to dense PyTorch is not promised. First-order gradients use native MPS
    operations; higher derivatives are unsupported. The MPS rulebook path
    uses deterministic one-writer Metal backward reductions, while the CPU
    baseline uses indexed MPS accumulation with unspecified reduction order.
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
    if rulebook_backend not in ("mps", "cpu"):
        raise ValueError("rulebook_backend must be 'mps' or 'cpu'")
    # Allocate a shape-only CPU feature fixture: no MPS feature transfer or
    # synchronization occurs while validating coordinates.
    coordinate_fixture = torch.empty(features.shape, dtype=torch.float32, device="cpu")
    sparse: SparseTensor3D = validate_sparse_tensor_3d(
        indices, coordinate_fixture, spatial_shape, batch_size
    )
    kernel_size = tuple(int(size) for size in weights.shape[2:])
    row_count = features.shape[0]
    out_channels = weights.shape[0]
    work = row_count * out_channels
    if work >= 2**32:
        raise ValueError("output scalar count exceeds the Metal 1-D dispatch limit")
    if row_count * features.shape[1] >= 2**32 or weights.numel() >= 2**32:
        raise ValueError("gradient scalar count exceeds the Metal 1-D dispatch limit")
    if rulebook_backend == "mps":
        rulebook_mps = generate_subm_rulebook_mps(
            sparse.indices.to(features.device),
            kernel_size=kernel_size,
            dilation=dilation,
        )
        ptr = rulebook_mps.output_ptr
        sources = rulebook_mps.output_sources
        offsets = rulebook_mps.output_offsets
        pair_sources = rulebook_mps.pairs
        pair_outputs = torch.empty(0, dtype=torch.int64, device=features.device)
        offset_ptr = rulebook_mps.offset_ptr
        return _SubmConv3d.apply(
            features, weights, bias, ptr, sources, offsets,
            pair_sources, pair_outputs, offset_ptr,
        )

    rulebook = generate_subm_rulebook(
        sparse, kernel_size=kernel_size, dilation=dilation
    )
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

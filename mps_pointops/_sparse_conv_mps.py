"""Private bounded MPS arithmetic for ordinary and keyed inverse SparseConv3d.

The validated coordinate rulebook is built on CPU. Float32 forward and first
backward run on Metal with one writer per output, input-gradient and weight-
gradient scalar. This is not an ``spconv`` shim or a public production API.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import MutableMapping, Sequence

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from ._ball_query_mps import _require_compile_shader
from ._sparse_conv_cpu import (
    MAX_CPU_CANDIDATES, MAX_CPU_KERNEL_VOLUME, MAX_CPU_POINTS,
    SparseConvIndiceData3D, SparseConvResult3D,
)
from ._sparse_rulebook import (
    SparseTensor3D, _triple, generate_sparse_conv_rulebook,
    validate_sparse_tensor_3d,
)
from ._subm_conv_mps import _library as _subm_math_library


@lru_cache(maxsize=1)
def _grad_library():
    _require_compile_shader()
    source = resources.files(__package__).joinpath("kernels", "sparse_conv_grad.metal").read_text()
    return torch.mps.compile_shader(source)


def _validate(sparse: SparseTensor3D, weights: Tensor, bias: Tensor | None,
              *, build: bool) -> tuple[int, int, int]:
    if not isinstance(sparse, SparseTensor3D):
        raise TypeError("sparse must be a SparseTensor3D")
    features = sparse.features
    if not isinstance(features, Tensor) or features.device.type != "mps" or features.dtype != torch.float32:
        raise ValueError("features must be MPS float32 [N,Cin]")
    if features.ndim != 2 or features.shape[1] < 1:
        raise ValueError("features must be MPS float32 [N,Cin]")
    if not isinstance(weights, Tensor) or (
        weights.device != features.device or weights.dtype != torch.float32
        or weights.ndim != 5 or weights.shape[0] < 1
        or weights.shape[1] != features.shape[1]
        or any(size < 1 for size in weights.shape[2:])
    ):
        raise ValueError("weights must be MPS float32 [Cout,Cin,K0,K1,K2]")
    if bias is not None and (
        not isinstance(bias, Tensor) or bias.device != features.device
        or bias.dtype != torch.float32 or bias.shape != (weights.shape[0],)
    ):
        raise ValueError("bias must be MPS float32 [Cout]")
    # Coordinate validation reads CPU int32 indices only, never MPS feature data.
    fixture = torch.empty(features.shape, dtype=torch.float32)
    validate_sparse_tensor_3d(sparse.indices, fixture, sparse.spatial_shape, sparse.batch_size)
    kernel = tuple(int(size) for size in weights.shape[2:])
    volume = kernel[0] * kernel[1] * kernel[2]
    if volume > MAX_CPU_KERNEL_VOLUME:
        raise ValueError(f"bounded rulebook supports kernel volume <= {MAX_CPU_KERNEL_VOLUME}")
    if sparse.indices.shape[0] > MAX_CPU_POINTS:
        raise ValueError(f"bounded rulebook supports at most {MAX_CPU_POINTS} active inputs")
    if build and sparse.indices.shape[0] * volume > MAX_CPU_CANDIDATES:
        raise ValueError(f"bounded rulebook supports at most {MAX_CPU_CANDIDATES} candidates")
    if weights.numel() >= 2**32 or features.numel() >= 2**32:
        raise ValueError("gradient scalar count exceeds Metal dispatch limit")
    return kernel  # type: ignore[return-value]


def _check_key(key: str | None, cache: MutableMapping | None, *, required: bool):
    if key is None and not required:
        return
    if not isinstance(key, str) or not key:
        raise ValueError("indice_key must be a nonempty string")
    if cache is None or not isinstance(cache, MutableMapping):
        raise ValueError("indice_cache must be a mutable mapping when indice_key is set")


def _csr(pairs: Tensor, source_col: int, target_col: int, target_count: int):
    rows = sorted(pairs.tolist(), key=lambda row: (row[target_col], row[0], row[source_col]))
    ptr = [0] * (target_count + 1)
    for row in rows:
        ptr[row[target_col] + 1] += 1
    for target in range(target_count):
        ptr[target + 1] += ptr[target]
    return (
        torch.tensor(ptr, dtype=torch.int64),
        torch.tensor([row[source_col] for row in rows], dtype=torch.int64),
        torch.tensor([row[0] for row in rows], dtype=torch.int64),
    )


def _offset_major(pairs: Tensor, source_col: int, target_col: int, volume: int):
    rows = sorted(pairs.tolist(), key=lambda row: (row[0], row[target_col], row[source_col]))
    ptr = [0] * (volume + 1)
    for row in rows:
        ptr[row[0] + 1] += 1
    for offset in range(volume):
        ptr[offset + 1] += ptr[offset]
    packed = torch.tensor(
        [(row[0], row[source_col], row[target_col]) for row in rows], dtype=torch.int64,
    ).reshape(-1, 3)
    return packed, torch.tensor(ptr, dtype=torch.int64)


class _SparseConv3d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, features, weights, bias, out_ptr, out_src, out_off,
                in_ptr, in_out, in_off, pairs, offset_ptr, output_count):
        input_count, in_channels = features.shape
        out_channels = weights.shape[0]
        volume = weights.shape[2] * weights.shape[3] * weights.shape[4]
        work = output_count * out_channels
        output = torch.empty((output_count, out_channels), dtype=torch.float32, device=features.device)
        if work:
            _subm_math_library().subm_conv3d_f32(
                features.contiguous(), weights.contiguous(), out_ptr, out_src, out_off,
                output if bias is None else bias.contiguous(), output,
                output_count, in_channels, out_channels, volume, int(bias is not None),
                threads=work, group_size=min(work, 256),
            )
        ctx.save_for_backward(features, weights, in_ptr, in_out, in_off, pairs, offset_ptr)
        ctx.has_bias = bias is not None
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        features, weights, in_ptr, in_out, in_off, pairs, offset_ptr = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        input_count, in_channels = features.shape
        out_channels = weights.shape[0]
        volume = weights.shape[2] * weights.shape[3] * weights.shape[4]
        dx = torch.zeros(features.shape, dtype=torch.float32, device=features.device) if ctx.needs_input_grad[0] else None
        dw = torch.zeros(weights.shape, dtype=torch.float32, device=weights.device) if ctx.needs_input_grad[1] else None
        db = grad_output.sum(dim=0) if ctx.has_bias and ctx.needs_input_grad[2] else None
        feature_work = input_count * in_channels
        if dx is not None and feature_work:
            _grad_library().sparse_conv3d_grad_features_f32(
                grad_output, weights.contiguous(), in_ptr, in_out, in_off, dx,
                input_count, in_channels, out_channels, volume,
                threads=feature_work, group_size=min(feature_work, 256),
            )
        if dw is not None and pairs.shape[0]:
            weight_work = weights.numel()
            _subm_math_library().subm_conv3d_grad_weights_f32(
                features.contiguous(), grad_output, pairs, offset_ptr, dw,
                in_channels, out_channels, volume,
                threads=weight_work, group_size=min(weight_work, 256),
            )
        return dx, dw, db, None, None, None, None, None, None, None, None, None


def _apply(features: Tensor, weights: Tensor, bias: Tensor | None, pairs: Tensor,
           *, input_count: int, output_count: int, inverse: bool) -> Tensor:
    source_col, target_col = (2, 1) if inverse else (1, 2)
    out_ptr, out_src, out_off = _csr(pairs, source_col, target_col, output_count)
    in_ptr, in_out, in_off = _csr(pairs, target_col, source_col, input_count)
    volume = weights.shape[2] * weights.shape[3] * weights.shape[4]
    packed, offset_ptr = _offset_major(pairs, source_col, target_col, volume)
    # The CPU builder creates short-lived CSR tensors. On PyTorch 2.7,
    # non_blocking=True here caused the second call in a process to wait
    # indefinitely at torch.mps.synchronize(). Blocking host-to-MPS transfers
    # avoid the observed queue hazard; the precise runtime cause is unknown.
    to_mps = lambda x: x.to(device=features.device)
    return _SparseConv3d.apply(
        features, weights, bias, to_mps(out_ptr), to_mps(out_src), to_mps(out_off),
        to_mps(in_ptr), to_mps(in_out), to_mps(in_off), to_mps(packed),
        to_mps(offset_ptr), output_count,
    )


def sparse_conv3d_forward_mps(
    sparse: SparseTensor3D, weights: Tensor, *,
    stride: int | Sequence[int] = 1, padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1, bias: Tensor | None = None,
    indice_key: str | None = None,
    indice_cache: MutableMapping[str, SparseConvIndiceData3D] | None = None,
) -> SparseConvResult3D:
    """Bounded ordinary sparse cross-correlation with Metal arithmetic."""
    kernel = _validate(sparse, weights, bias, build=True)
    _check_key(indice_key, indice_cache, required=False)
    if indice_key is not None and indice_key in indice_cache:
        raise ValueError(f"indice_key {indice_key!r} already exists")
    parent = sparse._lineage if isinstance(sparse, SparseConvResult3D) else ()
    if indice_key is not None and any(key == indice_key for key, _ in parent):
        raise ValueError(f"indice_key {indice_key!r} already exists in input lineage")
    steps = _triple("stride", stride, minimum=1)
    pads = _triple("padding", padding, minimum=0)
    dil = _triple("dilation", dilation, minimum=1)
    fixture = torch.empty(sparse.features.shape, dtype=torch.float32)
    rulebook = generate_sparse_conv_rulebook(
        SparseTensor3D(sparse.indices, fixture, sparse.spatial_shape, sparse.batch_size),
        kernel_size=kernel, stride=steps, padding=pads, dilation=dil,
    )
    output_count = rulebook.output_indices.shape[0]
    if output_count > MAX_CPU_POINTS:
        raise ValueError(f"bounded rulebook supports at most {MAX_CPU_POINTS} active outputs")
    if output_count * weights.shape[0] >= 2**32:
        raise ValueError("output scalar count exceeds Metal dispatch limit")
    values = _apply(sparse.features, weights, bias, rulebook.pairs,
                    input_count=sparse.indices.shape[0], output_count=output_count,
                    inverse=False)
    lineage: tuple[tuple[str, object], ...] = ()
    if indice_key is not None:
        token = object()
        lineage = parent + ((indice_key, token),)
        indice_cache[indice_key] = SparseConvIndiceData3D(
            input_indices=sparse.indices.clone(), output_indices=rulebook.output_indices.clone(),
            input_spatial_shape=sparse.spatial_shape,
            output_spatial_shape=rulebook.output_spatial_shape,
            batch_size=sparse.batch_size, kernel_size=kernel,
            stride=steps, padding=pads, dilation=dil, pairs=rulebook.pairs.clone(),
            lineage_token=token, parent_lineage=parent,
        )
    return SparseConvResult3D(
        rulebook.output_indices, values, rulebook.output_spatial_shape,
        sparse.batch_size, lineage,
    )


def sparse_inverse_conv3d_forward_mps(
    sparse: SparseConvResult3D, weights: Tensor, *, indice_key: str,
    indice_cache: MutableMapping[str, SparseConvIndiceData3D],
    bias: Tensor | None = None,
) -> SparseConvResult3D:
    """Invert a saved coordinate mapping onto its original active rows."""
    kernel = _validate(sparse, weights, bias, build=False)
    _check_key(indice_key, indice_cache, required=True)
    if indice_key not in indice_cache:
        raise KeyError(f"indice_key {indice_key!r} has no saved ordinary convolution")
    saved = indice_cache[indice_key]
    if not isinstance(saved, SparseConvIndiceData3D):
        raise ValueError("indice_key does not reference an ordinary 3D convolution")
    if (
        not isinstance(sparse, SparseConvResult3D)
        or len(sparse._lineage) != len(saved.parent_lineage) + 1
        or sparse._lineage[-1][0] != indice_key
        or sparse._lineage[-1][1] is not saved.lineage_token
        or any(
            key != saved_key or token is not saved_token
            for (key, token), (saved_key, saved_token)
            in zip(sparse._lineage[:-1], saved.parent_lineage)
        )
    ):
        raise ValueError("inverse indice_key lineage does not match saved downsample result")
    if kernel != saved.kernel_size:
        raise ValueError("inverse kernel size differs from saved convolution")
    if sparse.batch_size != saved.batch_size:
        raise ValueError("inverse batch size differs from saved convolution")
    if sparse.spatial_shape != saved.output_spatial_shape:
        raise ValueError("inverse spatial shape differs from saved convolution output")
    if not torch.equal(sparse.indices, saved.output_indices):
        raise ValueError("inverse indices or row order differ from saved convolution output")
    values = _apply(
        sparse.features, weights, bias, saved.pairs,
        input_count=sparse.indices.shape[0], output_count=saved.input_indices.shape[0],
        inverse=True,
    )
    return SparseConvResult3D(
        saved.input_indices.clone(), values, saved.input_spatial_shape,
        saved.batch_size, saved.parent_lineage,
    )

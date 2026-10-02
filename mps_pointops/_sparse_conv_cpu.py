"""Small CPU reference for ordinary and inverse sparse 3D convolution.

This private module uses the coordinate contract in ``_sparse_rulebook`` and
PyTorch's dense Conv3d weight layout. It is an autograd-friendly oracle, not a
spconv API or an MPS implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import MutableMapping, Sequence

import torch

from ._sparse_rulebook import (
    SparseTensor3D,
    Triplet,
    _triple,
    generate_sparse_conv_rulebook,
    validate_sparse_tensor_3d,
)


# The Python rulebook builder visits candidate coordinates on the host. These
# limits keep this deliberately small reference from becoming an accidental
# production fallback for large clouds.
MAX_CPU_POINTS = 1024
MAX_CPU_KERNEL_VOLUME = 125
MAX_CPU_CANDIDATES = 32768


@dataclass(frozen=True)
class SparseConvResult3D(SparseTensor3D):
    """Sparse result carrying the saved coordinate lineage for inverse reuse.

    Feature-only operations may call ``replace_features`` to preserve the
    lineage. Reconstructing a plain ``SparseTensor3D`` does not do so and is
    intentionally rejected by inverse, even if its coordinates happen to
    match a cached output.
    """

    _lineage: tuple[tuple[str, object], ...] = field(default=(), repr=False, compare=False)

    def replace_features(self, features: torch.Tensor) -> SparseConvResult3D:
        return SparseConvResult3D(
            self.indices, features, self.spatial_shape, self.batch_size, self._lineage
        )


@dataclass(frozen=True)
class SparseConvIndiceData3D:
    """Snapshot of one ordinary sparse convolution's coordinate mapping.

    ``pairs`` use the forward rulebook convention ``(offset, old_row,
    downsampled_row)``. An inverse operation reverses the row direction while
    keeping the same kernel offset. Coordinates are snapshotted so mutations
    of the original sparse tensors do not change the mapping. The caller-owned
    cache and the tensors stored in it must be treated as opaque.
    """

    input_indices: torch.Tensor
    output_indices: torch.Tensor
    input_spatial_shape: Triplet
    output_spatial_shape: Triplet
    batch_size: int
    kernel_size: Triplet
    stride: Triplet
    padding: Triplet
    dilation: Triplet
    pairs: torch.Tensor
    lineage_token: object = field(repr=False, compare=False)
    parent_lineage: tuple[tuple[str, object], ...] = field(repr=False, compare=False)


def _checked_sparse(sparse: SparseTensor3D) -> SparseTensor3D:
    if not isinstance(sparse, SparseTensor3D):
        raise TypeError("sparse must be a SparseTensor3D")
    validate_sparse_tensor_3d(
        sparse.indices, sparse.features, sparse.spatial_shape, sparse.batch_size
    )
    return sparse


def _checked_weights(
    sparse: SparseTensor3D,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    generate_candidates: bool,
) -> Triplet:
    if not isinstance(weights, torch.Tensor):
        raise TypeError("weights must be a torch.Tensor")
    if (
        weights.device.type != "cpu"
        or weights.dtype != sparse.features.dtype
        or weights.ndim != 5
        or weights.shape[0] < 1
        or weights.shape[1] != sparse.features.shape[1]
        or any(size < 1 for size in weights.shape[2:])
    ):
        raise ValueError(
            "weights must be CPU [Cout, Cin, K0, K1, K2] with the feature dtype"
        )
    if bias is not None and (
        not isinstance(bias, torch.Tensor)
        or bias.device.type != "cpu"
        or bias.dtype != sparse.features.dtype
        or bias.shape != (weights.shape[0],)
    ):
        raise ValueError("bias must be CPU [Cout] with the feature dtype")
    kernel = tuple(int(size) for size in weights.shape[2:])
    volume = kernel[0] * kernel[1] * kernel[2]
    if volume > MAX_CPU_KERNEL_VOLUME:
        raise ValueError(f"CPU reference kernel volume exceeds {MAX_CPU_KERNEL_VOLUME}")
    if sparse.indices.shape[0] > MAX_CPU_POINTS:
        raise ValueError(f"CPU reference input exceeds {MAX_CPU_POINTS} active points")
    if generate_candidates and sparse.indices.shape[0] * volume > MAX_CPU_CANDIDATES:
        raise ValueError(f"CPU reference exceeds {MAX_CPU_CANDIDATES} point-offset candidates")
    return kernel  # type: ignore[return-value]


def _checked_key(
    indice_key: str | None,
    indice_cache: MutableMapping[str, SparseConvIndiceData3D] | None,
    *,
    required: bool,
) -> None:
    if indice_key is None and not required:
        return
    if not isinstance(indice_key, str) or not indice_key:
        raise ValueError("indice_key must be a nonempty string")
    if indice_cache is None or not isinstance(indice_cache, MutableMapping):
        raise ValueError("indice_cache must be a mutable mapping when indice_key is set")


def _evaluate_pairs(
    features: torch.Tensor,
    weights: torch.Tensor,
    bias: torch.Tensor | None,
    pairs: torch.Tensor,
    output_count: int,
    *,
    inverse: bool,
) -> torch.Tensor:
    """Apply an offset-major rulebook with native PyTorch autograd operations."""

    out_channels, in_channels = weights.shape[:2]
    flat_weights = weights.reshape(out_channels, in_channels, -1)
    output = features.new_zeros((output_count, out_channels))
    if pairs.shape[0] == 0:
        # Keep zero gradients defined for features and weights on empty inputs.
        output = output + features.sum() * 0 + weights.sum() * 0
    else:
        source_column, target_column = (2, 1) if inverse else (1, 2)
        for offset in range(flat_weights.shape[2]):
            subset = pairs[pairs[:, 0] == offset]
            if subset.shape[0] == 0:
                continue
            sources = subset[:, source_column]
            targets = subset[:, target_column]
            contributions = features.index_select(0, sources) @ flat_weights[:, :, offset].T
            output = output.index_add(0, targets, contributions)
    if bias is not None:
        output = output + bias
    return output


def sparse_conv3d_forward_cpu(
    sparse: SparseTensor3D,
    weights: torch.Tensor,
    *,
    stride: int | Sequence[int] = 1,
    padding: int | Sequence[int] = 0,
    dilation: int | Sequence[int] = 1,
    bias: torch.Tensor | None = None,
    indice_key: str | None = None,
    indice_cache: MutableMapping[str, SparseConvIndiceData3D] | None = None,
) -> SparseConvResult3D:
    """Evaluate ordinary sparse cross-correlation and optionally save its pairs.

    Weights use PyTorch Conv3d layout ``[Cout,Cin,K0,K1,K2]``. A key may be
    written once to ``indice_cache`` and reused by
    ``sparse_inverse_conv3d_forward_cpu``. Output indices are lexicographically
    sorted, independent of input row order.
    """

    sparse = _checked_sparse(sparse)
    kernel = _checked_weights(sparse, weights, bias, generate_candidates=True)
    _checked_key(indice_key, indice_cache, required=False)
    if indice_key is not None and indice_key in indice_cache:
        raise ValueError(f"indice_key {indice_key!r} already exists")
    parent_lineage = sparse._lineage if isinstance(sparse, SparseConvResult3D) else ()
    if indice_key is not None and any(key == indice_key for key, _ in parent_lineage):
        raise ValueError(f"indice_key {indice_key!r} already exists in input lineage")
    steps = _triple("stride", stride, minimum=1)
    pads = _triple("padding", padding, minimum=0)
    dil = _triple("dilation", dilation, minimum=1)
    rulebook = generate_sparse_conv_rulebook(
        sparse, kernel_size=kernel, stride=steps, padding=pads, dilation=dil
    )
    if rulebook.output_indices.shape[0] > MAX_CPU_POINTS:
        raise ValueError(f"CPU reference output exceeds {MAX_CPU_POINTS} active points")
    output_features = _evaluate_pairs(
        sparse.features, weights, bias, rulebook.pairs,
        rulebook.output_indices.shape[0], inverse=False,
    )
    lineage: tuple[tuple[str, object], ...] = ()
    if indice_key is not None:
        token = object()
        lineage = parent_lineage + ((indice_key, token),)
        indice_cache[indice_key] = SparseConvIndiceData3D(
            input_indices=sparse.indices.clone(),
            output_indices=rulebook.output_indices.clone(),
            input_spatial_shape=sparse.spatial_shape,
            output_spatial_shape=rulebook.output_spatial_shape,
            batch_size=sparse.batch_size,
            kernel_size=kernel,
            stride=steps,
            padding=pads,
            dilation=dil,
            pairs=rulebook.pairs.clone(),
            lineage_token=token,
            parent_lineage=parent_lineage,
        )
    return SparseConvResult3D(
        rulebook.output_indices, output_features,
        rulebook.output_spatial_shape, sparse.batch_size, lineage,
    )


def sparse_inverse_conv3d_forward_cpu(
    sparse: SparseTensor3D,
    weights: torch.Tensor,
    *,
    indice_key: str,
    indice_cache: MutableMapping[str, SparseConvIndiceData3D],
    bias: torch.Tensor | None = None,
) -> SparseConvResult3D:
    """Reverse a saved sparse-convolution mapping onto its original active set.

    This reuses *exactly* the saved pairs and original coordinate row order.
    It never generates a transposed-convolution output set. The inverse layer
    has its own weights in ``[Cout,Cin,K0,K1,K2]`` layout.
    """

    sparse = _checked_sparse(sparse)
    kernel = _checked_weights(sparse, weights, bias, generate_candidates=False)
    _checked_key(indice_key, indice_cache, required=True)
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
    output_features = _evaluate_pairs(
        sparse.features, weights, bias, saved.pairs,
        saved.input_indices.shape[0], inverse=True,
    )
    return SparseConvResult3D(
        saved.input_indices.clone(), output_features,
        saved.input_spatial_shape, saved.batch_size, saved.parent_lineage,
    )

"""Private 3D sparse nn adapter for bounded backbone integration experiments.

This implements a tested subset of spconv 2.3.8's tensor/layer signatures with
independent coordinate oracles and Metal kernels. It never replaces an
installed ``spconv`` module. Coordinates are validated and stored on CPU;
MPS coordinate input therefore incurs an explicit host transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral

import torch
from torch import nn

from ._sparse_conv_cpu import (
    MAX_CPU_POINTS,
    SparseConvResult3D,
    _evaluate_pairs,
    sparse_conv3d_forward_cpu,
    sparse_inverse_conv3d_forward_cpu,
)
from ._sparse_rulebook import _triple, generate_subm_rulebook, validate_sparse_tensor_3d


class ConvAlgo(IntEnum):
    """Only the explicit, unquantized Native contract is accepted here."""

    Native = 0


MAX_DENSE_BYTES = 256 * 1024 * 1024


def _features(features, rows):
    if not isinstance(features, torch.Tensor):
        raise TypeError("features must be a Tensor")
    if features.ndim != 2 or features.shape[0] != rows or features.shape[1] < 1:
        raise ValueError("features must be [N, C] with C >= 1 and one row per coordinate")
    allowed = (torch.float32, torch.float64) if features.device.type == "cpu" else (torch.float32,)
    if features.device.type not in ("cpu", "mps") or features.dtype not in allowed:
        raise ValueError("features must be CPU float32/float64 or MPS float32")


def _spatial_shape(value):
    """Accept Python/NumPy integral dimensions without silently truncating floats."""
    try:
        dimensions = tuple(value)
    except TypeError as exc:
        raise ValueError("spatial_shape must contain three positive integers") from exc
    if len(dimensions) != 3 or any(
        not isinstance(item, Integral) or isinstance(item, bool)
        or not 1 <= item < 2**31
        for item in dimensions
    ):
        raise ValueError("spatial_shape must contain three positive int32 dimensions")
    return tuple(int(item) for item in dimensions)


class SparseConvTensor:
    """3D active rows with immutable feature replacement and shared key cache.

    Indices use [batch, axis0, axis1, axis2] int32 and unique coordinates.
    ``spatial_shape`` uses the same spatial axis order. The private adapter
    deliberately rejects duplicate coordinates and unsupported dtypes.
    """

    def __init__(self, features, indices, spatial_shape, batch_size, *, indice_dict=None):
        if not isinstance(indices, torch.Tensor):
            raise TypeError("indices must be a Tensor")
        if indices.device.type not in ("cpu", "mps"):
            raise ValueError("indices must be on CPU or MPS")
        if indices.ndim != 2 or indices.shape[1] != 4 or indices.dtype != torch.int32:
            raise ValueError("indices must be int32 [N, 4]")
        _features(features, indices.shape[0])
        shape = _spatial_shape(spatial_shape)
        if (not isinstance(batch_size, Integral) or isinstance(batch_size, bool)
                or not 0 <= batch_size < 2**31):
            raise ValueError("batch_size must be a nonnegative int32 integer")
        batch_size = int(batch_size)
        # Do not copy features to CPU just to validate integer coordinates.
        coordinates = indices.detach().to("cpu").clone()
        checked = validate_sparse_tensor_3d(
            coordinates, torch.empty((features.shape[0], 1)), shape, batch_size
        )
        if indice_dict is not None and not isinstance(indice_dict, dict):
            raise TypeError("indice_dict must be a dictionary")
        self.indices = checked.indices
        self.spatial_shape = list(checked.spatial_shape)
        self.batch_size = checked.batch_size
        self._features = features
        self.indice_dict = {} if indice_dict is None else indice_dict
        self._lineage = ()

    @property
    def features(self):
        return self._features

    @classmethod
    def _from_result(cls, result, cache):
        obj = object.__new__(cls)
        obj.indices = result.indices
        obj.spatial_shape = list(result.spatial_shape)
        obj.batch_size = result.batch_size
        obj._features = result.features
        obj.indice_dict = cache
        obj._lineage = result._lineage
        return obj

    def _result(self):
        return SparseConvResult3D(
            self.indices, self.features, tuple(self.spatial_shape), self.batch_size, self._lineage
        )

    def replace_feature(self, features):
        _features(features, self.indices.shape[0])
        if features.device != self.features.device or features.dtype != self.features.dtype:
            raise ValueError("replace_feature must preserve dtype and device")
        return self._from_result(self._result().replace_features(features), self.indice_dict)

    def shadow_copy(self):
        return self._from_result(self._result(), self.indice_dict)

    def dense(self, channels_first=True):
        # Unique validated coordinates ensure a single writer per active row.
        spatial_count = math.prod(self.spatial_shape)
        dense_bytes = self.batch_size * spatial_count * self.features.shape[1] * self.features.element_size()
        if dense_bytes > MAX_DENSE_BYTES:
            raise ValueError("dense conversion exceeds the private adapter's 256 MiB limit")
        flat = self.features.new_zeros((self.batch_size * spatial_count, self.features.shape[1]))
        coords = self.indices.to(dtype=torch.int64)
        ids = coords[:, 0]
        for axis, size in enumerate(self.spatial_shape):
            ids = ids * size + coords[:, axis + 1]
        flat = flat.index_copy(0, ids.to(self.features.device), self.features)
        result = flat.reshape(self.batch_size, *self.spatial_shape, self.features.shape[1])
        return result.permute(0, 4, 1, 2, 3).contiguous() if channels_first else result


class SparseModule(nn.Module):
    """Marker: a module consumes a sparse tensor, rather than its features."""


class SparseSequential(nn.Sequential, SparseModule):
    def forward(self, value):
        for module in self:
            if isinstance(value, SparseConvTensor) and not isinstance(module, SparseModule):
                if value.features.shape[0]:
                    value = value.replace_feature(module(value.features))
            else:
                value = module(value)
        return value


@dataclass(frozen=True)
class _SubMKey:
    indices: torch.Tensor
    shape: tuple
    batch_size: int
    kernel: tuple
    dilation: tuple


class SparseConvolution(SparseModule):
    """Common first-order 3D layer contract; weights use spconv KRSC layout."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None, *,
                 subm=False, inverse=False):
        super().__init__()
        for name, value in (("in_channels", in_channels), ("out_channels", out_channels)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(groups, int) or isinstance(groups, bool) or groups != 1:
            raise ValueError("the sparse adapter supports groups=1")
        if algo is not None and algo is not ConvAlgo.Native:
            raise ValueError("only the Native algorithm contract is supported")
        if indice_key is not None and (not isinstance(indice_key, str) or not indice_key):
            raise ValueError("indice_key must be a nonempty string")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size = _triple("kernel_size", kernel_size, minimum=1)
        self.stride = _triple("stride", stride, minimum=1)
        self.padding = _triple("padding", padding, minimum=0)
        self.dilation = _triple("dilation", dilation, minimum=1)
        self.subm, self.inverse, self.indice_key = subm, inverse, indice_key
        self.algo = ConvAlgo.Native
        if subm and (self.stride != (1, 1, 1) or any(k % 2 == 0 for k in self.kernel_size)):
            raise ValueError("SubM requires odd kernels and stride=1")
        if subm:
            implicit_center_padding = tuple(
                (kernel // 2) * dilation
                for kernel, dilation in zip(self.kernel_size, self.dilation)
            )
            # The private rulebook is centered on each active output. Pinned
            # OpenPCDet call sites pass either the spconv default 0 or this
            # explicit centered value; other padding would be silently lost.
            if self.padding not in ((0, 0, 0), implicit_center_padding):
                raise ValueError("SubM padding must be 0 or the implicit centered value")
        if inverse and indice_key is None:
            raise ValueError("inverse requires an indice_key")
        self.weight = nn.Parameter(torch.empty(out_channels, *self.kernel_size, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Explicit fan-in avoids interpreting KRSC as PyTorch's OI... layout.
        bound = 1 / math.sqrt(self.in_channels * math.prod(self.kernel_size))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value):
        if not isinstance(value, SparseConvTensor):
            raise TypeError("sparse convolution requires a SparseConvTensor")
        _features(value.features, value.indices.shape[0])
        if value.features.shape[1] != self.in_channels:
            raise ValueError("input channel count differs from the layer")
        if self.weight.dtype != value.features.dtype or self.weight.device != value.features.device:
            raise ValueError("layer parameters and features must have the same dtype and device")
        weights = self.weight.permute(0, 4, 1, 2, 3)
        sparse = value._result()
        cache = value.indice_dict.copy()
        if self.subm:
            if self.indice_key is not None:
                saved = cache.get(self.indice_key)
                if saved is not None and (
                    not isinstance(saved, _SubMKey) or saved.shape != sparse.spatial_shape
                    or saved.batch_size != sparse.batch_size or saved.kernel != self.kernel_size
                    or saved.dilation != self.dilation or not torch.equal(saved.indices, sparse.indices)
                ):
                    raise ValueError("SubM indice_key belongs to a different coordinate mapping")
                cache[self.indice_key] = _SubMKey(
                    sparse.indices.clone(), sparse.spatial_shape, sparse.batch_size,
                    self.kernel_size, self.dilation,
                )
            if value.features.device.type == "mps":
                from ._subm_conv_mps import subm_conv3d_forward_mps
                output = subm_conv3d_forward_mps(
                    sparse.indices, sparse.features, weights, sparse.spatial_shape,
                    sparse.batch_size, dilation=self.dilation, bias=self.bias,
                )
            else:
                if len(sparse.indices) > MAX_CPU_POINTS:
                    raise ValueError(f"CPU adapter reference exceeds {MAX_CPU_POINTS} active points")
                validate_sparse_tensor_3d(sparse.indices, sparse.features, sparse.spatial_shape, sparse.batch_size)
                pairs = generate_subm_rulebook(sparse, kernel_size=self.kernel_size, dilation=self.dilation).pairs
                output = _evaluate_pairs(sparse.features, weights, self.bias, pairs, len(sparse.indices), inverse=False)
            result = sparse.replace_features(output)
        else:
            if value.features.device.type == "mps":
                from ._sparse_conv_mps import sparse_conv3d_forward_mps, sparse_inverse_conv3d_forward_mps
                function = sparse_inverse_conv3d_forward_mps if self.inverse else sparse_conv3d_forward_mps
            else:
                function = sparse_inverse_conv3d_forward_cpu if self.inverse else sparse_conv3d_forward_cpu
            options = {} if self.inverse else dict(stride=self.stride, padding=self.padding, dilation=self.dilation)
            result = function(sparse, weights, bias=self.bias, indice_key=self.indice_key,
                              indice_cache=cache, **options)
        return SparseConvTensor._from_result(result, cache)


class SubMConv3d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, indice_key=None, algo=None):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding,
                         dilation, groups, bias, indice_key, algo, subm=True)


class SparseConv3d(SparseConvolution):
    pass


class SparseInverseConv3d(SparseConvolution):
    def __init__(self, in_channels, out_channels, kernel_size, indice_key,
                 bias=True, algo=None):
        super().__init__(in_channels, out_channels, kernel_size, bias=bias,
                         indice_key=indice_key, algo=algo, inverse=True)

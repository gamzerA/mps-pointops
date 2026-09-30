"""Drop-in stand-ins for the CUDA-only ``pointnet2_ops`` and ``knn_cuda``.

Call ``install()`` before importing code that uses them::

    import mps_pointops.compat
    mps_pointops.compat.install()

    from pointnet2_ops import pointnet2_utils  # served by mps_pointops
    from knn_cuda import KNN

``install()`` registers ``pointnet2_ops``, ``pointnet2_ops.pointnet2_utils``
and ``knn_cuda`` in ``sys.modules``. A name that is already importable, such
as the real package on a CUDA machine, is left alone unless ``force=True``.

Covered: ``furthest_point_sample``, ``gather_operation``,
``grouping_operation`` and ``ball_query`` from ``pointnet2_utils``, and
``KNN`` from ``knn_cuda``. Differences from the CUDA versions:

- Near ties can resolve differently. The kernels round each squared
  distance without FMA and break ties by the smaller index; the CUDA
  kernels use their own reduction order.
- ``ball_query`` runs the pure PyTorch reference, not a Metal kernel yet.
"""

from __future__ import annotations

import importlib.util
import sys
import types

import torch
from torch import Tensor

from . import ops, reference


# ---------------------------------------------------------------- pointnet2_ops


def furthest_point_sample(xyz: Tensor, npoint: int) -> Tensor:
    """Like ``pointnet2_ops.pointnet2_utils.furthest_point_sample``.

    (B, N, 3) float32 -> (B, npoint) int32. Starts at index 0 and never picks
    points with x^2 + y^2 + z^2 <= 1e-3, as pointnet2_ops does.
    """
    return ops.furthest_point_sample(xyz, npoint, skip_near_origin=True).int()


def gather_operation(features: Tensor, idx: Tensor) -> Tensor:
    """Like ``pointnet2_utils.gather_operation``: (B, C, N), (B, npoint) -> (B, C, npoint)."""
    B, C, _ = features.shape
    return features.gather(2, idx.long().unsqueeze(1).expand(B, C, -1))


def grouping_operation(features: Tensor, idx: Tensor) -> Tensor:
    """Like ``pointnet2_utils.grouping_operation``: (B, C, N), (B, npoint, nsample) -> (B, C, npoint, nsample)."""
    B, C, _ = features.shape
    _, npoint, nsample = idx.shape
    flat = idx.long().reshape(B, 1, npoint * nsample).expand(B, C, -1)
    return features.gather(2, flat).reshape(B, C, npoint, nsample)


def ball_query(radius: float, nsample: int, xyz: Tensor, new_xyz: Tensor) -> Tensor:
    """Like ``pointnet2_utils.ball_query``: (B, N, 3), (B, npoint, 3) -> (B, npoint, nsample) int32.

    Takes the first ``nsample`` points in input order with squared distance
    ``< radius**2``. Empty slots repeat the first neighbor, and a query with no
    neighbor gets all zeros, as in pointnet2_ops.
    """
    _, idx = reference.ball_query(new_xyz, xyz, radius, nsample)
    first = idx[..., :1].clamp(min=0)
    return torch.where(idx >= 0, idx, first).int()


# ---------------------------------------------------------------- knn_cuda


class KNN(torch.nn.Module):
    """Like ``knn_cuda.KNN``.

    With ``transpose_mode=True``, ``ref`` is (B, N, 3) and ``query`` is
    (B, M, 3), and ``dist`` and ``idx`` are (B, M, k). Otherwise ``ref`` is
    (B, 3, N), ``query`` is (B, 3, M), and the outputs are (B, k, M). ``dist``
    is Euclidean. No gradients flow, as in knn_cuda.
    """

    def __init__(self, k: int, transpose_mode: bool = False):
        super().__init__()
        self.k = k
        self.transpose_mode = transpose_mode

    def forward(self, ref: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
        if not self.transpose_mode:
            ref, query = ref.transpose(1, 2), query.transpose(1, 2)
        with torch.no_grad():
            dist, idx = ops.knn(query.float(), ref.float(), self.k)
        if not self.transpose_mode:
            dist, idx = dist.transpose(1, 2), idx.transpose(1, 2)
        return dist, idx


# ---------------------------------------------------------------- install


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def install(force: bool = False) -> list[str]:
    """Register the stand-in modules. Returns the names that were installed."""
    pointnet2_utils = _module(
        "pointnet2_ops.pointnet2_utils",
        furthest_point_sample=furthest_point_sample,
        gather_operation=gather_operation,
        grouping_operation=grouping_operation,
        ball_query=ball_query,
    )
    packages = {
        "pointnet2_ops": {
            "pointnet2_ops": _module("pointnet2_ops", pointnet2_utils=pointnet2_utils, __path__=[]),
            "pointnet2_ops.pointnet2_utils": pointnet2_utils,
        },
        "knn_cuda": {"knn_cuda": _module("knn_cuda", KNN=KNN)},
    }
    installed = []
    for top, modules in packages.items():
        if not force and (top in sys.modules or importlib.util.find_spec(top) is not None):
            continue
        sys.modules.update(modules)
        installed.extend(modules)
    return installed

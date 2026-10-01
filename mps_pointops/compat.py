"""Optional stand-ins for ``pointnet2_ops``, ``knn_cuda`` and ``torch_cluster``.

Call ``install()`` before importing code that uses them::

    import mps_pointops.compat
    mps_pointops.compat.install()

    from pointnet2_ops import pointnet2_utils  # served by mps_pointops
    from knn_cuda import KNN
    from torch_cluster import fps, knn, radius

``install()`` registers ``pointnet2_ops``, ``pointnet2_ops.pointnet2_utils``,
``knn_cuda`` and ``torch_cluster`` in ``sys.modules``. A name that is already
importable, such as the real package on a CUDA machine, is left alone unless
``force=True``. Call it before importing the packages that use these names.

Covered: ``furthest_point_sample``, ``gather_operation``,
``grouping_operation`` and ``ball_query`` from ``pointnet2_utils``, and
``KNN`` from ``knn_cuda``. Differences from the CUDA versions:

- Near ties can resolve differently because floating-point accumulation and
  CUDA reduction order differ. The ``nearest`` kernel reproduces the legacy
  1024-lane CUDA tie priority described in ``docs/nearest-contract.md``.
- ``ball_query`` uses the Metal kernel for MPS inputs and pads in the
  ``pointnet2_ops`` convention.
- The ``torch_cluster`` shim exposes ``fps``, ``knn``, ``radius``, ``nearest``,
  ``graclus_cluster``, and kNN/radius graph wrappers. FPS and radius use
  three-dimensional point coordinates; kNN also accepts float32 feature
  vectors on MPS. Experimental
  ``grid_cluster`` accepts float32 3D coordinates. Graclus has the documented
  CPU/Metal greedy matching subset. Other names needed for PyG 2.7 package
  import raise ``NotImplementedError`` when called. It does not
  register PyG's separate ``torch.ops.pyg`` operators.
"""

from __future__ import annotations

import importlib.util
from importlib.machinery import ModuleSpec
import sys
import types

import torch
from torch import Tensor

from . import flat, graclus, grid, nearest as nearest_ops, ops


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
    _, idx = ops.ball_query(new_xyz, xyz, radius, nsample)
    first = idx[..., :1].clamp(min=0)
    return torch.where(idx >= 0, idx, first).int()


# ---------------------------------------------------------------- knn_cuda


class KNN(torch.nn.Module):
    """Like ``knn_cuda.KNN``.

    With ``transpose_mode=True``, ``ref`` is (B, N, D) and ``query`` is
    (B, M, D), and ``dist`` and ``idx`` are (B, M, k). Otherwise ``ref`` is
    (B, D, N), ``query`` is (B, D, M), and the outputs are (B, k, M). ``dist``
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
    module.__spec__ = ModuleSpec(name, loader=None, is_package="__path__" in attrs)
    return module


def _unsupported_torch_cluster(name: str):
    def unsupported(*args, **kwargs):
        raise NotImplementedError(
            f"torch_cluster.{name} is outside the mps_pointops point-cloud subset"
        )

    unsupported.__name__ = name
    return unsupported


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
        "torch_cluster": {
            "torch_cluster": _module(
                "torch_cluster",
                fps=flat.fps,
                knn=flat.knn,
                radius=flat.radius,
                knn_graph=flat.knn_graph,
                radius_graph=flat.radius_graph,
                grid_cluster=grid.grid_cluster,
                graclus_cluster=graclus.graclus_cluster,
                random_walk=_unsupported_torch_cluster("random_walk"),
                nearest=nearest_ops.nearest,
            )
        },
    }
    installed = []
    for top, modules in packages.items():
        if not force and (top in sys.modules or importlib.util.find_spec(top) is not None):
            continue
        sys.modules.update(modules)
        installed.extend(modules)
    return installed

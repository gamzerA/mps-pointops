"""Reusable single-cloud spatial search with an explicit Metal BVH option.

The established dense and flat operators retain their own contracts. This
index presents a single-cloud interface and keeps the measured hierarchy
opt-in until cross-device and density-dependent performance are verified.
"""

from __future__ import annotations

import math
import os
import subprocess
from functools import lru_cache
from typing import Literal

import torch
from torch import Tensor

from . import ops
from ._ball_query_mps import _checked_radius_and_k

SpatialBackend = Literal["auto", "scan", "bvh"]
_MAX_BVH_K = 32
_MAX_BVH_N = 1_000_000


@lru_cache(maxsize=1)
def _measured_m5_pro() -> bool:
    try:
        chip = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return chip == "Apple M5 Pro"


class _BVHSquaredDistances(torch.autograd.Function):
    """Attach the existing first-order Ball Query coordinate derivative."""

    @staticmethod
    def forward(ctx, query: Tensor, points: Tensor, indices: Tensor, distances: Tensor) -> Tensor:
        ctx.save_for_backward(query, points, indices)
        return distances

    @staticmethod
    def backward(ctx, grad_distances: Tensor | None):
        if grad_distances is None:
            return None, None, None, None
        query, points, indices = ctx.saved_tensors
        if not indices.numel() or not len(points):
            grad_q = torch.zeros_like(query) if ctx.needs_input_grad[0] else None
            grad_p = torch.zeros_like(points) if ctx.needs_input_grad[1] else None
            return grad_q, grad_p, None, None

        valid = indices >= 0
        safe = indices.clamp_min(0)
        gathered = points.float().index_select(0, safe.reshape(-1)).reshape(*indices.shape, 3)
        delta = query.float().unsqueeze(1) - gathered
        delta = torch.where(valid.unsqueeze(-1), delta, torch.zeros_like(delta))
        grad = torch.where(valid, grad_distances, torch.zeros_like(grad_distances))
        contribution = 2.0 * grad.unsqueeze(-1) * delta
        grad_q = contribution.sum(dim=1).to(query.dtype) if ctx.needs_input_grad[0] else None
        grad_p = None
        if ctx.needs_input_grad[1]:
            grad_p = torch.zeros_like(points, dtype=torch.float32)
            grad_p.scatter_add_(
                0, safe.reshape(-1, 1).expand(-1, 3),
                -contribution.reshape(-1, 3),
            )
            grad_p = grad_p.to(points.dtype)
        return grad_q, grad_p, None, None


class SpatialIndex:
    """Search one point cloud using a reusable scan or experimental BVH path.

    ``points`` and queries have shape ``[N, 3]`` and ``[Q, 3]``. The index
    borrows ``points``; in-place mutation after construction is rejected.
    ``knn`` returns Euclidean distances like :func:`mps_pointops.knn`.
    ``ball_query`` returns squared distances and original-index first-K
    neighbors like :func:`mps_pointops.ball_query`, including its first-order
    coordinate gradient. No method differentiates neighbor selection.

    ``auto`` uses the BVH only for the measured M5 Pro, Safe Math, one-million
    point, K=16, 8192<=Q<=65536 kNN case, and rejects a concentrated Morton cell in
    a small deterministic sample. Other inputs retain the existing scan path.
    Radius search defaults to scanning until its hierarchy speed is measured.
    An explicit ``backend="bvh"`` is available for supported MPS float32
    inputs. Auto dispatch reads a small sample of Morton keys on the host;
    the search itself remains on MPS without copying the full point cloud.
    """

    def __init__(
        self,
        points: Tensor,
        *,
        backend: SpatialBackend = "auto",
        origin: Tensor | None = None,
        cell_size: float | None = None,
    ) -> None:
        if not isinstance(points, Tensor) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape [N, 3]")
        self._check_backend(backend)
        if origin is not None and (
            not isinstance(origin, Tensor) or origin.shape != (3,)
            or origin.device != points.device or origin.dtype != points.dtype
        ):
            raise ValueError("origin must be a [3] tensor with the same device and dtype as points")
        if cell_size is not None and (not math.isfinite(cell_size) or cell_size <= 0):
            raise ValueError("cell_size must be positive and finite")
        self.points = points
        self.backend = backend
        self.origin = origin
        self.cell_size = cell_size
        self._point_version = points._version
        self._bvh = None
        self._grid_cache: tuple[Tensor, float] | None = None
        self._hot_cell_fraction: float | None = None

    @staticmethod
    def _check_backend(backend: str) -> None:
        if backend not in ("auto", "scan", "bvh"):
            raise ValueError("backend must be 'auto', 'scan', or 'bvh'")

    def _check_query(self, query: Tensor) -> None:
        if not isinstance(query, Tensor) or query.ndim != 2 or query.shape[1] != 3:
            raise ValueError("query must have shape [Q, 3]")
        if query.device != self.points.device:
            raise ValueError("query and points must be on the same device")
        if query.dtype != self.points.dtype:
            raise TypeError("query and points must have the same dtype")
        if self.points._version != self._point_version:
            raise ValueError("points changed after SpatialIndex construction")

    def _grid_parameters(self) -> tuple[Tensor, float]:
        if self._grid_cache is not None:
            return self._grid_cache
        if self.origin is not None and self.cell_size is not None:
            self._grid_cache = (self.origin, self.cell_size)
            return self._grid_cache
        if not len(self.points):
            origin = torch.zeros(3, device=self.points.device, dtype=torch.float32)
            self._grid_cache = (self.origin if self.origin is not None else origin,
                                self.cell_size if self.cell_size is not None else 1.0)
            return self._grid_cache
        detached = self.points.detach()
        origin = self.origin if self.origin is not None else detached.amin(dim=0)
        if self.cell_size is not None:
            self._grid_cache = (origin, self.cell_size)
            return self._grid_cache
        # Morton order affects traversal speed, never the exact distance test.
        # A small deterministic sample gives fine cells for a dense cluster
        # without making a broad sparse background overflow its 21-bit domain.
        step = max(1, len(detached) // 4096)
        sample = detached[::step][:4096]
        sorted_sample = sample.sort(dim=0).values
        last = len(sorted_sample) - 1
        robust = (sorted_sample[int(0.9 * last)] - sorted_sample[int(0.1 * last)]).amax()
        full = (detached.amax(dim=0) - detached.amin(dim=0)).amax()
        robust_span = float(robust.item())
        full_span = float(full.item())
        cell_size = max(robust_span / 64.0, full_span / (1 << 20), 2.0**-100)
        if not math.isfinite(cell_size):
            raise ValueError("cannot derive a finite Morton cell size")
        self._grid_cache = (origin, cell_size)
        return self._grid_cache

    def _sample_hot_cell_fraction(self) -> float:
        if self._hot_cell_fraction is not None:
            return self._hot_cell_fraction
        from ._spatial_keys import morton_keys_f32

        origin, cell_size = self._grid_parameters()
        step = max(1, len(self.points) // 4096)
        sample = self.points.detach()[::step][:4096].contiguous()
        keys, invalid = morton_keys_f32(sample, origin, cell_size)
        if bool(invalid.any().item()):
            raise ValueError("sample exceeds the Morton domain")
        # A 32 KiB integer readback is the explicit cost of this conservative
        # host-side policy. It detects all-equal and other hot-cell cases that
        # make the serial tree much slower than the full scan.
        _, counts = torch.unique(keys.cpu(), return_counts=True)
        self._hot_cell_fraction = float(counts.max().item()) / len(sample)
        return self._hot_cell_fraction

    def _get_bvh(self):
        if self._bvh is None:
            from ._spatial_bvh import MortonTwoLevelBVH

            origin, cell_size = self._grid_parameters()
            self._bvh = MortonTwoLevelBVH.build(self.points, origin, cell_size)
        return self._bvh

    def _selected_backend(
        self, query: Tensor, k: int, backend: SpatialBackend | None, operation: str,
    ) -> str:
        choice = self.backend if backend is None else backend
        self._check_backend(choice)
        if choice == "auto":
            if (operation == "knn" and self.points.device.type == "mps"
                    and self.points.dtype == torch.float32 and query.dtype == torch.float32
                    and os.environ.get("PYTORCH_MPS_FAST_MATH") == "0"
                    and len(self.points) == 1_000_000
                    and 8192 <= len(query) <= 65_536
                    and k == 16 and _measured_m5_pro()):
                try:
                    from ._spatial_bvh import _validate_coordinate_domain

                    _validate_coordinate_domain(query, "query points")
                    if self._sample_hot_cell_fraction() < 0.05:
                        self._get_bvh()
                        return "bvh"
                except ValueError:
                    # Unsupported coordinate and Morton domains keep the
                    # existing MPS scan contract in automatic mode.
                    pass
            return "scan"
        if choice == "scan":
            return "scan"
        if self.points.device.type != "mps" or query.dtype != torch.float32 or self.points.dtype != torch.float32:
            raise ValueError("BVH requires MPS float32 points and queries")
        if len(self.points) > _MAX_BVH_N or k > _MAX_BVH_K:
            raise ValueError("BVH supports N<=1,000,000 and K<=32")
        if os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
            raise RuntimeError("BVH requires PYTORCH_MPS_FAST_MATH=0 before Python starts")
        return "bvh"

    def knn(
        self, query: Tensor, k: int, *, backend: SpatialBackend | None = None,
        parallel_microtrees: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return ``[Q,K]`` Euclidean distances and int64 reference indices."""
        self._check_query(query)
        if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k <= len(self.points):
            raise ValueError(f"k must be in [0, {len(self.points)}]")
        selected = self._selected_backend(query, k, backend, "knn")
        if selected == "scan":
            if parallel_microtrees is not None:
                raise ValueError("parallel_microtrees applies only to backend='bvh'")
            distances, indices = ops.knn(query.unsqueeze(0), self.points.unsqueeze(0), k)
            return distances.squeeze(0), indices.squeeze(0)
        split = len(query) <= 256 if parallel_microtrees is None else parallel_microtrees
        squared, indices, _ = self._get_bvh().knn(query, k, parallel_microtrees=split)
        return torch.sqrt(squared), indices

    def ball_query(
        self, query: Tensor, radius: float, K: int, *, backend: SpatialBackend | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return first-K original neighbors, squared distances, and 0/-1 padding."""
        self._check_query(query)
        _checked_radius_and_k(radius, K)
        selected = self._selected_backend(query, K, backend, "ball_query")
        if selected == "scan":
            distances, indices = ops.ball_query(
                query.unsqueeze(0), self.points.unsqueeze(0), radius, K,
            )
            return distances.squeeze(0), indices.squeeze(0)
        distances, indices, _ = self._get_bvh().radius(query, radius, K)
        if query.requires_grad or self.points.requires_grad:
            distances = _BVHSquaredDistances.apply(query, self.points, indices, distances)
        return distances, indices

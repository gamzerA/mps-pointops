"""PyTorch3D-style Ball Query call surface for three-dimensional points.

Import this module explicitly: ``from mps_pointops.pytorch3d import ball_query``.
It does not import, replace, or depend on PyTorch3D. The Metal path retains
the numerical and dtype limitations documented for ``mps_pointops.ball_query``.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from . import reference
from ._ball_query_mps import _checked_radius_and_k, _lengths_or_full
from ._ball_query_mps import ball_query as _metal_ball_query


class KNN(NamedTuple):
    """PyTorch3D-style ``(dists, idx, knn)`` result."""

    dists: torch.Tensor
    idx: torch.Tensor
    knn: torch.Tensor | None


def _gather_neighbors(points: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather valid neighbors, filling every ``-1`` slot with zero coordinates."""
    batch, queries, k = indices.shape
    if indices.numel() == 0:
        # Preserve the autograd connection for empty K, Q, or B dimensions.
        return points[:, :0, :].reshape(batch, queries, k, 3)
    if points.shape[1] == 0:
        return points.new_zeros((batch, queries, k, 3)) + points.sum() * 0.0
    valid = indices >= 0
    safe = indices.clamp_min(0)
    gathered = points.gather(1, safe.reshape(batch, -1, 1).expand(-1, -1, 3))
    gathered = gathered.reshape(batch, queries, k, 3)
    return torch.where(valid.unsqueeze(-1), gathered, torch.zeros_like(gathered))


def _reference_with_lengths(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: torch.Tensor,
    lengths2: torch.Tensor,
    radius: float,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the existing PyTorch reference per cloud to honor ragged lengths."""
    batch, queries, _ = p1.shape
    dists = torch.zeros((batch, queries, k), dtype=p1.dtype, device=p1.device)
    indices = torch.full((batch, queries, k), -1, dtype=torch.int64, device=p1.device)
    for b in range(batch):
        query_count = int(lengths1[b].item())
        point_count = int(lengths2[b].item())
        if query_count == 0 or point_count == 0 or k == 0:
            continue
        sub_dists, sub_indices = reference.ball_query(
            p1[b : b + 1, :query_count], p2[b : b + 1, :point_count], radius, k
        )
        dists[b : b + 1, :query_count] = sub_dists
        indices[b : b + 1, :query_count] = sub_indices
    return dists, indices


def ball_query(
    p1: torch.Tensor,
    p2: torch.Tensor,
    lengths1: torch.Tensor | None = None,
    lengths2: torch.Tensor | None = None,
    K: int = 500,
    radius: float = 0.2,
    return_nn: bool = True,
    skip_points_outside_cube: bool = False,
) -> KNN:
    """Find the first ``K`` points in ``p2`` within each ``p1`` radius.

    Inputs are matching float32 tensors of shape ``(B, Q, 3)`` and
    ``(B, P, 3)``. Length vectors, when supplied, are int64 tensors of shape
    ``(B,)`` on the same device. ``dists`` is squared distance, ``idx`` is a
    point index or ``-1`` padding, and ``knn`` contains gathered coordinates
    with zero padding when ``return_nn`` is true.

    ``skip_points_outside_cube`` is accepted as an optimization hint. This
    implementation performs the same radius search for either value; it does
    not switch the native shader's search strategy. Float32 boundary behavior
    and supported radii follow this package's Ball Query contract, so this
    adapter does not promise bitwise PyTorch3D CPU/CUDA equivalence. PyTorch3D
    supports other coordinate dimensions; this Metal adapter requires 3.
    """
    if not isinstance(p1, torch.Tensor) or not isinstance(p2, torch.Tensor):
        raise TypeError("p1 and p2 must be torch.Tensor values")
    if p1.ndim != 3 or p2.ndim != 3 or p1.shape[-1] != 3 or p2.shape[-1] != 3:
        raise ValueError("p1 and p2 must have shapes (B, Q, 3) and (B, P, 3)")
    if p1.shape[0] != p2.shape[0]:
        raise ValueError("p1 and p2 must have the same batch dimension")
    if p1.device != p2.device:
        raise ValueError("p1 and p2 must be on the same device")
    if p1.dtype != torch.float32 or p2.dtype != torch.float32:
        raise TypeError("PyTorch3D-style Ball Query currently requires float32 coordinates")
    if not isinstance(return_nn, bool) or not isinstance(skip_points_outside_cube, bool):
        raise TypeError("return_nn and skip_points_outside_cube must be bool")
    _checked_radius_and_k(radius, K)

    if p1.device.type == "mps":
        # The native entry point validates lengths, including their bounds.
        # Doing that here as well would add two extra GPU-to-CPU syncs.
        result = _metal_ball_query(
            p1, p2, radius=radius, k=K, query_lengths=lengths1, point_lengths=lengths2
        )
        dists, idx = result.distances, result.indices
    else:
        batch, queries, _ = p1.shape
        points = p2.shape[1]
        lengths1 = _lengths_or_full(
            lengths1, batch=batch, count=queries, device=p1.device, name="lengths1"
        )
        lengths2 = _lengths_or_full(
            lengths2, batch=batch, count=points, device=p1.device, name="lengths2"
        )
        dists, idx = _reference_with_lengths(p1, p2, lengths1, lengths2, radius, K)

    knn = _gather_neighbors(p2, idx) if return_nn else None
    return KNN(dists=dists, idx=idx, knn=knn)

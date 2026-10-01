"""Experimental legacy ``torch_cluster.random_walk`` call surface.

The implementation uses PyTorch tensor operators on CPU and MPS. Uniform
walks reproduce the upstream 1.6.3 CPU sampler's use of one float32 random
matrix when the CPU generator state and sorted input are the same. Biased
walks approximate the upstream CPU acceptance law; the upstream CUDA source
has a distinct rejection-loop state update, and neither random stream is
bitwise compatible with this implementation.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor


def _validate(row: Tensor, col: Tensor, start: Tensor, walk_length: int,
              p: float, q: float, num_nodes: Optional[int]) -> int:
    for name, value in (("row", row), ("col", col), ("start", start)):
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 1 or value.dtype != torch.int64:
            raise TypeError(f"{name} must be a one-dimensional int64 tensor")
        if value.device != row.device:
            raise ValueError("row, col, and start must be on the same device")
    if row.device.type not in ("cpu", "mps"):
        raise ValueError("random_walk supports CPU and MPS")
    if row.numel() != col.numel():
        raise ValueError("row and col must have the same number of edges")
    if isinstance(walk_length, bool) or not isinstance(walk_length, int) or walk_length < 0:
        raise ValueError("walk_length must be a nonnegative integer")
    for name, value in (("p", p), ("q", q)):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    inverse_weights = (1.0 / p, 1.0, 1.0 / q)
    if (not all(math.isfinite(weight) for weight in inverse_weights) or
            min(inverse_weights) / max(inverse_weights) < torch.finfo(torch.float32).tiny):
        raise ValueError("p and q yield transition weights outside the float32 normal range")
    if num_nodes is None:
        if not row.numel() or not col.numel() or not start.numel():
            raise ValueError("num_nodes is required when row, col, or start is empty")
        num_nodes = max(int(row.max()), int(col.max()), int(start.max())) + 1
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")
    if row.numel() and (bool((row < 0).any()) or bool((row >= num_nodes).any()) or
                        bool((col < 0).any()) or bool((col >= num_nodes).any())):
        raise ValueError("edge indices must lie in [0, num_nodes)")
    if start.numel() and (bool((start < 0).any()) or bool((start >= num_nodes).any())):
        raise ValueError("start indices must lie in [0, num_nodes)")
    return num_nodes


def _choose_uniform(current: Tensor, rowptr: Tensor, col: Tensor,
                    random_values: Tensor) -> tuple[Tensor, Tensor]:
    begin = rowptr[current]
    degree = rowptr[current + 1] - begin
    offset = torch.minimum((random_values * degree).to(torch.int64),
                           (degree - 1).clamp(min=0))
    edge = begin + offset
    valid = degree > 0
    if col.numel():
        next_node = torch.where(valid, col[edge.clamp(max=col.numel() - 1)], current)
    else:
        next_node = current
    return next_node, torch.where(valid, edge, torch.full_like(edge, -1))


def _choose_biased(previous: Tensor, current: Tensor, rowptr: Tensor,
                   col: Tensor, sorted_keys: Tensor, num_nodes: int,
                   p: float, q: float) -> tuple[Tensor, Tensor]:
    begin = rowptr[current]
    degree = rowptr[current + 1] - begin
    if col.numel() == 0:
        return current, torch.full_like(current, -1)
    width = int(degree.max()) if degree.numel() else 0
    if width == 0:
        return current, torch.full_like(current, -1)

    offset = torch.arange(width, device=current.device, dtype=torch.int64)
    valid = offset.unsqueeze(0) < degree.unsqueeze(1)
    edge = begin.unsqueeze(1) + offset.unsqueeze(0)
    candidate = col[edge.clamp(max=col.numel() - 1)]

    # The upstream CPU rejection sampler tests adjacency in the reverse direction:
    # candidate -> previous.  It then checks all three acceptance conditions
    # with the same random variate.  Their union is the maximum threshold.
    query_key = candidate * num_nodes + previous.unsqueeze(1)
    key_index = torch.searchsorted(sorted_keys, query_key.reshape(-1))
    adjacent = (sorted_keys[key_index.clamp(max=sorted_keys.numel() - 1)]
                == query_key.reshape(-1)).reshape(query_key.shape)
    scale = max(1.0 / p, 1.0, 1.0 / q)
    return_weight = 1.0 / p / scale
    neighbor_weight = 1.0 / scale
    outward_weight = 1.0 / q / scale
    weight = torch.full(candidate.shape, outward_weight,
                        device=current.device, dtype=torch.float32)
    weight = torch.where(adjacent, torch.maximum(
        weight, torch.full_like(weight, neighbor_weight)), weight)
    weight = torch.where(candidate == previous.unsqueeze(1), torch.maximum(
        weight, torch.full_like(weight, return_weight)), weight)
    weight = weight * valid
    cumulative = weight.cumsum(dim=1)
    total = cumulative[:, -1]
    random_values = torch.rand(current.shape, device=current.device) * total
    selected = (cumulative <= random_values.unsqueeze(1)).sum(dim=1)
    selected = selected.clamp(max=width - 1)
    chosen_edge = edge.gather(1, selected.unsqueeze(1)).squeeze(1)
    chosen_node = candidate.gather(1, selected.unsqueeze(1)).squeeze(1)
    has_edge = degree > 0
    return (torch.where(has_edge, chosen_node, current),
            torch.where(has_edge, chosen_edge, torch.full_like(chosen_edge, -1)))


def random_walk(
    row: Tensor,
    col: Tensor,
    start: Tensor,
    walk_length: int,
    p: float = 1,
    q: float = 1,
    coalesced: bool = True,
    num_nodes: Optional[int] = None,
    return_edge_indices: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Sample legacy ``torch_cluster``-style walks on CPU or MPS.

    ``coalesced=True`` sorts edges by ``row * num_nodes + col``.  With
    ``coalesced=False``, the caller must provide edges already grouped by row.
    Returned edge IDs index the (possibly sorted) edge sequence.  Isolated
    vertices repeat their current node and use edge ID ``-1``.

    Uniform walks have exact CPU parity for equal PyTorch CPU RNG state and
    duplicate-free edges.  MPS uses a device-specific RNG.  For ``p`` or
    ``q`` unequal to one, the transition probabilities are the float32
    cumulative-sum approximation of upstream 1.6.3 CPU rejection conditions,
    but the random stream differs from its C ``rand``
    or CUDA ``curand`` implementation.  The biased path uses O(W * d_max)
    temporary space per step for W walkers and maximum visited degree d_max.
    """
    num_nodes = _validate(row, col, start, walk_length, p, q, num_nodes)
    if coalesced and row.numel():
        perm = torch.argsort(row * num_nodes + col)
        row, col = row[perm], col[perm]
    elif row.numel() > 1 and bool((row[1:] < row[:-1]).any()):
        raise ValueError("coalesced=False requires edges grouped by row")

    degree = torch.zeros(num_nodes, dtype=torch.int64, device=row.device)
    if row.numel():
        degree.scatter_add_(0, row, torch.ones_like(row))
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=row.device)
    rowptr[1:] = torch.cumsum(degree, dim=0)

    nodes = [start]
    edges = []
    if start.numel() and walk_length:
        random_values = torch.rand((start.numel(), walk_length), device=start.device)
        sorted_keys = torch.sort(row * num_nodes + col).values if row.numel() else row
        for step in range(walk_length):
            current = nodes[-1]
            if step == 0 or (p == 1 and q == 1):
                next_node, edge = _choose_uniform(
                    current, rowptr, col, random_values[:, step])
            else:
                next_node, edge = _choose_biased(
                    nodes[-2], current, rowptr, col, sorted_keys, num_nodes, p, q)
            nodes.append(next_node)
            edges.append(edge)
    else:
        nodes.extend(start for _ in range(walk_length))
        edges.extend(torch.empty_like(start) for _ in range(walk_length))

    node_seq = torch.stack(nodes, dim=1)
    if not return_edge_indices:
        return node_seq
    edge_seq = torch.stack(edges, dim=1) if edges else start.new_empty((start.numel(), 0))
    return node_seq, edge_seq

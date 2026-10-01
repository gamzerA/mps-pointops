"""Greedy pair clustering for the legacy ``torch_cluster.graclus_cluster`` API.

This is an independently written implementation of the observable 1.6.3 CPU
contract. MPS uses a single Metal work item for the serial greedy decision;
the native path is intentionally a correctness baseline, not a speed claim.
"""

from __future__ import annotations

from functools import cache
from importlib import resources

import torch
from torch import Tensor


@cache
def _library():
    source = resources.files(__package__).joinpath("kernels", "graclus.metal").read_text()
    return torch.mps.compile_shader(source)


def _validate(row: Tensor, col: Tensor, weight: Tensor | None, num_nodes: int | None) -> int:
    if not isinstance(row, Tensor) or not isinstance(col, Tensor):
        raise TypeError("row and col must be tensors")
    if row.ndim != 1 or col.ndim != 1 or row.numel() != col.numel():
        raise ValueError("row and col must be one-dimensional with equal length")
    if row.dtype != torch.int64 or col.dtype != torch.int64:
        raise TypeError("row and col must be int64")
    if row.device != col.device or row.device.type not in ("cpu", "mps"):
        raise ValueError("row and col must be on the same CPU or MPS device")
    if weight is not None:
        if not isinstance(weight, Tensor):
            raise TypeError("weight must be a tensor or None")
        if weight.ndim != 1 or weight.numel() != row.numel():
            raise ValueError("weight must be one-dimensional with one value per edge")
        if weight.device != row.device or weight.dtype != torch.float32:
            raise TypeError("weight must be float32 on the same device as row")
        if not bool(torch.isfinite(weight).all().item()):
            raise ValueError("weight values must be finite")
    if num_nodes is None:
        if row.numel() == 0:
            raise ValueError("num_nodes is required when row and col are empty")
        num_nodes = max(int(row.max().item()), int(col.max().item())) + 1
    if isinstance(num_nodes, bool) or not isinstance(num_nodes, int) or num_nodes < 0:
        raise ValueError("num_nodes must be a nonnegative integer")
    if row.numel() and bool(((row < 0) | (col < 0) | (row >= num_nodes) | (col >= num_nodes)).any().item()):
        raise IndexError("row and col must contain node IDs in [0, num_nodes)")
    return num_nodes


def _csr_and_order(
    row: Tensor, col: Tensor, weight: Tensor | None, num_nodes: int
) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
    # Match the public wrapper's random edge permutation only when unweighted.
    # torch.argsort's equal-key order is intentionally left to the backend.
    keep = row != col
    row, col = row[keep], col[keep]
    if weight is None:
        edge_order = torch.randperm(row.numel(), dtype=torch.int64, device=row.device)
        row, col = row[edge_order], col[edge_order]
    else:
        weight = weight[keep]
    order = torch.argsort(row)
    row, col = row[order], col[order]
    if weight is not None:
        weight = weight[order]
    degree = torch.zeros(num_nodes, dtype=torch.int64, device=row.device)
    degree.scatter_add_(0, row, torch.ones_like(row))
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.int64, device=row.device)
    torch.cumsum(degree, 0, out=rowptr[1:])
    # The native 1.6.3 CPU op draws a node permutation after the CSR wrapper.
    node_order = torch.randperm(num_nodes, dtype=torch.int64, device=row.device)
    return rowptr, col.contiguous(), weight.contiguous() if weight is not None else None, node_order


def _greedy_cpu(rowptr: Tensor, col: Tensor, weight: Tensor | None, order: Tensor) -> Tensor:
    offsets = rowptr.tolist()
    targets = col.tolist()
    strengths = weight.tolist() if weight is not None else None
    labels = [-1] * order.numel()
    for center in order.tolist():
        if labels[center] != -1:
            continue
        chosen = center
        if strengths is None:
            for neighbor in targets[offsets[center]:offsets[center + 1]]:
                if labels[neighbor] == -1:
                    chosen = neighbor
                    break
        else:
            maximum = 0.0
            for edge in range(offsets[center], offsets[center + 1]):
                neighbor = targets[edge]
                if labels[neighbor] == -1 and strengths[edge] >= maximum:
                    maximum = strengths[edge]
                    chosen = neighbor
        identifier = min(center, chosen)
        labels[center] = identifier
        labels[chosen] = identifier
    return torch.tensor(labels, dtype=torch.int64, device=rowptr.device)


def graclus_cluster(
    row: Tensor,
    col: Tensor,
    weight: Tensor | None = None,
    num_nodes: int | None = None,
) -> Tensor:
    """Return a greedy matching partition as non-contiguous int64 labels.

    The four arguments and pair label ``min(u, v)`` follow torch_cluster 1.6.3.
    Self-loops are removed. An unweighted call randomly permutes the remaining
    edges; every call independently permutes nodes. A weighted call chooses
    the last still-available neighbor with maximal weight at least zero in its
    backend's CSR order. Thus negative-only edges leave a node unmatched.

    Exact CPU/MPS seeded parity is not promised: CPU and MPS use different RNG
    streams, and equal-key argsort order may differ. Weighted MPS supports
    finite float32 weights. Index range and shape errors are checked before
    native access; those checks synchronize scalar values on MPS. The Metal
    implementation serializes the greedy decision in one work item, so it is
    best suited to modest graphs until a parallel matching design is measured.
    """
    num_nodes = _validate(row, col, weight, num_nodes)
    rowptr, sorted_col, sorted_weight, node_order = _csr_and_order(row, col, weight, num_nodes)
    if row.device.type == "cpu":
        return _greedy_cpu(rowptr, sorted_col, sorted_weight, node_order)
    out = torch.empty(num_nodes, dtype=torch.int64, device=row.device)
    if num_nodes:
        _library().graclus_greedy(
            rowptr.contiguous(), sorted_col,
            sorted_weight if sorted_weight is not None else torch.empty(1, dtype=torch.float32, device=row.device),
            node_order.contiguous(), out, num_nodes, int(weight is not None),
            threads=1, group_size=1,
        )
    return out


__all__ = ["graclus_cluster"]

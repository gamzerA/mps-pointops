"""Flat point-cloud operators with the ``torch_cluster`` index convention.

``x`` contains reference points, ``y`` contains query points, and batch vectors
are sorted. Indices in the results refer to the original flat tensors. These
functions support three-dimensional point coordinates; they are also exposed
by the optional ``torch_cluster`` shim in :mod:`mps_pointops.compat`.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from . import reference


def _points(value: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {tuple(value.shape)}")
    if not value.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")


def _batch_size(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("batch_size must be a non-negative integer")
    return value


def _check_batch(batch: Tensor | None, count: int, device: torch.device, name: str) -> int:
    if batch is None:
        return 0
    if not isinstance(batch, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor or None")
    if batch.ndim != 1 or batch.numel() != count:
        raise ValueError(f"{name} must have shape ({count},)")
    if batch.device != device:
        raise ValueError(f"{name} must be on {device}")
    if batch.dtype not in (torch.int32, torch.int64):
        raise TypeError(f"{name} must have an integer dtype (int32 or int64)")
    if not count:
        return 0
    if bool((batch[0] < 0).item()) or bool((batch[1:] < batch[:-1]).any().item()):
        raise ValueError(f"{name} must be sorted with non-negative batch IDs")
    return int(batch[-1].item()) + 1


def _ptr_from_batch(
    batch: Tensor | None, count: int, device: torch.device, size: int
) -> Tensor:
    if batch is None:
        # A missing batch vector means one point cloud, even if it is empty.
        return torch.tensor([0, count], dtype=torch.long, device=device)
    if size == 0:
        return torch.zeros(1, dtype=torch.long, device=device)
    boundaries = torch.arange(size + 1, dtype=torch.long, device=device)
    return torch.searchsorted(batch.contiguous().long(), boundaries).long()


def _pair_ptrs(
    x: Tensor,
    y: Tensor,
    batch_x: Tensor | None,
    batch_y: Tensor | None,
    batch_size: int | None,
) -> tuple[Tensor, Tensor]:
    explicit_size = _batch_size(batch_size)
    size_x = _check_batch(batch_x, len(x), x.device, "batch_x")
    size_y = _check_batch(batch_y, len(y), y.device, "batch_y")
    size = max(1, size_x, size_y, explicit_size or 0)
    if size > 1 and (batch_x is None or batch_y is None):
        raise ValueError("batch_x and batch_y are both required for multiple batches")
    if explicit_size is not None and explicit_size < max(size_x, size_y):
        raise ValueError("batch_size is smaller than the largest batch ID")
    return (
        _ptr_from_batch(batch_x, len(x), x.device, size),
        _ptr_from_batch(batch_y, len(y), y.device, size),
    )


def _fps_ptr(x: Tensor, batch: Tensor | None, batch_size: int | None, ptr: Tensor | Sequence[int] | None) -> Tensor:
    if ptr is not None:
        if isinstance(ptr, Tensor):
            if ptr.device != x.device or ptr.dtype not in (torch.int32, torch.int64):
                raise ValueError("ptr must be an integer tensor on the same device as x")
            result = ptr.long()
        elif isinstance(ptr, (list, tuple)):
            result = torch.tensor(ptr, dtype=torch.long, device=x.device)
        else:
            raise TypeError("ptr must be a tensor, a sequence of integers, or None")
        if result.ndim != 1 or len(result) == 0:
            raise ValueError("ptr must be a non-empty one-dimensional offset array")
        if int(result[0].item()) != 0 or int(result[-1].item()) != len(x):
            raise ValueError("ptr must start at 0 and end at len(x)")
        if bool((result[1:] < result[:-1]).any().item()):
            raise ValueError("ptr must be non-decreasing")
        return result.contiguous()
    explicit_size = _batch_size(batch_size)
    inferred_size = _check_batch(batch, len(x), x.device, "batch")
    if explicit_size is not None and explicit_size < inferred_size:
        raise ValueError("batch_size is smaller than the largest batch ID")
    size = max(1, inferred_size, explicit_size or 0) if batch is None else max(inferred_size, explicit_size or 0)
    return _ptr_from_batch(batch, len(x), x.device, size)


def _ratio(value: float | Tensor | None, dtype: torch.dtype) -> Tensor:
    """Return the ratio as the CPU scalar tensor torch_cluster would use.

    torch_cluster turns a Python ratio into a tensor of the points' dtype and
    keeps a tensor ratio as given. The sample counts are then computed in that
    precision, so the dtype is part of the contract.
    """
    if value is None:
        value = 0.5
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError("ratio tensor must contain one value")
        result = value.detach().to("cpu").reshape(())
        if not result.is_floating_point():
            result = result.to(torch.float32)
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("ratio must be a number or scalar tensor")
    else:
        result = torch.tensor(float(value), dtype=dtype)
    if not math.isfinite(float(result)) or float(result) < 0:
        raise ValueError("ratio must be finite and non-negative")
    return result


def _sample_counts(lengths: list[int], ratio: Tensor) -> list[int]:
    """Samples per cloud, ``ceil(float32(N_b) * ratio)``, as in torch_cluster.

    The product is rounded in the ratio's precision before the ceiling. In
    float32 that is not always ``ceil`` of the exact product: 25 points at
    ratio 0.6 give 16 samples, because 25 * float32(0.6) rounds just above 15.
    """
    sizes = torch.tensor(lengths, dtype=torch.float32)
    return torch.ceil(sizes * ratio).to(torch.long).tolist()


def fps(
    x: Tensor,
    batch: Tensor | None = None,
    ratio: float | Tensor | None = None,
    random_start: bool = True,
    batch_size: int | None = None,
    ptr: Tensor | Sequence[int] | None = None,
) -> Tensor:
    """Sample ``ceil(float32(N_b) * ratio)`` global point indices per batch.

    The count is computed as torch_cluster does, with the ratio in the dtype
    of ``x`` (or of the ratio tensor). ``ratio=None`` means 0.5. ``ptr`` takes
    precedence over ``batch`` if
    supplied. ``random_start=False`` starts from the first point in each
    non-empty batch. The output is an int64 tensor on ``x.device``.
    """
    _points(x, "x")
    if x.device.type == "mps" and x.dtype != torch.float32:
        raise TypeError("x must be float32 on MPS")
    sample_ratio = _ratio(ratio, x.dtype)
    offsets = _fps_ptr(x, batch, batch_size, ptr)
    offset_values = offsets.cpu().tolist()
    lengths = [hi - lo for lo, hi in zip(offset_values, offset_values[1:])]
    counts = _sample_counts(lengths, sample_ratio)
    output_offsets = [0]
    for count in counts:
        output_offsets.append(output_offsets[-1] + count)
    if output_offsets[-1] == 0:
        return torch.empty(0, dtype=torch.long, device=x.device)

    if random_start:
        length_tensor = torch.tensor(lengths, dtype=torch.long, device=x.device)
        starts = (torch.rand(len(lengths), device=x.device) * length_tensor).long()
        starts = torch.minimum(starts, (length_tensor - 1).clamp_min(0))
    else:
        starts = torch.zeros(len(lengths), dtype=torch.long, device=x.device)

    if x.device.type == "mps":
        from ._flat_fps_mps import fps_flat

        return fps_flat(
            x.contiguous(),
            offsets,
            torch.tensor(output_offsets, dtype=torch.long, device=x.device),
            starts,
        )

    samples = []
    for batch_id, (lo, hi) in enumerate(zip(offset_values, offset_values[1:])):
        count = counts[batch_id]
        if count:
            selected = reference.furthest_point_sample(
                x[lo:hi].unsqueeze(0), count, start_idx=int(starts[batch_id].item())
            )[0]
            samples.append(selected + lo)
    return torch.cat(samples) if samples else torch.empty(0, dtype=torch.long, device=x.device)


def _search_inputs(x: Tensor, y: Tensor) -> None:
    _points(x, "x")
    _points(y, "y")
    if x.device != y.device:
        raise ValueError(f"x and y must be on the same device, got {x.device} and {y.device}")
    if x.dtype != y.dtype:
        raise TypeError(f"x and y must have the same dtype, got {x.dtype} and {y.dtype}")


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _edges(indices: Tensor) -> Tensor:
    if indices.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=indices.device)
    valid = indices >= 0
    queries = torch.arange(indices.shape[0], device=indices.device, dtype=torch.long)
    queries = queries[:, None].expand_as(indices)[valid]
    return torch.stack((queries, indices[valid]), dim=0)


def _sqdist(query: Tensor, refs: Tensor) -> Tensor:
    d = refs - query
    return (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]


def _reference_search(
    x: Tensor, y: Tensor, ptr_x: Tensor, ptr_y: Tensor, limit: int, radius: float | None
) -> Tensor:
    indices = torch.full((len(y), limit), -1, dtype=torch.long, device=x.device)
    if limit == 0 or len(x) == 0 or len(y) == 0:
        return indices
    x_offsets = ptr_x.cpu().tolist()
    y_offsets = ptr_y.cpu().tolist()
    r2 = radius * radius if radius is not None else None
    for (x_lo, x_hi), (y_lo, y_hi) in zip(
        zip(x_offsets, x_offsets[1:]), zip(y_offsets, y_offsets[1:])
    ):
        if x_lo == x_hi:
            continue
        refs = x[x_lo:x_hi]
        take = min(limit, x_hi - x_lo)
        for query_idx in range(y_lo, y_hi):
            d2 = _sqdist(y[query_idx], refs)
            if r2 is None:
                ordered = torch.argsort(d2, stable=True)
                local = ordered[torch.isfinite(d2[ordered])][:take]
            else:
                local = torch.nonzero(d2 < r2, as_tuple=False).flatten()[:take]
            indices[query_idx, : len(local)] = local + x_lo
    return indices


def knn(
    x: Tensor,
    y: Tensor,
    k: int,
    batch_x: Tensor | None = None,
    batch_y: Tensor | None = None,
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: int | None = None,
) -> Tensor:
    """Return global ``[query, reference]`` edges for the nearest points.

    The ``batch_size`` argument sets the number of batch slots, including
    empty ones. ``cosine=True`` is not implemented for 3D point coordinates.
    ``num_workers`` is accepted for call-site compatibility and has no effect
    for MPS tensors or batched inputs.
    """
    _search_inputs(x, y)
    k = _nonnegative_int(k, "k")
    if cosine:
        raise NotImplementedError("cosine kNN is not supported")
    _nonnegative_int(num_workers, "num_workers")
    if x.device.type == "mps" and x.dtype != torch.float32:
        raise TypeError("x and y must be float32 for kNN on MPS")
    if k == 0 or len(x) == 0 or len(y) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    ptr_x, ptr_y = _pair_ptrs(x, y, batch_x, batch_y, batch_size)
    width = min(k, len(x))
    if x.device.type == "mps" and width <= 256:
        from ._flat_search_mps import knn_indices

        indices = knn_indices(x.contiguous(), y.contiguous(), ptr_x, ptr_y, width)
    else:
        indices = _reference_search(x, y, ptr_x, ptr_y, width, radius=None)
    return _edges(indices)


def radius(
    x: Tensor,
    y: Tensor,
    r: float,
    batch_x: Tensor | None = None,
    batch_y: Tensor | None = None,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    batch_size: int | None = None,
    ignore_same_index: bool = False,
) -> Tensor:
    """Return global ``[query, reference]`` edges with squared distance ``< r²``.

    Neighbors are selected in reference input order and limited to
    ``max_num_neighbors`` per query. Unused padded kernel slots are removed.
    ``ignore_same_index=True`` is not implemented. ``num_workers`` is
    accepted for call-site compatibility and has no effect on MPS or with
    batches.
    """
    _search_inputs(x, y)
    limit = _nonnegative_int(max_num_neighbors, "max_num_neighbors")
    _nonnegative_int(num_workers, "num_workers")
    if ignore_same_index:
        raise NotImplementedError("ignore_same_index=True is not supported")
    if isinstance(r, bool) or not isinstance(r, (int, float)):
        raise ValueError("r must be a finite non-negative number")
    r = float(r)
    if not math.isfinite(r) or r < 0:
        raise ValueError("r must be a finite non-negative number")
    if x.device.type == "mps" and x.dtype not in (torch.float16, torch.float32):
        raise TypeError("x and y must be float16 or float32 for radius on MPS")
    if limit == 0 or len(x) == 0 or len(y) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    ptr_x, ptr_y = _pair_ptrs(x, y, batch_x, batch_y, batch_size)
    width = min(limit, len(x))
    if x.device.type == "mps":
        from ._flat_search_mps import radius_indices

        indices = radius_indices(x.contiguous(), y.contiguous(), ptr_x, ptr_y, r, width)
    else:
        indices = _reference_search(x, y, ptr_x, ptr_y, width, radius=r)
    return _edges(indices)


def _graph_edges(edges: Tensor, loop: bool, flow: str) -> Tensor:
    if flow not in ("source_to_target", "target_to_source"):
        raise ValueError("flow must be 'source_to_target' or 'target_to_source'")
    if flow == "source_to_target":
        edges = edges.flip(0)
    if not loop:
        edges = edges[:, edges[0] != edges[1]]
    return edges


def knn_graph(
    x: Tensor,
    k: int,
    batch: Tensor | None = None,
    loop: bool = False,
    flow: str = "source_to_target",
    cosine: bool = False,
    num_workers: int = 1,
    batch_size: int | None = None,
) -> Tensor:
    """Build a kNN graph with ``torch_cluster`` 1.6.3 edge orientation."""
    k = _nonnegative_int(k, "k")
    if flow not in ("source_to_target", "target_to_source"):
        raise ValueError("flow must be 'source_to_target' or 'target_to_source'")
    edges = knn(
        x, x, k if loop else k + 1, batch, batch,
        cosine=cosine, num_workers=num_workers, batch_size=batch_size,
    )
    return _graph_edges(edges, loop, flow)


def radius_graph(
    x: Tensor,
    r: float,
    batch: Tensor | None = None,
    loop: bool = False,
    max_num_neighbors: int = 32,
    flow: str = "source_to_target",
    num_workers: int = 1,
    batch_size: int | None = None,
) -> Tensor:
    """Build a radius graph with ``torch_cluster`` 1.6.3 edge orientation."""
    limit = _nonnegative_int(max_num_neighbors, "max_num_neighbors")
    if flow not in ("source_to_target", "target_to_source"):
        raise ValueError("flow must be 'source_to_target' or 'target_to_source'")
    edges = radius(
        x, x, r, batch, batch, limit if loop else limit + 1,
        num_workers=num_workers, batch_size=batch_size,
    )
    return _graph_edges(edges, loop, flow)


__all__ = ["fps", "knn", "radius", "knn_graph", "radius_graph"]

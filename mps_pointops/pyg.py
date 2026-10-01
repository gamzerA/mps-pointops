"""MPS dispatch kernels for the point-cloud operators used by PyG 2.8.

PyG 2.8 checks for :mod:`pyg_lib` before calling ``torch.ops.pyg``. Install
the matching pyg-lib wheel, then call :func:`register_mps` once in the process.
This module adds only MPS implementations to pyg-lib's existing operator
schemas; pyg-lib keeps ownership of its CPU and CUDA implementations.
"""

from __future__ import annotations

from functools import wraps
from importlib.metadata import PackageNotFoundError, version

import torch
from torch import Tensor

from . import flat


_MPS_LIBRARY: torch.library.Library | None = None


def _install_pyg28_ptr_bridge() -> None:
    """Use searchsorted where PyG 2.8's CSR conversion has no MPS kernel.

    PyG 2.8's ``_batch_to_ptr`` calls ``index2ptr``, which reaches
    ``aten::_convert_indices_from_coo_to_csr.out``. PyTorch 2.12 has no MPS
    implementation for that operator. Keep PyG's original helper for all
    non-MPS inputs and replace only its MPS batch-vector conversion.
    """
    try:
        pyg_version = version("torch-geometric")
    except PackageNotFoundError:
        return
    if not pyg_version.startswith("2.8."):
        return

    import torch_geometric.nn.pool as pool

    original = pool._batch_to_ptr
    if getattr(original, "_mps_pointops_ptr_bridge", False):
        return

    @wraps(original)
    def batch_to_ptr(batch: Tensor | None, batch_size: int | None = None) -> Tensor | None:
        if batch is None or batch.device.type != "mps":
            return original(batch, batch_size)
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1 if batch.numel() else 0
        boundaries = torch.arange(batch_size + 1, dtype=batch.dtype, device=batch.device)
        return torch.searchsorted(batch.contiguous(), boundaries).long()

    batch_to_ptr._mps_pointops_ptr_bridge = True  # type: ignore[attr-defined]
    pool._batch_to_ptr = batch_to_ptr


def _ptrs(
    x: Tensor, y: Tensor, ptr_x: Tensor | None, ptr_y: Tensor | None,
    *, feature_space: bool = False,
) -> tuple[Tensor, Tensor]:
    flat._search_inputs(x, y, feature_space=feature_space)
    if ptr_x is None:
        # pyg-lib treats an unbatched x as the full reference set even when
        # y has a batch pointer. Its ptr_y is ignored in this case.
        ptr_x = torch.tensor([0, len(x)], dtype=torch.long, device=x.device)
        ptr_y = torch.tensor([0, len(y)], dtype=torch.long, device=y.device)
    elif ptr_y is None:
        raise ValueError("ptr_y is required when ptr_x is given")
    return ptr_x, ptr_y


def _knn_mps(
    x: Tensor,
    y: Tensor,
    ptr_x: Tensor | None = None,
    ptr_y: Tensor | None = None,
    k: int = 1,
    cosine: bool = False,
    num_workers: int = 1,
) -> Tensor:
    """Return pyg-lib's query-major ``[query, reference]`` edge index."""
    from ._flat_search_mps import _check_inputs, knn_indices

    ptr_x, ptr_y = _ptrs(x, y, ptr_x, ptr_y, feature_space=True)
    _check_inputs(x, y, ptr_x, ptr_y, feature_space=True)
    k = flat._nonnegative_int(k, "k")
    flat._nonnegative_int(num_workers, "num_workers")
    if cosine:
        raise NotImplementedError("cosine kNN is not supported for MPS point clouds")
    if x.dtype != torch.float32:
        raise TypeError("x and y must be float32 for kNN on MPS")
    if k == 0 or len(x) == 0 or len(y) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    width = min(k, len(x))
    if x.shape[1] != 3 and width > 256:
        raise ValueError("feature-space kNN supports at most 256 neighbors on MPS")
    if width <= 256:
        indices = knn_indices(x.contiguous(), y.contiguous(), ptr_x, ptr_y, width)
    else:
        indices = flat._reference_search(x, y, ptr_x, ptr_y, width, radius=None)
    return flat._edges(indices)


def _radius_mps(
    x: Tensor,
    y: Tensor,
    ptr_x: Tensor | None = None,
    ptr_y: Tensor | None = None,
    r: float = 1.0,
    max_num_neighbors: int = 32,
    num_workers: int = 1,
    ignore_same_index: bool = False,
) -> Tensor:
    """Return the first neighbors in x order, optionally omitting equal IDs."""
    from ._flat_search_mps import _check_inputs, radius_indices

    ptr_x, ptr_y = _ptrs(x, y, ptr_x, ptr_y)
    _check_inputs(x, y, ptr_x, ptr_y)
    limit = flat._nonnegative_int(max_num_neighbors, "max_num_neighbors")
    flat._nonnegative_int(num_workers, "num_workers")
    if x.dtype not in (torch.float16, torch.float32):
        raise TypeError("x and y must be float16 or float32 for radius on MPS")
    if limit == 0 or len(x) == 0 or len(y) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=x.device)
    width = min(limit, len(x))
    indices = radius_indices(
        x.contiguous(), y.contiguous(), ptr_x, ptr_y, r, width,
        ignore_same_index=ignore_same_index,
    )
    return flat._edges(indices)


def _fps_mps(src: Tensor, ptr: Tensor, ratio: float = 0.5, random_start: bool = True) -> Tensor:
    """Return FPS indices using pyg-lib's pointer-shaped call surface."""
    return flat.fps(src, ratio=ratio, random_start=random_start, ptr=ptr)


def register_mps() -> list[str]:
    """Register MPS kernels for pyg-lib's ``fps``, ``knn`` and ``radius``.

    Requires pyg-lib 0.6 or newer so PyG 2.8's feature checks and operator
    schemas are genuine. Existing MPS kernels, if any, are left unchanged.
    Returns the operator names newly registered by this call.
    """
    global _MPS_LIBRARY
    if _MPS_LIBRARY is not None:
        _install_pyg28_ptr_bridge()
        return []
    try:
        import pyg_lib
    except ImportError as exc:
        raise ImportError(
            "PyG 2.8 integration requires pyg-lib>=0.6 with a wheel matching "
            "your PyTorch version; install it from https://data.pyg.org/whl/"
        ) from exc

    implementations = {
        "fps": _fps_mps,
        "knn": _knn_mps,
        "radius": _radius_mps,
    }
    for name in implementations:
        if not hasattr(pyg_lib.ops, name) or not hasattr(torch.ops.pyg, name):
            raise RuntimeError(f"pyg-lib does not provide pyg::{name}; pyg-lib>=0.6 is required")

    library = torch.library.Library("pyg", "IMPL", "MPS")
    registered = []
    for name, implementation in implementations.items():
        if not torch._C._dispatch_has_kernel_for_dispatch_key(f"pyg::{name}", "MPS"):
            library.impl(name, implementation)
            registered.append(name)
    _MPS_LIBRARY = library  # torch.library registrations live as long as the Library.
    _install_pyg28_ptr_bridge()
    return registered

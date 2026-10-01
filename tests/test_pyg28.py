"""Integration checks against PyG 2.8 and pyg-lib's real operator schemas."""

import importlib.util
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch

from mps_pointops import pyg


def _pyg_28_available() -> bool:
    if importlib.util.find_spec("pyg_lib") is None:
        return False
    try:
        return version("torch-geometric").startswith("2.8.")
    except PackageNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or not _pyg_28_available(),
    reason="requires MPS, torch-geometric 2.8 and pyg-lib>=0.6",
)


def test_register_real_pyg_schemas_and_pool_functions():
    names = {"fps", "knn", "radius", "grid_cluster"}
    assert set(pyg.register_mps()) <= names
    assert all(torch._C._dispatch_has_kernel_for_dispatch_key(f"pyg::{name}", "MPS") for name in names)
    assert pyg.register_mps() == []

    from torch_geometric.nn.pool import fps, knn, knn_graph, radius, radius_graph

    x = torch.tensor(
        [[0.0, 0, 0], [1.0, 0, 0], [10.0, 0, 0], [11.0, 0, 0]],
        device="mps",
    )
    y = torch.tensor([[0.1, 0, 0], [10.1, 0, 0]], device="mps")
    batch_x = torch.tensor([0, 0, 1, 1], dtype=torch.long, device="mps")
    batch_y = torch.tensor([0, 1], dtype=torch.long, device="mps")

    assert fps(x, batch_x, ratio=0.5, random_start=False).cpu().tolist() == [0, 2]
    expected = [[0, 0, 1, 1], [0, 1, 2, 3]]
    assert knn(x, y, 2, batch_x, batch_y).cpu().tolist() == expected
    assert radius(x, y, 1.0, batch_x, batch_y).cpu().tolist() == expected

    graph = [[1, 0, 3, 2], [0, 1, 2, 3]]
    assert knn_graph(x, 1, batch_x, loop=False).cpu().tolist() == graph
    assert radius_graph(x, 2.0, batch_x, loop=False, max_num_neighbors=1).cpu().tolist() == graph


def test_radius_ignores_equal_global_indices_before_neighbor_limit():
    pyg.register_mps()
    x = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.2, 0, 0]], device="mps")
    y = torch.tensor([[0.0, 0, 0]], device="mps")
    edges = torch.ops.pyg.radius(x, y, None, None, 1.0, 1, 1, True)
    assert edges.cpu().tolist() == [[0], [1]]


def test_pyg28_batch_pointer_bridge_keeps_empty_batch_slots():
    pyg.register_mps()
    import torch_geometric.nn.pool as pool

    batch = torch.tensor([0, 0, 2], dtype=torch.long, device="mps")
    ptr = pool._batch_to_ptr(batch, batch_size=4)
    assert ptr.cpu().tolist() == [0, 2, 2, 3, 3]


def test_direct_pyg_operators_accept_schema_defaults():
    pyg.register_mps()
    x = torch.tensor([[0.0, 0, 0], [2.0, 0, 0]], device="mps")
    y = torch.tensor([[0.1, 0, 0]], device="mps")
    assert torch.ops.pyg.knn(x, y).cpu().tolist() == [[0], [0]]
    assert torch.ops.pyg.radius(x, y).cpu().tolist() == [[0], [0]]


def test_feature_space_knn_uses_registered_pyg_operator():
    pyg.register_mps()
    x = torch.zeros((4, 64), device="mps")
    x[:, 0] = torch.tensor([0.0, 1.0, 10.0, 11.0], device="mps")
    y = x[[0, 2]] + 0.1
    ptr_x = torch.tensor([0, 2, 4], device="mps")
    ptr_y = torch.tensor([0, 1, 2], device="mps")
    edges = torch.ops.pyg.knn(x, y, ptr_x, ptr_y, 2)
    assert edges.cpu().tolist() == [[0, 0, 1, 1], [0, 1, 2, 3]]


@pytest.mark.parametrize("dimensions", [3, 64])
def test_pyg_knn_large_effective_k_is_explicit_error(dimensions):
    pyg.register_mps()
    x = torch.arange(257, dtype=torch.float32, device="mps")[:, None].expand(-1, dimensions)
    y = x[:1]
    with pytest.raises(ValueError, match="effective k must be at most 256 on MPS"):
        torch.ops.pyg.knn(x, y, None, None, 257)
    assert torch.ops.pyg.knn(x[:256], y, None, None, 300).shape == (2, 256)


def test_query_batch_without_reference_batch_uses_all_references():
    pyg.register_mps()
    x = torch.tensor([[0.0, 0, 0], [10.0, 0, 0]], device="mps")
    y = x.clone()
    ptr_y = torch.tensor([0, 1, 2], dtype=torch.long, device="mps")
    expected = [[0, 1], [0, 1]]
    assert torch.ops.pyg.knn(x, y, None, ptr_y, 1).cpu().tolist() == expected
    assert torch.ops.pyg.radius(x, y, None, ptr_y, 1.0, 2).cpu().tolist() == expected

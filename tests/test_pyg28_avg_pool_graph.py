"""Pinned PyG 2.8 graph coarsening parity on CPU and Apple MPS."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import pytest
import torch


def _pinned_reference_available() -> bool:
    try:
        return (
            version("torch-geometric") == "2.8.0"
            and version("pyg-lib").startswith("0.7.0")
        )
    except PackageNotFoundError:
        return False


requires_reference = pytest.mark.skipif(
    not _pinned_reference_available(),
    reason="requires torch-geometric 2.8.0 and pyg-lib 0.7.0",
)
requires_mps_reference = pytest.mark.skipif(
    not _pinned_reference_available() or not torch.backends.mps.is_available(),
    reason="requires MPS, torch-geometric 2.8.0 and pyg-lib 0.7.0",
)


def _graph(device: str, *, empty: bool = False):
    from torch_geometric.data import Data

    if empty:
        cluster = torch.empty(0, dtype=torch.long, device=device)
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        edge_attr = torch.empty((0, 2), dtype=torch.float32, device=device)
        x = torch.empty((0, 3), dtype=torch.float32, device=device)
        pos = torch.empty((0, 2), dtype=torch.float32, device=device)
        batch = torch.empty(0, dtype=torch.long, device=device)
    else:
        # Raw IDs are intentionally nonconsecutive. Batch 1 is empty, while
        # batches 0 and 2 have four and six nodes respectively.
        cluster = torch.tensor([10, 10, 20, 20, 30, 30, 40, 40, 40, 40], device=device)
        edge_index = torch.tensor([
            [0, 1, 0, 1, 1, 2, 3, 4, 5, 4, 5, 6, 7, 0, 6, 7],
            [1, 0, 2, 3, 2, 0, 0, 5, 4, 6, 7, 4, 4, 0, 7, 6],
        ], device=device)
        edge_attr = torch.arange(32, device=device, dtype=torch.float32).view(16, 2) / 7
        x = torch.arange(30, device=device, dtype=torch.float32).view(10, 3) / 11
        pos = torch.arange(20, device=device, dtype=torch.float32).view(10, 2) / 13
        batch = torch.tensor([0, 0, 0, 0, 2, 2, 2, 2, 2, 2], device=device)
    x = x.detach().requires_grad_(True)
    pos = pos.detach().requires_grad_(True)
    edge_attr = edge_attr.detach().requires_grad_(True)
    return cluster, Data(x=x, pos=pos, edge_index=edge_index,
                         edge_attr=edge_attr, batch=batch)


def _compare_pooled(actual, expected) -> None:
    assert actual.x.device.type == "mps"
    assert actual.edge_index.device.type == "mps"
    for name in ("edge_index", "batch"):
        assert torch.equal(getattr(actual, name).cpu(), getattr(expected, name))
    for name in ("x", "pos", "edge_attr"):
        torch.testing.assert_close(
            getattr(actual, name).detach().cpu(), getattr(expected, name).detach(),
            rtol=1e-5, atol=1e-6,
        )


def _loss(pooled):
    return (
        pooled.x.square().sum()
        + pooled.pos.square().sum()
        + pooled.edge_attr.square().sum()
    )


def _compare_input_gradients(cpu_data, mps_data) -> None:
    for name in ("x", "pos", "edge_attr"):
        torch.testing.assert_close(
            getattr(mps_data, name).grad.cpu(), getattr(cpu_data, name).grad,
            rtol=1e-5, atol=1e-6,
        )


@requires_reference
def test_cpu_graph_coarsening_contract() -> None:
    from torch_geometric.nn import avg_pool

    cluster, data = _graph("cpu")
    pooled = avg_pool(cluster, data)
    assert pooled.x.shape == (4, 3)
    assert pooled.pos.shape == (4, 2)
    assert pooled.edge_attr.shape == (4, 2)
    assert pooled.edge_index.tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]
    assert pooled.batch.tolist() == [0, 0, 2, 2]
    # Collapsed self-loops disappear. Three distinct input edges become 0->1;
    # their edge attributes are summed, then edges are sorted by source/target.
    torch.testing.assert_close(
        pooled.edge_attr,
        torch.tensor([[18, 21], [22, 24], [38, 40], [46, 48]], dtype=torch.float32) / 7,
        rtol=1e-6, atol=1e-6,
    )


@requires_mps_reference
def test_graph_coarsening_cpu_mps_forward_backward() -> None:
    from torch_geometric.nn import avg_pool

    cpu_cluster, cpu_data = _graph("cpu")
    mps_cluster, mps_data = _graph("mps")
    cpu = avg_pool(cpu_cluster, cpu_data)
    mps = avg_pool(mps_cluster, mps_data)
    _compare_pooled(mps, cpu)
    assert mps.edge_index.cpu().tolist() == [[0, 1, 2, 3], [1, 0, 3, 2]]

    _loss(cpu).backward()
    _loss(mps).backward()
    torch.mps.synchronize()
    _compare_input_gradients(cpu_data, mps_data)
    # Removed self-loops must not leak gradients to their edge attributes.
    assert torch.count_nonzero(mps_data.edge_attr.grad[[0, 1, 7, 8, 13, 14, 15]]) == 0


@requires_mps_reference
def test_empty_graph_coarsening_cpu_mps() -> None:
    from torch_geometric.nn import avg_pool

    cpu_cluster, cpu_data = _graph("cpu", empty=True)
    mps_cluster, mps_data = _graph("mps", empty=True)
    cpu = avg_pool(cpu_cluster, cpu_data)
    mps = avg_pool(mps_cluster, mps_data)
    _compare_pooled(mps, cpu)
    cpu_loss = _loss(cpu)
    mps_loss = _loss(mps)
    torch.testing.assert_close(mps_loss.cpu(), cpu_loss, rtol=1e-5, atol=1e-6)
    cpu_loss.backward()
    mps_loss.backward()
    torch.mps.synchronize()
    _compare_input_gradients(cpu_data, mps_data)
    assert mps.x.shape == (0, 3)
    assert mps.pos.shape == (0, 2)
    assert mps.edge_index.shape == (2, 0)
    assert mps.edge_attr.shape == (0, 2)
    assert mps.batch.shape == (0,)


@requires_mps_reference
def test_voxel_grid_to_graph_coarsening_cpu_mps() -> None:
    import pyg_lib  # noqa: F401 - installs the real pyg::grid_cluster schema
    from torch_geometric.nn import avg_pool, voxel_grid

    from mps_pointops.pyg import register_mps

    register_mps()
    _, cpu_data = _graph("cpu")
    _, mps_data = _graph("mps")
    cpu_cluster = voxel_grid(cpu_data.pos, size=1.0, batch=cpu_data.batch,
                             start=0.0, end=2.0)
    mps_cluster = voxel_grid(mps_data.pos, size=1.0, batch=mps_data.batch,
                             start=0.0, end=2.0)
    assert torch.equal(mps_cluster.cpu(), cpu_cluster)
    cpu = avg_pool(cpu_cluster, cpu_data)
    mps = avg_pool(mps_cluster, mps_data)
    _compare_pooled(mps, cpu)
    cpu_loss = _loss(cpu)
    mps_loss = _loss(mps)
    torch.testing.assert_close(mps_loss.cpu(), cpu_loss, rtol=1e-5, atol=1e-6)
    cpu_loss.backward()
    mps_loss.backward()
    torch.mps.synchronize()
    _compare_input_gradients(cpu_data, mps_data)

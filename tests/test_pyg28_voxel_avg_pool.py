"""PyG 2.8 voxel-grid feature pooling, against its pinned CPU implementation."""

from __future__ import annotations

from copy import deepcopy
from importlib.metadata import PackageNotFoundError, version

import pytest
import torch

from mps_pointops.pyg import register_mps


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


@requires_reference
def test_cpu_upstream_compact_and_fixed_size_contract() -> None:
    import pyg_lib  # noqa: F401 - loads the real pyg::grid_cluster schema
    from torch_geometric.nn import avg_pool_x, voxel_grid

    pos = torch.tensor([
        [0.0, 0.0], [0.2, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0],
    ])
    batch = torch.tensor([0, 0, 0, 2, 2])
    x = torch.arange(10, dtype=torch.float32).view(5, 2)
    cluster = voxel_grid(pos, size=1.0, batch=batch, start=0.0, end=2.0)

    assert cluster.tolist() == [0, 0, 4, 18, 19]
    compact, compact_batch = avg_pool_x(cluster, x, batch)
    assert compact.tolist() == [[1.0, 2.0], [4.0, 5.0], [6.0, 7.0], [8.0, 9.0]]
    assert compact_batch.tolist() == [0, 0, 2, 2]

    fixed, fixed_batch = avg_pool_x(cluster, x, batch, batch_size=3, size=9)
    assert fixed.shape == (27, 2)
    assert fixed_batch is None
    assert torch.equal(fixed[[0, 4, 18, 19]], compact)
    assert torch.count_nonzero(fixed[[1, 9, 26]]) == 0


@requires_mps_reference
@pytest.mark.parametrize("spatial_dimensions", [1, 2, 3])
@pytest.mark.parametrize("fixed_size", [False, True])
def test_voxel_avg_pool_model_cpu_mps_forward_backward(
    spatial_dimensions: int,
    fixed_size: bool,
) -> None:
    import pyg_lib  # noqa: F401 - loads the real pyg::grid_cluster schema
    from torch_geometric.nn import avg_pool_x, voxel_grid

    register_mps()
    pos = torch.tensor([
        [-0.25] * spatial_dimensions,
        [0.0] * spatial_dimensions,
        [0.5] * spatial_dimensions,
        [1.0] * spatial_dimensions,
        [0.0] * spatial_dimensions,
        [1.0] * spatial_dimensions,
    ], dtype=torch.float32)
    if spatial_dimensions == 1:
        pos = pos[:, 0]  # PyG 2.8 also accepts a flat 1D position vector.
    batch = torch.tensor([0, 0, 0, 0, 2, 2], dtype=torch.long)
    features = torch.arange(24, dtype=torch.float32).view(6, 4) / 7.0
    cpu_x = features.detach().clone().requires_grad_(True)
    mps_x = features.to("mps").detach().requires_grad_(True)
    cpu_model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        cpu_model.weight.copy_(torch.arange(12).view(3, 4) / 19.0)
        cpu_model.bias.copy_(torch.tensor([-0.25, 0.5, 0.75]))
    mps_model = deepcopy(cpu_model).to("mps")
    torch.testing.assert_close(mps_model.weight.cpu(), cpu_model.weight, rtol=0, atol=0)
    torch.testing.assert_close(mps_model.bias.cpu(), cpu_model.bias, rtol=0, atol=0)

    cpu_cluster = voxel_grid(pos, size=1.0, batch=batch, start=0.0, end=2.0)
    mps_cluster = voxel_grid(
        pos.to("mps"), size=1.0, batch=batch.to("mps"), start=0.0, end=2.0,
    )
    kwargs = {"batch_size": 3, "size": 3**spatial_dimensions} if fixed_size else {}
    cpu_pooled, cpu_batch = avg_pool_x(cpu_cluster, cpu_x, batch, **kwargs)
    mps_pooled, mps_batch = avg_pool_x(
        mps_cluster, mps_x, batch.to("mps"), **kwargs,
    )
    cpu_output = cpu_model(cpu_pooled)
    mps_output = mps_model(mps_pooled)
    torch.testing.assert_close(mps_pooled.cpu(), cpu_pooled, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mps_output.cpu(), cpu_output, rtol=1e-5, atol=1e-6)
    cpu_loss = cpu_output.square().mean()
    mps_loss = mps_output.square().mean()
    cpu_loss.backward()
    mps_loss.backward()
    torch.mps.synchronize()

    assert mps_cluster.device.type == "mps" and mps_pooled.device.type == "mps"
    assert mps_pooled.dtype == torch.float32
    assert torch.equal(mps_cluster.cpu(), cpu_cluster)
    if fixed_size:
        assert cpu_batch is mps_batch is None
    else:
        assert torch.equal(mps_batch.cpu(), cpu_batch)
    torch.testing.assert_close(mps_pooled.cpu(), cpu_pooled, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mps_loss.cpu(), cpu_loss, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(mps_x.grad.cpu(), cpu_x.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        mps_model.weight.grad.cpu(), cpu_model.weight.grad, rtol=1e-5, atol=1e-6,
    )
    torch.testing.assert_close(
        mps_model.bias.grad.cpu(), cpu_model.bias.grad, rtol=1e-5, atol=1e-6,
    )


@requires_mps_reference
def test_hot_voxel_mean_feature_gradient() -> None:
    import pyg_lib  # noqa: F401 - loads the real pyg::grid_cluster schema
    from torch_geometric.nn import avg_pool_x, voxel_grid

    register_mps()
    count = 256
    pos = torch.zeros((count, 3), dtype=torch.float32)
    batch = torch.zeros(count, dtype=torch.long)
    features = torch.arange(count * 2, dtype=torch.float32).view(count, 2) / 13.0
    cpu_x = features.detach().clone().requires_grad_(True)
    mps_x = features.to("mps").detach().requires_grad_(True)

    cpu_cluster = voxel_grid(pos, size=1.0, batch=batch)
    mps_cluster = voxel_grid(pos.to("mps"), size=1.0, batch=batch.to("mps"))
    cpu_pooled, cpu_batch = avg_pool_x(cpu_cluster, cpu_x, batch)
    mps_pooled, mps_batch = avg_pool_x(
        mps_cluster, mps_x, batch.to("mps"),
    )
    cpu_pooled.sum().backward()
    mps_pooled.sum().backward()
    torch.mps.synchronize()

    assert torch.equal(mps_cluster.cpu(), cpu_cluster)
    assert torch.equal(mps_batch.cpu(), cpu_batch)
    torch.testing.assert_close(mps_pooled.cpu(), cpu_pooled, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(mps_x.grad.cpu(), cpu_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        mps_x.grad.cpu(), torch.full_like(cpu_x, 1.0 / count), rtol=0, atol=0,
    )

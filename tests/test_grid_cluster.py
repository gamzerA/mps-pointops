"""Grid-cluster contract checks; the upstream comparison is optional."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from mps_pointops.grid import grid_cluster


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    return torch.device(request.param)


def _vectors(device):
    size = torch.tensor([1.0, 2.0, 0.5], device=device)
    start = torch.tensor([0.0, -2.0, 1.0], device=device)
    end = torch.tensor([3.0, 4.0, 2.0], device=device)
    return size, start, end


def test_explicit_bounds_negative_coordinates_and_cell_boundaries(device):
    size, start, end = _vectors(device)
    points = torch.tensor(
        [
            [0.0, -2.0, 1.0],
            [1.0, 0.0, 1.5],
            [-0.5, -2.0, 1.0],
            [-1.0, -2.0, 1.0],
            [3.0, 4.0, 2.0],
        ],
        device=device,
    )
    # Extents are (4, 4, 3); int conversion truncates -0.5 toward zero.
    expected = torch.tensor([0, 21, 0, -1, 47], dtype=torch.long)
    result = grid_cluster(points, size, start, end)
    assert result.device.type == device.type and result.dtype == torch.long
    assert torch.equal(result.cpu(), expected)


def test_implicit_bounds_and_repeated_voxels(device):
    points = torch.tensor(
        [[0.0, 0.0, 0.0], [11.0, 9.0, 0.0], [2.0, 8.0, 0.0],
         [2.0, 2.0, 0.0], [8.0, 3.0, 0.0]],
        device=device,
    )
    result = grid_cluster(points, torch.tensor([5.0, 5.0, 1.0], device=device))
    assert torch.equal(result.cpu(), torch.tensor([0, 5, 3, 0, 1]))


def test_empty_is_rejected_like_upstream(device):
    empty = torch.empty((0, 3), device=device)
    with pytest.raises(ValueError, match="at least one point"):
        grid_cluster(empty, torch.ones(3, device=device))
    with pytest.raises(ValueError, match="at least one point"):
        grid_cluster(
            empty, torch.ones(3, device=device),
            torch.zeros(3, device=device), torch.ones(3, device=device),
        )


def test_shape_dtype_and_device_are_checked():
    points = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="shape"):
        grid_cluster(torch.zeros((2, 2)), torch.ones(3))
    with pytest.raises(TypeError, match="float32"):
        grid_cluster(points.double(), torch.ones(3))
    with pytest.raises(TypeError, match="torch.Tensor"):
        grid_cluster(points, 1.0)
    with pytest.raises(ValueError, match="shape"):
        grid_cluster(points, torch.ones(2))
    with pytest.raises(TypeError, match="batch"):
        grid_cluster(points, torch.ones(3), batch=torch.zeros(2))


def test_upstream_163_differential_when_installed(device):
    spec = importlib.util.find_spec("torch_cluster")
    if spec is None or spec.origin is None:
        pytest.skip("real torch_cluster is not installed")
    import torch_cluster

    if getattr(torch_cluster, "__version__", "") != "1.6.3":
        pytest.skip("requires real torch_cluster 1.6.3")

    generator = torch.Generator().manual_seed(416)
    size = torch.tensor([0.5, 1.0, 2.0])
    start = torch.tensor([-4.0, -8.0, -2.0])
    end = torch.tensor([8.0, 4.0, 12.0])
    for n in (1, 7, 103):
        points = torch.rand((n, 3), generator=generator) * 9.0 - 3.0
        for low, high in ((None, None), (start, end)):
            got = grid_cluster(
                points.to(device), size.to(device),
                low.to(device) if low is not None else None,
                high.to(device) if high is not None else None,
            )
            expected = torch_cluster.grid_cluster(points, size, low, high)
            assert torch.equal(got.cpu(), expected)

    if device.type == "cpu":
        # 100 deterministic seeds and 5,050 points exercise mixed-radix
        # strides across implicit and explicit bounds against the real op.
        for seed in range(100):
            points = torch.randn(
                ((seed % 199) + 1, 3), generator=torch.Generator().manual_seed(seed)
            ) * 3.0
            low = start if seed % 3 != 1 else None
            high = end if seed % 3 == 0 else None
            assert torch.equal(
                grid_cluster(points, size, low, high),
                torch_cluster.grid_cluster(points, size, low, high),
            )

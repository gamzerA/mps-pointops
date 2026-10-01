"""Compact voxel API: exact integer maps and differentiable aggregation."""

from __future__ import annotations

import pytest
import torch

from mps_pointops.pyg_grid import grid_cluster_ids
from mps_pointops.voxel import voxel_downsample, voxelize


@pytest.fixture(params=["cpu", "mps"])
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    return request.param


def _cpu(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "mps":
        torch.mps.synchronize()
    return tensor.cpu()


def test_uneven_unsorted_batches_negative_cells_and_bidirectional_map(device: str) -> None:
    pos = torch.tensor([
        [0.0, 0.0], [-1.0, 0.0], [0.0, 0.0],
        [1.0, 1.0], [-1.0, 0.0], [0.5, -0.5],
    ], device=device)
    batch = torch.tensor([2, 0, 0, 2, 0, 2], device=device)
    result = voxelize(pos, 1.0, batch)

    assert result.voxel_coords.dtype == result.inverse.dtype == torch.int64
    assert all(t.device.type == device for t in (
        result.voxel_coords, result.batch, result.inverse,
        result.counts, result.point_order, result.ptr,
    ))
    assert _cpu(result.voxel_coords).tolist() == [
        [-1, 0], [0, 0], [0, -1], [0, 0], [1, 1],
    ]
    assert _cpu(result.batch).tolist() == [0, 0, 2, 2, 2]
    assert _cpu(result.inverse).tolist() == [3, 0, 1, 4, 0, 2]
    assert _cpu(result.counts).tolist() == [2, 1, 1, 1, 1]
    assert _cpu(result.point_order).tolist() == [1, 4, 2, 5, 0, 3]
    assert _cpu(result.ptr).tolist() == [0, 2, 3, 4, 5, 6]


@pytest.mark.parametrize("dimensions", [1, 2, 3])
def test_exact_cell_boundaries_and_half_open_end(device: str, dimensions: int) -> None:
    pos = torch.tensor([
        [-2.0] * dimensions, [-1.5] * dimensions,
        [0.0] * dimensions, [1.5] * dimensions,
    ], device=device)
    voxels = voxelize(pos, 0.5, start=-2.0, end=2.0)
    assert _cpu(voxels.voxel_coords).tolist() == [
        [0] * dimensions, [1] * dimensions,
        [4] * dimensions, [7] * dimensions,
    ]
    assert _cpu(voxels.inverse).tolist() == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="half-open"):
        voxelize(torch.full((1, dimensions), 2.0, device=device), 0.5,
                 start=-2.0, end=2.0)
    with pytest.raises(ValueError, match="half-open"):
        voxelize(torch.full((1, dimensions), -2.5, device=device), 0.5,
                 start=-2.0, end=2.0)


def test_empty_input_and_empty_feature_output(device: str) -> None:
    pos = torch.empty((0, 3), device=device, requires_grad=True)
    features = torch.empty((0, 2), device=device, requires_grad=True)
    result = voxel_downsample(pos, [1.0, 2.0, 3.0],
                              torch.empty((0,), dtype=torch.long, device=device),
                              features)
    assert result.voxels.voxel_coords.shape == (0, 3)
    assert result.voxels.batch.shape == result.voxels.inverse.shape == (0,)
    assert result.voxels.point_order.shape == result.voxels.counts.shape == (0,)
    assert _cpu(result.voxels.ptr).tolist() == [0]
    assert result.pos.shape == (0, 3)
    assert result.features is not None and result.features.shape == (0, 2)
    assert result.pos.requires_grad and result.features.requires_grad


@pytest.mark.parametrize("feature_reduce", ["mean", "sum"])
def test_position_feature_aggregation_and_first_order_gradients(
    device: str, feature_reduce: str,
) -> None:
    # Dyadic values and counts of two make the expected map and gradients exact.
    pos = torch.tensor([[0.0], [0.5], [2.0], [2.5]], device=device,
                       requires_grad=True)
    features = torch.tensor([[2.0, 4.0], [6.0, 8.0], [10.0, 12.0],
                             [14.0, 16.0]], device=device, requires_grad=True)
    result = voxel_downsample(pos, 1.0, features=features,
                              feature_reduce=feature_reduce)
    assert result.pos.device.type == device
    assert result.features is not None and result.features.device.type == device
    assert _cpu(result.pos).tolist() == [[0.25], [2.25]]
    expected_features = ([[4.0, 6.0], [12.0, 14.0]] if feature_reduce == "mean"
                         else [[8.0, 12.0], [24.0, 28.0]])
    assert _cpu(result.features).tolist() == expected_features
    (result.pos * torch.tensor([[2.0], [4.0]], device=device)).sum().backward(
        retain_graph=True,
    )
    assert _cpu(pos.grad).tolist() == [[1.0], [1.0], [2.0], [2.0]]
    (result.features * torch.tensor([[2.0, 4.0], [6.0, 8.0]], device=device)).sum().backward()
    multiplier = 0.5 if feature_reduce == "mean" else 1.0
    assert _cpu(features.grad).tolist() == [
        [2.0 * multiplier, 4.0 * multiplier],
        [2.0 * multiplier, 4.0 * multiplier],
        [6.0 * multiplier, 8.0 * multiplier],
        [6.0 * multiplier, 8.0 * multiplier],
    ]


def test_cpu_mps_randomized_integer_parity_and_aggregation() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    gen = torch.Generator().manual_seed(1047)
    for dimensions in (1, 2, 3):
        # Half-integer coordinates with size 1 avoid ambiguous division bands.
        pos = torch.randint(-12, 13, (257, dimensions), generator=gen).float() / 2
        batch = torch.randint(0, 5, (257,), generator=gen)
        features = torch.randint(-8, 9, (257, 4), generator=gen).float() / 4
        expected = voxel_downsample(pos, 1.0, batch, features)
        actual = voxel_downsample(pos.to("mps"), 1.0, batch.to("mps"),
                                  features.to("mps"))
        for name in ("voxel_coords", "batch", "inverse", "counts", "point_order", "ptr"):
            assert torch.equal(_cpu(getattr(actual.voxels, name)),
                               getattr(expected.voxels, name)), name
        torch.testing.assert_close(_cpu(actual.pos), expected.pos, rtol=1e-5, atol=1e-6)
        assert actual.features is not None and expected.features is not None
        torch.testing.assert_close(_cpu(actual.features), expected.features,
                                   rtol=1e-5, atol=1e-6)


def test_batch_isolation_and_pyg_raw_id_contract_are_distinct() -> None:
    pos = torch.tensor([[-0.5], [0.5], [0.5]], dtype=torch.float32)
    batch = torch.tensor([0, 0, 7])
    ours = voxelize(pos, 1.0, batch)
    assert ours.voxel_coords[:, 0].tolist() == [-1, 0, 0]
    assert ours.batch.tolist() == [0, 0, 7]
    assert ours.inverse.tolist() == [0, 1, 2]
    # The existing PyG 2.8 operator computes truncation-toward-zero raw IDs,
    # and this new compact, floor-based API does not claim to match them.
    pyg_like = grid_cluster_ids(
        torch.cat((pos, batch.float().unsqueeze(1)), dim=1),
        torch.ones(2), torch.zeros(2), torch.tensor([1.0, 7.0]),
    )
    assert pyg_like[0].item() == pyg_like[1].item()
    assert ours.inverse[0].item() != ours.inverse[1].item()


def test_bad_inputs_rejected_without_silent_conversion(device: str) -> None:
    pos = torch.tensor([[0.0], [1.0]], device=device)
    with pytest.raises(TypeError, match="float32"):
        voxelize(pos.half(), 1.0)
    with pytest.raises(ValueError, match="size must be positive"):
        voxelize(pos, 0.0)
    with pytest.raises(ValueError, match="finite"):
        voxelize(torch.tensor([[float("nan")]], device=device), 1.0)
    with pytest.raises(ValueError, match="nonnegative"):
        voxelize(pos, 1.0, torch.tensor([0, -1], device=device))
    with pytest.raises(ValueError, match="int64"):
        voxelize(pos, 1.0, torch.tensor([0, 1], device=device, dtype=torch.int32))
    with pytest.raises(ValueError, match="end must exceed"):
        voxelize(pos, 1.0, end=0.0)
    with pytest.raises(ValueError, match="feature_reduce"):
        voxel_downsample(pos, 1.0, feature_reduce="max")
    with pytest.raises(ValueError, match="features must be float32"):
        voxel_downsample(pos, 1.0, features=pos.half())

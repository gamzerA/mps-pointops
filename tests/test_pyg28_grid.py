"""Pinned PyG 2.8 / pyg-lib 0.7 grid-clustering parity checks."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import pytest
import torch

from mps_pointops import pyg
from mps_pointops.pyg_grid import grid_cluster_ids


def _pinned_reference_available() -> bool:
    try:
        return (
            version("torch-geometric").startswith("2.8.")
            and version("pyg-lib").startswith("0.7.0")
        )
    except PackageNotFoundError:
        return False


requires_reference = pytest.mark.skipif(
    not _pinned_reference_available(),
    reason="requires torch-geometric 2.8 and pyg-lib 0.7 reference",
)
requires_mps_reference = pytest.mark.skipif(
    not _pinned_reference_available() or not torch.backends.mps.is_available(),
    reason="requires MPS, torch-geometric 2.8 and pyg-lib 0.7",
)


def test_mixed_radix_ids_truncate_negative_coordinates() -> None:
    points = torch.tensor([
        [-0.25, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.75, 0.0, 1.0],
    ])
    size = torch.ones(3)
    start = torch.zeros(3)
    end = torch.tensor([2.0, 2.0, 1.0])
    ids = grid_cluster_ids(points, size, start, end)
    assert ids.dtype == torch.long
    assert ids.tolist() == [0, 0, 4, 9, 10]


def test_grid_contract_rejects_unsupported_shapes_and_dtype() -> None:
    points = torch.zeros((2, 3))
    with pytest.raises(ValueError, match="D in"):
        grid_cluster_ids(torch.zeros((2, 5)), torch.ones(5))
    with pytest.raises(TypeError, match="float32"):
        grid_cluster_ids(points.double(), torch.ones(3))
    with pytest.raises(ValueError, match="shape"):
        grid_cluster_ids(points, torch.ones(2))
    with pytest.raises(ValueError, match="at least one"):
        grid_cluster_ids(torch.empty((0, 3)), torch.ones(3))


@requires_reference
def test_direct_operator_matches_pinned_cpu_reference() -> None:
    import pyg_lib

    for dimensions in (2, 3, 4):
        size = torch.arange(1, dimensions + 1, dtype=torch.float32) * 0.5
        start = torch.arange(dimensions, dtype=torch.float32) - 2.0
        end = start + 7.0
        for seed in range(32):
            generator = torch.Generator().manual_seed(seed + 1000 * dimensions)
            points = torch.rand((23, dimensions), generator=generator) * 8.0 - 2.0
            for low, high in ((None, None), (start, end)):
                actual = grid_cluster_ids(points, size, low, high)
                expected = pyg_lib.ops.grid_cluster(points, size, low, high)
                assert torch.equal(actual, expected), (dimensions, seed, low is not None)


@requires_reference
def test_pyg_wrapper_cpu_batch_and_scalar_parameters() -> None:
    from torch_geometric.nn.pool import voxel_grid

    points = torch.tensor([
        [-0.25, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.75, 0.0],
    ])
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    actual = voxel_grid(points, size=1.0, batch=batch, start=0.0, end=2.0)
    assert actual.tolist() == [0, 0, 4, 9, 10]


@requires_reference
def test_pinned_cpu_exact_cell_boundaries_and_adjacent_ulps() -> None:
    import pyg_lib

    one = torch.tensor(1.0)
    zero = torch.tensor(0.0)
    below_one = torch.nextafter(one, zero)
    above_one = torch.nextafter(one, torch.tensor(2.0))
    above_negative_one = torch.nextafter(-one, zero)
    points = torch.stack((
        torch.stack((below_one, zero)),
        torch.stack((one, zero)),
        torch.stack((above_one, zero)),
        torch.stack((-one, zero)),
        torch.stack((above_negative_one, zero)),
    ))
    size = torch.ones(2)
    start = torch.zeros(2)
    end = torch.tensor([2.0, 0.0])
    expected = pyg_lib.ops.grid_cluster(points, size, start, end)
    assert expected.tolist() == [0, 1, 1, -1, 0]
    assert torch.equal(grid_cluster_ids(points, size, start, end), expected)


@requires_mps_reference
def test_direct_schema_mps_matches_pinned_cpu() -> None:
    import pyg_lib  # noqa: F401 - loads the real pyg::grid_cluster schema

    pyg.register_mps()
    assert str(torch.ops.pyg.grid_cluster.default._schema) == (
        "pyg::grid_cluster(Tensor pos, Tensor size, Tensor? start=None, "
        "Tensor? end=None) -> Tensor"
    )
    for dimensions in (2, 3, 4):
        size = torch.arange(1, dimensions + 1, dtype=torch.float32) * 0.5
        start = torch.arange(dimensions, dtype=torch.float32) - 2.0
        end = start + 7.0
        points = torch.tensor([
            [0.0] * dimensions,
            [1.0] * dimensions,
            [-0.25] * dimensions,
            [2.0] * dimensions,
        ], dtype=torch.float32)
        for low, high in ((None, None), (start, end)):
            expected = torch.ops.pyg.grid_cluster(points, size, low, high)
            actual = torch.ops.pyg.grid_cluster(
                points.to("mps"), size.to("mps"),
                None if low is None else low.to("mps"),
                None if high is None else high.to("mps"),
            )
            assert actual.device.type == "mps" and actual.dtype == torch.long
            assert torch.equal(actual.cpu(), expected)


@requires_mps_reference
def test_voxel_grid_mps_matches_pyg_28_cpu_for_batches() -> None:
    from torch_geometric.nn.pool import voxel_grid

    pyg.register_mps()
    points = torch.tensor([
        [-0.25, 0.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
        [1.75, 0.0],
    ])
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    for parameters in (
        {"size": 1.0, "batch": batch, "start": 0.0, "end": 2.0},
        {"size": [1.0, 0.5], "batch": batch},
        {
            "size": torch.tensor([1.0, 0.5]),
            "batch": batch,
            "start": torch.tensor([0.0, 0.0]),
            "end": torch.tensor([2.0, 2.0]),
        },
        {"size": 1.0, "batch": None},
    ):
        expected = voxel_grid(points, **parameters)
        mps_parameters = {
            key: value.to("mps") if isinstance(value, torch.Tensor) else value
            for key, value in parameters.items()
        }
        actual = voxel_grid(points.to("mps"), **mps_parameters)
        assert actual.device.type == "mps" and actual.dtype == torch.long
        assert torch.equal(actual.cpu(), expected)


@requires_mps_reference
@pytest.mark.parametrize("spatial_dimensions", [1, 3])
def test_voxel_grid_mps_spatial_dimensions_and_missing_batch(
    spatial_dimensions: int,
) -> None:
    from torch_geometric.nn.pool import voxel_grid

    pyg.register_mps()
    points = torch.tensor([
        [0.0] * spatial_dimensions,
        [0.5] * spatial_dimensions,
        [1.0] * spatial_dimensions,
        [0.0] * spatial_dimensions,
    ])
    if spatial_dimensions == 1:
        points = points[:, 0]  # PyG also accepts a one-dimensional vector.
    batch = torch.tensor([0, 0, 2, 2], dtype=torch.long)
    for bound in (None, 0.0):
        expected = voxel_grid(points, size=0.5, batch=batch, start=bound)
        actual = voxel_grid(
            points.to("mps"), size=0.5, batch=batch.to("mps"), start=bound,
        )
        assert torch.equal(actual.cpu(), expected)

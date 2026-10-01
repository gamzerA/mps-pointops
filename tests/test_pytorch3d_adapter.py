"""Functional checks for the optional PyTorch3D-style Ball Query call surface."""

from __future__ import annotations

import pytest
import torch

from mps_pointops.pytorch3d import KNN, ball_query


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_defaults_named_fields_and_zero_padded_neighbors(device: str) -> None:
    queries = torch.zeros((1, 1, 3), device=device)
    points = torch.tensor([[[0.125, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device)
    result = ball_query(queries, points)

    assert isinstance(result, KNN)
    assert result._fields == ("dists", "idx", "knn")
    assert result.dists.shape == result.idx.shape == (1, 1, 500)
    assert result.knn is not None and result.knn.shape == (1, 1, 500, 3)
    assert result.dists.dtype == torch.float32
    assert result.idx.dtype == torch.int64
    assert result.knn.dtype == torch.float32
    assert result.idx[0, 0, 0].item() == 0
    assert torch.all(result.idx[0, 0, 1:] == -1)
    assert result.dists[0, 0, 0].item() == 0.015625
    assert torch.all(result.dists[0, 0, 1:] == 0)
    torch.testing.assert_close(result.knn[0, 0, 0].cpu(), points[0, 0].cpu(), rtol=0, atol=0)
    assert torch.all(result.knn[0, 0, 1:] == 0)


@pytest.mark.parametrize("device", DEVICES)
def test_ragged_lengths_first_k_return_nn_and_cube_hint(device: str) -> None:
    queries = torch.tensor(
        [[[0.0, 0, 0], [0.0, 0, 0]], [[3.0, 0, 0], [0.0, 0, 0]]],
        device=device,
    )
    points = torch.tensor(
        [
            [[0.5, 0, 0], [0.25, 0, 0], [0.125, 0, 0], [2.0, 0, 0]],
            [[0.125, 0, 0], [0.25, 0, 0], [2.0, 0, 0], [3.0, 0, 0]],
        ],
        device=device,
    )
    lengths1 = torch.tensor([1, 2], device=device, dtype=torch.int64)
    lengths2 = torch.tensor([2, 1], device=device, dtype=torch.int64)

    default = ball_query(queries, points, lengths1, lengths2, 3, 1.0, True, False)
    cube = ball_query(queries, points, lengths1, lengths2, 3, 1.0, True, True)
    no_nn = ball_query(queries, points, lengths1, lengths2, 3, 1.0, False, True)

    expected_idx = torch.tensor(
        [[[0, 1, -1], [-1, -1, -1]], [[-1, -1, -1], [0, -1, -1]]]
    )
    expected_dists = torch.tensor(
        [[[0.25, 0.0625, 0], [0, 0, 0]], [[0, 0, 0], [0.015625, 0, 0]]]
    )
    torch.testing.assert_close(default.idx.cpu(), expected_idx, rtol=0, atol=0)
    torch.testing.assert_close(default.dists.cpu(), expected_dists, rtol=0, atol=0)
    torch.testing.assert_close(cube.idx.cpu(), default.idx.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(cube.dists.cpu(), default.dists.cpu(), rtol=0, atol=0)
    assert no_nn.knn is None
    assert default.knn is not None
    assert default.knn[0, 0, 0, 0].item() == 0.5
    assert default.knn[0, 0, 1, 0].item() == 0.25
    assert torch.all(default.knn[default.idx == -1] == 0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "batch,query_count,point_count,k",
    [(1, 2, 0, 3), (1, 2, 2, 0), (1, 0, 2, 3), (0, 2, 2, 3)],
)
def test_empty_dimensions(
    device: str, batch: int, query_count: int, point_count: int, k: int
) -> None:
    queries = torch.zeros((batch, query_count, 3), device=device)
    points = torch.zeros((batch, point_count, 3), device=device)
    result = ball_query(queries, points, K=k, radius=1.0)
    assert result.dists.shape == result.idx.shape == (batch, query_count, k)
    assert result.knn is not None and result.knn.shape == (batch, query_count, k, 3)
    assert torch.all(result.idx == -1)
    assert torch.all(result.dists == 0)
    assert torch.all(result.knn == 0)


@pytest.mark.parametrize("device", DEVICES)
def test_coordinate_gradients_include_returned_neighbors(device: str) -> None:
    queries = torch.zeros((1, 1, 3), device=device, requires_grad=True)
    points = torch.tensor([[[0.25, 0.0, 0.0]]], device=device, requires_grad=True)
    result = ball_query(queries, points, K=2, radius=1.0)
    assert result.knn is not None
    (result.dists.sum() + result.knn.sum()).backward()

    torch.testing.assert_close(
        queries.grad.cpu(), torch.tensor([[[-0.5, 0.0, 0.0]]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        points.grad.cpu(), torch.tensor([[[1.5, 1.0, 1.0]]]), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("point_count,k", [(0, 2), (2, 0)])
def test_empty_neighbor_gather_keeps_zero_gradient(
    device: str, point_count: int, k: int
) -> None:
    queries = torch.zeros((1, 1, 3), device=device)
    points = torch.zeros((1, point_count, 3), device=device, requires_grad=True)
    result = ball_query(queries, points, K=k, radius=1.0)
    assert result.knn is not None
    result.knn.sum().backward()
    assert points.grad is not None
    assert torch.all(points.grad == 0)


@pytest.mark.parametrize("device", DEVICES)
def test_rejects_unsupported_dtype_and_invalid_lengths(device: str) -> None:
    points = torch.zeros((1, 2, 3), device=device)
    with pytest.raises(TypeError, match="float32"):
        ball_query(points.half(), points.half(), K=1, radius=1.0)
    with pytest.raises(ValueError, match="lengths1|query_lengths"):
        ball_query(
            points,
            points,
            lengths1=torch.tensor([3], device=device, dtype=torch.int64),
            K=1,
            radius=1.0,
        )
    with pytest.raises(TypeError, match="bool"):
        ball_query(points, points, K=1, radius=1.0, skip_points_outside_cube=1)

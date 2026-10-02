"""Experimental Morton-brick kNN correctness, not a release performance gate."""

import sys
import types
from pathlib import Path

import pytest
import torch

if "bench" not in sys.modules:
    package = types.ModuleType("bench")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "bench")]
    sys.modules["bench"] = package

from bench.spatial_bricks import MortonBrickIndex


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("sorted_input", [False, True])
def test_brick_knn_matches_native_flat_indices_and_squared_distances(sorted_input):
    from mps_pointops._flat_search_mps import knn_indices

    generator = torch.Generator().manual_seed(20261002)
    points = torch.randint(0, 128, (2049, 3), generator=generator).float() / 8
    if sorted_input:
        points = points[torch.argsort(points[:, 0], stable=True)]
    queries = points[::37][:40].clone()
    points = points.to("mps")
    queries = queries.to("mps")
    index = MortonBrickIndex.build(points, torch.zeros(3, device="mps"), 1.0)
    ptr_x = torch.tensor([0, len(points)], device="mps")
    ptr_y = torch.tensor([0, len(queries)], device="mps")
    for k in (1, 8, 16, 32):
        distances, actual = index.knn(queries, k)
        expected = knn_indices(points, queries, ptr_x, ptr_y, k)
        torch.mps.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        selected = points[actual]
        # This fixture uses eighth-unit coordinates, so all intermediate
        # differences, squares, and sums are exactly representable in f32.
        d = selected - queries[:, None, :]
        expected_sq = d[..., 0] * d[..., 0]
        expected_sq = expected_sq + d[..., 1] * d[..., 1]
        expected_sq = expected_sq + d[..., 2] * d[..., 2]
        torch.testing.assert_close(distances.cpu(), expected_sq.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_knn_repeated_ties_across_bricks_choose_lowest_original_indices():
    # Stable Morton sort leaves all duplicate points in input order, crossing
    # four 128-point bricks; each independent dispatch must choose IDs 0..15.
    points = torch.zeros((513, 3), dtype=torch.float32, device="mps")
    queries = torch.zeros((17, 3), dtype=torch.float32, device="mps")
    index = MortonBrickIndex.build(points, torch.zeros(3, device="mps"), 1.0)
    for _ in range(32):
        distances, actual = index.knn(queries, 16)
        torch.mps.synchronize()
        assert actual.cpu().tolist() == [list(range(16))] * len(queries)
        assert torch.count_nonzero(distances).item() == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_knn_empty_padding_and_invalid_query():
    empty = torch.empty((0, 3), dtype=torch.float32, device="mps")
    origin = torch.zeros(3, device="mps")
    index = MortonBrickIndex.build(empty, origin, 1.0)
    query = torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]], device="mps")
    distances, indices = index.knn(query, 4)
    torch.mps.synchronize()
    assert indices.cpu().tolist() == [[-1] * 4, [-1] * 4]
    assert torch.isinf(distances).all().item()
    assert index.knn(query, 0)[0].shape == (2, 0)
    with pytest.raises(ValueError, match=r"\[0, 32\]"):
        index.knn(query, 33)

"""Differential tests for the experimental sorted-Morton radius path."""

import sys
import types
from pathlib import Path

import pytest
import torch

if "bench" not in sys.modules:
    package = types.ModuleType("bench")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "bench")]
    sys.modules["bench"] = package

from bench.spatial_radius import SortedMortonIndex


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("sorted_input", [False, True])
def test_sorted_morton_radius_matches_flat_first_k(sorted_input):
    from mps_pointops._flat_search_mps import radius_indices
    generator = torch.Generator().manual_seed(202609)
    points = torch.randint(0, 64, (1000, 3), generator=generator).float() / 8
    if sorted_input:
        points = points[torch.argsort(points[:, 0], stable=True)]
    queries = points[::7][:120].clone()
    points_mps = points.to("mps")
    queries_mps = queries.to("mps")
    origin = torch.zeros(3, device="mps")
    ptr_x = torch.tensor([0, len(points)], device="mps")
    ptr_y = torch.tensor([0, len(queries)], device="mps")
    index = SortedMortonIndex.build(points_mps, origin, 1.0)
    actual, status = index.radius(queries_mps, 0.75, 16)
    expected = radius_indices(points_mps, queries_mps, ptr_x, ptr_y, 0.75, 16)
    torch.mps.synchronize()
    assert not status.any().item()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_sorted_morton_radius_reports_unhandled_domain_and_large_ball():
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="mps")
    index = SortedMortonIndex.build(points, torch.zeros(3, device="mps"), 1.0)
    queries = torch.tensor([[-4.0, 0.0, 0.0], [0.0, 0.0, 0.0]], device="mps")
    _, status = index.radius(queries, 1.0, 2)
    _, wide_status = index.radius(queries[1:], 100.0, 2)
    torch.mps.synchronize()
    assert status.cpu().tolist() == [1, 0]
    assert wide_status.cpu().tolist() == [1]

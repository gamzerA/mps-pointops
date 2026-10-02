"""Contract checks for the reusable single-cloud spatial index."""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

import pytest
import torch

from mps_pointops import SpatialIndex, ball_query, knn


def test_spatial_shaders_are_packaged() -> None:
    kernels = resources.files("mps_pointops").joinpath("kernels")
    assert kernels.joinpath("spatial_keys.metal").is_file()
    assert kernels.joinpath("spatial_bvh.metal").is_file()
    # The benchmark and installed wheel must compile the same research kernel.
    for name in ("spatial_bvh.metal", "spatial_keys.metal"):
        research = Path(__file__).resolve().parents[1] / "bench" / name
        assert research.read_bytes() == kernels.joinpath(name).read_bytes()


def test_cpu_auto_preserves_existing_scan_contract() -> None:
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]])
    query = torch.tensor([[0.2, 0, 0], [2.8, 0, 0]])
    index = SpatialIndex(points)
    d, i = index.knn(query, 2)
    expected_d, expected_i = knn(query[None], points[None], 2)
    assert torch.equal(i, expected_i[0])
    assert torch.equal(d, expected_d[0])
    d2, i2 = index.ball_query(query, 1.1, 2)
    expected_d2, expected_i2 = ball_query(query[None], points[None], 1.1, 2)
    assert torch.equal(i2, expected_i2[0])
    assert torch.equal(d2, expected_d2[0])
    with pytest.raises(ValueError, match="MPS float32"):
        index.knn(query, 2, backend="bvh")


def test_cpu_float64_and_empty_public_shapes() -> None:
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], dtype=torch.float64)
    query = points[:1].clone()
    distance, indices = SpatialIndex(points).ball_query(query, 2.0, 3)
    assert distance.dtype == torch.float64
    assert indices.tolist() == [[0, 1, -1]]
    empty_points = points[:0]
    empty_query = query[:0]
    for p, q, k in ((empty_points, query, 3), (points, empty_query, 3), (points, query, 0)):
        actual = SpatialIndex(p).ball_query(q, 0.0, k)
        expected = ball_query(q.unsqueeze(0), p.unsqueeze(0), 0.0, k)
        assert actual[0].shape == (len(q), k)
        assert torch.equal(actual[0], expected[0][0])
        assert torch.equal(actual[1], expected[1][0])


def test_spatial_index_rejects_mutated_reference_and_invalid_backend() -> None:
    points = torch.zeros((4, 3))
    with pytest.raises(ValueError, match="backend"):
        SpatialIndex(points, backend="other")
    index = SpatialIndex(points)
    points[0, 0] = 1.0
    with pytest.raises(ValueError, match="changed"):
        index.knn(points, 1)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0"
    or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0",
    reason="requires isolated MPS Safe Math with fallback disabled",
)
def test_mps_bvh_public_knn_and_ball_query_match_scan() -> None:
    generator = torch.Generator().manual_seed(2402)
    points = torch.rand((513, 3), generator=generator).to("mps") * 4
    query = torch.cat((points[::47][:7], torch.rand((10, 3), generator=generator).to("mps") * 4))
    index = SpatialIndex(points, backend="bvh")
    for k in (1, 8, 32):
        distance, indices = index.knn(query, k)
        scan_distance, scan_indices = SpatialIndex(points, backend="scan").knn(query, k)
        assert torch.equal(indices, scan_indices)
        torch.testing.assert_close(distance, scan_distance, rtol=2e-6, atol=2e-6)
    for radius in (0.1, 0.5, 2.0):
        distance, indices = index.ball_query(query, radius, 8)
        scan_distance, scan_indices = SpatialIndex(points, backend="scan").ball_query(query, radius, 8)
        assert torch.equal(indices, scan_indices)
        assert torch.equal(distance.view(torch.int32), scan_distance.view(torch.int32))


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0"
    or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0",
    reason="requires isolated MPS Safe Math with fallback disabled",
)
def test_mps_bvh_ball_query_first_order_gradient_matches_scan() -> None:
    torch.manual_seed(94)
    source = torch.rand((257, 3), device="mps") * 2
    query_source = torch.cat((source[:3], torch.rand((12, 3), device="mps") * 2))
    points_bvh = source.clone().detach().requires_grad_()
    query_bvh = query_source.clone().detach().requires_grad_()
    points_scan = source.clone().detach().requires_grad_()
    query_scan = query_source.clone().detach().requires_grad_()
    squared_bvh, idx_bvh = SpatialIndex(points_bvh, backend="bvh").ball_query(query_bvh, 0.9, 12)
    squared_scan, idx_scan = SpatialIndex(points_scan, backend="scan").ball_query(query_scan, 0.9, 12)
    assert torch.equal(idx_bvh, idx_scan)
    assert torch.equal(squared_bvh.view(torch.int32), squared_scan.view(torch.int32))
    weight = torch.arange(1, squared_bvh.numel() + 1, device="mps", dtype=torch.float32).reshape_as(squared_bvh) / 100
    (squared_bvh * weight).sum().backward()
    (squared_scan * weight).sum().backward()
    torch.testing.assert_close(query_bvh.grad, query_scan.grad, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(points_bvh.grad, points_scan.grad, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0"
    or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0",
    reason="requires isolated MPS Safe Math with fallback disabled",
)
def test_mps_public_bvh_empty_and_zero_radius_match_scan() -> None:
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], device="mps")
    query = points[:1].clone()
    for p, q, k in ((points[:0], query, 3), (points, query[:0], 3),
                    (points, query, 0), (points, query, 3)):
        for radius in (0.0, 2.0):
            actual = SpatialIndex(p, backend="bvh").ball_query(q, radius, k)
            expected = SpatialIndex(p, backend="scan").ball_query(q, radius, k)
            assert actual[0].shape == (len(q), k)
            assert torch.equal(actual[0].view(torch.int32), expected[0].view(torch.int32))
            assert torch.equal(actual[1], expected[1])


@pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("PYTORCH_MPS_FAST_MATH") != "1",
    reason="requires isolated MPS Fast Math",
)
def test_fast_math_auto_scans_and_forced_bvh_rejects() -> None:
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], device="mps")
    query = points[:1]
    index = SpatialIndex(points)
    assert index.knn(query, 1)[1].tolist() == [[0]]
    with pytest.raises(RuntimeError, match="PYTORCH_MPS_FAST_MATH=0"):
        index.knn(query, 1, backend="bvh")

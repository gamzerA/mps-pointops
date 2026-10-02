"""Exact first-K Ball Query gates for the private two-level Metal BVH."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest
import torch

if "bench" not in sys.modules:
    package = types.ModuleType("bench")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "bench")]
    sys.modules["bench"] = package

from bench.spatial_bvh import MortonTwoLevelBVH
from mps_pointops.ops import ball_query


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0"
    or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0",
    reason="requires a physical MPS GPU in isolated Safe Math with fallback disabled",
)


def _build(points: torch.Tensor, cell_size: float = 1.0) -> MortonTwoLevelBVH:
    return MortonTwoLevelBVH.build(points, torch.zeros(3, device="mps"), cell_size)


def _assert_native_parity(index: MortonTwoLevelBVH, query: torch.Tensor,
                          radius: float, k: int, *, audit: bool = True) -> torch.Tensor:
    actual_dist, actual_idx, stats = index.radius(query, radius, k, audit=audit)
    expected_dist, expected_idx = ball_query(
        query.unsqueeze(0), index.points.unsqueeze(0), radius, k
    )
    torch.mps.synchronize()
    assert actual_idx.shape == (len(query), k)
    assert actual_dist.dtype == torch.float32
    assert actual_idx.dtype == torch.int64
    assert torch.equal(actual_idx, expected_idx[0])
    assert torch.equal(actual_dist.view(torch.int32), expected_dist[0].view(torch.int32))
    assert stats.shape == (len(query), 5)
    assert torch.all(stats[:, 3:] == 0)
    return stats


@pytest.mark.parametrize("distribution", ["uniform", "dense_sparse"])
def test_radius_matches_native_metal_and_audits_prunes(distribution: str) -> None:
    torch.manual_seed(49)
    if distribution == "uniform":
        points = torch.rand((1600, 3), device="mps") * 8
        query = torch.cat((points[:8], torch.rand((8, 3), device="mps") * 8))
        radius, cell = 0.35, 0.25
    else:
        points = torch.cat((torch.rand((1450, 3), device="mps") * 0.25 + 4,
                            torch.rand((150, 3), device="mps") * 8))
        query = torch.cat((points[:8], torch.rand((8, 3), device="mps") * 8))
        radius, cell = 0.2, 0.125
    stats = _assert_native_parity(_build(points, cell), query, radius, 8)
    assert int(stats[:, 2].sum().item()) > 0


def test_first_k_uses_original_index_not_morton_or_distance_order() -> None:
    # Nearby points appear in scattered Morton bricks; their original source
    # indices, rather than key or distance order, define the output.
    points = torch.full((600, 3), 20.0, device="mps")
    points[0] = torch.tensor([2.0, 0.0, 0.0], device="mps")
    points[2] = torch.tensor([0.0, 0.0, 0.0], device="mps")
    points[127] = torch.tensor([1.0, 0.0, 0.0], device="mps")
    points[128] = points[127]
    points[255] = points[127]
    query = torch.zeros((2, 3), device="mps")
    index = _build(points)
    for _ in range(8):
        _assert_native_parity(index, query, 3.0, 4)
    assert index.radius(query, 3.0, 4)[1][0].tolist() == [0, 2, 127, 128]


def test_strict_boundary_and_fma_sensitive_components() -> None:
    below = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()
    above = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    points = torch.tensor(
        [[below, 0, 0], [1.0, 0, 0], [above, 0, 0], [0, 0, 0],
         [4096.0, 1.0, 1.0], [4096.0, 0, 0]],
        dtype=torch.float32, device="mps",
    )
    query = torch.tensor([[0.0, 0, 0], [4096.0, 0, 0]], device="mps")
    index = _build(points, 1.0)
    _assert_native_parity(index, query, 1.0, 4)
    assert index.radius(query[:1], 1.0, 4)[1][0].tolist() == [0, 3, -1, -1]

    r32 = torch.tensor(0.1, dtype=torch.float32)
    below_r = torch.nextafter(r32, torch.tensor(0.0)).item()
    above_r = torch.nextafter(r32, torch.tensor(1.0)).item()
    edge_points = torch.tensor([[below_r, 0, 0], [r32.item(), 0, 0],
                                [above_r, 0, 0], [0, 0, 0]], device="mps")
    _assert_native_parity(_build(edge_points, 0.125), query[:1], 0.1, 4)


def test_tiny_normalized_and_overflowing_square_follow_dense_path() -> None:
    small = 2.0**-60
    points = torch.tensor([[0, 0, 0], [small / 4, 0, 0],
                           [small, 0, 0], [1, 0, 0]], device="mps")
    query = torch.zeros((1, 3), device="mps")
    _assert_native_parity(_build(points), query, small, 4)

    huge = torch.tensor([[0, 0, 0], [1, 0, 0], [2.0**60, 0, 0]], device="mps")
    _assert_native_parity(_build(huge, 2.0**40), query, 2.0**65, 4)


def test_normalized_component_permutations_and_minimum_supported_radius() -> None:
    # A small component square flushes before comparison in a direct path.
    # Exercise its placement in all three coordinates against the dense kernel.
    radius = 2.0**-62
    near = 0.98 * radius
    small = radius / 4
    points = torch.tensor(
        [[small, near, 0], [near, small, 0], [near, 0, small],
         [0, near, small], [0, 0, 0]], device="mps"
    )
    query = torch.zeros((1, 3), device="mps")
    _assert_native_parity(_build(points), query, radius, 5)
    minimum = 2.0**-112
    tiny_points = torch.tensor([[0, 0, 0], [minimum / 2, 0, 0]], device="mps")
    _assert_native_parity(_build(tiny_points), query, minimum, 4)


def test_signed_domain_differential_fuzz_across_k_and_radius() -> None:
    origin = torch.full((3,), -16.0, device="mps")
    for seed in (3, 11, 29):
        torch.manual_seed(seed)
        points = (torch.rand((577, 3), device="mps") - 0.5) * 16
        queries = torch.cat((points[::87], (torch.rand((9, 3), device="mps") - 0.5) * 16))
        index = MortonTwoLevelBVH.build(points, origin, 0.125)
        for radius, k in ((0.01, 1), (0.5, 7), (3.0, 32)):
            _assert_native_parity(index, queries, radius, k)


def test_macro_tree_with_over_100k_points_matches_native() -> None:
    torch.manual_seed(117)
    points = torch.rand((100_003, 3), device="mps") * 16
    query = torch.stack((points[0], points[50_000], points[-1],
                         torch.tensor([8.0, 8.0, 8.0], device="mps")))
    index = _build(points, 0.125)
    assert index.padded_leaves > 64  # Exercise the macro build and traversal.
    _assert_native_parity(index, query, 0.1, 8)
    _assert_native_parity(index, query, 32.0, 8)


def test_empty_short_zero_radius_zero_k_and_zero_queries_write_every_slot() -> None:
    query = torch.zeros((3, 3), device="mps")
    for count in (0, 2):
        points = torch.arange(count, device="mps", dtype=torch.float32).unsqueeze(1).expand(-1, 3).contiguous()
        index = _build(points)
        for radius in (0.0, 10.0):
            _assert_native_parity(index, query, radius, 4)
        assert index.radius(query, 10.0, 0)[0].shape == (3, 0)
        assert index.radius(query[:0], 10.0, 4)[1].shape == (0, 4)


def test_stack_overflow_resets_row_to_exact_original_order_scan() -> None:
    from bench import spatial_bvh

    points = torch.arange(300, device="mps", dtype=torch.float32).unsqueeze(1).expand(-1, 3).contiguous()
    query = torch.tensor([[5.0, 6.0, 7.0]], device="mps")
    index = _build(points)
    source = Path(spatial_bvh.__file__).with_suffix(".metal").read_text()
    assert "STACK_CAPACITY = 32" in source
    shader = torch.mps.compile_shader(source.replace("STACK_CAPACITY = 32", "STACK_CAPACITY = 1"))
    actual_dist = torch.empty((1, 8), device="mps")
    actual_idx = torch.empty((1, 8), dtype=torch.int64, device="mps")
    stats = torch.empty((1, 5), dtype=torch.int32, device="mps")
    shader.bvh_radius_f32(
        query, points, index.sorted_indices, index.bounds, index.min_index,
        actual_dist, actual_idx, stats, 1, len(points), index.padded_leaves,
        8, 1000.0, 1_000_000.0, 1,
        threads=[128, 1, 1], group_size=[128, 1, 1],
    )
    expected_dist, expected_idx = ball_query(query.unsqueeze(0), points.unsqueeze(0), 1000.0, 8)
    torch.mps.synchronize()
    assert stats[0, 4].item() == 1
    assert stats[0, 3].item() == 0
    assert torch.equal(actual_idx, expected_idx[0])
    assert torch.equal(actual_dist.view(torch.int32), expected_dist[0].view(torch.int32))


def test_validation_and_mutation_guard() -> None:
    points = torch.zeros((3, 3), device="mps")
    index = _build(points)
    query = torch.zeros((1, 3), device="mps")
    with pytest.raises(ValueError, match="limit"):
        index.radius(query, 1.0, 33)
    with pytest.raises(ValueError, match="radius"):
        index.radius(query, -1.0, 2)
    with pytest.raises(ValueError, match="normal-or-zero"):
        invalid = query.clone()
        invalid.view(torch.int32)[0, 0] = 1
        index.radius(invalid, 1.0, 2)
    points.add_(1)
    with pytest.raises(ValueError, match="changed"):
        index.radius(query, 1.0, 2)

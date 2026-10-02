"""Small differential gates for the private two-level Metal BVH."""

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
from bench.spatial_selected_distances import selected_sqdist_f32
from mps_pointops._flat_search_mps import knn_indices
from mps_pointops.ops import knn as dense_knn


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available() or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0"
    or os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0",
    reason="requires a physical MPS GPU in isolated Safe Math with fallback disabled",
)


def _oracle(points: torch.Tensor, query: torch.Tensor, k: int) -> torch.Tensor:
    px = torch.tensor([0, len(points)], device="mps")
    py = torch.tensor([0, len(query)], device="mps")
    return knn_indices(points, query, px, py, k)


@pytest.mark.parametrize("kind", ["uniform", "dense_sparse"])
@pytest.mark.parametrize("parallel", [False, True])
def test_exact_index_parity_and_pruning_audit(kind: str, parallel: bool) -> None:
    torch.manual_seed(21)
    if kind == "uniform":
        points = torch.rand((1279, 3)).to("mps") * 32
        query = torch.cat((points[:8], torch.rand((8, 3)).to("mps") * 32))
        cell = 1.0
    else:
        cluster = (torch.rand((1152, 3)).to("mps") * 0.5) + 16
        background = torch.rand((127, 3)).to("mps") * 32
        points = torch.cat((cluster, background))
        query = torch.cat((cluster[:8], background[:4],
                           torch.rand((4, 3)).to("mps") * 32))
        cell = 0.015625
    index = MortonTwoLevelBVH.build(points, torch.zeros(3, device="mps"), cell)
    distances, actual, stats = index.knn(query, 16, audit=not parallel,
                                         parallel_microtrees=parallel)
    expected = _oracle(points, query, 16)
    torch.mps.synchronize()
    assert torch.equal(actual, expected)
    direct = selected_sqdist_f32(query, points, actual)
    torch.mps.synchronize()
    assert torch.equal(distances.view(torch.int32), direct.view(torch.int32))
    assert torch.all(stats[:, 3:] == 0)  # no missed winner, no fallback
    assert torch.all(distances.isfinite())
    assert int(stats[:, 2].sum().item()) > 0


def test_cross_leaf_exact_ties_choose_lowest_original_index() -> None:
    points = torch.full((300, 3), 10.0, device="mps")
    points[0] = torch.tensor([1.0, 0.0, 0.0], device="mps")
    points[127] = points[0]
    points[128] = points[0]
    points[255] = points[0]
    query = torch.zeros((1, 3), device="mps")
    index = MortonTwoLevelBVH.build(points, torch.zeros(3, device="mps"), 1.0)
    for parallel in (False, True):
        for _ in range(16):
            distances, actual, stats = index.knn(query, 3, audit=not parallel,
                                                 parallel_microtrees=parallel)
            assert actual.tolist() == [[0, 127, 128]]
            assert torch.equal(actual, _oracle(points, query, 3))
            assert stats[0, 3:].tolist() == [0, 0]
            assert distances[0].tolist() == [1.0, 1.0, 1.0]


def test_explicit_full_scan_on_bounded_stack_overflow() -> None:
    # Compile a deliberately tiny stack from the *same* source to drive the
    # otherwise unreachable balanced-tree fallback in a small fixture.
    from bench import spatial_bvh

    points = torch.arange(300, device="mps", dtype=torch.float32).unsqueeze(1).expand(-1, 3).contiguous()
    query = torch.tensor([[5.0, 6.0, 7.0]], device="mps")
    index = MortonTwoLevelBVH.build(points, torch.zeros(3, device="mps"), 1.0)
    source = Path(spatial_bvh.__file__).with_suffix(".metal").read_text()
    assert "STACK_CAPACITY = 32" in source
    shader = torch.mps.compile_shader(source.replace("STACK_CAPACITY = 32", "STACK_CAPACITY = 1"))
    distances = torch.empty((1, 4), device="mps")
    actual = torch.empty((1, 4), device="mps", dtype=torch.int64)
    stats = torch.empty((1, 5), device="mps", dtype=torch.int32)
    shader.bvh_knn_f32(
        query, points, index.sorted_indices, index.bounds, index.min_index,
        distances, actual, stats, 1, len(points), index.padded_leaves, 4, 0,
        threads=[128, 1, 1], group_size=[128, 1, 1],
    )
    torch.mps.synchronize()
    assert stats[0, 4].item() == 1
    assert torch.equal(actual, _oracle(points, query, 4))


def test_reject_subnormal_coordinate_bits() -> None:
    points = torch.zeros((2, 3), device="mps")
    points.view(torch.int32)[0, 0] = 1
    with pytest.raises(ValueError, match="normal-or-zero"):
        MortonTwoLevelBVH.build(points, torch.zeros(3, device="mps"), 1.0)


def test_finite_coordinates_with_infinite_squared_distance_keep_indices() -> None:
    # float32 distance overflows even though all coordinates are finite and
    # within the BVH's documented coordinate domain.
    huge = float(2**60)
    points = torch.tensor([[0.0, 0.0, 0.0], [huge, 0.0, 0.0],
                           [-huge, 0.0, 0.0]], device="mps")
    query = points[:1].clone()
    index = MortonTwoLevelBVH.build(
        points, torch.tensor([-huge, 0.0, 0.0], device="mps"), float(2**41)
    )
    expected_d, expected_i = dense_knn(query.unsqueeze(0), points.unsqueeze(0), 3)
    assert expected_i[0].tolist() == [[0, 1, 2]]
    for parallel in (False, True):
        squared, indices, stats = index.knn(query, 3, parallel_microtrees=parallel,
                                             audit=not parallel)
        assert torch.equal(indices, expected_i[0])
        assert torch.equal(torch.sqrt(squared).view(torch.int32),
                           expected_d[0].view(torch.int32))
        assert torch.all(stats[:, 3:] == 0)


def test_infinite_distance_ties_merge_across_microtrees() -> None:
    huge = float(2**60)
    points = torch.zeros((8193, 3), device="mps")
    points[::2, 0] = -huge
    points[1::2, 0] = huge
    query = torch.zeros((1, 3), device="mps")
    index = MortonTwoLevelBVH.build(
        points, torch.tensor([-huge, 0.0, 0.0], device="mps"), float(2**41)
    )
    assert index.padded_leaves > 64
    expected_d, expected_i = dense_knn(query.unsqueeze(0), points.unsqueeze(0), 16)
    assert expected_i[0].tolist() == [list(range(16))]
    for parallel in (False, True):
        squared, indices, stats = index.knn(query, 16, parallel_microtrees=parallel,
                                             audit=not parallel)
        assert torch.equal(indices, expected_i[0])
        assert torch.equal(torch.sqrt(squared).view(torch.int32),
                           expected_d[0].view(torch.int32))
        assert torch.all(stats[:, 3:] == 0)


def test_empty_and_short_reference_rows_are_fully_written() -> None:
    origin = torch.zeros(3, device="mps")
    query = torch.zeros((3, 3), device="mps")
    for count in (0, 2):
        points = torch.arange(count, device="mps", dtype=torch.float32).unsqueeze(1).expand(-1, 3).contiguous()
        index = MortonTwoLevelBVH.build(points, origin, 1.0)
        for parallel in (False, True):
            distance, actual, _ = index.knn(query, 4, parallel_microtrees=parallel)
            torch.mps.synchronize()
            if count == 0:
                assert actual.tolist() == [[-1] * 4] * len(query)
                assert torch.isinf(distance).all().item()
            else:
                assert actual.tolist() == [[0, 1, -1, -1]] * len(query)
                assert distance[:, :2].tolist() == [[0.0, 3.0]] * len(query)
                assert torch.isinf(distance[:, 2:]).all().item()
        assert index.knn(query, 0)[0].shape == (len(query), 0)


def test_boundary_and_underflow_cases_match_native_metal() -> None:
    below = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()
    above = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    tiny_normal = 2.0**-70
    pattern = torch.tensor(
        [[below, 0, 0], [1.0, 0, 0], [above, 0, 0],
         [4096.0, 1.0, 1.0], [4096.0, 0, 0],
         [tiny_normal, 0, 0], [0, tiny_normal, 0], [0, 0, tiny_normal],
         [0, 0, 0]], dtype=torch.float32,
    )
    points = pattern.repeat((40, 1)).flip(0).to("mps")
    query = torch.tensor([[1.0, 0, 0], [0, 0, 0], [4096.0, 0, 0],
                          [2.0**-65, 0, 0]], device="mps")
    index = MortonTwoLevelBVH.build(points, torch.zeros(3, device="mps"), 1.0)
    for parallel in (False, True):
        distance, actual, stats = index.knn(query, 16, audit=not parallel,
                                            parallel_microtrees=parallel)
        direct = selected_sqdist_f32(query, points, actual)
        expected = _oracle(points, query, 16)
        torch.mps.synchronize()
        assert torch.equal(actual, expected)
        assert torch.equal(distance.view(torch.int32), direct.view(torch.int32))
        assert torch.all(stats[:, 3:] == 0)


def test_collapsed_cluster_cannot_prune_equal_distance_without_tie_certificate() -> None:
    points = torch.ones((10_000, 3), device="mps")
    query = torch.ones((4, 3), device="mps")
    index = MortonTwoLevelBVH.build(points, torch.zeros(3, device="mps"), 1.0)
    expected = _oracle(points, query, 16)
    assert expected[0].tolist() == list(range(16))
    for parallel in (False, True):
        _, actual, stats = index.knn(query, 16, audit=not parallel,
                                     parallel_microtrees=parallel)
        assert torch.equal(actual, expected)
        assert torch.all(stats[:, 3:] == 0)
        # Strict bound equals the zero Kth distance everywhere, so this
        # prototype must visit all points until tie-aware equality pruning.
        assert torch.all(stats[:, 1] == len(points))

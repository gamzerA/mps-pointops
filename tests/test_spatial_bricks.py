"""Experimental Morton-brick kNN correctness, not a release performance gate."""

import sys
import types
from pathlib import Path

import pytest
import torch
import os

if "bench" not in sys.modules:
    package = types.ModuleType("bench")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "bench")]
    sys.modules["bench"] = package

from bench.spatial_bricks import MortonBrickIndex
from bench.spatial_selected_distances import selected_sqdist_f32


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
        same_mode_distances = selected_sqdist_f32(queries, points, actual)
        torch.mps.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
        assert torch.equal(distances.cpu().view(torch.int32),
                           same_mode_distances.cpu().view(torch.int32))
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
    query = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device="mps")
    distances, indices = index.knn(query, 4)
    torch.mps.synchronize()
    assert indices.cpu().tolist() == [[-1] * 4, [-1] * 4]
    assert torch.isinf(distances).all().item()
    assert index.knn(query, 0)[0].shape == (2, 0)
    with pytest.raises(ValueError, match=r"\[0, 32\]"):
        index.knn(query, 33)
    with pytest.raises(ValueError, match="finite"):
        index.knn(torch.tensor([[float("nan"), 0.0, 0.0]], device="mps"), 1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_pruning_policy_and_mutated_reference_rejection():
    points = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]], device="mps")
    index = MortonBrickIndex.build(points, torch.zeros(3, device="mps"), 1.0)
    assert index.pruning_enabled is (os.environ.get("PYTORCH_MPS_FAST_MATH") == "0")
    points[0, 0] = 4.0
    with pytest.raises(ValueError, match="changed"):
        index.knn(torch.zeros((1, 3), device="mps"), 1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_knn_adversarial_boundaries_ftz_and_staged_sum():
    from mps_pointops._flat_search_mps import knn_indices

    low = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()
    high = torch.nextafter(torch.tensor(1.0), torch.tensor(2.0)).item()
    tiny = 2.0**-70
    points_cpu = torch.tensor(
        [[low, 0, 0], [1.0, 0, 0], [high, 0, 0],
         [4096.0, 1.0, 1.0], [4096.0, 0.0, 0.0],
         [tiny, 0, 0], [0.0, tiny, 0], [0.0, 0.0, tiny],
         [0.0, 0.0, 0.0]] * 31,
        dtype=torch.float32,
    )
    # Reverse source order to keep Morton traversal order distinct from the
    # required lowest-original-index rule across multiple bricks.
    points = points_cpu.flip(0).to("mps")
    queries = torch.tensor([[1.0, 0, 0], [0.0, 0, 0],
                            [4096.0, 0, 0], [2.0**-65, 0, 0]], device="mps")
    index = MortonBrickIndex.build(points, torch.zeros(3, device="mps"), 1.0)
    assert len(index.brick_bounds) >= 3
    permutation = index.sorted_indices.cpu()
    bounds = index.brick_bounds.cpu()
    for brick in range(len(bounds)):
        source = points_cpu.flip(0)[permutation[brick * 128:(brick + 1) * 128]]
        torch.testing.assert_close(bounds[brick, :3], source.amin(dim=0), rtol=0, atol=0)
        torch.testing.assert_close(bounds[brick, 3:], source.amax(dim=0), rtol=0, atol=0)
    ptr_x = torch.tensor([0, len(points)], device="mps")
    ptr_y = torch.tensor([0, len(queries)], device="mps")
    for k in (1, 16, 32):
        _, actual = index.knn(queries, k)
        expected = knn_indices(points, queries, ptr_x, ptr_y, k)
        torch.mps.synchronize()
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_knn_large_finite_coordinate_domain():
    from mps_pointops._flat_search_mps import knn_indices

    large = float(2**60)
    points = torch.tensor([[-large, 0, 0], [large, 0, 0], [0, 0, 0]], device="mps")
    query = torch.tensor([[large, 0, 0]], device="mps")
    origin = torch.tensor([-large, 0, 0], device="mps")
    index = MortonBrickIndex.build(points, origin, large)
    distance, actual = index.knn(query, 3)
    expected = knn_indices(points, query, torch.tensor([0, 3], device="mps"),
                           torch.tensor([0, 1], device="mps"), 3)
    torch.mps.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert torch.isfinite(distance).all().item()
    with pytest.raises(ValueError, match=r"2\*\*60"):
        index.knn(torch.tensor([[2.0**61, 0, 0]], device="mps"), 1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_audit_checks_all_points_in_each_pruned_box():
    points = torch.cat((torch.zeros((128, 3)),
                        torch.tensor([[100.0, 0, 0]]).repeat(128, 1),
                        torch.tensor([[200.0, 0, 0]]).repeat(128, 1))).to("mps")
    query = torch.zeros((1, 3), device="mps")
    index = MortonBrickIndex.build(points, torch.zeros(3, device="mps"), 1.0)
    audit = index.audit_pruning(query, 1)
    torch.mps.synchronize()
    pruned, would_win = audit.cpu().tolist()[0]
    if index.pruning_enabled:
        assert pruned == 2
    else:
        assert pruned == 0  # Fast scans all bricks.
    assert would_win == 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_staged_sum_association_matches_native_mode():
    from mps_pointops._flat_search_mps import knn_indices

    points = torch.tensor([[4096.0, 1.0, 1.0], [4096.0, 0.0, 0.0]], device="mps")
    query = torch.zeros((1, 3), device="mps")
    index = MortonBrickIndex.build(points, torch.zeros(3, device="mps"), 1.0)
    distances, actual = index.knn(query, 2)
    expected = knn_indices(points, query, torch.tensor([0, 2], device="mps"),
                           torch.tensor([0, 1], device="mps"), 2)
    direct = selected_sqdist_f32(query, points, actual)
    torch.mps.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
    assert torch.equal(distances.cpu().view(torch.int32), direct.cpu().view(torch.int32))
    if index.pruning_enabled:
        assert actual.cpu().tolist() == [[0, 1]]
        assert distances.cpu().tolist() == [[float(2**24)] * 2]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_domain_rejects_subnormal_coordinate_bits_even_in_fast_math():
    subnormal = 2.0**-130
    origin = torch.zeros(3, device="mps")
    with pytest.raises(ValueError, match="subnormal"):
        MortonBrickIndex.build(torch.tensor([[subnormal, 0, 0]], device="mps"), origin, 1.0)
    index = MortonBrickIndex.build(torch.zeros((1, 3), device="mps"), origin, 1.0)
    with pytest.raises(ValueError, match="subnormal"):
        index.knn(torch.tensor([[subnormal, 0, 0]], device="mps"), 1)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_brick_randomized_mixed_density_audit_and_native_differential():
    from mps_pointops._flat_search_mps import knn_indices

    pruned_total = 0
    for seed in (7, 23, 101):
        generator = torch.Generator().manual_seed(seed)
        dense = torch.rand((700, 3), generator=generator) * 0.5 + 1.0
        sparse = torch.rand((325, 3), generator=generator) * 8.0
        points = torch.cat((dense, sparse)).to("mps")
        queries = torch.cat((dense[::70][:10],
                             torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]]))).to("mps")
        ptr_x = torch.tensor([0, len(points)], device="mps")
        ptr_y = torch.tensor([0, len(queries)], device="mps")
        for cell_size in (0.125, 1.0, 4.0):
            index = MortonBrickIndex.build(points, torch.zeros(3, device="mps"), cell_size)
            for k in (1, 8, 16):
                distance, actual = index.knn(queries, k)
                expected = knn_indices(points, queries, ptr_x, ptr_y, k)
                independent = selected_sqdist_f32(queries, points, actual)
                audit = index.audit_pruning(queries, k)
                torch.mps.synchronize()
                torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)
                assert torch.equal(distance.cpu().view(torch.int32),
                                   independent.cpu().view(torch.int32))
                counts = audit.cpu()
                assert int(counts[:, 1].sum()) == 0
                pruned_total += int(counts[:, 0].sum())
    if os.environ.get("PYTORCH_MPS_FAST_MATH") == "0":
        assert pruned_total > 0
    else:
        assert pruned_total == 0

"""Correctness tests for the CPU-only Morton/AABB hierarchy experiment."""

import math
import random
import sys
import types
from pathlib import Path
from fractions import Fraction

import pytest

if "bench" not in sys.modules:
    package = types.ModuleType("bench")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "bench")]
    sys.modules["bench"] = package

from bench.spatial_grid_reference import brute_knn, brute_radius
from bench.spatial_hierarchy_reference import (
    HierarchicalMortonReference, brute_knn_exact, brute_radius_exact,
)
from bench.spatial_hierarchy_reference import _distance_sq, _exact


def test_random_differential_and_original_index_contract():
    rng = random.Random(9074)
    # Dyadic coordinates make this exact oracle and the existing binary64
    # brute-force reference agree away from deliberately ambiguous boundaries.
    points = [(rng.randrange(-64, 65) / 32, rng.randrange(-64, 65) / 32,
               rng.randrange(-64, 65) / 32) for _ in range(192)]
    index = HierarchicalMortonReference(points, leaf_size=7, bits_per_axis=6)
    queries = points[:8] + [(rng.randrange(-48, 49) / 32,
                              rng.randrange(-48, 49) / 32,
                              rng.randrange(-48, 49) / 32) for _ in range(16)]
    for query in queries:
        for radius in (0.0, 0.125, 0.5, 2.0, 4.0):
            for limit in (None, 0, 1, 8):
                expected = brute_radius_exact(points, query, radius, limit)
                assert index.radius(query, radius, limit) == expected
                assert expected == brute_radius(points, query, radius, limit)
        for k in (0, 1, 3, 16, len(points) + 1):
            expected = brute_knn_exact(points, query, k)
            assert index.knn(query, k) == expected
            assert expected == brute_knn(points, query, k)
    assert index.radius_fallbacks == index.knn_fallbacks == 0


def test_ties_coincidences_and_first_k_across_brick_order():
    points = [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0),
              (0.0, 1.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 0.0),
              (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)]
    index = HierarchicalMortonReference(points, leaf_size=1, bits_per_axis=2)
    for _ in range(32):
        assert index.knn((0, 0, 0), 5) == [1, 5, 0, 2, 3]
        assert index.radius((0, 0, 0), 2, 4) == [0, 1, 2, 3]
        assert index.radius((0, 0, 0), 1, None) == [1, 5]


def test_dense_cluster_sparse_background_and_outside_queries():
    rng = random.Random(421)
    dense = [(rng.randrange(-4, 5) / 1024, rng.randrange(-4, 5) / 1024,
              rng.randrange(-4, 5) / 1024) for _ in range(220)]
    sparse = [(float(i * 17), float(i % 3) * 13, -float(i % 5) * 7)
              for i in range(36)]
    # Interleave so Morton traversal order strongly disagrees with source order.
    points = [point for pair in zip(dense[:36], sparse) for point in pair] + dense[36:]
    index = HierarchicalMortonReference(points, leaf_size=8, bits_per_axis=4)
    queries = dense[::11] + sparse[::6] + [(1000.0, 1000.0, 1000.0)]
    for query in queries:
        for radius in (0.001, 0.01, 1.0, 100.0):
            for limit in (None, 1, 16, 64):
                assert index.radius(query, radius, limit) == brute_radius_exact(points, query, radius, limit)
        for k in (1, 3, 16, 64):
            assert index.knn(query, k) == brute_knn_exact(points, query, k)


def test_boxes_contain_every_descendant_actual_coordinate():
    points = [(0.0, 0.0, 0.0), (math.nextafter(0.0, math.inf), 0.0, 0.0),
              (math.nextafter(1.0, -math.inf), 1.0, 0.0), (1.0, 1.0, 0.0),
              (2.0, 2.0, 2.0), (float(2 ** -126), 0.0, 0.0)]
    index = HierarchicalMortonReference(points, leaf_size=1, bits_per_axis=1)

    def descendant_indices(node_id):
        node = index.nodes[node_id]
        if node.indices:
            indices = node.indices
        else:
            indices = descendant_indices(node.left) + descendant_indices(node.right)
        for point_id in indices:
            for axis in range(3):
                assert node.lo[axis] <= points[point_id][axis] <= node.hi[axis]
        assert node.min_index == min(indices)
        for query in (points[0], points[2], (0.5, 0.5, 0.0)):
            q = _exact(query)
            bound = index._box_lower_sq(q, node)
            assert isinstance(bound, Fraction)
            assert all(bound <= _distance_sq(q, _exact(points[point_id]))
                       for point_id in indices)
        return indices

    assert set(descendant_indices(index.root)) == set(range(len(points)))
    for query in (points[0], points[2], (0.5, 0.5, 0.0)):
        for radius in (math.nextafter(1.0, -math.inf), 1.0, math.nextafter(1.0, math.inf)):
            assert index.radius(query, radius, 3) == brute_radius_exact(points, query, radius, 3)
        assert index.knn(query, 4) == brute_knn_exact(points, query, 4)


def test_extreme_float32_values_do_not_make_aabb_underconservative():
    minimum_subnormal = float(2 ** -149)
    maximum_finite = 3.4028234663852886e38
    points = [(-maximum_finite, 0.0, 0.0), (maximum_finite, 0.0, 0.0),
              (minimum_subnormal, 0.0, 0.0), (0.0, minimum_subnormal, 0.0),
              (0.0, 0.0, 0.0)]
    index = HierarchicalMortonReference(points, leaf_size=1, bits_per_axis=21)
    for query in ((0.0, 0.0, 0.0), (maximum_finite, 0.0, 0.0),
                  (-maximum_finite, 0.0, 0.0)):
        for radius in (minimum_subnormal, 1.0, maximum_finite):
            assert index.radius(query, radius, 3) == brute_radius_exact(points, query, radius, 3)
        for k in (1, 3, 5):
            assert index.knn(query, k) == brute_knn_exact(points, query, k)


def test_visit_budget_has_visible_exact_fallback():
    points = [(float(i), 0.0, 0.0) for i in range(30)]
    index = HierarchicalMortonReference(points, leaf_size=2, max_node_visits=1)
    assert index.knn((14.2, 0, 0), 3) == brute_knn_exact(points, (14.2, 0, 0), 3)
    assert index.radius((14.2, 0, 0), 3, 2) == brute_radius_exact(points, (14.2, 0, 0), 3, 2)
    assert index.knn_fallbacks == 1
    assert index.radius_fallbacks == 1
    assert index.last_knn_nodes == index.last_radius_nodes == 1


def test_empty_invalid_and_nonfinite_inputs():
    index = HierarchicalMortonReference([])
    assert index.knn((0, 0, 0), 4) == []
    assert index.radius((0, 0, 0), 1) == []
    with pytest.raises(ValueError, match="finite"):
        HierarchicalMortonReference([(math.nan, 0, 0)])
    with pytest.raises(ValueError, match="finite"):
        index.knn((math.inf, 0, 0), 1)
    with pytest.raises(ValueError, match="nonnegative"):
        index.knn((0, 0, 0), -1)
    with pytest.raises(ValueError, match="nonnegative"):
        index.radius((0, 0, 0), -1)
    with pytest.raises(ValueError, match="positive"):
        HierarchicalMortonReference([], leaf_size=0)

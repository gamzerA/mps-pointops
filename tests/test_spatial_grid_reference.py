"""Exact CPU research oracle for the proposed v0.9 spatial-index layout."""

import random
import sys
import types
from pathlib import Path

import pytest

if "bench" not in sys.modules:
    package = types.ModuleType("bench")
    package.__path__ = [str(Path(__file__).resolve().parents[1] / "bench")]
    sys.modules["bench"] = package

from bench.spatial_grid_reference import UniformGridReference, brute_knn, brute_radius


def test_uniform_grid_matches_full_scan_for_random_and_boundary_queries():
    rng = random.Random(913)
    points = [(rng.randrange(-20, 21) / 16, rng.randrange(-20, 21) / 16,
               rng.randrange(-20, 21) / 16) for _ in range(240)]
    queries = points[:10] + [(rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.5),
                              rng.uniform(-1.5, 1.5)) for _ in range(10)]
    index = UniformGridReference(points, 0.25)
    for query in queries:
        for radius in (0.0, 0.125, 0.25, 0.7, 4.0):
            for limit in (None, 0, 1, 8):
                assert index.radius(query, radius, limit) == brute_radius(points, query, radius, limit)
        for k in (0, 1, 3, 16, len(points) + 1):
            assert index.knn(query, k) == brute_knn(points, query, k)


def test_first_indices_and_ties_survive_cell_traversal_order():
    # The nearest cells are not necessarily in source order. The public flat
    # radius API nevertheless requires the first K source rows.
    points = [(0.9, 0.0, 0.0), (0.1, 0.0, 0.0), (-0.1, 0.0, 0.0),
              (0.0, 0.0, 0.0), (0.9, 0.0, 0.0)]
    index = UniformGridReference(points, 0.2)
    assert index.radius((0.0, 0.0, 0.0), 0.5, 2) == [1, 2]
    assert index.knn((0.0, 0.0, 0.0), 4) == [3, 1, 2, 0]
    assert index.radius((0.0, 0.0, 0.0), 0.1) == [3]


def test_degenerate_scans_use_visible_exact_fallback():
    points = [(float(i), 0.0, 0.0) for i in range(20)]
    index = UniformGridReference(points, 0.01, max_radius_cells=1, max_knn_shells=0)
    assert index.radius((0.0, 0.0, 0.0), 10, 3) == [0, 1, 2]
    assert index.radius_fallbacks == 1
    assert index.knn((100.0, 0.0, 0.0), 2) == [19, 18]
    assert index.knn_fallbacks == 1


def test_empty_and_invalid_inputs():
    index = UniformGridReference([], 1.0)
    assert index.radius((0, 0, 0), 1) == []
    assert index.knn((0, 0, 0), 1) == []
    with pytest.raises(ValueError, match="positive"):
        UniformGridReference([], 0)
    with pytest.raises(ValueError, match="finite"):
        UniformGridReference([(float("nan"), 0, 0)], 1)
    with pytest.raises(ValueError, match="nonnegative"):
        index.knn((0, 0, 0), -1)

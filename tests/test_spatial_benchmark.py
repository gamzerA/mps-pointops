"""Small contract checks for the bounded 1M spatial benchmark oracle."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "bench" / "bench_spatial_1m.py"
SPEC = importlib.util.spec_from_file_location("bench_spatial_1m", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


def test_knn_oracle_breaks_ties_by_original_index():
    points = np.array([[1, 0, 0], [-1, 0, 0], [0.5, 0, 0], [1, 0, 0]], dtype=np.float32)
    idx, distance = bench.brute_knn(np.zeros(3, dtype=np.float32), points, 4)
    assert idx.tolist() == [2, 0, 1, 3]
    np.testing.assert_array_equal(distance, np.array([0.5, 1, 1, 1], dtype=np.float32))


def test_radius_oracle_first_k_strict_boundary_and_padding():
    points = np.array([[1, 0, 0], [0.75, 0, 0], [0, 0, 0], [-0.5, 0, 0]], dtype=np.float32)
    idx, d2, ambiguous = bench.brute_ball_query(np.zeros(3, dtype=np.float32), points, 1.0, 4)
    assert idx.tolist() == [1, 2, 3, -1]
    np.testing.assert_array_equal(d2, np.array([0.5625, 0, 0.25, 0], dtype=np.float32))
    assert ambiguous == 1  # the point exactly on the strict radius boundary


@pytest.mark.parametrize("n,q,requested,expected", [
    (20_000, 1024, 8, 8),
    (100_000, 1024, 8, 8),
    (1_000_000, 1024, 8, 8),
    (1_000_000, 1_000_000, 1_000_000, 16),
])
def test_verification_plan_bounds_brute_pairs(n, q, requested, expected):
    rows = bench.verification_rows(q, n, requested)
    assert len(rows) == expected
    assert len(rows) * n <= bench.MAX_VERIFY_PAIRS
    assert rows[0] == 0 and rows[-1] == q - 1


def test_timed_workload_rejects_one_million_by_one_million():
    bench.check_timed_pair_budget(1_000_000, 1024)
    with pytest.raises(ValueError, match="timed brute-force pair cap"):
        bench.check_timed_pair_budget(1_000_000, 1_000_000)


def test_scipy_radius_output_is_first_k_by_input_index():
    scipy = pytest.importorskip("scipy.spatial")
    points = np.array([[0.5, 0, 0], [2, 0, 0], [0.25, 0, 0], [0, 0, 0]], dtype=np.float32)
    queries = np.zeros((1, 3), dtype=np.float32)
    idx, distances = bench.scipy_search(scipy.cKDTree(points), queries, "ball_query", 2, 1.0, 2, 1)
    assert idx.tolist() == [[0, 2]]
    assert distances is None


def test_outside_queries_make_a_no_hit_radius_case():
    scipy = pytest.importorskip("scipy.spatial")
    points, queries = bench.make_cloud(20_000, 8, 3, "shell", "random", "outside")
    tree = scipy.cKDTree(points)
    idx, _ = bench.scipy_search(tree, queries, "ball_query", 8, 0.1, 8, 1)
    assert (idx == -1).all()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
@pytest.mark.parametrize("n", [20_000, 100_000, 1_000_000])
def test_sampled_dense_mps_matches_bounded_brute_oracle(n):
    """Run 8 queries at 1M; never allocate a full QxN reference matrix."""
    from mps_pointops import ops
    points, queries = bench.make_cloud(n, 8, 9, "shell", "random")
    ref = torch.from_numpy(points).unsqueeze(0).to("mps")
    q = torch.from_numpy(queries).unsqueeze(0).to("mps")
    knn_distance, knn_index = ops.knn(q, ref, 8)
    ball_distance, ball_index = ops.ball_query(q, ref, 0.1, 8)
    torch.mps.synchronize()
    knn_distance = knn_distance[0].cpu().numpy()
    knn_index = knn_index[0].cpu().numpy()
    ball_distance = ball_distance[0].cpu().numpy()
    ball_index = ball_index[0].cpu().numpy()
    for row in range(8):
        want_idx, want_distance = bench.brute_knn(queries[row], points, 8)
        np.testing.assert_array_equal(knn_index[row], want_idx)
        np.testing.assert_allclose(knn_distance[row], want_distance, rtol=1e-6, atol=1e-7)
        want_idx, want_d2, ambiguous = bench.brute_ball_query(queries[row], points, 0.1, 8)
        if ambiguous == 0:
            np.testing.assert_array_equal(ball_index[row], want_idx)
        np.testing.assert_allclose(ball_distance[row], want_d2, rtol=1e-5, atol=1e-6)

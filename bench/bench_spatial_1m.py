#!/usr/bin/env python3
"""Reproducible 20k/100k/1M baseline for exact 3D spatial search.

The current Metal search is a full scan. SciPy cKDTree is timed with its tree
build and query separated. Verification uses at most 16 million query/reference
pairs and keeps only one query's distances in memory at a time; it never builds
an N-by-N matrix.

Example:
    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_spatial_1m.py --output bench/results/local/spatial-safe.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.spatial import cKDTree
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import ops  # noqa: E402

MAX_VERIFY_PAIRS = 16_000_000
MAX_TIMED_BRUTE_PAIRS = 1_024_000_000


def make_cloud(n: int, query_count: int, seed: int, distribution: str, order: str,
               query_placement: str = "on-cloud") -> tuple[np.ndarray, np.ndarray]:
    """Create finite float32 references and reproducible query placements."""
    rng = np.random.default_rng(seed)
    if distribution == "shell":
        points = rng.standard_normal((n, 3), dtype=np.float32)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        points += np.float32(0.01) * rng.standard_normal((n, 3), dtype=np.float32)
    elif distribution == "uniform":
        points = rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32)
    else:
        raise ValueError(f"unknown distribution: {distribution}")
    if order == "x-sorted":
        points = points[np.argsort(points[:, 0], kind="stable")]
    elif order != "random":
        raise ValueError(f"unknown order: {order}")
    points = np.ascontiguousarray(points)
    selected = rng.choice(n, size=query_count, replace=query_count > n)
    queries = np.ascontiguousarray(points[selected])
    if query_placement == "outside":
        # All supported clouds occupy roughly [-1, 1]^3, so this creates a
        # no-hit radius fixture at the default radius and forces a full scan.
        queries[:, 0] += np.float32(4.0)
    elif query_placement != "on-cloud":
        raise ValueError(f"unknown query placement: {query_placement}")
    return points, queries


def radius_threshold(radius: float) -> tuple[float, float]:
    """Dense Ball Query uses fl32(r), then fl32(fl32(r) * fl32(r))."""
    r32 = np.float32(radius)
    return float(r32), float(np.float32(r32 * r32))


def direct_sqdist_f32(query: np.ndarray, points: np.ndarray) -> np.ndarray:
    """One float32 subtraction/product/addition at a time, without a QxN matrix."""
    dx = np.subtract(points[:, 0], query[0], dtype=np.float32)
    dy = np.subtract(points[:, 1], query[1], dtype=np.float32)
    dz = np.subtract(points[:, 2], query[2], dtype=np.float32)
    out = np.multiply(dx, dx, dtype=np.float32)
    out = np.add(out, np.multiply(dy, dy, dtype=np.float32), dtype=np.float32)
    return np.add(out, np.multiply(dz, dz, dtype=np.float32), dtype=np.float32)


def brute_knn(query: np.ndarray, points: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Select by (rounded squared distance, original index), including ties."""
    if not 1 <= k <= len(points):
        raise ValueError("k must lie in [1, N]")
    if len(points) >= 2**32:
        raise ValueError("the packed key requires N < 2**32")
    d2 = direct_sqdist_f32(query, points)
    if not np.isfinite(d2).all() or (d2 < 0).any():
        raise ValueError("this finite-input oracle requires finite squared distances")
    key = (d2.view(np.uint32).astype(np.uint64) << np.uint64(32)) | np.arange(len(points), dtype=np.uint64)
    first = np.argpartition(key, k - 1)[:k]
    first = first[np.argsort(key[first], kind="stable")]
    return first.astype(np.int64), np.sqrt(d2[first]).astype(np.float32)


def brute_ball_query(query: np.ndarray, points: np.ndarray, radius: float, limit: int) -> tuple[np.ndarray, np.ndarray, int]:
    """First-K strict-radius oracle and count of numerically ambiguous points.

    The oracle uses separate float32 operations. Metal's Ball Query uses FMA,
    so points within four float32 ULPs of the threshold are reported instead
    of being called an unconditional cross-device failure.
    """
    if limit < 0:
        raise ValueError("limit must be nonnegative")
    _, r2 = radius_threshold(radius)
    d2 = direct_sqdist_f32(query, points)
    hits = np.flatnonzero(d2 < np.float32(r2))[:limit]
    delta64 = points.astype(np.float64) - query.astype(np.float64)
    exact_d2 = np.sum(delta64 * delta64, axis=1)
    band = 4.0 * float(np.spacing(np.float32(r2)))
    ambiguous = int(np.count_nonzero(np.abs(exact_d2 - r2) <= band))
    out_idx = np.full(limit, -1, dtype=np.int64)
    out_d2 = np.zeros(limit, dtype=np.float32)
    out_idx[:len(hits)] = hits
    out_d2[:len(hits)] = d2[hits]
    return out_idx, out_d2, ambiguous


def verification_rows(query_count: int, n: int, requested: int) -> np.ndarray:
    """Bound verification work even when the timed query batch is very large."""
    if not 1 <= n <= MAX_VERIFY_PAIRS or query_count < 1 or requested < 1:
        raise ValueError("N, query_count and requested verification count must be positive; N <= 16M")
    count = min(query_count, requested, MAX_VERIFY_PAIRS // n)
    return np.unique(np.linspace(0, query_count - 1, num=count, dtype=np.int64))


def check_timed_pair_budget(n: int, query_count: int) -> None:
    """Reject accidental 1M-by-1M brute-force timing requests."""
    if n * query_count > MAX_TIMED_BRUTE_PAIRS:
        raise ValueError(f"N * Q exceeds the {MAX_TIMED_BRUTE_PAIRS:,} timed brute-force pair cap")


def scipy_search(tree: cKDTree, queries: np.ndarray, op: str, k: int, radius: float, limit: int, workers: int) -> tuple[np.ndarray, np.ndarray | None]:
    if op == "knn":
        distances, indices = tree.query(queries, k=k, eps=0.0, workers=workers)
        return np.asarray(indices, dtype=np.int64).reshape(len(queries), k), np.asarray(distances).reshape(len(queries), k)
    hits = tree.query_ball_point(queries, r=radius, eps=0.0, workers=workers, return_sorted=True)
    indices = np.full((len(queries), limit), -1, dtype=np.int64)
    for row, found in enumerate(hits):
        selected = found[:limit]
        indices[row, :len(selected)] = selected
    return indices, None


def _ms_summary(samples_ns: list[int]) -> dict[str, Any]:
    samples = [value / 1e6 for value in samples_ns]
    return {"samples_ms": samples, "median_ms": statistics.median(samples), "min_ms": min(samples)}


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _mps_memory_sample() -> dict[str, int] | None:
    if not torch.backends.mps.is_available():
        return None
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def measure_scipy(points: np.ndarray, queries: np.ndarray, op: str, args: argparse.Namespace) -> tuple[dict[str, Any], np.ndarray, np.ndarray | None]:
    for _ in range(args.warmup):
        tree = cKDTree(points)
        scipy_search(tree, queries, op, args.k, args.radius, args.ball_k, args.workers)
    builds, queries_ns, totals = [], [], []
    output: tuple[np.ndarray, np.ndarray | None] | None = None
    for _ in range(args.repeat):
        t0 = time.perf_counter_ns()
        tree = cKDTree(points)
        t1 = time.perf_counter_ns()
        output = scipy_search(tree, queries, op, args.k, args.radius, args.ball_k, args.workers)
        t2 = time.perf_counter_ns()
        builds.append(t1 - t0)
        queries_ns.append(t2 - t1)
        totals.append(t2 - t0)
    assert output is not None
    return {
        "build": _ms_summary(builds),
        "query": _ms_summary(queries_ns),
        "build_plus_query": _ms_summary(totals),
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }, output[0], output[1]


def measure_mps(points: np.ndarray, queries: np.ndarray, op: str, args: argparse.Namespace) -> tuple[dict[str, Any], tuple[torch.Tensor, torch.Tensor]]:
    def run(ref: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return ops.knn(q, ref, args.k) if op == "knn" else ops.ball_query(q, ref, args.radius, args.ball_k)

    ref = torch.from_numpy(points).unsqueeze(0).to("mps")
    q = torch.from_numpy(queries).unsqueeze(0).to("mps")
    torch.mps.synchronize()
    for _ in range(args.warmup):
        run(ref, q)
        torch.mps.synchronize()
    resident_ns: list[int] = []
    sampled_memory: list[dict[str, int]] = []
    output: tuple[torch.Tensor, torch.Tensor] | None = None
    for _ in range(args.repeat):
        torch.mps.synchronize()
        t0 = time.perf_counter_ns()
        output = run(ref, q)
        torch.mps.synchronize()
        resident_ns.append(time.perf_counter_ns() - t0)
        sampled_memory.append(_mps_memory_sample() or {})
    assert output is not None
    cpu_output = (output[0].cpu(), output[1].cpu())
    del output, ref, q
    torch.mps.empty_cache()

    # This second mode includes both host-to-device inputs and device-to-host
    # outputs. It is recorded separately from resident search latency.
    transfer_ns: list[int] = []
    for _ in range(args.repeat):
        torch.mps.synchronize()
        t0 = time.perf_counter_ns()
        ref_once = torch.from_numpy(points).unsqueeze(0).to("mps")
        q_once = torch.from_numpy(queries).unsqueeze(0).to("mps")
        result = run(ref_once, q_once)
        result[0].cpu(), result[1].cpu()
        torch.mps.synchronize()
        transfer_ns.append(time.perf_counter_ns() - t0)
        sampled_memory.append(_mps_memory_sample() or {})
        del ref_once, q_once, result
    return {
        "resident_query": _ms_summary(resident_ns),
        "host_to_device_query_device_to_host": _ms_summary(transfer_ns),
        "sampled_mps_current_allocated_max_bytes": max((m.get("current_allocated_bytes", 0) for m in sampled_memory), default=0),
        "sampled_mps_driver_allocated_max_bytes": max((m.get("driver_allocated_bytes", 0) for m in sampled_memory), default=0),
        "process_peak_rss_bytes": _peak_rss_bytes(),
    }, cpu_output


def verify(points: np.ndarray, queries: np.ndarray, op: str, args: argparse.Namespace,
           scipy_indices: np.ndarray, scipy_distances: np.ndarray | None,
           mps_output: tuple[torch.Tensor, torch.Tensor] | None) -> dict[str, Any]:
    sampled = verification_rows(len(queries), len(points), args.verify_queries)
    cpu_mismatches = mps_mismatches = 0
    mps_distance_mismatches = 0
    cpu_max_distance_error = mps_max_distance_error = 0.0
    boundary_points = 0
    topk_ties = cutoff_tie_queries = 0
    for row in sampled:
        if op == "knn":
            want_idx, want_dist = brute_knn(queries[row], points, args.k)
            ranked_d2 = direct_sqdist_f32(queries[row], points)
            selected_d2 = ranked_d2[want_idx]
            topk_ties += int(np.count_nonzero(selected_d2[1:] == selected_d2[:-1]))
            cutoff_tie_queries += int(np.count_nonzero(ranked_d2 == selected_d2[-1]) >
                                      np.count_nonzero(selected_d2 == selected_d2[-1]))
            cpu_mismatches += int(np.count_nonzero(scipy_indices[row] != want_idx))
            assert scipy_distances is not None
            cpu_max_distance_error = max(cpu_max_distance_error, float(np.max(np.abs(scipy_distances[row] - want_dist))))
        else:
            want_idx, want_dist, ambiguous = brute_ball_query(queries[row], points, args.radius, args.ball_k)
            boundary_points += ambiguous
            cpu_mismatches += int(np.count_nonzero(scipy_indices[row] != want_idx))
        if mps_output is not None:
            got_dist = mps_output[0][0, row].numpy()
            got_idx = mps_output[1][0, row].numpy()
            mps_mismatches += int(np.count_nonzero(got_idx != want_idx))
            mps_max_distance_error = max(mps_max_distance_error, float(np.max(np.abs(got_dist - want_dist))))
            tolerance = (1e-6, 1e-7) if op == "knn" else (1e-5, 1e-6)
            mps_distance_mismatches += int(np.count_nonzero(
                ~np.isclose(got_dist, want_dist, rtol=tolerance[0], atol=tolerance[1])))
    return {
        "sample_query_rows": sampled.tolist(),
        "brute_force_pairs": int(len(sampled) * len(points)),
        "oracle": "float32 direct squared distance; kNN sorted by (distance, index); radius first K with strict d2 < fl32(fl32(r)^2)",
        "radius_boundary_band_points": boundary_points,
        "radius_boundary_band": "4 float32 ULP around threshold; FMA and SciPy float64 can differ there",
        "float32_knn_adjacent_topk_ties": topk_ties if op == "knn" else None,
        "float32_knn_cutoff_tie_queries": cutoff_tie_queries if op == "knn" else None,
        "scipy_index_mismatches_diagnostic": cpu_mismatches,
        "scipy_max_distance_abs_error_diagnostic": cpu_max_distance_error if op == "knn" else None,
        "mps_index_mismatches": mps_mismatches if mps_output is not None else None,
        "mps_max_distance_abs_error": mps_max_distance_error if mps_output is not None else None,
        "mps_distance_mismatches_at_tolerance": mps_distance_mismatches if mps_output is not None else None,
        "mps_distance_tolerance": "knn rtol=1e-6 atol=1e-7; ball_query rtol=1e-5 atol=1e-6",
        "strict_mps_index_gate": (mps_mismatches == 0 and boundary_points == 0) if mps_output is not None else None,
        "strict_mps_distance_gate": (mps_distance_mismatches == 0) if mps_output is not None else None,
    }


def _source_hashes() -> dict[str, str]:
    paths = [
        "bench/bench_spatial_1m.py", "mps_pointops/ops.py", "mps_pointops/_ball_query_mps.py",
        "mps_pointops/kernels/knn.metal", "mps_pointops/kernels/ball_query.metal",
    ]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def _git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.k < 1 or args.ball_k < 1 or args.queries < 1 or args.verify_queries < 1:
        raise ValueError("k, ball-k, queries and verify-queries must be positive")
    if args.radius < 0 or not np.isfinite(args.radius):
        raise ValueError("radius must be finite and nonnegative")
    if args.repeat < 1 or args.warmup < 0:
        raise ValueError("repeat must be positive and warmup nonnegative")
    if any(n < args.k or n > MAX_VERIFY_PAIRS for n in args.sizes):
        raise ValueError("every N must satisfy k <= N <= 16,000,000")
    for n in args.sizes:
        check_timed_pair_budget(n, args.queries)
    has_mps = bool(torch.backends.mps.is_available())
    if args.require_mps and not has_mps:
        raise RuntimeError("MPS is not available")
    rows = []
    for n in args.sizes:
        points, queries = make_cloud(n, args.queries, args.seed, args.distribution, args.order,
                                     args.query_placement)
        for op in args.ops:
            cpu_times, cpu_indices, cpu_distances = measure_scipy(points, queries, op, args)
            mps_times = mps_output = None
            if has_mps:
                mps_times, mps_output = measure_mps(points, queries, op, args)
            verification = verify(points, queries, op, args, cpu_indices, cpu_distances, mps_output)
            row = {"n": n, "q": len(queries), "op": op, "scipy_ckdtree": cpu_times,
                   "mps_dense_brute": mps_times, "verification": verification}
            if op == "ball_query":
                row["radius_selected_cap_fraction"] = float(np.mean(np.all(cpu_indices >= 0, axis=1)))
            rows.append(row)
            cpu_ms = cpu_times["build_plus_query"]["median_ms"]
            mps_ms = mps_times["resident_query"]["median_ms"] if mps_times else None
            print(f"{op:10s} N={n:,} Q={len(queries):,} cKDTree build+query={cpu_ms:.3f} ms "
                  f"MPS resident={f'{mps_ms:.3f} ms' if mps_ms is not None else 'unavailable'} "
                  f"sample={len(verification['sample_query_rows'])}", flush=True)
    result = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "chip": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip() or platform.processor(),
            "platform": platform.platform(), "python": platform.python_version(),
            "torch": torch.__version__, "scipy": scipy.__version__, "numpy": np.__version__,
            "mps_available": has_mps,
            "PYTORCH_MPS_FAST_MATH": os.getenv("PYTORCH_MPS_FAST_MATH", "unset"),
            "PYTORCH_ENABLE_MPS_FALLBACK": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            "git_commit": _git_commit(), "source_sha256": _source_hashes(),
        },
        "parameters": {"sizes": args.sizes, "queries": args.queries, "k": args.k,
                       "ball_k": args.ball_k, "radius": args.radius,
                       "radius_float32_and_square": radius_threshold(args.radius),
                       "seed": args.seed, "distribution": args.distribution, "order": args.order,
                       "query_placement": args.query_placement,
                       "warmup": args.warmup, "repeat": args.repeat, "workers": args.workers,
                       "verify_queries_requested": args.verify_queries,
                       "max_brute_force_verification_pairs": MAX_VERIFY_PAIRS,
                       "max_timed_brute_force_pairs": MAX_TIMED_BRUTE_PAIRS},
        "measurement_policy": {
            "cpu": "cKDTree build and query timed separately and together; float32 input converted by SciPy; eps=0; workers as recorded; all radius hits returned then first K retained",
            "mps_resident": "float32 inputs already on MPS; torch.mps.synchronize before and after each timed public API call; output allocation included; H2D/D2H excluded",
            "mps_transfer_inclusive": "CPU float32 inputs to MPS, public API, output to CPU and synchronize in one timed interval",
            "memory": "process_peak_rss_bytes is process-lifetime ru_maxrss, not a per-case delta; MPS byte counts are samples after dispatch, not true interval peaks",
            "numerical": "SciPy uses float64 distance and Python float radius; its indices/distances are diagnostics. Direct float32 brute oracle is the Metal index gate; radius rows with points in the boundary band are inconclusive for strict cross-device equality.",
        },
        "rows": rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=int, nargs="+", default=[20_000, 100_000, 1_000_000])
    parser.add_argument("--ops", nargs="+", choices=["knn", "ball_query"], default=["knn", "ball_query"])
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--ball-k", type=int, default=64)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--distribution", choices=["shell", "uniform"], default="shell")
    parser.add_argument("--order", choices=["random", "x-sorted"], default="random")
    parser.add_argument("--query-placement", choices=["on-cloud", "outside"], default="on-cloud")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=-1, help="SciPy query workers; -1 uses all cores")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--verify-queries", type=int, default=8)
    parser.add_argument("--require-mps", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

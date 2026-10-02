"""Synchronized Safe Math BVH versus flat brick and native Metal kNN.

This is a private feasibility experiment, not a release performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.spatial_bvh import MortonTwoLevelBVH  # noqa: E402
from bench.spatial_bricks import MortonBrickIndex  # noqa: E402
from mps_pointops._flat_search_mps import knn_indices  # noqa: E402


def _git(*args: str) -> str:
    result = subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                            text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _fixture(args) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(args.seed)
    if args.distribution == "uniform":
        points = rng.uniform(0, args.extent, size=(args.points, 3)).astype(np.float32)
        if args.query_source == "independent":
            query = rng.uniform(0, args.extent, size=(args.queries, 3)).astype(np.float32)
        else:
            rows = rng.choice(args.points, size=args.queries, replace=args.queries > args.points)
            query = points[rows].copy()
        detail = {"query_source": args.query_source, "uniform_cube": [0, args.extent]}
    elif args.distribution == "cluster-sparse":
        clustered = int(args.points * 0.9)
        cluster = (args.extent / 2 + rng.uniform(0, 1, size=(clustered, 3))).astype(np.float32)
        background = rng.uniform(0, args.extent, size=(args.points - clustered, 3)).astype(np.float32)
        points = np.concatenate((cluster, background), axis=0)
        q_cluster = args.queries // 2
        if args.query_source == "independent":
            cq = (args.extent / 2 + rng.uniform(0, 1, size=(q_cluster, 3))).astype(np.float32)
            bq = rng.uniform(0, args.extent,
                             size=(args.queries - q_cluster, 3)).astype(np.float32)
        else:
            ca = rng.choice(len(cluster), size=q_cluster, replace=q_cluster > len(cluster))
            ba = rng.choice(len(background), size=args.queries - q_cluster,
                            replace=args.queries - q_cluster > len(background))
            cq, bq = cluster[ca], background[ba]
        query = np.concatenate((cq, bq), axis=0).copy()
        detail = {"query_source": args.query_source, "cluster_fraction": 0.9,
                  "cluster_side": 1.0, "cluster_queries": q_cluster,
                  "background_queries": args.queries - q_cluster}
    else:
        points = np.full((args.points, 3), args.extent / 2, dtype=np.float32)
        query = np.full((args.queries, 3), args.extent / 2, dtype=np.float32)
        detail = {"query_source": "all coincident with all references",
                  "cluster_fraction": 1.0, "cluster_side": 0.0}
    return np.ascontiguousarray(points), np.ascontiguousarray(query), detail


def _time(fn, repeat: int, mps: bool):
    result = None
    samples = []
    for _ in range(repeat):
        if mps:
            torch.mps.synchronize()
        start = time.perf_counter_ns()
        result = fn()
        if mps:
            torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return result, samples


def _percentiles(values: np.ndarray) -> dict:
    return {key: float(np.percentile(values, pct)) for key, pct in
            (("p50", 50), ("p95", 95), ("max", 100))}


def _hashes() -> dict[str, str]:
    paths = ["bench/bench_spatial_bvh.py", "bench/spatial_bvh.py",
             "bench/spatial_bvh.metal", "bench/spatial_bricks.py",
             "bench/spatial_bricks.metal", "bench/spatial_keys.py",
             "bench/spatial_keys.metal", "mps_pointops/_flat_search_mps.py",
             "mps_pointops/kernels/flat_search.metal"]
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=("uniform", "cluster-sparse", "collapsed"), required=True)
    parser.add_argument("--query-source", choices=("sampled-reference", "independent"),
                        default="sampled-reference")
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--extent", type=float, default=1024.0)
    parser.add_argument("--cell-size", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
        raise RuntimeError("requires PYTORCH_ENABLE_MPS_FALLBACK=0 and PYTORCH_MPS_FAST_MATH=0")
    if not torch.backends.mps.is_available():
        raise RuntimeError("requires physical MPS device")
    if not 1 <= args.points <= 1_000_000 or not 1 <= args.queries <= 65_536 or not 1 <= args.k <= 32:
        raise ValueError("unsupported N, Q or K")
    if not 1 <= args.repeats <= 5:
        raise ValueError("repeats must be 1..5")
    points_cpu, query_cpu, detail = _fixture(args)
    start = time.perf_counter_ns()
    points = torch.from_numpy(points_cpu).to("mps")
    query = torch.from_numpy(query_cpu).to("mps")
    torch.mps.synchronize()
    transfer_ms = (time.perf_counter_ns() - start) / 1e6
    origin = torch.zeros(3, dtype=torch.float32, device="mps")
    px = torch.tensor([0, len(points)], device="mps")
    py = torch.tensor([0, len(query)], device="mps")
    py_warm = torch.tensor([0, 1], device="mps")

    # Compile and warm each path outside timed samples.
    bvh_warm = MortonTwoLevelBVH.build(points, origin, args.cell_size)
    brick_warm = MortonBrickIndex.build(points, origin, args.cell_size)
    bvh_warm.knn(query[:1], args.k)
    if args.queries <= 256:
        bvh_warm.knn(query[:1], args.k, parallel_microtrees=True)
    brick_warm.knn(query[:1], args.k)
    knn_indices(points, query[:1], px, py_warm, args.k)
    torch.mps.synchronize()

    bvh, bvh_build_ms = _time(
        lambda: MortonTwoLevelBVH.build(points, origin, args.cell_size), args.repeats, True)
    brick, brick_build_ms = _time(
        lambda: MortonBrickIndex.build(points, origin, args.cell_size), args.repeats, True)
    (bvh_dist, bvh_idx, bvh_stats), bvh_query_ms = _time(
        lambda: bvh.knn(query, args.k), args.repeats, True)
    split_result = None
    split_query_ms = None
    if args.queries <= 256:
        split_result, split_query_ms = _time(
            lambda: bvh.knn(query, args.k, parallel_microtrees=True),
            args.repeats, True)
    (brick_dist, brick_idx), brick_query_ms = _time(
        lambda: brick.knn(query, args.k), args.repeats, True)
    native_idx, native_query_ms = _time(
        lambda: knn_indices(points, query, px, py, args.k), args.repeats, True)
    tree, cpu_build_ms = _time(lambda: cKDTree(points_cpu), args.repeats, False)
    _, cpu_query_ms = _time(lambda: tree.query(query_cpu, k=args.k, workers=1), args.repeats, False)
    torch.mps.synchronize()

    bvh_brute_mismatch = int(torch.count_nonzero(bvh_idx != native_idx).item())
    split_brute_mismatch = (int(torch.count_nonzero(split_result[1] != native_idx).item())
                            if split_result is not None else None)
    brick_brute_mismatch = int(torch.count_nonzero(brick_idx != native_idx).item())
    bvh_brick_distance_bit_mismatch = int(torch.count_nonzero(
        bvh_dist.view(torch.int32) != brick_dist.view(torch.int32)).item())
    split_brick_distance_bit_mismatch = (int(torch.count_nonzero(
        split_result[0].view(torch.int32) != brick_dist.view(torch.int32)).item())
        if split_result is not None else None)
    stats = bvh_stats.cpu().numpy().astype(np.uint32)
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git("rev-parse", "HEAD"),
        "source_dirty": bool(_git("status", "--porcelain")),
        "source_sha256": _hashes(),
        "hardware": platform.processor(), "macos": platform.mac_ver()[0],
        "torch": torch.__version__, "scipy": __import__("scipy").__version__,
        "mps_fast_math": "0", "mps_fallback": "0",
        "fixture": {"kind": args.distribution, "points": args.points,
                    "queries": args.queries, "k": args.k, "extent": args.extent,
                    "cell_size": args.cell_size, "seed": args.seed, **detail},
        "repeats": args.repeats,
        "input_transfer_ms": transfer_ms,
        "bvh_build_ms": bvh_build_ms, "brick_build_ms": brick_build_ms,
        "bvh_query_ms": bvh_query_ms, "brick_query_ms": brick_query_ms,
        "bvh_split_query_ms": split_query_ms,
        "native_brute_query_ms": native_query_ms,
        "cpu_ckdtree_build_ms": cpu_build_ms, "cpu_ckdtree_query_ms": cpu_query_ms,
        "bvh_index_mismatch_vs_native": bvh_brute_mismatch,
        "bvh_split_index_mismatch_vs_native": split_brute_mismatch,
        "brick_index_mismatch_vs_native": brick_brute_mismatch,
        "bvh_brick_squared_distance_bit_mismatch": bvh_brick_distance_bit_mismatch,
        "bvh_split_brick_squared_distance_bit_mismatch": split_brick_distance_bit_mismatch,
        "bvh_node_visits": _percentiles(stats[:, 0]),
        "bvh_point_visits": _percentiles(stats[:, 1]),
        "bvh_pruned_nodes": _percentiles(stats[:, 2]),
        "bvh_total_fallbacks": int(stats[:, 4].sum()),
        "bvh_node_bytes": int(bvh.bounds.numel() * bvh.bounds.element_size()
                              + bvh.min_index.numel() * bvh.min_index.element_size()),
        "timing_scope": "preloaded inputs, synchronized host wall, compilation excluded; wrappers include validation/synchronization",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for key in ("bvh_build_ms", "brick_build_ms", "bvh_query_ms", "brick_query_ms",
                "native_brute_query_ms", "cpu_ckdtree_build_ms", "cpu_ckdtree_query_ms"):
        print(key, round(statistics.median(result[key]), 3), "ms")
    if split_query_ms is not None:
        print("bvh_split_query_ms", round(statistics.median(split_query_ms), 3), "ms")
    print("index mismatches: BVH", bvh_brute_mismatch, "brick", brick_brute_mismatch,
          "distance bits BVH/brick", bvh_brick_distance_bit_mismatch)
    print("BVH visits", result["bvh_node_visits"], result["bvh_point_visits"])
    print("raw", args.output)
    if (bvh_brute_mismatch or brick_brute_mismatch or bvh_brick_distance_bit_mismatch
            or split_brute_mismatch or split_brick_distance_bit_mismatch):
        raise AssertionError("prototype differs from native full scan or brick distance")


if __name__ == "__main__":
    main()

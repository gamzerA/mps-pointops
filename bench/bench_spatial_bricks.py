"""Synchronized 1M-point Morton-brick kNN research benchmark.

Run Safe and Fast in separate processes with MPS fallback disabled. Example:

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_spatial_bricks.py --distribution uniform \
      --output bench/results/local/brick-uniform-safe.json

This is not a release benchmark or an LBVH. It measures preloaded CPU/MPS
coordinates, separates input transfer, and records sampled allocator memory
(which is not a peak measurement). The Fast path scans every brick because
float reassociation invalidates the current Safe AABB-pruning proof.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
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
from bench.spatial_bricks import MortonBrickIndex  # noqa: E402
from bench.spatial_selected_distances import selected_sqdist_f32  # noqa: E402
from mps_pointops._flat_search_mps import knn_indices  # noqa: E402

MAX_TIMED_PAIRS = 2_048_000_000  # N=1M, Q=2048; higher Q needs an explicit study.
MAX_OPT_IN_PAIRS = 65_536_000_000  # N=1M, Q=65536.
MAX_OPT_IN_REPEATS = 5


def _command(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False, cwd=ROOT)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _source_hashes() -> dict[str, str]:
    names = (
        "bench/bench_spatial_bricks.py", "bench/spatial_bricks.py",
        "bench/spatial_bricks.metal", "bench/spatial_keys.py",
        "bench/spatial_keys.metal", "bench/spatial_selected_distances.py",
        "bench/spatial_selected_distances.metal", "mps_pointops/_flat_search_mps.py",
        "mps_pointops/kernels/flat_search.metal",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def _time(fn, *, repeats: int, mps: bool):
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        if mps:
            torch.mps.synchronize()
        start = time.perf_counter_ns()
        result = fn()
        if mps:
            torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return result, samples


def _memory() -> dict[str, int]:
    return {
        "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
        "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _sampled_highwater_pass(points_mps, query_mps, origin, ptr_x, ptr_y, args):
    """Sample allocation after stages, outside the timed repetitions."""
    samples = [{"stage": "before", **_memory()}]
    index = MortonBrickIndex.build(points_mps, origin, args.cell_size)
    torch.mps.synchronize()
    samples.append({"stage": "after_brick_build", **_memory()})
    index.knn(query_mps, args.k)
    torch.mps.synchronize()
    samples.append({"stage": "after_brick_query", **_memory()})
    knn_indices(points_mps, query_mps, ptr_x, ptr_y, args.k)
    torch.mps.synchronize()
    samples.append({"stage": "after_native_brute_query", **_memory()})
    highwater = {
        key: max(sample[key] for sample in samples)
        for key in ("current_allocated_bytes", "driver_allocated_bytes")
    }
    return samples, highwater


def _fixture(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    rng = np.random.default_rng(args.seed)
    extent = np.float32(args.extent)
    if args.distribution == "uniform":
        points = rng.uniform(0, float(extent), size=(args.points, 3)).astype(np.float32)
        if args.query_source == "sampled-reference":
            chosen = rng.choice(args.points, size=args.queries, replace=args.queries > args.points)
            query = points[chosen].copy()
        else:
            query = rng.uniform(0, float(extent), size=(args.queries, 3)).astype(np.float32)
        params = {"cube": [0.0, float(extent)], "query_source": args.query_source}
    else:
        clustered = int(args.points * args.cluster_fraction)
        center = extent / np.float32(2)
        cluster = (center + rng.uniform(0, args.cluster_side, size=(clustered, 3))).astype(np.float32)
        background = rng.uniform(0, float(extent), size=(args.points - clustered, 3)).astype(np.float32)
        points = np.concatenate((cluster, background), axis=0)
        cluster_queries = args.queries // 2
        if args.query_source == "sampled-reference":
            cluster_rows = rng.choice(clustered, size=cluster_queries,
                                      replace=cluster_queries > clustered)
            background_rows = rng.choice(len(background), size=args.queries - cluster_queries,
                                         replace=args.queries - cluster_queries > len(background))
            query = np.concatenate((cluster[cluster_rows], background[background_rows]), axis=0).copy()
        else:
            dense_query = (center + rng.uniform(0, args.cluster_side,
                                               size=(cluster_queries, 3))).astype(np.float32)
            background_query = rng.uniform(0, float(extent),
                                           size=(args.queries - cluster_queries, 3)).astype(np.float32)
            query = np.concatenate((dense_query, background_query), axis=0)
        params = {
            "cube": [0.0, float(extent)], "cluster_center": [float(center)] * 3,
            "cluster_side": args.cluster_side, "cluster_fraction": args.cluster_fraction,
            "cluster_points": clustered, "background_points": len(background),
            "cluster_queries": cluster_queries, "background_queries": args.queries - cluster_queries,
            "query_source": args.query_source,
        }
    return np.ascontiguousarray(points), np.ascontiguousarray(query), params


def _selected_sqdist_f32(points: np.ndarray, query: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.empty(indices.shape, dtype=np.float32)
    for row in range(len(query)):
        selected = points[indices[row]]
        dx = np.subtract(selected[:, 0], query[row, 0], dtype=np.float32)
        dy = np.subtract(selected[:, 1], query[row, 1], dtype=np.float32)
        dz = np.subtract(selected[:, 2], query[row, 2], dtype=np.float32)
        d = np.multiply(dx, dx, dtype=np.float32)
        d = np.add(d, np.multiply(dy, dy, dtype=np.float32), dtype=np.float32)
        result[row] = np.add(d, np.multiply(dz, dz, dtype=np.float32), dtype=np.float32)
    return result


def _classify_scipy_differences(points: np.ndarray, query: np.ndarray,
                                metal: np.ndarray, scipy: np.ndarray) -> dict:
    """Separate exact-tie ordering differences from distance disagreements."""
    rows, slots = np.nonzero(metal != scipy)
    equal_f32 = 0
    equal_f64 = 0
    examples = []
    for row, slot in zip(rows, slots):
        a, b = int(metal[row, slot]), int(scipy[row, slot])
        d32 = _selected_sqdist_f32(points, query[row:row + 1],
                                    np.array([[a, b]], dtype=np.int64))[0]
        da = points[a].astype(np.float64) - query[row].astype(np.float64)
        db = points[b].astype(np.float64) - query[row].astype(np.float64)
        d64a, d64b = float(np.dot(da, da)), float(np.dot(db, db))
        equal_f32 += int(d32[0] == d32[1])
        equal_f64 += int(d64a == d64b)
        if len(examples) < 8:
            examples.append({"query": int(row), "slot": int(slot),
                             "metal_index": a, "scipy_index": b,
                             "metal_squared_f32": float(d32[0]),
                             "scipy_squared_f32": float(d32[1]),
                             "metal_squared_f64": d64a,
                             "scipy_squared_f64": d64b})
    return {"mismatched_slots": len(rows), "equal_squared_float32": equal_f32,
            "equal_squared_float64": equal_f64, "examples": examples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=("uniform", "cluster-sparse"), required=True)
    parser.add_argument("--query-source", choices=("sampled-reference", "independent"),
                        default="sampled-reference")
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=16)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--extent", type=float, default=1024.0)
    parser.add_argument("--cell-size", type=float, default=16.0)
    parser.add_argument("--cluster-fraction", type=float, default=0.9)
    parser.add_argument("--cluster-side", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--allow-large-q", action="store_true",
                        help="permit up to 65.536B pairs per run, at most five repeats")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    if os.environ.get("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
        raise RuntimeError("set PYTORCH_MPS_FAST_MATH=0 or 1 before starting Python")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if args.points < 1 or args.queries < 1 or args.repeats < 1:
        raise ValueError("points, queries, and repeats must be positive")
    pairs = args.points * args.queries
    if pairs > MAX_TIMED_PAIRS and not (args.allow_large_q and args.repeats <= MAX_OPT_IN_REPEATS
                                       and pairs <= MAX_OPT_IN_PAIRS):
        raise ValueError(f"N*Q exceeds the {MAX_TIMED_PAIRS:,} default guard; "
                         f"use --allow-large-q --repeats <= {MAX_OPT_IN_REPEATS} "
                         f"up to {MAX_OPT_IN_PAIRS:,} pairs per repeat")
    if not 1 <= args.k <= min(32, args.points):
        raise ValueError("k must be in [1, min(32, points)]")
    if not 0 < args.cluster_fraction < 1 or not 0 < args.cluster_side < args.extent:
        raise ValueError("cluster parameters must fit the positive extent")
    if args.distribution == "cluster-sparse":
        clustered = int(args.points * args.cluster_fraction)
        if clustered < 1 or args.points - clustered < 1:
            raise ValueError("cluster and background must both contain points")
    if not 0 < args.cell_size <= args.extent:
        raise ValueError("cell size must be in (0, extent]")

    points, query, distribution = _fixture(args)
    started = time.perf_counter_ns()
    points_mps = torch.from_numpy(points).to("mps")
    query_mps = torch.from_numpy(query).to("mps")
    torch.mps.synchronize()
    transfer_ms = (time.perf_counter_ns() - started) / 1e6
    ptr_x = torch.tensor([0, args.points], device="mps")
    ptr_y = torch.tensor([0, args.queries], device="mps")
    origin = torch.zeros(3, device="mps")

    # Compile both Metal kernels before timing. Index construction includes
    # validity checks, stable sort, and AABB build. Query includes the wrapper
    # coordinate validation and its synchronization.
    warm = MortonBrickIndex.build(points_mps, origin, args.cell_size)
    warm.knn(query_mps[:1], args.k)
    knn_indices(points_mps, query_mps[:1], ptr_x, torch.tensor([0, 1], device="mps"), args.k)
    torch.mps.synchronize()
    memory_before = _memory()
    index, mps_build = _time(
        lambda: MortonBrickIndex.build(points_mps, origin, args.cell_size),
        repeats=args.repeats, mps=True,
    )
    assert isinstance(index, MortonBrickIndex)
    memory_after_build = _memory()
    (brick_dist, brick_idx), brick_query = _time(
        lambda: index.knn(query_mps, args.k), repeats=args.repeats, mps=True,
    )
    native_idx, native_query = _time(
        lambda: knn_indices(points_mps, query_mps, ptr_x, ptr_y, args.k),
        repeats=args.repeats, mps=True,
    )
    memory_after_query = _memory()
    memory_pass, sampled_highwater = _sampled_highwater_pass(
        points_mps, query_mps, origin, ptr_x, ptr_y, args,
    )

    tree, cpu_build = _time(lambda: cKDTree(points), repeats=args.repeats, mps=False)
    assert isinstance(tree, cKDTree)
    (cpu_dist, cpu_idx), cpu_query = _time(
        lambda: tree.query(query, k=args.k, eps=0.0, workers=1),
        repeats=args.repeats, mps=False,
    )
    torch.mps.synchronize()
    brick_idx_np = brick_idx.cpu().numpy()
    native_idx_np = native_idx.cpu().numpy()
    brick_d_np = brick_dist.cpu().numpy()
    direct_mps_sq = selected_sqdist_f32(query_mps, points_mps, brick_idx)
    torch.mps.synchronize()
    direct_mps_sq_np = direct_mps_sq.cpu().numpy()
    cpu_idx_np = np.asarray(cpu_idx).reshape(args.queries, args.k)
    expected_sq = _selected_sqdist_f32(points, query, brick_idx_np)
    mismatched_native = int(np.count_nonzero(brick_idx_np != native_idx_np))
    mismatched_scipy = int(np.count_nonzero(brick_idx_np != cpu_idx_np))
    scipy_diagnostics = _classify_scipy_differences(points, query, brick_idx_np, cpu_idx_np)
    mismatched_sq_bits = int(np.count_nonzero(brick_d_np.view(np.uint32) != expected_sq.view(np.uint32)))
    mismatched_same_mode_sq_bits = int(np.count_nonzero(
        brick_d_np.view(np.uint32) != direct_mps_sq_np.view(np.uint32)))
    max_sq_error = float(np.max(np.abs(brick_d_np - expected_sq)))

    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(_command("git", "status", "--porcelain")),
        "source_sha256": _source_hashes(),
        "hardware": _command("sysctl", "-n", "machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0], "python": platform.python_version(),
        "torch": torch.__version__, "scipy": __import__("scipy").__version__,
        "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "fixture": {"kind": args.distribution, "points": args.points, "queries": args.queries,
                    "k": args.k, "extent": args.extent, "cell_size": args.cell_size,
                    "seed": args.seed, **distribution},
        "large_q_opt_in": args.allow_large_q,
        "pruning_enabled": index.pruning_enabled,
        "input_transfer_ms": transfer_ms,
        "mps_brick_build_ms": mps_build,
        "mps_brick_query_ms": brick_query,
        "mps_native_brute_query_ms": native_query,
        "cpu_ckdtree_build_ms": cpu_build,
        "cpu_ckdtree_query_ms": cpu_query,
        "memory_before_build": memory_before,
        "memory_after_build": memory_after_build,
        "memory_after_query": memory_after_query,
        "memory_pass_samples": memory_pass,
        "sampled_allocator_highwater_bytes": sampled_highwater,
        "sampled_allocator_highwater_is_true_gpu_peak": False,
        "cpu_process_lifetime_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "mismatched_indices_vs_native_brute": mismatched_native,
        "mismatched_indices_vs_ckdtree_diagnostic": mismatched_scipy,
        "ckdtree_difference_classification": scipy_diagnostics,
        "mismatched_sqdist_bits_vs_cpu_float32_diagnostic": mismatched_sq_bits,
        "mismatched_sqdist_bits_vs_independent_metal_probe": mismatched_same_mode_sq_bits,
        "max_abs_sqdist_error_vs_cpu_float32": max_sq_error,
        "timing_scope": "preloaded resident coordinates; synchronized host wall; shader compilation excluded; separate allocator sampling pass",
        "memory_scope": "sampled after stages; not true transient GPU peak; CPU RSS covers process lifetime",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for key in ("mps_brick_build_ms", "mps_brick_query_ms", "mps_native_brute_query_ms",
                "cpu_ckdtree_build_ms", "cpu_ckdtree_query_ms"):
        print(f"{key}: median={statistics.median(result[key]):.3f} ms; raw={result[key]}")
    print(f"native index mismatch={mismatched_native}; SciPy diagnostic mismatch={mismatched_scipy}; "
          f"CPU-f32 squared-bit mismatch={mismatched_sq_bits}; "
          f"same-mode Metal squared-bit mismatch={mismatched_same_mode_sq_bits}")
    print(f"output: {args.output}")
    if mismatched_native:
        raise AssertionError(f"brick kNN differs from native full-scan Metal in {mismatched_native} slots")


if __name__ == "__main__":
    main()

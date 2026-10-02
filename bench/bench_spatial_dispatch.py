"""Measure public SpatialIndex dispatch against explicit scan and BVH paths.

Run one fixture per fresh Safe-Math process. Inputs are already on MPS for
query timings; the initial public call is reported separately because it can
sample Morton keys and build an index. This is a measured-policy audit, not a
claim that BVH is faster for every point cloud or Apple Silicon device.
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
from typing import Callable, TypeVar

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.bench_spatial_bvh import _fixture  # noqa: E402
from mps_pointops.spatial import SpatialIndex  # noqa: E402

T = TypeVar("T")


def _command(*argv: str) -> str:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _hashes() -> dict[str, str]:
    paths = (
        "bench/bench_spatial_dispatch.py", "bench/bench_spatial_bvh.py",
        "mps_pointops/spatial.py", "mps_pointops/_spatial_bvh.py",
        "mps_pointops/_spatial_keys.py", "mps_pointops/ops.py",
        "mps_pointops/kernels/spatial_bvh.metal",
        "mps_pointops/kernels/spatial_keys.metal",
        "mps_pointops/kernels/knn.metal",
    )
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in paths}


def _memory() -> dict[str, int]:
    return {
        "current_tensor_bytes": int(torch.mps.current_allocated_memory()),
        "driver_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _time(fn: Callable[[], T], repeat: int) -> tuple[T, list[float], list[dict[str, int]]]:
    result: T
    samples = []
    allocator_peaks = []
    peak_api = all(hasattr(torch.accelerator, name) for name in
                   ("reset_peak_memory_stats", "max_memory_allocated",
                    "max_memory_reserved"))
    for _ in range(repeat):
        torch.mps.synchronize()
        if peak_api:
            torch.accelerator.reset_peak_memory_stats()
        start = time.perf_counter_ns()
        result = fn()
        torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
        if peak_api:
            allocator_peaks.append({
                "tensor_bytes": int(torch.accelerator.max_memory_allocated()),
                "reserved_bytes": int(torch.accelerator.max_memory_reserved()),
            })
    return result, samples, allocator_peaks


def _max_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    if not actual.numel():
        return 0.0
    return float((actual - reference).abs().max().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=("uniform", "cluster-sparse", "collapsed"),
                        required=True)
    parser.add_argument("--query-source", choices=("sampled-reference", "independent"),
                        default="sampled-reference")
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--extent", type=float, default=1024.0)
    parser.add_argument("--cell-size", type=float,
                        help="default: 16 for uniform/collapsed; 1/64 for cluster-sparse")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 and PYTORCH_MPS_FAST_MATH=0 before Python starts")
    if not torch.backends.mps.is_available():
        raise RuntimeError("requires a physical MPS device")
    if not 1 <= args.points <= 1_000_000 or not 1 <= args.queries <= 65_536 or not 1 <= args.k <= 32:
        raise ValueError("requires N=1..1M, Q=1..65536, K=1..32")
    if not 1 <= args.repeats <= 5:
        raise ValueError("repeats must be 1..5")
    if args.cell_size is None:
        args.cell_size = 1 / 64 if args.distribution == "cluster-sparse" else 16.0
    if args.output.exists():
        raise FileExistsError(args.output)

    points_cpu, query_cpu, detail = _fixture(args)
    # Compilation is excluded from all measured stages.
    warm_points = torch.zeros((128, 3), device="mps")
    warm_query = torch.zeros((1, 3), device="mps")
    warm_origin = torch.zeros(3, device="mps")
    SpatialIndex(warm_points, backend="scan").knn(warm_query, args.k)
    SpatialIndex(warm_points, backend="bvh", origin=warm_origin,
                 cell_size=1.0).knn(warm_query, args.k)
    torch.mps.synchronize()
    del warm_points, warm_query, warm_origin
    torch.mps.empty_cache()
    torch.mps.synchronize()
    memory_after_warmup = _memory()

    def transfer():
        points = torch.from_numpy(points_cpu).to("mps")
        query = torch.from_numpy(query_cpu).to("mps")
        origin = torch.zeros(3, device="mps", dtype=torch.float32)
        return points, query, origin

    (points, query, origin), transfer_ms, transfer_peaks = _time(transfer, 1)
    memory_after_transfer = _memory()
    common = {"origin": origin, "cell_size": args.cell_size}
    auto = SpatialIndex(points, backend="auto", **common)
    scan = SpatialIndex(points, backend="scan", **common)
    bvh = SpatialIndex(points, backend="bvh", **common)

    # First call includes adaptive decision and, if chosen, BVH construction.
    auto_first, auto_first_ms, auto_first_peaks = _time(
        lambda: auto.knn(query, args.k), 1)
    auto_selected = "bvh" if auto._bvh is not None else "scan"
    hot_cell_fraction = auto._hot_cell_fraction
    memory_after_auto_first = _memory()
    _, bvh_build_ms, bvh_build_peaks = _time(lambda: bvh._get_bvh(), 1)
    memory_after_bvh_build = _memory()
    scan_result, scan_query_ms, scan_peaks = _time(
        lambda: scan.knn(query, args.k), args.repeats)
    bvh_result, bvh_query_ms, bvh_query_peaks = _time(
        lambda: bvh.knn(query, args.k), args.repeats)
    auto_result, auto_query_ms, auto_query_peaks = _time(
        lambda: auto.knn(query, args.k), args.repeats)
    torch.mps.synchronize()
    memory_after_queries = _memory()

    auto_first_d, auto_first_i = auto_first
    scan_d, scan_i = scan_result
    bvh_d, bvh_i = bvh_result
    auto_d, auto_i = auto_result
    mismatch = {
        "auto_first_vs_scan_indices": int(torch.count_nonzero(auto_first_i != scan_i).item()),
        "auto_vs_scan_indices": int(torch.count_nonzero(auto_i != scan_i).item()),
        "bvh_vs_scan_indices": int(torch.count_nonzero(bvh_i != scan_i).item()),
    }
    distance_error = {
        "auto_first_vs_scan_max_abs": _max_error(auto_first_d, scan_d),
        "auto_vs_scan_max_abs": _max_error(auto_d, scan_d),
        "bvh_vs_scan_max_abs": _max_error(bvh_d, scan_d),
    }
    storage = bvh._get_bvh()
    bvh_storage = sum(t.numel() * t.element_size() for t in
                      (storage.sorted_indices, storage.bounds, storage.min_index))
    source_status = _command("git", "status", "--porcelain")
    result = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(source_status),
        "source_status": source_status.splitlines(),
        "source_sha256": _hashes(),
        "hardware": _command("sysctl", "-n", "machdep.cpu.brand_string"),
        "physical_ram_bytes": _command("sysctl", "-n", "hw.memsize"),
        "macos": platform.mac_ver()[0],
        "torch": torch.__version__,
        "mps_fast_math": "0", "mps_fallback": "0",
        "fixture": {"distribution": args.distribution, "points": args.points,
                    "queries": args.queries, "k": args.k, "extent": args.extent,
                    "cell_size": args.cell_size, "seed": args.seed, **detail},
        "repeats": args.repeats,
        "adaptive_decision": {
            "selected_backend": auto_selected,
            "sample_hot_cell_fraction": hot_cell_fraction,
            "policy_note": "M5 Pro Safe Math only; N=1M, K=16, Q>=8192 and sampled Morton max-cell fraction<0.05",
        },
        "input_transfer_ms": transfer_ms,
        "auto_first_call_ms_including_decision_and_optional_build": auto_first_ms,
        "explicit_bvh_build_ms": bvh_build_ms,
        "scan_steady_query_ms": scan_query_ms,
        "bvh_steady_query_ms": bvh_query_ms,
        "auto_steady_query_ms": auto_query_ms,
        "index_mismatch": mismatch,
        "euclidean_distance_max_abs_error": distance_error,
        "bvh_resident_index_tensor_bytes": bvh_storage,
        "memory_samples": {
            "after_warmup": memory_after_warmup,
            "after_transfer": memory_after_transfer,
            "after_auto_first": memory_after_auto_first,
            "after_bvh_build": memory_after_bvh_build,
            "after_queries": memory_after_queries,
        },
        "torch_allocator_peaks_by_stage": {
            "transfer": transfer_peaks, "auto_first": auto_first_peaks,
            "explicit_bvh_build": bvh_build_peaks,
            "scan_query": scan_peaks, "bvh_query": bvh_query_peaks,
            "auto_query": auto_query_peaks,
        },
        "timing_scope": "preloaded MPS inputs; synchronize before and after each host-wall sample; compilation excluded; query includes public wrapper and its validation",
        "memory_scope": "PyTorch allocator tensor/reserved peak counters and stage-sampled MPS driver bytes; neither is total GPU memory or transient driver peak",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.distribution} N={args.points} Q={args.queries} K={args.k}: auto={auto_selected}, hot-cell={hot_cell_fraction}")
    for key in ("auto_first_call_ms_including_decision_and_optional_build",
                "explicit_bvh_build_ms", "scan_steady_query_ms",
                "bvh_steady_query_ms", "auto_steady_query_ms"):
        print(key, round(statistics.median(result[key]), 3), "ms")
    print("index mismatches", mismatch, "distance max abs", distance_error)
    print("raw", args.output)
    if any(mismatch.values()):
        raise AssertionError("public SpatialIndex paths disagree on neighbor indices")


if __name__ == "__main__":
    main()

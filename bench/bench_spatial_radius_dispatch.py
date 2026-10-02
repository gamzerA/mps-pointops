"""Public first-K Ball Query: MPS full scan versus the guarded Metal BVH.

One fresh Safe-Math process measures one synthetic fixture. Uniform and mixed
queries are independent of references. The all-coincident stress fixture uses
half coincident and half displaced queries to exercise both full and empty
rows; these measurements are not a model workload performance claim.

Use ``--allow-homogeneous-rows`` when probing a large-radius case in which
every query fills its first-K row. The option only relaxes the fixture-shape
assertion; index, distance-bit, and output-order checks still apply.
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

import numpy as np
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
        "bench/bench_spatial_radius_dispatch.py", "bench/bench_spatial_bvh.py",
        "mps_pointops/spatial.py", "mps_pointops/_spatial_bvh.py",
        "mps_pointops/_spatial_keys.py", "mps_pointops/ops.py",
        "mps_pointops/_ball_query_mps.py",
        "mps_pointops/kernels/spatial_bvh.metal",
        "mps_pointops/kernels/spatial_keys.metal",
        "mps_pointops/kernels/ball_query.metal",
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
    samples: list[float] = []
    peaks: list[dict[str, int]] = []
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
            peaks.append({
                "tensor_bytes": int(torch.accelerator.max_memory_allocated()),
                "reserved_bytes": int(torch.accelerator.max_memory_reserved()),
            })
    return result, samples, peaks


def _row_counts(indices: torch.Tensor) -> dict[str, int]:
    valid = (indices >= 0).sum(dim=1)
    k = indices.shape[1]
    return {
        "zero_neighbor_rows": int((valid == 0).sum().item()),
        "partially_filled_rows": int(((valid > 0) & (valid < k)).sum().item()),
        "first_k_full_rows": int((valid == k).sum().item()),
        "total_valid_slots": int(valid.sum().item()),
    }


def _order_violations(indices: torch.Tensor) -> int:
    if indices.shape[1] < 2:
        return 0
    left = indices[:, :-1]
    right = indices[:, 1:]
    return int(((right >= 0) & (left >= right)).sum().item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distribution", choices=("uniform", "cluster-sparse", "collapsed"),
                        required=True)
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--radius", type=float, required=True)
    parser.add_argument("--extent", type=float, default=1024.0)
    parser.add_argument("--cell-size", type=float,
                        help="default: 16 for uniform/collapsed; 1/64 for cluster-sparse")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--allow-homogeneous-rows", action="store_true",
                        help="permit all-full or all-empty rows in radius crossover probes")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 and PYTORCH_MPS_FAST_MATH=0 before Python starts")
    if not torch.backends.mps.is_available():
        raise RuntimeError("requires a physical MPS device")
    if not 1 <= args.points <= 1_000_000 or not 1 <= args.queries <= 65_536 or not 1 <= args.limit <= 32:
        raise ValueError("requires N=1..1M, Q=1..65536, limit=1..32")
    if not 0.0 < args.radius <= args.extent:
        raise ValueError("radius must be in (0,extent]")
    if not 1 <= args.repeats <= 5:
        raise ValueError("repeats must be 1..5")
    if args.cell_size is None:
        args.cell_size = 1 / 64 if args.distribution == "cluster-sparse" else 16.0
    if args.output.exists():
        raise FileExistsError(args.output)

    # Independent queries expose zero-hit rows without changing the source
    # distribution. All-coincident references need a separate displaced half.
    args.query_source = "independent"
    points_cpu, query_cpu, detail = _fixture(args)
    if args.distribution == "collapsed":
        query_cpu[args.queries // 2:, 0] = np.float32(args.extent / 2 + 1)
        detail = {"query_source": "first half coincident, second half displaced +1 on x",
                  "cluster_fraction": 1.0, "cluster_side": 0.0}
    query_cpu = np.ascontiguousarray(query_cpu)

    # Compile both shader paths once on a tiny cloud outside timed samples.
    warm_points = torch.zeros((128, 3), device="mps")
    warm_query = torch.zeros((2, 3), device="mps")
    warm_origin = torch.zeros(3, device="mps")
    SpatialIndex(warm_points, backend="scan").ball_query(warm_query, args.radius, args.limit)
    SpatialIndex(warm_points, backend="bvh", origin=warm_origin,
                 cell_size=1.0).ball_query(warm_query, args.radius, args.limit)
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
    scan = SpatialIndex(points, backend="scan", **common)
    bvh = SpatialIndex(points, backend="bvh", **common)
    auto = SpatialIndex(points, backend="auto", **common)

    bvh_first, bvh_first_ms, bvh_first_peaks = _time(
        lambda: bvh.ball_query(query, args.radius, args.limit), 1)
    memory_after_bvh_first = _memory()
    scan_result, scan_ms, scan_peaks = _time(
        lambda: scan.ball_query(query, args.radius, args.limit), args.repeats)
    bvh_result, bvh_ms, bvh_peaks = _time(
        lambda: bvh.ball_query(query, args.radius, args.limit), args.repeats)
    auto_result, auto_ms, auto_peaks = _time(
        lambda: auto.ball_query(query, args.radius, args.limit), args.repeats)
    torch.mps.synchronize()
    memory_after_queries = _memory()

    scan_d, scan_i = scan_result
    bvh_d, bvh_i = bvh_result
    auto_d, auto_i = auto_result
    first_d, first_i = bvh_first
    mismatch = {
        "bvh_first_vs_scan_indices": int(torch.count_nonzero(first_i != scan_i).item()),
        "bvh_steady_vs_scan_indices": int(torch.count_nonzero(bvh_i != scan_i).item()),
        "auto_vs_scan_indices": int(torch.count_nonzero(auto_i != scan_i).item()),
        "bvh_first_vs_scan_squared_distance_bits": int(torch.count_nonzero(
            first_d.view(torch.int32) != scan_d.view(torch.int32)).item()),
        "bvh_steady_vs_scan_squared_distance_bits": int(torch.count_nonzero(
            bvh_d.view(torch.int32) != scan_d.view(torch.int32)).item()),
        "auto_vs_scan_squared_distance_bits": int(torch.count_nonzero(
            auto_d.view(torch.int32) != scan_d.view(torch.int32)).item()),
    }
    row_counts = _row_counts(scan_i)
    order_violations = _order_violations(scan_i)
    bvh_storage = bvh._get_bvh()
    bvh_storage_bytes = sum(t.numel() * t.element_size() for t in
                            (bvh_storage.sorted_indices, bvh_storage.bounds,
                             bvh_storage.min_index))
    source_status = _command("git", "status", "--porcelain")
    radius32 = np.float32(args.radius)
    radius2 = np.float32(radius32 * radius32)
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
                    "queries": args.queries, "limit": args.limit,
                    "radius_python_float_repr": repr(args.radius),
                    "radius_float32": float(radius32),
                    "radius_squared_float32": float(radius2),
                    "extent": args.extent, "cell_size": args.cell_size,
                    "seed": args.seed,
                    **detail},
        "repeats": args.repeats,
        "mixed_rows_required": not args.allow_homogeneous_rows,
        "auto_selected_backend": "bvh" if auto._bvh is not None else "scan",
        "row_counts": row_counts,
        "original_index_order_violations": order_violations,
        "mismatch": mismatch,
        "input_transfer_ms": transfer_ms,
        "bvh_first_call_including_build_ms": bvh_first_ms,
        "scan_steady_query_ms": scan_ms,
        "bvh_steady_query_ms": bvh_ms,
        "auto_steady_query_ms": auto_ms,
        "bvh_resident_index_tensor_bytes": bvh_storage_bytes,
        "memory_samples": {
            "after_warmup": memory_after_warmup,
            "after_transfer": memory_after_transfer,
            "after_bvh_first": memory_after_bvh_first,
            "after_queries": memory_after_queries,
        },
        "torch_allocator_peaks_by_stage": {
            "transfer": transfer_peaks,
            "bvh_first": bvh_first_peaks,
            "scan_query": scan_peaks,
            "bvh_query": bvh_peaks,
            "auto_query": auto_peaks,
        },
        "timing_scope": "preloaded MPS inputs; torch.mps.synchronize before/after every sample; shader compilation excluded; first BVH call includes index construction and validation",
        "memory_scope": "PyTorch tensor/reserved allocator peaks and stage-sampled driver bytes; not whole-system GPU memory or transient driver peak",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.distribution} Q={args.queries} R={args.radius}: rows={row_counts}, auto={result['auto_selected_backend']}")
    for key in ("bvh_first_call_including_build_ms", "scan_steady_query_ms",
                "bvh_steady_query_ms", "auto_steady_query_ms"):
        print(key, round(statistics.median(result[key]), 3), "ms")
    print("mismatch", mismatch, "order violations", order_violations)
    print("raw", args.output)
    if any(mismatch.values()) or order_violations:
        raise AssertionError("public Ball Query paths differ in indices, squared bits, or first-K order")
    if (not args.allow_homogeneous_rows and
            (row_counts["first_k_full_rows"] == 0 or row_counts["zero_neighbor_rows"] == 0)):
        raise AssertionError("fixture did not exercise both full first-K and empty rows")


if __name__ == "__main__":
    main()

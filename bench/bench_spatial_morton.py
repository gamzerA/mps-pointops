"""Reproducible v0.9 experimental sorted-Morton radius comparison.

Run in a fresh process per math mode, for example:

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_spatial_morton.py --output bench/results/morton-safe.json

The input is uniform synthetic float32 XYZ; it is not a real point cloud.
Build and query times use preloaded CPU/MPS coordinates and synchronize MPS.
The MPS index is experimental, single-cloud only, and uses one thread/query.
The CPU result includes cKDTree build. No CPU/MPS transfer is counted in
build-plus-query; the separate transfer timing is reported in the JSON.
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
from bench.spatial_radius import SortedMortonIndex  # noqa: E402
from mps_pointops._flat_search_mps import radius_indices  # noqa: E402


def _command(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _sync() -> None:
    torch.mps.synchronize()


def _timed(fn, repeats: int, *, mps: bool) -> tuple[object, list[float]]:
    samples = []
    result = None
    for _ in range(repeats):
        if mps:
            _sync()
        started = time.perf_counter_ns()
        result = fn()
        if mps:
            _sync()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return result, samples


def _sources() -> dict[str, str]:
    files = (
        "bench/bench_spatial_morton.py", "bench/spatial_keys.py",
        "bench/spatial_keys.metal", "bench/spatial_radius.py",
        "bench/spatial_radius.metal", "mps_pointops/kernels/flat_search.metal",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--extent", type=float, default=1024.0)
    parser.add_argument("--distribution", choices=("uniform", "clustered"), default="uniform")
    parser.add_argument("--radius", type=float, default=16.0)
    parser.add_argument("--cell-size", type=float, default=16.0)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    if os.environ.get("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
        raise RuntimeError("set PYTORCH_MPS_FAST_MATH=0 or 1")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if args.points < 1 or args.queries < 1 or args.repeats < 1:
        raise ValueError("points, queries, and repeats must be positive")
    if not 0 < args.radius <= args.extent or not 0 < args.cell_size <= args.extent:
        raise ValueError("radius and cell size must be positive and at most extent")
    if not 1 <= args.limit <= 256:
        raise ValueError("limit must be in [1, 256]")

    generator = torch.Generator().manual_seed(20261002)
    if args.distribution == "uniform":
        ref_cpu = (torch.rand((args.points, 3), generator=generator) * args.extent).contiguous()
        query_cpu = (torch.rand((args.queries, 3), generator=generator) * args.extent).contiguous()
    else:
        # All references occupy a single coarse cell when cell_size exceeds
        # extent / 1024. This exposes long serial cell-list scans.
        center = args.extent / 2
        span = min(args.cell_size / 16, args.extent / 1024)
        ref_cpu = (torch.rand((args.points, 3), generator=generator) * span + center).contiguous()
        query_cpu = (torch.rand((args.queries, 3), generator=generator) * span + center).contiguous()
    started = time.perf_counter_ns()
    ref_mps = ref_cpu.to("mps")
    query_mps = query_cpu.to("mps")
    _sync()
    input_transfer_ms = (time.perf_counter_ns() - started) / 1e6

    ref_numpy = ref_cpu.numpy()
    query_numpy = query_cpu.numpy()
    tree, cpu_build = _timed(lambda: cKDTree(ref_numpy), args.repeats, mps=False)
    assert isinstance(tree, cKDTree)
    cpu_neighbors, cpu_query = _timed(
        lambda: tree.query_ball_point(query_numpy, args.radius, workers=1,
                                      return_sorted=True), args.repeats, mps=False
    )

    origin = torch.zeros(3, dtype=torch.float32, device="mps")
    # Compile and warm both shaders before measured runs.
    warm_index = SortedMortonIndex.build(ref_mps, origin, args.cell_size)
    warm_index.radius(query_mps[:1], args.radius, args.limit)
    _sync()
    index, gpu_build = _timed(
        lambda: SortedMortonIndex.build(ref_mps, origin, args.cell_size),
        args.repeats, mps=True,
    )
    assert isinstance(index, SortedMortonIndex)
    (actual, status), gpu_query = _timed(
        lambda: index.radius(query_mps, args.radius, args.limit),
        args.repeats, mps=True,
    )
    _sync()
    status_cpu = status.cpu().numpy()
    if np.any(status_cpu):
        raise AssertionError(f"spatial query requested fallback for {int(np.count_nonzero(status_cpu))} rows")
    actual_cpu = actual.cpu().numpy()
    expected = np.full((args.queries, args.limit), -1, dtype=np.int64)
    for row, candidates in enumerate(cpu_neighbors):
        # SciPy distance arithmetic is float64. Random fixtures should have
        # no borderline pairs; retain any mismatch as a failed experiment.
        kept = candidates[:args.limit]
        expected[row, :len(kept)] = kept
    mismatches = int(np.count_nonzero(actual_cpu != expected))
    if mismatches:
        raise AssertionError(f"sorted-Morton vs cKDTree first-K mismatch: {mismatches} slots")

    # A small direct check preserves the package's actual float32/FMA radius
    # contract without performing Q=N=1M brute force.
    sample_queries = min(args.queries, 8)
    ptr_x = torch.tensor([0, args.points], device="mps")
    ptr_y = torch.tensor([0, sample_queries], device="mps")
    direct = radius_indices(ref_mps, query_mps[:sample_queries], ptr_x, ptr_y,
                            args.radius, args.limit)
    _sync()
    direct_equal = bool(torch.equal(direct.cpu(), actual[:sample_queries].cpu()))
    if not direct_equal:
        raise AssertionError("sorted-Morton differs from native brute-force radius on sample queries")

    result = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(_command("git", "status", "--porcelain")),
        "source_sha256": _sources(),
        "hardware": _command("sysctl", "-n", "machdep.cpu.brand_string"),
        "macos": platform.mac_ver()[0], "python": platform.python_version(),
        "torch": torch.__version__, "scipy": __import__("scipy").__version__,
        "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "fixture": {"kind": f"{args.distribution} synthetic float32", "seed": 20261002,
                    "points": args.points, "queries": args.queries,
                    "extent": args.extent, "radius": args.radius,
                    "cell_size": args.cell_size, "limit": args.limit},
        "input_transfer_ms": input_transfer_ms,
        "cpu_ckdtree_build_ms": cpu_build,
        "cpu_ckdtree_query_ms": cpu_query,
        "mps_morton_build_ms": gpu_build,
        "mps_morton_query_ms": gpu_query,
        "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
        "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory(),
        "mismatched_slots_vs_ckdtree": mismatches,
        "native_brute_sample_queries": sample_queries,
        "native_brute_sample_equal": direct_equal,
        "timing_scope": "preloaded coordinates; synchronized host wall time; shader compilation excluded",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name in ("cpu_ckdtree_build_ms", "cpu_ckdtree_query_ms",
                 "mps_morton_build_ms", "mps_morton_query_ms"):
        print(f"{name}: median={statistics.median(result[name]):.3f} ms; raw={result[name]}")
    print(f"parity: {mismatches} mismatched cKDTree slots; native brute sample={direct_equal}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()

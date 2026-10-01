#!/usr/bin/env python3
"""Paired old-vs-SIMD dense Ball Query Metal kernel benchmark.

The baseline shader is read from a fixed Git commit. The candidate is read
from this checkout. Both are compiled and timed in one Python process against
the same preallocated buffers and synthetic point clouds. This script does
not time the public API, autograd, SciPy, or shader compilation.

    PYTORCH_ENABLE_MPS_FALLBACK=0 python bench/bench_ball_query_simd_ablation.py
    PYTORCH_ENABLE_MPS_FALLBACK=0 python bench/bench_ball_query_simd_ablation.py \
        --sizes 20000 100000 --warmup 3 --repeat 12
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import statistics
import struct
import subprocess
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SHADER_PATH = "mps_pointops/kernels/ball_query.metal"
BASELINE_COMMIT = "ea07179dd13bd8545753f9275e628ba4974c6639"
BASELINE_SHA256 = "331d385610346131ef73f306411bf2a9bfe3f532bab6bd2e6cf41dcd48d3816f"
SIMD_SHA256 = "2856fa046cbbafb71ec0cd688f071c7dc5d2280e9f04d00471241e24d6f85ba3"
SIMD_WIDTH = 32
QUERIES_PER_GROUP = 8
GROUP_SIZE = SIMD_WIDTH * QUERIES_PER_GROUP


def command(*args: str) -> str:
    try:
        return subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shader_sources() -> tuple[str, str]:
    try:
        baseline = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{SHADER_PATH}"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"baseline commit {BASELINE_COMMIT} is unavailable; fetch repository history"
        ) from exc
    candidate = (ROOT / SHADER_PATH).read_bytes()
    for label, source, expected in (
        ("baseline", baseline, BASELINE_SHA256),
        ("SIMD candidate", candidate, SIMD_SHA256),
    ):
        actual = sha256(source)
        if actual != expected:
            raise RuntimeError(f"{label} shader SHA-256 changed: {actual} != {expected}")
    return baseline.decode(), candidate.decode()


def cloud(n: int, seed: int, order: str) -> torch.Tensor:
    """Match bench_pointops.py's unit-sphere shell and optional x sort."""
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn((1, n, 3), generator=generator)
    points = points / points.norm(dim=-1, keepdim=True)
    points = points + 0.01 * torch.randn((1, n, 3), generator=generator)
    if order == "sorted":
        points = torch.stack([sample[sample[:, 0].argsort()] for sample in points])
    return points.float().contiguous()


def query_points(points: torch.Tensor, count: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + 1)
    selection = torch.randperm(points.shape[1], generator=generator)[:count]
    return points[:, selection].contiguous()


def radius_f32_and_square(radius: float) -> tuple[float, float]:
    radius_f32 = struct.unpack("f", struct.pack("f", radius))[0]
    radius_sq = struct.unpack("f", struct.pack("f", float(radius_f32) ** 2))[0]
    return radius_f32, radius_sq


def time_kernel(kernel, arguments: tuple, threads: int) -> float:
    torch.mps.synchronize()
    start = time.perf_counter()
    kernel(*arguments, threads=[threads, 1, 1], group_size=[GROUP_SIZE, 1, 1])
    torch.mps.synchronize()
    return (time.perf_counter() - start) * 1000.0


def measure_case(
    baseline_kernel, candidate_kernel, *, n: int, order: str,
    queries: int, k: int, radius: float, seed: int, warmup: int, repeat: int,
) -> dict:
    cpu_points = cloud(n, seed, order)
    cpu_queries = query_points(cpu_points, queries, seed)
    input_hashes = {
        "points_float32_sha256": sha256(cpu_points.numpy().tobytes()),
        "queries_float32_sha256": sha256(cpu_queries.numpy().tobytes()),
    }
    points = cpu_points.to("mps")
    query = cpu_queries.to("mps")
    lengths1 = torch.tensor([queries], dtype=torch.int64, device="mps")
    lengths2 = torch.tensor([n], dtype=torch.int64, device="mps")
    radius_f32, radius_sq = radius_f32_and_square(radius)
    baseline_threads = queries
    candidate_threads = ((queries + QUERIES_PER_GROUP - 1) // QUERIES_PER_GROUP) * GROUP_SIZE
    runs = {}
    for label, kernel, threads in (
        ("baseline", baseline_kernel, baseline_threads),
        ("simd", candidate_kernel, candidate_threads),
    ):
        indices = torch.empty((1, queries, k), dtype=torch.int64, device="mps")
        distances = torch.empty((1, queries, k), dtype=torch.float32, device="mps")
        arguments = (
            query, points, lengths1, lengths2, indices, distances,
            1, queries, n, k, radius_sq, radius_f32,
        )
        runs[label] = {"kernel": kernel, "threads": threads, "arguments": arguments,
                       "indices": indices, "distances": distances, "times_ms": []}

    # Alternate invocation order to limit a systematic warmup or clock bias.
    for round_index in range(warmup + repeat):
        labels = ("baseline", "simd") if round_index % 2 == 0 else ("simd", "baseline")
        for label in labels:
            run = runs[label]
            elapsed = time_kernel(run["kernel"], run["arguments"], run["threads"])
            if round_index >= warmup:
                run["times_ms"].append(elapsed)

    baseline_indices = runs["baseline"]["indices"].cpu()
    simd_indices = runs["simd"]["indices"].cpu()
    baseline_distances = runs["baseline"]["distances"].cpu()
    simd_distances = runs["simd"]["distances"].cpu()
    indices_equal = baseline_indices.numpy().tobytes() == simd_indices.numpy().tobytes()
    distance_bits_equal = (
        baseline_distances.numpy().tobytes() == simd_distances.numpy().tobytes()
    )
    if not indices_equal or not distance_bits_equal:
        raise AssertionError(
            f"{order} n={n}: baseline and SIMD outputs are not bitwise identical"
        )
    baseline_median = statistics.median(runs["baseline"]["times_ms"])
    simd_median = statistics.median(runs["simd"]["times_ms"])
    return {
        "order": order,
        "points": n,
        "queries": queries,
        "k": k,
        "radius_input": radius,
        "radius_float32": radius_f32,
        "radius_squared_float32": radius_sq,
        "seed": seed,
        "input_hashes": input_hashes,
        "baseline": {
            "threads": baseline_threads,
            "group_size": GROUP_SIZE,
            "times_ms": runs["baseline"]["times_ms"],
            "median_ms": baseline_median,
        },
        "simd": {
            "threads": candidate_threads,
            "group_size": GROUP_SIZE,
            "times_ms": runs["simd"]["times_ms"],
            "median_ms": simd_median,
        },
        "speedup_baseline_over_simd": baseline_median / simd_median,
        "indices_bitwise_equal": indices_equal,
        "distances_bitwise_equal": distance_bits_equal,
        "indices_sha256": sha256(baseline_indices.numpy().tobytes()),
        "distances_sha256": sha256(baseline_distances.numpy().tobytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[20_000, 100_000])
    parser.add_argument("--orders", nargs="+", choices=("sorted", "random"),
                        default=["sorted", "random"])
    parser.add_argument("--queries", type=int, default=1024)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--radius", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("Apple Silicon MPS and torch.mps.compile_shader are required")
    if any(size <= 0 for size in args.sizes) or args.queries <= 0 or args.k <= 0:
        raise ValueError("sizes, queries, and k must be positive")
    if any(args.queries > size for size in args.sizes):
        raise ValueError("queries must not exceed any point count")
    if args.radius <= 0 or args.warmup < 0 or args.repeat <= 0:
        raise ValueError("radius and repeat must be positive; warmup must be non-negative")

    baseline_source, candidate_source = shader_sources()
    baseline_kernel = torch.mps.compile_shader(baseline_source).ball_query_f32
    candidate_kernel = torch.mps.compile_shader(candidate_source).ball_query_f32
    chip = command("sysctl", "-n", "machdep.cpu.brand_string")
    memory = command("sysctl", "-n", "hw.memsize")
    slug = re.sub(r"[^a-z0-9]+", "-", chip.lower()).strip("-")
    path = args.out or ROOT / "bench" / "results" / (
        f"{dt.date.today().isoformat()}-{slug}-ball-query-simd-ablation.json"
    )
    results = {
        "environment": {
            "chip": chip,
            "memory_gib": round(int(memory) / 2**30, 1) if memory.isdigit() else "unknown",
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "pytorch_mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "unset"),
            "pytorch_enable_mps_fallback": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK", "unset"
            ),
        },
        "sources": {
            "baseline_commit": BASELINE_COMMIT,
            "baseline_path": SHADER_PATH,
            "baseline_sha256": BASELINE_SHA256,
            "simd_path": SHADER_PATH,
            "simd_sha256": SIMD_SHA256,
            "checkout_head": command("git", "rev-parse", "HEAD"),
            "benchmark_path": "bench/bench_ball_query_simd_ablation.py",
            "benchmark_sha256": sha256(Path(__file__).read_bytes()),
        },
        "method": {
            "distribution": "float32 noisy unit-sphere shell; sorted means ascending x coordinate",
            "queries": "sampled from point cloud by torch.randperm with seed+1",
            "batch": 1,
            "warmup_per_kernel": args.warmup,
            "repeat_per_kernel": args.repeat,
            "invocation_order": "alternated every round",
            "timing": "time.perf_counter around Metal dispatch and torch.mps.synchronize; "
                      "allocation, transfers, shader compilation, and checks excluded",
            "scipy_comparison": {
                "performed": False,
                "reason": "This paired ablation isolates two Metal shaders; SciPy is measured separately by bench_pointops.py.",
            },
        },
        "cases": [],
    }
    for order in args.orders:
        for n in args.sizes:
            case = measure_case(
                baseline_kernel, candidate_kernel, n=n, order=order,
                queries=args.queries, k=args.k, radius=args.radius,
                seed=args.seed, warmup=args.warmup, repeat=args.repeat,
            )
            results["cases"].append(case)
            print(
                f"{order:6s} N={n:>7d}: baseline {case['baseline']['median_ms']:.3f} ms, "
                f"SIMD {case['simd']['median_ms']:.3f} ms, "
                f"speedup {case['speedup_baseline_over_simd']:.2f}x, bitwise equal",
                flush=True,
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"saved to {path}")


if __name__ == "__main__":
    main()

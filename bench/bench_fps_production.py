#!/usr/bin/env python3
"""Paired single-vs-multigroup FPS timings through the public API.

Example:
    python bench/bench_fps_production.py --sizes 500000 1000000 \
        --output bench/results/2026-10-01-apple-m5-pro-fps-production.json

Each timed call includes Python and shader launch overhead. MPS is explicitly
synchronized immediately before and after it. The call order alternates to
reduce order and thermal bias; both strategies receive the same input tensor.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import furthest_point_sample  # noqa: E402


def command(*args: str) -> str:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500_000, 1_000_000])
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    if min(args.sizes) < 1 or args.samples < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("sizes/samples/repeat must be positive and warmup nonnegative")

    results = []
    for n in args.sizes:
        generator = torch.Generator().manual_seed(args.seed)
        xyz = torch.randn((1, n, 3), generator=generator).to("mps")
        times = {"single": [], "multigroup": []}
        outputs = {}
        mismatches = 0
        for iteration in range(args.warmup + args.repeat):
            order = ("single", "multigroup") if iteration % 2 == 0 else ("multigroup", "single")
            for strategy in order:
                torch.mps.synchronize()
                start = time.perf_counter()
                outputs[strategy] = furthest_point_sample(xyz, args.samples, strategy=strategy)
                torch.mps.synchronize()
                if iteration >= args.warmup:
                    times[strategy].append((time.perf_counter() - start) * 1000)
            pair_mismatches = int((outputs["single"] != outputs["multigroup"]).sum().item())
            mismatches += pair_mismatches
            if pair_mismatches:
                raise AssertionError(
                    f"{pair_mismatches} FPS indices differ at N={n}, iteration={iteration}"
                )
        row = {
            "N": n,
            "B": 1,
            "npoint": args.samples,
            "single_ms": times["single"],
            "multigroup_ms": times["multigroup"],
            "single_median_ms": statistics.median(times["single"]),
            "multigroup_median_ms": statistics.median(times["multigroup"]),
            "speedup": statistics.median(times["single"]) / statistics.median(times["multigroup"]),
            "index_mismatches": mismatches,
        }
        results.append(row)
        print(json.dumps(row), flush=True)

    memory = command("sysctl", "-n", "hw.memsize")
    report = {
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_gib": round(int(memory) / 2**30, 1) if memory.isdigit() else "unknown",
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pytorch_mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "unset"),
            "commit": command("git", "rev-parse", "HEAD"),
            "dirty": bool(command("git", "status", "--porcelain")),
            "script_sha256": sha256(Path(__file__)),
            "ops_sha256": sha256(ROOT / "mps_pointops/ops.py"),
            "single_shader_sha256": sha256(ROOT / "mps_pointops/kernels/fps.metal"),
            "multigroup_shader_sha256": sha256(ROOT / "mps_pointops/kernels/fps_multigroup.metal"),
        },
        "method": {
            "input": "float32 standard-normal point cloud, seed 42 unless overridden",
            "shape": "(1, N, 3)",
            "timing": "wall-clock ms, Python launch included, torch.mps.synchronize immediately before and after",
            "measurement_order": "single/multigroup and multigroup/single alternate",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
        },
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

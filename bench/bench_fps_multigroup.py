#!/usr/bin/env python3
"""Reproducible B=1 FPS dispatch experiment on Apple Silicon.

This script compares the released single-threadgroup kernel with an
experimental two-dispatch-per-sample Metal implementation. It is deliberately
kept in bench/ and does not change the public FPS API.

    python bench/bench_fps_multigroup.py --sizes 100000 250000 500000 1000000
    python bench/bench_fps_multigroup.py --sizes 100000 --samples 256 --repeat 1
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
from mps_pointops import ops  # noqa: E402


def command(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment() -> dict:
    memory = command("sysctl", "-n", "hw.memsize")
    source_path = Path(__file__).with_name("fps_multigroup.metal")
    return {
        "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gib": round(int(memory) / 2**30, 1) if memory.isdigit() else "unknown",
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "pytorch_mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "unset"),
        "commit": command("git", "rev-parse", "HEAD"),
        "dirty": bool(command("git", "status", "--porcelain")),
        "source": "bench/fps_multigroup.metal",
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "benchmark_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


class MultiGroupFPS:
    def __init__(self, group_size: int = 256, chunk: int = 4096):
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is required")
        if group_size < 32 or group_size > 1024 or group_size % 32:
            raise ValueError("group_size must be a multiple of 32 in [32, 1024]")
        if chunk < group_size:
            raise ValueError("chunk must be at least group_size")
        self.group_size = group_size
        self.chunk = chunk
        source = Path(__file__).with_name("fps_multigroup.metal").read_text()
        self.lib = torch.mps.compile_shader(source)

    def __call__(
        self, xyz: torch.Tensor, npoint: int, start_idx: int = 0,
        skip_near_origin: bool = False,
    ) -> torch.Tensor:
        if xyz.device.type != "mps" or xyz.dtype != torch.float32 or xyz.ndim != 3:
            raise ValueError("xyz must be a float32 MPS tensor of shape (1, N, 3)")
        if xyz.shape[0] != 1 or xyz.shape[-1] != 3:
            raise ValueError("experimental path requires shape (1, N, 3)")
        N = xyz.shape[1]
        if npoint < 0 or (npoint and (N == 0 or not 0 <= start_idx < N)):
            raise ValueError("invalid npoint or start_idx")
        if N >= 2**32:
            raise ValueError("experimental uint32 point indices require N < 2**32")
        out = torch.full((1, npoint), start_idx, dtype=torch.long, device="mps")
        if npoint < 2:
            return out
        xyz = xyz.contiguous()
        min_d2 = torch.empty(N, dtype=torch.float32, device="mps")
        parts = (N + self.chunk - 1) // self.chunk
        partial_d2 = torch.empty(parts, dtype=torch.float32, device="mps")
        partial_idx = torch.empty(parts, dtype=torch.int32, device="mps")
        for step in range(npoint - 1):
            self.lib.fps_update_partials(
                xyz, min_d2, out, partial_d2, partial_idx,
                N, step, self.chunk, int(skip_near_origin),
                threads=parts * self.group_size, group_size=self.group_size,
            )
            self.lib.fps_reduce_partials(
                partial_d2, partial_idx, out, parts, step,
                threads=self.group_size, group_size=self.group_size,
            )
        return out


def cloud(n: int, seed: int, order: str) -> torch.Tensor:
    # Match the synthetic shell distribution in bench_pointops.py.
    generator = torch.Generator().manual_seed(seed)
    points = torch.randn((1, n, 3), generator=generator)
    points = points / points.norm(dim=-1, keepdim=True)
    points += 0.01 * torch.randn((1, n, 3), generator=generator)
    if order == "sorted":
        points = points[:, points[0, :, 0].argsort(), :]
    return points.contiguous().to("mps")


def measure_pair(single, multi, warmup: int, repeat: int):
    """Alternate first-run order per repetition to limit thermal/order bias."""
    funcs = {"single": single, "multi": multi}
    outputs = {}
    timings = {"single": [], "multi": []}
    for j in range(warmup + repeat):
        labels = ("single", "multi") if j % 2 == 0 else ("multi", "single")
        for label in labels:
            torch.mps.synchronize()
            start = time.perf_counter()
            outputs[label] = funcs[label]()
            torch.mps.synchronize()
            if j >= warmup:
                timings[label].append((time.perf_counter() - start) * 1000)
    return outputs["single"], outputs["multi"], timings["single"], timings["multi"]


def check_ties(impl: MultiGroupFPS) -> list[dict]:
    cases = []
    g = torch.Generator().manual_seed(17)
    base = torch.rand((1, 7, 3), generator=g)
    repeated = base[:, torch.arange(129) % 7].contiguous().to("mps")
    all_zero = torch.zeros((1, 129, 3), dtype=torch.float32, device="mps")
    for label, pts, start, skip in [
        ("duplicates", repeated, 3, False),
        ("all_skipped", all_zero, 3, True),
    ]:
        got = impl(pts, 20, start_idx=start, skip_near_origin=skip)
        want = ops.furthest_point_sample(pts, 20, start_idx=start, skip_near_origin=skip)
        torch.mps.synchronize()
        mismatch = int((got != want).sum().item())
        cases.append({"case": label, "mismatch_count": mismatch, "total": 20})
        if mismatch:
            raise AssertionError(f"tie/fallback mismatch for {label}: {mismatch}/20")
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[100000, 250000, 500000, 1000000])
    parser.add_argument("--samples", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("--orders", choices=["random", "sorted"], nargs="+", default=["random"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(n < 1 for n in args.sizes) or any(k < 1 for k in args.samples):
        parser.error("all sizes and sample counts must be positive")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be >= 0 and repeat >= 1")
    impl = MultiGroupFPS(args.group_size, args.chunk)
    tie_checks = check_ties(impl)
    rows = []
    for order in args.orders:
        for N in args.sizes:
            pts = cloud(N, args.seed, order)
            for npoint in args.samples:
                single = lambda: ops.furthest_point_sample(pts, npoint)
                multi = lambda: impl(pts, npoint)
                single_out, multi_out, single_ms, multi_ms = measure_pair(
                    single, multi, args.warmup, args.repeat,
                )
                mismatch = int((single_out != multi_out).sum().item())
                row = {
                    "order": order,
                    "N": N,
                    "B": 1,
                    "npoint": npoint,
                    "groups": (N + args.chunk - 1) // args.chunk,
                    "dispatches_multi": 2 * (npoint - 1),
                    "single_ms": single_ms,
                    "multi_ms": multi_ms,
                    "single_median_ms": statistics.median(single_ms),
                    "multi_median_ms": statistics.median(multi_ms),
                    "speedup_multi_over_single": statistics.median(single_ms) / statistics.median(multi_ms),
                    "idx_mismatches": mismatch,
                    "idx_total": npoint,
                }
                rows.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if mismatch:
                    raise AssertionError(f"FPS output mismatch at N={N}, npoint={npoint}: {mismatch}")
    result = {
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": environment(),
        "method": {
            "input": "synthetic noisy unit sphere, B=1, float32, seed 42 unless overridden",
            "single": "package fps.metal, one threadgroup and one dispatch per batch",
            "multi": "experimental fps_multigroup.metal, 2*(npoint-1) dispatches; contiguous chunks",
            "timing": "wall-clock milliseconds including Python launch overhead, torch.mps.synchronize before and after",
            "measurement_order": "single/multi and multi/single alternate each repetition",
            "group_size": args.group_size,
            "chunk": args.chunk,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "orders": args.orders,
            "sizes": args.sizes,
            "samples": args.samples,
        },
        "tie_checks": tie_checks,
        "results": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

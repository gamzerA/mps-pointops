"""Synchronized dense and flat feature-kNN timings on the local Apple GPU.

Run Safe and Fast Math in different Python processes, for example::

  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
    python bench/bench_feature_knn.py --output bench/results/feature-knn-safe.json
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
    python bench/bench_feature_knn.py --output bench/results/feature-knn-fast.json

Each call includes the public API's dispatch, allocation, and result enqueue.
The MPS timer synchronizes both before and after the call. CPU timing uses the
same API on the same shapes. This is a CPU reference cost, not an upstream CUDA
or optimized feature-space search benchmark.
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
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import flat, knn  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _head() -> str | None:
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _timed(call, *, device: str, warmups: int, repeats: int) -> dict:
    for _ in range(warmups):
        call()
        if device == "mps":
            torch.mps.synchronize()
    samples = []
    for _ in range(repeats):
        if device == "mps":
            torch.mps.synchronize()
        start = time.perf_counter_ns()
        call()
        if device == "mps":
            torch.mps.synchronize()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--references", type=int, default=1024)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1601)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if min(args.warmups, args.repeats, args.queries, args.references, args.k) < 1:
        parser.error("warmups, repeats, queries, references, and k must be positive")
    if args.k > min(args.references, 256):
        parser.error("k must be <= min(references, 256)")

    files = [
        "bench/bench_feature_knn.py", "mps_pointops/kernels/feature_knn.metal",
        "mps_pointops/ops.py", "mps_pointops/flat.py",
        "mps_pointops/_flat_search_mps.py", "tests/test_feature_knn.py",
    ]
    results = {
        "environment": {
            "device": torch.backends.mps.get_name(), "os": platform.platform(),
            "python": platform.python_version(), "torch": torch.__version__,
            "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH", "0"),
            "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
            "git_head": _head(),
        },
        "method": {
            "warmups": args.warmups, "repeats": args.repeats,
            "timing": "perf_counter_ns around public API with torch.mps.synchronize before and after MPS calls",
            "source_sha256": {name: _sha(ROOT / name) for name in files},
        },
        "cases": [],
    }
    for dim in (64, 128):
        gen = torch.Generator().manual_seed(args.seed + dim)
        x_cpu = torch.randn((1, args.references, dim), generator=gen)
        q_cpu = torch.randn((1, args.queries, dim), generator=gen)
        x_mps, q_mps = x_cpu.to("mps"), q_cpu.to("mps")
        for kind in ("dense", "flat"):
            if kind == "dense":
                call_cpu = lambda: knn(q_cpu, x_cpu, args.k)
                call_mps = lambda: knn(q_mps, x_mps, args.k)
            else:
                call_cpu = lambda: flat.knn(x_cpu[0], q_cpu[0], args.k)
                call_mps = lambda: flat.knn(x_mps[0], q_mps[0], args.k)
            case = {
                "kind": kind, "batch": 1, "queries": args.queries,
                "references": args.references, "dimension": dim, "k": args.k,
                "cpu": _timed(call_cpu, device="cpu", warmups=args.warmups, repeats=args.repeats),
                "mps": _timed(call_mps, device="mps", warmups=args.warmups, repeats=args.repeats),
            }
            if kind == "dense":
                # The ordinary PyTorch MPS path materializes Q*N distances;
                # timing it gives a useful same-device baseline. Selection
                # bits near ties may differ because cdist can use GEMM.
                case["mps_cdist_topk"] = _timed(
                    lambda: torch.cdist(q_mps, x_mps).topk(args.k, largest=False, sorted=True),
                    device="mps", warmups=args.warmups, repeats=args.repeats,
                )
            results["cases"].append(case)
            print(f"{kind} D={dim}: CPU {case['cpu']['median_ms']:.3f} ms; MPS {case['mps']['median_ms']:.3f} ms", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()

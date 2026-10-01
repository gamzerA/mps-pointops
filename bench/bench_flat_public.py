"""Synchronized public flat FPS, kNN, and radius timings on CPU and MPS.

Run Safe and Fast Math in separate processes, for example::

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_flat_public.py --output bench/results/flat-safe.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
      python bench/bench_flat_public.py --output bench/results/flat-fast.json

Inputs are finite dyadic float32 points, randomly ordered within two uneven
batches numbered 0 and 2. Timings include validation, Metal dispatch, and
edge compaction, but exclude input transfers, shader compilation, and output
readback. CPU is this package's PyTorch reference path, not an optimized CPU
library. The fixture is synthetic and does not represent every point cloud.
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

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import flat  # noqa: E402


def _command(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _fixture(n: int) -> tuple[torch.Tensor, ...]:
    if n < 64 or n % 32:
        raise ValueError("N must be a multiple of 32 and at least 64")
    generator = torch.Generator().manual_seed(7102 + n)
    lengths = (3 * n // 4, n // 4)
    query_lengths = (3 * n // 128, n // 128)
    if min(query_lengths) < 1:
        raise ValueError("each batch must have at least one query")
    points, queries, batch_x, batch_y = [], [], [], []
    for batch_id, length, query_count in zip((0, 2), lengths, query_lengths):
        # Integer coordinates / 16 keep the arithmetic away from ambiguous
        # cell/radius boundaries while retaining a randomly ordered 3D cloud.
        cloud = torch.randint(-128, 129, (length, 3), generator=generator).float() / 16
        query_rows = torch.arange(query_count) * length // query_count
        points.append(cloud)
        queries.append(cloud[query_rows])
        batch_x.append(torch.full((length,), batch_id, dtype=torch.long))
        batch_y.append(torch.full((query_count,), batch_id, dtype=torch.long))
    return tuple(torch.cat(parts) for parts in (points, queries, batch_x, batch_y))


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _measure(fn, device: str, warmups: int, repeats: int) -> dict[str, object]:
    for _ in range(warmups):
        fn()
        _sync(device)
    samples = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter_ns()
        fn()
        _sync(device)
        samples.append((time.perf_counter_ns() - start) / 1_000_000)
    return {
        "median_ms": round(statistics.median(samples), 6),
        "samples_ms": [round(sample, 6) for sample in samples],
    }


def _sources() -> dict[str, str]:
    names = (
        "bench/bench_flat_public.py",
        "mps_pointops/flat.py",
        "mps_pointops/reference.py",
        "mps_pointops/_flat_fps_mps.py",
        "mps_pointops/_flat_search_mps.py",
        "mps_pointops/kernels/fps_flat.metal",
        "mps_pointops/kernels/flat_search.metal",
    )
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 before launching")
    if os.environ.get("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
        raise RuntimeError("set PYTORCH_MPS_FAST_MATH=0 or 1 before launching")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if args.warmups < 0 or args.repeats < 1:
        raise ValueError("warmups must be nonnegative and repeats positive")

    cases = []
    for n in args.sizes:
        cpu = _fixture(n)
        mps = tuple(value.to("mps") for value in cpu)
        configs = {"cpu": cpu, "mps": mps}
        calls = {}
        for device, (x, y, batch_x, batch_y) in configs.items():
            calls[device] = {
                "fps": lambda x=x, batch_x=batch_x: flat.fps(
                    x, batch_x, ratio=1 / 64, random_start=False, batch_size=3
                ),
                "knn": lambda x=x, y=y, batch_x=batch_x, batch_y=batch_y: flat.knn(
                    x, y, 16, batch_x, batch_y, batch_size=3
                ),
                "radius": lambda x=x, y=y, batch_x=batch_x, batch_y=batch_y: flat.radius(
                    x, y, 2.0, batch_x, batch_y, max_num_neighbors=16, batch_size=3
                ),
            }
        parity, timings = {}, {"cpu": {}, "mps": {}}
        for operation in ("fps", "knn", "radius"):
            expected = calls["cpu"][operation]()
            actual = calls["mps"][operation]()
            _sync("mps")
            parity[operation] = bool(torch.equal(expected, actual.cpu()))
            if not parity[operation]:
                raise AssertionError(f"CPU/MPS {operation} output differs for N={n}")
            for device in ("cpu", "mps"):
                timings[device][operation] = _measure(
                    calls[device][operation], device, args.warmups, args.repeats
                )
        cases.append({
            "n": n, "queries": len(cpu[1]),
            "batch_ids": [0, 2], "batch_size": 3,
            "fps_ratio": 1 / 64, "knn_k": 16,
            "radius": 2.0, "max_num_neighbors": 16,
            "exact_cpu_mps_parity": parity, "timings": timings,
        })

    memory_bytes = _command("sysctl", "-n", "hw.memsize")
    result = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(_command("git", "status", "--porcelain")),
        "source_sha256": _sources(),
        "chip": _command("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gib": round(int(memory_bytes) / 2**30, 2) if memory_bytes.isdigit() else None,
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
        "warmups": args.warmups, "repeats": args.repeats,
        "timing_scope": "public API; preloaded tensors; synchronized host wall time",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "cases": cases}, indent=2))


if __name__ == "__main__":
    main()

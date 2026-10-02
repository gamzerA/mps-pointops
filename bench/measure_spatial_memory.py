"""Sample MPS allocator memory during one spatial-search workload.

Each invocation is one fresh process and one search path. PyTorch's unified
accelerator peak counters capture temporary tensor allocations that polling
can miss. These are allocator peaks, *not* whole-process or total GPU memory
peaks. For a Metal resource/footprint trace, use Instruments as described in
docs/spatial-memory-v090.md.
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
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.bench_spatial_bvh import _fixture  # noqa: E402
from bench.spatial_bvh import MortonTwoLevelBVH  # noqa: E402
from mps_pointops._flat_search_mps import knn_indices  # noqa: E402


def _command(*argv: str) -> str:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _memory() -> tuple[int, int]:
    return (int(torch.mps.current_allocated_memory()),
            int(torch.mps.driver_allocated_memory()))


class _Sampler:
    """Polling observer; its largest sample is only a lower bound on a peak."""

    def __init__(self, period_ms: float, read: Callable[[], tuple[int, int]] = _memory):
        self.period_s = period_ms / 1000
        self.read = read
        self.samples: list[tuple[int, int, int]] = []
        self.error: BaseException | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        tensor, driver = self.read()
        self.samples.append((time.perf_counter_ns(), tensor, driver))

    def _loop(self) -> None:
        try:
            while not self.stop.is_set():
                self._sample()
                self.stop.wait(self.period_s)
        except BaseException as exc:
            self.error = exc
            self.stop.set()

    def __enter__(self) -> "_Sampler":
        self._sample()
        self.thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop.set()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("MPS memory sampler failed") from self.error
        self._sample()

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            raise RuntimeError("no MPS memory samples")
        gaps = [(b[0] - a[0]) / 1e6 for a, b in zip(self.samples, self.samples[1:])]
        return {
            "sample_count": len(self.samples),
            "sampling_gap_ms_median": statistics.median(gaps) if gaps else None,
            "sampling_gap_ms_max": max(gaps) if gaps else None,
            "first": {"tensor_bytes": self.samples[0][1],
                      "driver_bytes": self.samples[0][2]},
            "last": {"tensor_bytes": self.samples[-1][1],
                     "driver_bytes": self.samples[-1][2]},
            "sampled_high_water": {
                "tensor_bytes": max(row[1] for row in self.samples),
                "driver_bytes": max(row[2] for row in self.samples),
            },
        }


def _stage(action: Callable[[], Any], period_ms: float) -> tuple[Any, dict[str, Any]]:
    torch.mps.synchronize()
    torch.accelerator.reset_peak_memory_stats()
    started_unix_ns = time.time_ns()
    start = time.perf_counter_ns()
    with _Sampler(period_ms) as sampler:
        value = action()
        torch.mps.synchronize()
    summary = sampler.summary()
    summary["allocator_peak"] = {
        "tensor_bytes": int(torch.accelerator.max_memory_allocated()),
        "reserved_bytes": int(torch.accelerator.max_memory_reserved()),
    }
    summary["started_unix_ns"] = started_unix_ns
    summary["finished_unix_ns"] = time.time_ns()
    summary["host_wall_ms_with_sampling"] = (time.perf_counter_ns() - start) / 1e6
    return value, summary


def _hashes() -> dict[str, str]:
    paths = ("bench/measure_spatial_memory.py", "bench/bench_spatial_bvh.py",
             "bench/spatial_bvh.py", "bench/spatial_bvh.metal",
             "bench/spatial_keys.py", "bench/spatial_keys.metal",
             "mps_pointops/_flat_search_mps.py", "mps_pointops/kernels/flat_search.metal")
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in paths}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", choices=("bvh-serial", "bvh-split", "native-full-scan"), required=True)
    parser.add_argument("--distribution", choices=("uniform", "cluster-sparse", "collapsed"), required=True)
    parser.add_argument("--query-source", choices=("sampled-reference", "independent"),
                        default="sampled-reference")
    parser.add_argument("--points", type=int, default=1_000_000)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--extent", type=float, default=1024.0)
    parser.add_argument("--cell-size", type=float, required=True)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--poll-ms", type=float, default=0.5)
    parser.add_argument("--pause-before-s", type=float, default=0.0,
                        help="seconds to allow Instruments to attach after warmup")
    parser.add_argument("--pause-after-s", type=float, default=0.0,
                        help="seconds to keep the process alive for Instruments after measurement")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
        raise RuntimeError("requires PYTORCH_ENABLE_MPS_FALLBACK=0 and PYTORCH_MPS_FAST_MATH=0 before Python starts")
    if not torch.backends.mps.is_available():
        raise RuntimeError("requires a physical MPS device")
    if not all(hasattr(torch.accelerator, name) for name in
               ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved")):
        raise RuntimeError("requires PyTorch unified accelerator peak memory APIs (verified on 2.14.1)")
    if not 1 <= args.points <= 1_000_000 or not 1 <= args.queries <= 65_536 or not 1 <= args.k <= 32:
        raise ValueError("requires N=1..1M, Q=1..65536, K=1..32")
    if args.path == "bvh-split" and args.queries > 256:
        raise ValueError("the experimental split path supports Q<=256")
    if not 0.1 <= args.poll_ms <= 100:
        raise ValueError("--poll-ms must be in [0.1, 100]")
    if not 0 <= args.pause_before_s <= 600 or not 0 <= args.pause_after_s <= 600:
        raise ValueError("Instruments pause values must be in [0, 600] seconds")
    if args.output.exists():
        raise FileExistsError(args.output)

    points_cpu, query_cpu, detail = _fixture(args)
    # Compile a tiny workload before measurement so shader compilation is not
    # accidentally reported as index construction or query allocation.
    warm_points = torch.zeros((128, 3), device="mps")
    warm_query = torch.zeros((1, 3), device="mps")
    warm_origin = torch.zeros(3, device="mps")
    if args.path.startswith("bvh"):
        warm_index = MortonTwoLevelBVH.build(warm_points, warm_origin, 1.0)
        warm_index.knn(warm_query, 1, parallel_microtrees=args.path == "bvh-split")
        del warm_index
    else:
        warm_ptr = torch.tensor([0, 128], device="mps")
        warm_qptr = torch.tensor([0, 1], device="mps")
        knn_indices(warm_points, warm_query, warm_ptr, warm_qptr, 1)
        del warm_ptr, warm_qptr
    torch.mps.synchronize()
    del warm_points, warm_query, warm_origin
    torch.mps.empty_cache()
    torch.mps.synchronize()
    baseline_tensor, baseline_driver = _memory()
    if args.pause_before_s:
        print(f"INSTRUMENTS_ATTACH_READY pid={os.getpid()} wait={args.pause_before_s}s", flush=True)
        time.sleep(args.pause_before_s)

    def transfer() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        points = torch.from_numpy(points_cpu).to("mps")
        query = torch.from_numpy(query_cpu).to("mps")
        origin = torch.zeros(3, dtype=torch.float32, device="mps")
        ptr_x = torch.tensor([0, args.points], device="mps")
        ptr_y = torch.tensor([0, args.queries], device="mps")
        return points, query, origin, ptr_x, ptr_y

    (points, query, origin, ptr_x, ptr_y), transfer_summary = _stage(transfer, args.poll_ms)
    if args.path.startswith("bvh"):
        index, build_summary = _stage(
            lambda: MortonTwoLevelBVH.build(points, origin, args.cell_size), args.poll_ms)
        index_storage_bytes = sum(t.numel() * t.element_size() for t in
                                  (index.sorted_indices, index.bounds, index.min_index))
        result, query_summary = _stage(
            lambda: index.knn(query, args.k, parallel_microtrees=args.path == "bvh-split"),
            args.poll_ms)
        result_storage_bytes = sum(t.numel() * t.element_size() for t in result)
    else:
        build_summary = None
        index_storage_bytes = 0
        result, query_summary = _stage(
            lambda: knn_indices(points, query, ptr_x, ptr_y, args.k), args.poll_ms)
        result_storage_bytes = result.numel() * result.element_size()

    stages = {"input_transfer": transfer_summary, "index_build": build_summary,
              "query": query_summary}
    high_water = {
        key: max(stage["sampled_high_water"][key] for stage in stages.values() if stage is not None)
        for key in ("tensor_bytes", "driver_bytes")
    }
    allocator_high_water = {
        key: max(stage["allocator_peak"][key] for stage in stages.values() if stage is not None)
        for key in ("tensor_bytes", "reserved_bytes")
    }
    record = {
        "utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _command("git", "rev-parse", "HEAD"),
        "source_dirty": bool(_command("git", "status", "--porcelain")),
        "source_sha256": _hashes(),
        "hardware": _command("sysctl", "-n", "machdep.cpu.brand_string"),
        "physical_ram_bytes": _command("sysctl", "-n", "hw.memsize"),
        "macos": platform.mac_ver()[0],
        "torch": torch.__version__,
        "mps_recommended_max_memory_bytes": int(torch.mps.recommended_max_memory()),
        "mps_fast_math": "0", "mps_fallback": "0",
        "fixture": {"path": args.path, "distribution": args.distribution,
                    "points": args.points, "queries": args.queries, "k": args.k,
                    "extent": args.extent, "cell_size": args.cell_size,
                    "seed": args.seed, **detail},
        "poll_requested_ms": args.poll_ms,
        "instruments_pause_before_s": args.pause_before_s,
        "instruments_pause_after_s": args.pause_after_s,
        "baseline_after_warmup_and_empty_cache": {
            "tensor_bytes": baseline_tensor, "driver_bytes": baseline_driver},
        "stages": stages,
        "sampled_high_water_bytes": high_water,
        "allocator_high_water_bytes": allocator_high_water,
        "index_tensor_storage_bytes": index_storage_bytes,
        "result_tensor_storage_bytes": result_storage_bytes,
        "allocator_high_water_is_total_gpu_peak": False,
        "sampled_driver_high_water_is_true_driver_peak": False,
        "memory_scope": "PyTorch allocator peak counters include transient tensor allocations; driver counter is polled and may miss transients; neither is whole-process/OS GPU memory peak; external Instruments captures, if any, are archived separately",
        "timing_scope": "sampling perturbs wall time; do not use host_wall_ms_with_sampling as a performance benchmark",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"path": args.path, "fixture": record["fixture"],
                      "allocator_high_water_bytes": allocator_high_water,
                      "sampled_high_water_bytes": high_water,
                      "stage_samples": {name: stage["sample_count"] if stage else None
                                        for name, stage in stages.items()},
                      "output": str(args.output)}, indent=2))
    if args.pause_after_s:
        print(f"INSTRUMENTS_CAPTURE_COMPLETE pid={os.getpid()} wait={args.pause_after_s}s", flush=True)
        time.sleep(args.pause_after_s)


if __name__ == "__main__":
    main()

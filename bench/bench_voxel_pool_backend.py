"""Compare full compact voxel forward/backward for the default and fused backends.

Run Safe and Fast Math in distinct processes with MPS fallback disabled. Every
case/backend pair gets a fresh worker process. The recorded MPS allocator
values are synchronized checkpoints, not a true transient device-memory peak.
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
from bench.bench_voxel_api import _inputs, _loss, _memory, _sync  # noqa: E402
from mps_pointops.voxel import voxel_downsample  # noqa: E402


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _worker(nodes: int, shape: str, occupancy: str, backend: str,
            warmups: int, repeats: int) -> dict[str, object]:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    baseline = _memory("mps")
    cpu_pos, cpu_batch, cpu_features, expected_voxels, lengths, digest = _inputs(
        nodes, shape, occupancy,
    )
    pos = cpu_pos.to("mps").requires_grad_(True)
    batch = cpu_batch.to("mps")
    features = cpu_features.to("mps").requires_grad_(True)
    del cpu_pos, cpu_batch, cpu_features
    _sync("mps")
    after_inputs = _memory("mps")
    timings = []
    checkpoints = []
    for iteration in range(warmups + repeats):
        pos.grad = None
        features.grad = None
        _sync("mps")
        start = time.perf_counter()
        output = voxel_downsample(
            pos, 0.5, batch=batch, features=features, pool_backend=backend,
        )
        assert output.features is not None
        _loss(output.pos, output.features).backward()
        _sync("mps")
        elapsed_ms = (time.perf_counter() - start) * 1000
        if iteration >= warmups:
            timings.append(elapsed_ms)
            checkpoints.append(_memory("mps"))
        if output.voxels.counts.numel() != expected_voxels:
            raise AssertionError("unexpected voxel count")
        if pos.grad is None or features.grad is None:
            raise AssertionError("missing gradient")
        del output
    return {
        "nodes": nodes,
        "batch_shape": shape,
        "occupancy": occupancy,
        "backend": backend,
        "batch_lengths": lengths,
        "expected_voxels": expected_voxels,
        "input_sha256": digest,
        "full_step": _summary(timings),
        "memory": {
            "baseline": baseline,
            "after_inputs": after_inputs,
            "post_step_checkpoints": checkpoints,
            "max_observed_mps_current_allocated_bytes": max(
                item["mps_current_allocated_bytes"] for item in checkpoints
            ),
            "max_observed_mps_driver_allocated_bytes": max(
                item["mps_driver_allocated_bytes"] for item in checkpoints
            ),
            "process_peak_rss_bytes": max(
                item["rss_process_peak_bytes"] for item in checkpoints
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[20000, 100000, 500000])
    parser.add_argument("--batch-shapes", nargs="+", choices=["uniform", "ragged"],
                        default=["uniform", "ragged"])
    parser.add_argument("--occupancies", nargs="+", choices=["dense", "sparse"],
                        default=["dense", "sparse"])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shape", help=argparse.SUPPRESS)
    parser.add_argument("--occupancy", help=argparse.SUPPRESS)
    parser.add_argument("--backend", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("PYTORCH_ENABLE_MPS_FALLBACK=0 is required")
    if os.getenv("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
        parser.error("PYTORCH_MPS_FAST_MATH=0 or 1 is required")
    if args.warmups < 0 or args.repeats < 1 or any(n <= 0 for n in args.nodes):
        parser.error("nodes/repeats must be positive and warmups nonnegative")
    if args.worker:
        print(json.dumps(_worker(
            args.nodes[0], args.shape, args.occupancy, args.backend,
            args.warmups, args.repeats,
        )))
        return
    if args.output is None:
        parser.error("--output is required")
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    result: dict[str, object] = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "benchmark_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "voxel_source_sha256": hashlib.sha256(
            (ROOT / "mps_pointops" / "voxel.py").read_bytes()
        ).hexdigest(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "macos_version": platform.mac_ver()[0],
        "cpu_brand": subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
        ).strip(),
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
        "warmups": args.warmups,
        "repeats": args.repeats,
        "requested_matrix": {
            "nodes": args.nodes,
            "batch_shapes": args.batch_shapes,
            "occupancies": args.occupancies,
            "backends": ["index_add", "fused_csr"],
        },
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for nodes in args.nodes:
        for shape in args.batch_shapes:
            for occupancy in args.occupancies:
                for backend in ("index_add", "fused_csr"):
                    command = [
                        sys.executable, str(Path(__file__)), "--worker", "--nodes", str(nodes),
                        "--shape", shape, "--occupancy", occupancy,
                        "--backend", backend, "--warmups", str(args.warmups),
                        "--repeats", str(args.repeats),
                    ]
                    worker = subprocess.run(command, cwd=ROOT, capture_output=True,
                                            text=True, check=False)
                    if worker.returncode:
                        result["incomplete_case"] = {
                            "nodes": nodes, "shape": shape, "occupancy": occupancy,
                            "backend": backend, "stderr": worker.stderr[-4000:],
                        }
                        args.output.write_text(json.dumps(result, indent=2) + "\n")
                        raise RuntimeError(f"worker failed: {worker.stderr}")
                    result["cases"].append(json.loads(worker.stdout))
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(f"completed {nodes}/{shape}/{occupancy}/{backend}",
                          file=sys.stderr, flush=True)
    print(f"wrote {args.output} ({len(result['cases'])} cases)")


if __name__ == "__main__":
    main()

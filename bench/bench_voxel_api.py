"""Measure the compact voxel API on CPU and MPS with per-case memory workers.

Safe and Fast Math require separate launches, for example:

  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
    python bench/bench_voxel_api.py --output docs/bench-voxel-api-safe.json
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
    python bench/bench_voxel_api.py --output docs/bench-voxel-api-fast.json

By default, each launch covers 20k/100k/500k points, uniform/ragged batches,
dense/sparse voxels, and CPU/MPS. Each matrix row runs in a fresh subprocess
so its process peak RSS is specific to that row. The stage measurements sync
at stage boundaries. The separate end-to-end step has one outer synchronization
pair and no explicit synchronization between forward and backward.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops.voxel import VoxelDownsample, Voxelization, voxel_downsample, voxelize  # noqa: E402

BATCH_IDS = (0, 3, 11, 29)
FEATURES = 32
CELL_SIZE = 0.5
SEED = 2701
OCCUPANCY = {"dense": 16, "sparse": 1}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True, check=True,
    )
    return result.stdout.strip()


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _batch_lengths(nodes: int, shape: str) -> tuple[int, int, int, int]:
    if shape == "uniform":
        if nodes % 4:
            raise ValueError("uniform batches require a point count divisible by four")
        return (nodes // 4,) * 4
    if shape == "ragged":
        lengths = (nodes // 100, nodes // 10, nodes // 4)
        return (*lengths, nodes - sum(lengths))
    raise ValueError(f"unknown batch shape: {shape}")


def _inputs(nodes: int, shape: str, occupancy: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, list[int], str]:
    """Generate exactly one or up to sixteen points per occupied cell.

    Dyadic 0.5 cells and interior offsets keep every generated point away from
    a cell boundary. Global permutation exercises unsorted, gapped batches.
    """
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    lengths = _batch_lengths(nodes, shape)
    points_per_cell = OCCUPANCY[occupancy]
    positions = []
    batches = []
    expected_voxels = 0
    for batch_id, length in zip(BATCH_IDS, lengths):
        cell = torch.arange(length, dtype=torch.int64) // points_per_cell
        coords = torch.stack((cell % 128, (cell // 128) % 128, cell // (128 * 128)), dim=1)
        pos = coords.to(torch.float32) * CELL_SIZE
        pos += 0.125 + torch.rand((length, 3), generator=generator) * 0.25
        positions.append(pos)
        batches.append(torch.full((length,), batch_id, dtype=torch.int64))
        expected_voxels += (length + points_per_cell - 1) // points_per_cell
    permutation = torch.randperm(nodes, generator=generator)
    pos = torch.cat(positions)[permutation]
    batch = torch.cat(batches)[permutation]
    features = torch.randn((nodes, FEATURES), generator=generator)[permutation]
    digest = hashlib.sha256()
    for tensor in (pos, batch, features):
        digest.update(memoryview(tensor.numpy()))
    return pos, batch, features, expected_voxels, list(lengths), digest.hexdigest()


def _aggregate(pos: torch.Tensor, features: torch.Tensor, voxels: Voxelization) -> tuple[torch.Tensor, torch.Tensor]:
    """The aggregation portion of voxel_downsample, with a fixed voxel map."""
    rows = voxels.counts.numel()
    means = pos.new_zeros((rows, 3)).index_add_(0, voxels.inverse, pos)
    means = means / voxels.counts.to(pos.dtype).unsqueeze(1)
    pooled = features.new_zeros((rows, FEATURES)).index_add_(0, voxels.inverse, features)
    pooled = pooled / voxels.counts.to(features.dtype).unsqueeze(1)
    return means, pooled


def _loss(pos: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    return pos.square().mean() + features.square().mean()


def _summary(samples: list[float]) -> dict[str, object]:
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def _current_rss_bytes() -> int:
    # psutil uses task_info on macOS and works in restricted benchmark shells.
    # ps is a fallback for environments without the optional psutil package.
    try:
        import psutil
    except ImportError:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, check=True,
        )
        return int(result.stdout.strip()) * 1024
    return psutil.Process().memory_info().rss


def _memory(device: str) -> dict[str, int]:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    snapshot = {
        "rss_current_bytes": _current_rss_bytes(),
        "rss_process_peak_bytes": peak if sys.platform == "darwin" else peak * 1024,
    }
    if device == "mps":
        snapshot["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        snapshot["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
    return snapshot


def _record_observed_mps_max(snapshot: dict[str, int], observed: dict[str, int]) -> None:
    for name in ("mps_current_allocated_bytes", "mps_driver_allocated_bytes"):
        if name in snapshot:
            observed[name] = max(observed.get(name, 0), snapshot[name])


def _measure(
    make_step, device: str, warmups: int, repeats: int,
    observed: dict[str, int],
) -> dict[str, object]:
    samples = []
    for iteration in range(warmups + repeats):
        # make_step may build an autograd graph. Only the returned callable is
        # timed; this keeps backward samples separate from graph construction.
        step = make_step()
        _sync(device)
        start = time.perf_counter()
        output = step()
        _sync(device)
        elapsed = (time.perf_counter() - start) * 1000
        if iteration >= warmups:
            samples.append(elapsed)
        _record_observed_mps_max(_memory(device), observed)
        del output, step
    return _summary(samples)


def _worker(nodes: int, shape: str, occupancy: str, device: str, warmups: int, repeats: int) -> dict[str, object]:
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable in this process")
    _sync(device)
    baseline = _memory(device)
    cpu_pos, cpu_batch, cpu_features, expected_voxels, lengths, input_sha256 = _inputs(nodes, shape, occupancy)
    pos = cpu_pos.to(device).detach().requires_grad_(True)
    batch = cpu_batch.to(device)
    features = cpu_features.to(device).detach().requires_grad_(True)
    del cpu_pos, cpu_batch, cpu_features
    gc.collect()
    _sync(device)
    after_inputs = _memory(device)
    observed: dict[str, int] = {}
    _record_observed_mps_max(after_inputs, observed)

    def clear_grads() -> None:
        pos.grad = None
        features.grad = None

    # The API's integer map is checked before samples, including the exact
    # voxel count implied by the fixture. This check is outside timing.
    check = voxelize(pos, CELL_SIZE, batch)
    if check.counts.numel() != expected_voxels or int(check.counts.sum()) != nodes:
        raise AssertionError("fixture produced an unexpected voxel count")
    del check
    _sync(device)

    def simple(factory):
        def make_step():
            clear_grads()
            return factory
        return make_step

    timings: dict[str, dict[str, object]] = {}
    timings["stage_voxelize_forward"] = _measure(
        simple(lambda: voxelize(pos, CELL_SIZE, batch)), device, warmups, repeats, observed,
    )
    voxels = voxelize(pos, CELL_SIZE, batch)
    _sync(device)
    timings["stage_aggregate_forward"] = _measure(
        simple(lambda: _aggregate(pos, features, voxels)), device, warmups, repeats, observed,
    )

    def stage_backward():
        clear_grads()
        mean_pos, pooled = _aggregate(pos, features, voxels)
        loss = _loss(mean_pos, pooled)
        return loss.backward

    timings["stage_aggregate_backward"] = _measure(stage_backward, device, warmups, repeats, observed)
    del voxels
    gc.collect()
    _sync(device)

    timings["e2e_forward"] = _measure(
        simple(lambda: voxel_downsample(pos, CELL_SIZE, batch, features)),
        device, warmups, repeats, observed,
    )

    def e2e_backward():
        clear_grads()
        output = voxel_downsample(pos, CELL_SIZE, batch, features)
        assert isinstance(output, VoxelDownsample) and output.features is not None
        loss = _loss(output.pos, output.features)
        return loss.backward

    timings["e2e_backward"] = _measure(e2e_backward, device, warmups, repeats, observed)

    def e2e_step():
        clear_grads()

        def step():
            output = voxel_downsample(pos, CELL_SIZE, batch, features)
            assert output.features is not None
            _loss(output.pos, output.features).backward()

        return step

    timings["e2e_forward_backward"] = _measure(e2e_step, device, warmups, repeats, observed)
    _sync(device)
    after_timing = _memory(device)
    _record_observed_mps_max(after_timing, observed)
    if pos.grad is None or features.grad is None:
        raise AssertionError("missing position or feature gradient")
    if not bool(torch.isfinite(pos.grad).all()) or not bool(torch.isfinite(features.grad).all()):
        raise AssertionError("nonfinite position or feature gradient")
    clear_grads()
    del pos, batch, features
    gc.collect()
    _sync(device)
    after_cleanup = _memory(device)
    _record_observed_mps_max(after_cleanup, observed)
    return {
        "nodes": nodes,
        "batch_shape": shape,
        "occupancy": occupancy,
        "points_per_cell": OCCUPANCY[occupancy],
        "device": device,
        "batch_ids": list(BATCH_IDS),
        "batch_lengths": lengths,
        "voxels": expected_voxels,
        "input_sha256": input_sha256,
        "timings": timings,
        "memory": {
            "baseline": baseline,
            "after_inputs": after_inputs,
            "after_timing": after_timing,
            "after_cleanup": after_cleanup,
            "observed_mps_boundary_max_bytes": observed,
        },
    }


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    brand = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
        text=True, check=False,
    ).stdout.strip()
    distributions = sorted(
        f"{item.metadata['Name'].lower()}=={item.version}"
        for item in importlib.metadata.distributions()
        if item.metadata.get("Name")
    )
    return {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_commit(),
        "benchmark_script_sha256": _sha256(Path(__file__)),
        "voxel_source_sha256": _sha256(ROOT / "mps_pointops" / "voxel.py"),
        "python_executable": Path(sys.executable).name,
        "python_executable_path_policy": "basename_only",
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "installed_distributions_sha256": hashlib.sha256(
            ("\n".join(distributions) + "\n").encode()
        ).hexdigest(),
        "macos_version": platform.mac_ver()[0],
        "machine": platform.machine(),
        "cpu_brand": brand or "unavailable",
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
        "seed": SEED,
        "position_dimensions": 3,
        "feature_channels": FEATURES,
        "cell_size": CELL_SIZE,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "requested_matrix": {
            "nodes": args.nodes,
            "batch_shapes": args.batch_shapes,
            "occupancies": args.occupancies,
            "devices": args.devices,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[20000, 100000, 500000])
    parser.add_argument("--batch-shapes", nargs="+", choices=["uniform", "ragged"], default=["uniform", "ragged"])
    parser.add_argument("--occupancies", nargs="+", choices=["dense", "sparse"], default=["dense", "sparse"])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "mps"], default=["cpu", "mps"])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shape", choices=["uniform", "ragged"], help=argparse.SUPPRESS)
    parser.add_argument("--occupancy", choices=["dense", "sparse"], help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=["cpu", "mps"], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("set PYTORCH_ENABLE_MPS_FALLBACK=0 before launching")
    if os.getenv("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
        parser.error("set PYTORCH_MPS_FAST_MATH=0 or 1 before launching")
    if args.warmups < 0 or args.repeats < 1 or any(n <= 0 for n in args.nodes):
        parser.error("nodes/repeats must be positive and warmups nonnegative")
    if args.worker:
        if args.shape is None or args.occupancy is None or args.device is None:
            parser.error("worker requires --shape, --occupancy, and --device")
        print(json.dumps(_worker(args.nodes[0], args.shape, args.occupancy, args.device, args.warmups, args.repeats)))
        return
    if args.output is None:
        parser.error("--output is required")
    if "mps" in args.devices and not torch.backends.mps.is_available():
        parser.error("MPS is unavailable in this process; select --devices cpu for a CPU-only run")
    result = _metadata(args)
    cases = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for nodes in args.nodes:
        for shape in args.batch_shapes:
            for occupancy in args.occupancies:
                for device in args.devices:
                    command = [
                        sys.executable, str(Path(__file__)), "--worker", "--nodes", str(nodes),
                        "--shape", shape, "--occupancy", occupancy, "--device", device,
                        "--warmups", str(args.warmups), "--repeats", str(args.repeats),
                    ]
                    worker = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
                    if worker.returncode:
                        result["cases"] = cases
                        result["incomplete_case"] = {
                            "nodes": nodes, "batch_shape": shape,
                            "occupancy": occupancy, "device": device,
                            "returncode": worker.returncode,
                            "stderr": worker.stderr[-4000:],
                        }
                        args.output.write_text(json.dumps(result, indent=2) + "\n")
                        raise RuntimeError(
                            f"case {nodes}/{shape}/{occupancy}/{device} failed:\n{worker.stderr}"
                        )
                    cases.append(json.loads(worker.stdout))
                    result["cases"] = cases
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
                    print(f"completed {nodes}/{shape}/{occupancy}/{device}", file=sys.stderr, flush=True)
    print(f"wrote {args.output} ({len(cases)} cases)")


if __name__ == "__main__":
    main()

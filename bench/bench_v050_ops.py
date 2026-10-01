#!/usr/bin/env python3
"""Reproduce M5 Pro timings for the experimental v0.5.0 source operators.

Run Safe and Fast Math in separate Python processes::

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_v050_ops.py --output bench/results/v050-safe.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
      python bench/bench_v050_ops.py --output bench/results/v050-fast.json

The measurements are synchronized host wall times for the public APIs, not
isolated Metal kernel timings. No CPU/GPU speedup is inferred from this run.
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

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
    raise SystemExit("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python")
if os.environ.get("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
    raise SystemExit("Set PYTORCH_MPS_FAST_MATH=0 or 1 before starting Python")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import chamfer_distance, three_interpolate, three_nn  # noqa: E402


def command(*argv: str) -> str:
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronized_call(call):
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    value = call()
    torch.mps.synchronize()
    return value, (time.perf_counter_ns() - start) / 1e6


def summarize(samples: list[float]) -> dict:
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def make_three_nn(batch: int, queries: int, known: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    unknown = torch.randn((batch, queries, 3), generator=generator).to("mps")
    points = torch.randn((batch, known, 3), generator=generator).to("mps")
    distances, indices = three_nn(unknown, points)
    torch.mps.synchronize()
    assert distances.shape == indices.shape == (batch, queries, 3)
    assert distances.dtype == torch.float32 and indices.dtype == torch.int32
    assert torch.isfinite(distances).all().item()
    assert ((indices >= 0) & (indices < known)).all().item()
    return {
        "name": "three_nn",
        "shape": {"B": batch, "N_unknown": queries, "M_known": known, "D": 3},
        "inputs": (unknown, points),
        "validation": {"shape_dtype_finite_and_index_bounds": True},
    }


def make_three_interpolate(batch: int, channels: int, source: int, queries: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn((batch, channels, source), generator=generator).to("mps").requires_grad_()
    indices = torch.randint(source, (batch, queries, 3), generator=generator, dtype=torch.int32).to("mps")
    weights = torch.rand((batch, queries, 3), generator=generator).to("mps")
    weights = weights / weights.sum(dim=-1, keepdim=True)
    grad = torch.ones((batch, channels, queries), device="mps", dtype=torch.float32)
    output = three_interpolate(features, indices, weights)
    output.backward(grad)
    torch.mps.synchronize()
    assert output.shape == grad.shape and torch.isfinite(output).all().item()
    assert features.grad is not None and torch.isfinite(features.grad).all().item()
    features.grad = None
    return {
        "name": "three_interpolate",
        "shape": {"B": batch, "C": channels, "M_source": source, "N_target": queries, "neighbors": 3},
        "inputs": (features, indices, weights),
        "gradient": grad,
        "validation": {"shape_dtype_finite_and_finite_feature_gradient": True},
    }


def make_chamfer(batch: int, points: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((batch, points, 3), generator=generator).to("mps").requires_grad_()
    y = torch.randn((batch, points, 3), generator=generator).to("mps").requires_grad_()
    loss, normals = chamfer_distance(x, y)
    assert normals is None
    loss.backward()
    torch.mps.synchronize()
    assert loss.ndim == 0 and torch.isfinite(loss).item()
    assert x.grad is not None and y.grad is not None
    assert torch.isfinite(x.grad).all().item() and torch.isfinite(y.grad).all().item()
    x.grad = y.grad = None
    return {
        "name": "chamfer_distance",
        "shape": {"B": batch, "P": points, "Q": points, "D": 3},
        "inputs": (x, y),
        "validation": {"finite_loss_and_both_finite_coordinate_gradients": True},
    }


def measure(case: dict, warmup: int, repeat: int) -> dict:
    name = case["name"]
    forward_samples, backward_samples = [], []
    for iteration in range(warmup + repeat):
        if name == "three_nn":
            _, forward = synchronized_call(lambda: three_nn(*case["inputs"]))
        elif name == "three_interpolate":
            features, indices, weights = case["inputs"]
            features.grad = None  # This reset and any prior work are outside both timers.
            output, forward = synchronized_call(lambda: three_interpolate(features, indices, weights))
            _, backward = synchronized_call(lambda: output.backward(case["gradient"]))
        else:
            x, y = case["inputs"]
            x.grad = y.grad = None
            result, forward = synchronized_call(lambda: chamfer_distance(x, y))
            _, backward = synchronized_call(lambda: result[0].backward())
        if iteration >= warmup:
            forward_samples.append(forward)
            if name != "three_nn":
                backward_samples.append(backward)
    return {
        "operator": name,
        "shape": case["shape"],
        "validation": case["validation"],
        "forward": summarize(forward_samples),
        "backward": summarize(backward_samples) if backward_samples else None,
    }


def render_markdown(report: dict) -> str:
    env = report["environment"]
    lines = [
        "# Experimental PointNet++ and Chamfer MPS benchmark",
        "",
        f"- UTC: {report['date_utc']}",
        f"- Device: {env['chip']}; macOS {env['macos']}; PyTorch {env['torch']}",
        f"- Source base commit: `{env['source_commit']}`; tracked operator sources clean: `{env['operator_sources_clean']}`",
        f"- Math mode: `{env['pytorch_mps_fast_math']}`; CPU fallback: `{env['pytorch_enable_mps_fallback']}`",
        f"- Reproduce: `{report['command']}`",
        f"- {report['method']['warmup']} warmups and {report['method']['repeat']} measured calls per case; seed {report['method']['seed']}.",
        "",
        "## Median synchronized wall time",
        "",
        "| Operator | Shape | Forward (ms) | Backward (ms) |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in report["results"]:
        shape = ", ".join(f"{key}={value}" for key, value in row["shape"].items())
        backward = "—" if row["backward"] is None else f"{row['backward']['median_ms']:.3f}"
        lines.append(f"| `{row['operator']}` | {shape} | {row['forward']['median_ms']:.3f} | {backward} |")
    lines += [
        "",
        "## Method and scope",
        "",
        "- Inputs are independently seeded float32 standard-normal coordinates/features; interpolation uses seeded random valid indices and normalized positive weights. Input construction, shader compilation, and initial finite-output checks precede timed calls.",
        "- `three_nn` is forward-only by contract. `three_interpolate` backward uses a preallocated all-ones output gradient. Chamfer uses its default bidirectional squared-L2, point mean, and batch mean with no padding, normals, or weights.",
        "- Each public call is bracketed by `torch.mps.synchronize()`. Python validation, allocation, dispatch, and autograd are included. These are not isolated shader timings. Cases run in table order, so thermal state may affect comparisons.",
        "- No CPU baseline or cross-operator speedup claim is made. The JSON records every raw sample and exact SHA-256 hashes of the Python modules, Metal shaders, and this benchmark script.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=27001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("PyTorch MPS is unavailable")
    if args.warmup < 0 or args.repeat < 1 or args.output.suffix != ".json":
        parser.error("warmup must be nonnegative, repeat positive, output a .json path")

    cases = [
        make_three_nn(2, 512, 1024, args.seed),
        make_three_nn(2, 2048, 4096, args.seed + 1),
        make_three_interpolate(2, 32, 512, 2048, args.seed + 2),
        make_three_interpolate(2, 64, 2048, 8192, args.seed + 3),
        make_chamfer(2, 256, args.seed + 4),
        make_chamfer(2, 1024, args.seed + 5),
    ]
    results = []
    for case in cases:
        row = measure(case, args.warmup, args.repeat)
        results.append(row)
        print(json.dumps({"operator": row["operator"], "shape": row["shape"], "forward_median_ms": row["forward"]["median_ms"], "backward_median_ms": row["backward"]["median_ms"] if row["backward"] else None}), flush=True)

    memory = command("sysctl", "-n", "hw.memsize")
    hardware = command("system_profiler", "SPHardwareDataType")
    chip = next((line.partition(":")[2].strip() for line in hardware.splitlines() if line.strip().startswith("Chip:")), "unknown")
    source_files = [
        "mps_pointops/__init__.py",
        "mps_pointops/pointnet2.py",
        "mps_pointops/kernels/pointnet2.metal",
        "mps_pointops/chamfer.py",
        "mps_pointops/kernels/chamfer_nn.metal",
    ]
    report = {
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "chip": chip,
            "memory_gib": round(int(memory) / 2**30, 1) if memory.isdigit() else "unknown",
            "macos": platform.mac_ver()[0],
            "architecture": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_mps_is_available": torch.backends.mps.is_available(),
            "pytorch_mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
            "pytorch_enable_mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "source_commit": command("git", "rev-parse", "HEAD"),
            "operator_sources_clean": not bool(command("git", "diff", "--name-only", "HEAD", "--", *source_files)),
            "source_sha256": {name: digest(ROOT / name) for name in source_files},
            "script_sha256": digest(Path(__file__)),
        },
        "method": {
            "device": "mps",
            "dtype": "float32",
            "input": "seeded independent standard-normal coordinates/features; interpolation has random valid indices and normalized positive weights",
            "timing": "host perf_counter_ns wall time; torch.mps.synchronize immediately before and after each public forward/backward call",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "case_order": "table order, all repeats of one case before the next",
            "chamfer": "bidirectional squared-L2, point mean and batch mean, no padding/normals/weights",
        },
        "command": f"PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH={os.environ['PYTORCH_MPS_FAST_MATH']} python bench/bench_v050_ops.py --warmup {args.warmup} --repeat {args.repeat} --seed {args.seed} --output {args.output.as_posix()}",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(report))
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

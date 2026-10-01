#!/usr/bin/env python3
"""Compare public bidirectional Chamfer on CPU and Apple MPS.

Use separate processes for Safe and Fast Math (the setting is cached)::

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_chamfer_cpu_mps.py --output bench/results/chamfer-m1-safe.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
      python bench/bench_chamfer_cpu_mps.py --output bench/results/chamfer-m1-fast.json

The same finite, independently generated float32 points are used on both
devices. This times the full public API, including input validation, the
native MPS nearest search or tiled CPU search, reductions and autograd.
Inputs are resident on each device before timing; no transfer is timed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

# These checks must run before importing torch. Fast Math is process-cached.
if os.environ.get("PYTORCH_MPS_FAST_MATH") not in {"0", "1"}:
    raise SystemExit("set PYTORCH_MPS_FAST_MATH=0 or =1 before starting Python")
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").lower() not in {"0", "false"}:
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import chamfer_distance  # noqa: E402

RTOL = 1e-5
ATOL = 1e-5
SOURCE_FILES = (
    "bench/bench_chamfer_cpu_mps.py",
    "mps_pointops/__init__.py",
    "mps_pointops/chamfer.py",
    "mps_pointops/_ball_query_mps.py",
    "mps_pointops/kernels/chamfer_nn.metal",
)


def _command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, check=True, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip()


def _chip() -> str | None:
    hardware = _command("system_profiler", "SPHardwareDataType")
    if hardware:
        for line in hardware.splitlines():
            if line.strip().startswith("Chip:"):
                return line.partition(":")[2].strip()
    return _command("sysctl", "-n", "machdep.cpu.brand_string")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(samples: list[float]) -> dict:
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def _make_points(size: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed + size)
    x = torch.randn((1, size, 3), generator=generator, dtype=torch.float32)
    y = torch.randn((1, size, 3), generator=generator, dtype=torch.float32)
    assert bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
    return x, y


def _check_parity(
    x_values: torch.Tensor, y_values: torch.Tensor
) -> tuple[dict, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    inputs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    observed: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for device in ("cpu", "mps"):
        x = x_values.to(device).detach().requires_grad_()
        y = y_values.to(device).detach().requires_grad_()
        loss, normals = chamfer_distance(x, y, single_directional=False)
        if normals is not None or loss.ndim != 0:
            raise AssertionError(f"unexpected public Chamfer result on {device}")
        loss.backward()
        if device == "mps":
            torch.mps.synchronize()
        if x.grad is None or y.grad is None:
            raise AssertionError(f"missing first-order gradient on {device}")
        observed[device] = (loss.detach().cpu(), x.grad.detach().cpu(), y.grad.detach().cpu())
        x.grad = y.grad = None
        inputs[device] = (x, y)

    cpu_loss, cpu_x_grad, cpu_y_grad = observed["cpu"]
    mps_loss, mps_x_grad, mps_y_grad = observed["mps"]
    # The repository's Chamfer parity tests use this tolerance. A selection
    # difference will normally be exposed by the coordinate gradients.
    for name, expected, actual in (
        ("loss", cpu_loss, mps_loss),
        ("x_gradient", cpu_x_grad, mps_x_grad),
        ("y_gradient", cpu_y_grad, mps_y_grad),
    ):
        try:
            torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
        except AssertionError as exc:
            raise AssertionError(f"CPU/MPS Chamfer {name} parity failed: {exc}") from exc
    return {
        "passed": True,
        "rtol": RTOL,
        "atol": ATOL,
        "cpu_loss": float(cpu_loss.item()),
        "mps_loss": float(mps_loss.item()),
        "max_abs_loss_error": float((mps_loss - cpu_loss).abs().item()),
        "max_abs_x_gradient_error": float((mps_x_grad - cpu_x_grad).abs().max().item()),
        "max_abs_y_gradient_error": float((mps_y_grad - cpu_y_grad).abs().max().item()),
    }, inputs


def _measure_once(device: str, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    x.grad = y.grad = None
    if device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter_ns()
    loss, _ = chamfer_distance(x, y, single_directional=False)
    if device == "mps":
        torch.mps.synchronize()
    forward_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    loss.backward()
    if device == "mps":
        torch.mps.synchronize()
    backward_ms = (time.perf_counter_ns() - start) / 1e6
    return forward_ms, backward_ms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024], help="points per cloud; e.g. 256 1024 4096")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if not args.sizes or min(args.sizes) < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("sizes and repeat must be positive; warmup must be nonnegative")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be unique")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "chip": _chip(),
            "hardware_model": _command("sysctl", "-n", "hw.model"),
            "physical_memory_bytes": _command("sysctl", "-n", "hw.memsize"),
            "macos": platform.mac_ver()[0],
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
            "mps_device": torch.backends.mps.get_name(),
            "pytorch_mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
            "pytorch_enable_mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "git_head": _command("git", "-C", str(ROOT), "rev-parse", "HEAD"),
            "source_sha256": {name: _sha256(ROOT / name) for name in SOURCE_FILES},
        },
        "method": {
            "public_api": "mps_pointops.chamfer_distance(x, y, single_directional=False)",
            "shape": "(1, N, 3) for both x and y",
            "dtype": "torch.float32",
            "distribution": "independent standard-normal x and y from a fixed CPU generator",
            "seed": args.seed,
            "point_reduction": "mean",
            "batch_reduction": "mean",
            "warmup": args.warmup,
            "repeat": args.repeat,
            "timer": "perf_counter_ns; MPS synchronized before forward and after each forward/backward phase",
            "data_transfer_in_timed_region": False,
            "device_order": "CPU then MPS on even iterations; MPS then CPU on odd iterations",
            "parity_before_timing": "CPU versus MPS loss and first-order x/y gradients, rtol=atol=1e-5",
            "comparison_scope": "same public implementation on two devices; no SciPy or upstream PyTorch3D baseline",
        },
        "invocation": {"argv": [sys.executable, *sys.argv], "sizes": args.sizes},
        "cases": [],
    }

    for size in args.sizes:
        x_values, y_values = _make_points(size, args.seed)
        parity, inputs = _check_parity(x_values, y_values)
        samples = {device: {"forward": [], "backward": []} for device in ("cpu", "mps")}
        for iteration in range(args.warmup + args.repeat):
            order = ("cpu", "mps") if iteration % 2 == 0 else ("mps", "cpu")
            for device in order:
                forward_ms, backward_ms = _measure_once(device, *inputs[device])
                if iteration >= args.warmup:
                    samples[device]["forward"].append(forward_ms)
                    samples[device]["backward"].append(backward_ms)
        case = {
            "batch": 1,
            "points_x": size,
            "points_y": size,
            "parity": parity,
            "cpu": {phase: _summary(values) for phase, values in samples["cpu"].items()},
            "mps": {phase: _summary(values) for phase, values in samples["mps"].items()},
        }
        case["cpu_over_mps_forward"] = case["cpu"]["forward"]["median_ms"] / case["mps"]["forward"]["median_ms"]
        case["cpu_over_mps_backward"] = case["cpu"]["backward"]["median_ms"] / case["mps"]["backward"]["median_ms"]
        report["cases"].append(case)
        print(
            f"N={size}: forward CPU {case['cpu']['forward']['median_ms']:.3f} ms, "
            f"MPS {case['mps']['forward']['median_ms']:.3f} ms; "
            f"backward CPU {case['cpu']['backward']['median_ms']:.3f} ms, "
            f"MPS {case['mps']['backward']['median_ms']:.3f} ms; parity passed",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

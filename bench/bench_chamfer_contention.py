#!/usr/bin/env python3
"""Measure the native-Metal/PyTorch Chamfer path under scatter contention.

Run Safe and Fast Math in *separate processes*, for example::

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_chamfer_contention.py --output bench/results/chamfer-safe.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
      python bench/bench_chamfer_contention.py --output bench/results/chamfer-fast.json

This uses single-directional squared-L2 Chamfer with point and batch means so
one backward scatter has a controlled destination pattern. The full public
forward and backward are timed separately, including Python/dispatch overhead.
An additional preallocated scatter_add_ control times only that operation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

# An explicitly enabled CPU fallback would invalidate the GPU comparison.
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0").lower() not in ("0", "false"):
    raise SystemExit("PYTORCH_ENABLE_MPS_FALLBACK must be 0")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops.chamfer import _nearest_mps, chamfer_distance  # noqa: E402


def command(*args: str) -> str:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chip_name() -> str:
    hardware = command("system_profiler", "SPHardwareDataType")
    for line in hardware.splitlines():
        if line.strip().startswith("Chip:"):
            return line.partition(":")[2].strip()
    return command("sysctl", "-n", "machdep.cpu.brand_string")


def make_case(batch: int, n: int, pattern: str) -> dict:
    locations = torch.arange(n, dtype=torch.float32)
    y_cpu = torch.zeros((batch, n, 3), dtype=torch.float32)
    x_cpu = torch.zeros_like(y_cpu)
    y_cpu[:, :, 0] = 4 * locations
    if pattern == "uniform":
        x_cpu[:, :, 0] = 4 * locations + 0.5
        expected_index = torch.arange(n, dtype=torch.long).expand(batch, n).clone()
    else:
        # Every x lies near y[0] and at least 3.4 from y[1]. The small
        # deterministic variation keeps nonzero gradients without NN ties.
        x_cpu[:, :, 0] = 0.5 + 0.01 * (locations % 7)
        expected_index = torch.zeros((batch, n), dtype=torch.long)
    return {
        "x_cpu": x_cpu,
        "y_cpu": y_cpu,
        "x": x_cpu.to("mps").requires_grad_(),
        "y": y_cpu.to("mps").requires_grad_(),
        "expected_index": expected_index,
        "index": expected_index.to("mps"),
        "lengths": torch.full((batch,), n, dtype=torch.long, device="mps"),
    }


def verify_case(case: dict, batch: int, n: int, pattern: str) -> dict:
    x, y, lengths = case["x"], case["y"], case["lengths"]
    distances, actual_index = _nearest_mps(x.detach(), y.detach(), lengths, lengths)
    torch.mps.synchronize()
    actual_index_cpu = actual_index.cpu()
    expected_index = case["expected_index"]
    torch.testing.assert_close(actual_index_cpu, expected_index, rtol=0, atol=0)
    assert distances.shape == (batch, n)

    x.grad = y.grad = None
    loss, _ = chamfer_distance(x, y, single_directional=True)
    loss.backward()
    torch.mps.synchronize()
    selected = case["y_cpu"].gather(1, expected_index.unsqueeze(-1).expand(-1, -1, 3))
    delta = case["x_cpu"] - selected
    expected_loss = delta.square().sum(dim=-1).mean()
    expected_x_grad = 2 * delta / (batch * n)
    expected_y_grad = torch.zeros_like(case["y_cpu"])
    expected_y_grad.scatter_add_(1, expected_index.unsqueeze(-1).expand(-1, -1, 3), -expected_x_grad)
    torch.testing.assert_close(loss.cpu(), expected_loss, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(x.grad.cpu(), expected_x_grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(y.grad.cpu(), expected_y_grad, rtol=1e-4, atol=1e-5)
    counts = torch.bincount(actual_index_cpu[0], minlength=n)
    observed_occupied = int((counts > 0).sum().item())
    observed_max_hits = int(counts.max().item())
    assert observed_occupied == (n if pattern == "uniform" else 1)
    assert observed_max_hits == (1 if pattern == "uniform" else n)
    result = {
        "query_shape": list(x.shape),
        "reference_shape": list(y.shape),
        "dtype": str(x.dtype),
        "valid_points_per_cloud": n,
        "index_mismatches": 0,
        "occupied_references_per_batch": observed_occupied,
        "max_queries_per_reference_per_batch": observed_max_hits,
        "loss": float(loss.item()),
        "max_abs_x_gradient_error": float((x.grad.cpu() - expected_x_grad).abs().max().item()),
        "max_abs_y_gradient_error": float((y.grad.cpu() - expected_y_grad).abs().max().item()),
    }
    x.grad = y.grad = None
    return result


def measure_chamfer(case: dict) -> tuple[float, float]:
    x, y = case["x"], case["y"]
    x.grad = y.grad = None
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    loss, _ = chamfer_distance(x, y, single_directional=True)
    torch.mps.synchronize()
    forward_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    loss.backward()
    torch.mps.synchronize()
    backward_ms = (time.perf_counter_ns() - start) / 1e6
    return forward_ms, backward_ms


def measure_scatter(case: dict, destination: torch.Tensor, source: torch.Tensor) -> float:
    destination.zero_()  # Reset outside the measured region.
    torch.mps.synchronize()
    start = time.perf_counter_ns()
    destination.scatter_add_(1, case["index"].unsqueeze(-1).expand(-1, -1, 3), source)
    torch.mps.synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def summary(samples: list[float]) -> dict:
    return {"samples_ms": samples, "median_ms": statistics.median(samples)}


def render_markdown(report: dict) -> str:
    env, method = report["environment"], report["method"]
    lines = [
        "# MPS Chamfer scatter contention benchmark",
        "",
        f"- UTC: {report['date_utc']}",
        f"- Device: {env['chip']}; macOS {env['macos']}; PyTorch {env['torch']}",
        f"- Math mode: `{env['pytorch_mps_fast_math']}`; CPU fallback: `{env['pytorch_enable_mps_fallback']}`",
        f"- Reproduction: `{report['command']}`",
        f"- {method['warmup']} warmups and {method['repeat']} measured runs per case; synchronized wall time, Python launch included.",
        "",
        "## Results",
        "",
        "All timings are medians in milliseconds. F/B is backward divided by forward. "
        "Concentrated / uniform is the median of paired per-iteration ratios.",
        "",
        "| B | N | Pattern | Forward | Backward | F/B | Direct scatter |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        for pattern in ("uniform", "concentrated"):
            result = row[pattern]
            lines.append(
                f"| {row['B']} | {row['N']} | {pattern} | "
                f"{result['forward']['median_ms']:.3f} | {result['backward']['median_ms']:.3f} | "
                f"{result['backward_over_forward']:.2f}× | {result['direct_scatter']['median_ms']:.3f} |"
            )
        lines.append(
            f"| | | concentrated / uniform | | {row['ratios']['backward_concentrated_over_uniform']:.2f}× | "
            f"| {row['ratios']['scatter_concentrated_over_uniform']:.2f}× |"
        )
    lines += [
        "",
        "## Method and limits",
        "",
        "- Same batch, query/reference shapes, float32 dtype and valid point counts in both patterns. "
        "Uniform selects each reference once per batch; concentrated selects reference 0 for every query. "
        "Expected indices and analytic mean-reduction gradients are checked before timing.",
        "- The timed Chamfer path is `chamfer_distance(x, y, single_directional=True)` with default "
        "point and batch means. Its forward includes public input validation and Metal nearest search; "
        "its backward includes autograd, elementwise gradient work, allocation and PyTorch `scatter_add_`. "
        "The direct control times only `scatter_add_` into a preallocated destination, reset outside the timer.",
        "- Forward and backward are synchronized separately before/after timing. "
        "Case order alternates. Compilation, CPU tensor creation and correctness checks are outside timed runs.",
        "- Timing ratios alone cannot establish atomic contention as the cause, nor prove that a custom Metal "
        "reduction kernel will be faster. End-to-end autograd overhead, allocator behavior and thermal state "
        "also affect these wall times. This run has no padding, weights or reverse Chamfer term.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 2048])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True, help="JSON path; matching .md is also written")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    if args.batch < 1 or not args.sizes or min(args.sizes) < 1 or args.warmup < 0 or args.repeat < 1:
        parser.error("batch/sizes/repeat must be positive and warmup nonnegative")
    if args.repeat % 2:
        parser.error("repeat must be even so both case orders have equal measured runs")
    if args.output.suffix != ".json":
        parser.error("--output must end in .json")

    results = []
    for n in args.sizes:
        cases = {name: make_case(args.batch, n, name) for name in ("uniform", "concentrated")}
        checks = {name: verify_case(case, args.batch, n, name) for name, case in cases.items()}
        timings = {
            name: {"forward": [], "backward": [], "direct_scatter": []}
            for name in cases
        }
        for iteration in range(args.warmup + args.repeat):
            order = ("uniform", "concentrated") if iteration % 2 == 0 else ("concentrated", "uniform")
            for name in order:
                forward_ms, backward_ms = measure_chamfer(cases[name])
                if iteration >= args.warmup:
                    timings[name]["forward"].append(forward_ms)
                    timings[name]["backward"].append(backward_ms)

        source = torch.ones((args.batch, n, 3), device="mps", dtype=torch.float32)
        destinations = {name: torch.zeros_like(source) for name in cases}
        for iteration in range(args.warmup + args.repeat):
            order = ("uniform", "concentrated") if iteration % 2 == 0 else ("concentrated", "uniform")
            for name in order:
                scatter_ms = measure_scatter(cases[name], destinations[name], source)
                if iteration >= args.warmup:
                    timings[name]["direct_scatter"].append(scatter_ms)
        for name in cases:
            expected = torch.ones_like(source) if name == "uniform" else torch.zeros_like(source)
            if name == "concentrated":
                expected[:, 0, :] = n
            torch.testing.assert_close(destinations[name].cpu(), expected.cpu(), rtol=0, atol=0)

        row = {"B": args.batch, "N": n}
        for name in cases:
            row[name] = {key: summary(values) for key, values in timings[name].items()}
            row[name]["backward_over_forward"] = (
                row[name]["backward"]["median_ms"] / row[name]["forward"]["median_ms"]
            )
            row[name]["correctness"] = checks[name]
        row["ratios"] = {
            "aggregation": "median of paired per-iteration ratios",
            "backward_concentrated_over_uniform": statistics.median(
                concentrated / uniform
                for concentrated, uniform in zip(
                    timings["concentrated"]["backward"], timings["uniform"]["backward"]
                )
            ),
            "scatter_concentrated_over_uniform": statistics.median(
                concentrated / uniform
                for concentrated, uniform in zip(
                    timings["concentrated"]["direct_scatter"],
                    timings["uniform"]["direct_scatter"],
                )
            ),
        }
        results.append(row)
        print(json.dumps({"B": args.batch, "N": n, "ratios": row["ratios"]}), flush=True)

    report = {
        "schema": "mps-pointops.chamfer-contention.v1",
        "date_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": "PYTORCH_ENABLE_MPS_FALLBACK=0 "
        f"PYTORCH_MPS_FAST_MATH={os.environ.get('PYTORCH_MPS_FAST_MATH', '0')} "
        + shlex.join(["python", str(Path(__file__).resolve().relative_to(ROOT)), *sys.argv[1:]]),
        "environment": {
            "chip": chip_name(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pytorch_mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "unset"),
            "pytorch_enable_mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "source_commit": command("git", "rev-parse", "HEAD"),
            "script_sha256": sha256(Path(__file__)),
            "chamfer_sha256": sha256(ROOT / "mps_pointops/chamfer.py"),
            "metal_sha256": sha256(ROOT / "mps_pointops/kernels/chamfer_nn.metal"),
        },
        "method": {
            "batch": args.batch,
            "sizes": args.sizes,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "input": "deterministic x/y coordinates on x axis; same shapes, float32 and full lengths; "
            "uniform selects each reference once; concentrated selects reference 0 for every query",
            "chamfer": "single_directional=True, point_reduction=mean, batch_reduction=mean",
            "timing": "wall-clock perf_counter_ns around each call, torch.mps.synchronize immediately before and after; "
            "forward and backward separately; Python launch overhead included",
            "direct_scatter": "preallocated output zeroed outside timed region; time only scatter_add_ and completion",
            "measurement_order": "uniform/concentrated and concentrated/uniform alternate",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(report))
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

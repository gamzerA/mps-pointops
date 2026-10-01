#!/usr/bin/env python3
"""Measure native PyTorch MPS scatter sensitivity to destination fan-in.

Run Safe/Fast in separate fresh processes. This is synchronized host-wall
timing of native PyTorch calls, not GPU kernel or Metal atomic profiling.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
    raise SystemExit("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch")
if os.environ.get("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
    raise SystemExit("Set PYTORCH_MPS_FAST_MATH=0 or 1 before importing torch")

import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("uniform", "hub", "single_hot")
OPERATIONS = ("add_c32", "amax_scalar")


def command(*args: str) -> str:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def chip_name() -> str:
    for line in command("system_profiler", "SPHardwareDataType").splitlines():
        if line.strip().startswith("Chip:"):
            return line.partition(":")[2].strip()
    return "unknown"


def synchronize() -> None:
    torch.mps.synchronize()


def timed(call) -> float:
    synchronize()
    start = time.perf_counter_ns()
    call()
    synchronize()
    return (time.perf_counter_ns() - start) / 1e6


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "p10_ms": ordered[max(0, round(0.1 * (len(ordered) - 1)))],
        "p90_ms": ordered[min(len(ordered) - 1, round(0.9 * (len(ordered) - 1)))],
    }


def destinations(edges: int, nodes: int, pattern: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + edges)
    if pattern == "single_hot":
        return torch.zeros(edges, dtype=torch.int64)
    index = torch.randint(nodes, (edges,), generator=generator, dtype=torch.int64)
    if pattern == "hub":
        mask = torch.rand(edges, generator=generator) < 0.8
        index[mask] = torch.randint(max(1, nodes // 100), (int(mask.sum()),), generator=generator)
    return index


def make_source(edges: int, channels: int, operation: str, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed + edges + 17)
    if operation == "add_c32":
        # Every partial integer sum has magnitude <= 3*edges. For the default
        # maximum 2^21 edges, /256 preserves exact float32 addition order.
        return torch.randint(-3, 4, (edges, channels), generator=generator).to(torch.float32) / 256
    # Unique, positive and exactly representable float32 values: no max ties
    # and no interaction with an output initialized to zero.
    return (torch.randperm(edges, generator=generator).to(torch.float32).unsqueeze(1) + 1) / 2**24


def scatter(output: torch.Tensor, index: torch.Tensor, source: torch.Tensor, operation: str) -> None:
    if operation == "add_c32":
        output.scatter_add_(0, index, source)
    else:
        output.scatter_reduce_(0, index, source, reduce="amax", include_self=False)


def validate(
    nodes: int, index_cpu: torch.Tensor, source_cpu: torch.Tensor,
    index_mps: torch.Tensor, source_mps: torch.Tensor, operation: str,
) -> dict:
    channels = source_cpu.shape[1]
    upstream_cpu = torch.ones((nodes, channels), dtype=torch.float32)
    upstream_mps = upstream_cpu.to("mps")
    cpu_leaf = source_cpu.detach().clone().requires_grad_()
    mps_leaf = source_mps.detach().clone().requires_grad_()
    cpu_output = torch.zeros((nodes, channels), dtype=torch.float32)
    mps_output = torch.zeros((nodes, channels), dtype=torch.float32, device="mps")
    scatter(cpu_output, index_cpu, cpu_leaf, operation)
    scatter(mps_output, index_mps, mps_leaf, operation)
    cpu_output.backward(upstream_cpu)
    mps_output.backward(upstream_mps)
    synchronize()
    actual_output = mps_output.detach().cpu()
    actual_grad = mps_leaf.grad.detach().cpu()
    expected_grad = cpu_leaf.grad.detach()
    output_equal = torch.equal(actual_output, cpu_output.detach())
    grad_equal = torch.equal(actual_grad, expected_grad)
    if not output_equal or not grad_equal:
        raise AssertionError(
            f"CPU/MPS exact fixture mismatch in {operation}: output={output_equal}, gradient={grad_equal}"
        )
    return {
        "cpu_mps_output_bitwise_equal": output_equal,
        "cpu_mps_source_gradient_bitwise_equal": grad_equal,
        "output_all_finite": bool(torch.isfinite(actual_output).all()),
        "gradient_all_finite": bool(torch.isfinite(actual_grad).all()),
    }


def benchmark_case(
    edges: int, degree: int, channels: int, pattern: str, operation: str,
    warmup: int, repeat: int, seed: int,
) -> dict:
    nodes = edges // degree
    index_cpu = destinations(edges, nodes, pattern, seed)
    source_cpu = make_source(edges, channels, operation, seed)
    width = source_cpu.shape[1]
    counts = torch.bincount(index_cpu, minlength=nodes)
    index_cpu = index_cpu.unsqueeze(1).expand(-1, width)
    index_mps = index_cpu.to("mps")
    source_mps = source_cpu.to("mps")
    validation = validate(nodes, index_cpu, source_cpu, index_mps, source_mps, operation)
    output = torch.empty((nodes, width), dtype=torch.float32, device="mps")
    forward_samples = []
    for iteration in range(warmup + repeat):
        output.zero_()
        synchronize()  # Reset is outside the timed region.
        elapsed = timed(lambda: scatter(output, index_mps, source_mps, operation))
        if iteration >= warmup:
            forward_samples.append(elapsed)
    del output

    upstream = torch.ones((nodes, width), dtype=torch.float32, device="mps")
    source_leaf = source_mps.detach().requires_grad_()
    backward_samples = []
    for iteration in range(warmup + repeat):
        source_leaf.grad = None
        output = torch.zeros((nodes, width), dtype=torch.float32, device="mps")
        scatter(output, index_mps, source_leaf, operation)
        synchronize()  # Allocation and forward are outside the timed region.
        elapsed = timed(lambda: output.backward(upstream))
        if iteration >= warmup:
            backward_samples.append(elapsed)
        del output
    return {
        "shape": {
            "edges": edges,
            "nodes": nodes,
            "degree": degree,
            "channels": width,
            "pattern": pattern,
            "operation": operation,
            "occupied_destinations": int((counts > 0).sum()),
            "max_in_degree": int(counts.max()),
            "p99_in_degree": float(torch.quantile(counts.to(torch.float32), 0.99)),
            "top_one_percent_edge_fraction": float(
                counts.topk(max(1, nodes // 100)).values.sum() / edges
            ),
        },
        "validation": validation,
        "forward": summarize(forward_samples),
        "backward": summarize(backward_samples),
    }


def render_markdown(report: dict) -> str:
    env = report["environment"]
    lines = [
        "# Large native MPS scatter fan-in probe",
        "",
        f"Captured `{report['captured_utc']}` on {env['chip']}, macOS {env['macos']}, "
        f"PyTorch {env['torch']}, Fast Math `{env['fast_math']}`.",
        f"Base commit `{env['base_commit']}`; script SHA-256 `{env['script_sha256']}`.",
        f"Run: `{report['command']}`",
        "",
        f"{report['method']['warmup']} warmups and {report['method']['repeat']} timed calls per case. "
        "Times are synchronized host-wall milliseconds; full samples and validation are in the paired JSON.",
        "",
        "| Edges | Nodes | Channels | Destinations | Max degree | Top 1% share | Native op | Forward median | Backward median |",
        "| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for case in report["cases"]:
        shape = case["shape"]
        lines.append(
            f"| {shape['edges']} | {shape['nodes']} | {shape['channels']} | {shape['pattern']} | "
            f"{shape['max_in_degree']} | {shape['top_one_percent_edge_fraction']:.3f} | "
            f"{shape['operation']} | {case['forward']['median_ms']:.3f} | "
            f"{case['backward']['median_ms']:.3f} |"
        )
    lines += [
        "",
        "`add_c32` times native `scatter_add_` on channel-wide graph messages; "
        "`amax_scalar` times native `scatter_reduce_(amax, include_self=False)` "
        "on one attention-like value per edge. Uniform, 80%-to-1%-hub, and "
        "all-to-one destination mappings use the same edge count and source values "
        "within each operation and scale. Source/output transfer, output reset, "
        "and forward construction for backward are outside timed regions. "
        "The fixture uses exact float32 lattice sums or unique positive max "
        "values, and every case checks bitwise CPU/MPS output and source gradient.",
        "",
        "These observations describe sensitivity to destination fan-in in this "
        "PyTorch/MPS version. Host dispatch, synchronization, allocation around "
        "backward, power state, and other processes can affect timings. This "
        "script does not inspect GPU kernel duration, Metal atomics, or the "
        "backend primitive used by PyTorch. It cannot by itself justify a custom "
        "Metal reduction kernel or a universal PyG speed claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=int, nargs="+", default=[262144, 1048576, 2097152])
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--patterns", choices=PATTERNS, nargs="+", default=list(PATTERNS))
    parser.add_argument("--operations", choices=OPERATIONS, nargs="+", default=list(OPERATIONS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    if args.degree <= 0 or args.channels <= 0 or args.warmup < 0 or args.repeat < 1:
        parser.error("degree and channels must be positive; warmup >= 0 and repeat >= 1")
    if any(edges <= 0 or edges % args.degree or edges > 2**21 for edges in args.edges):
        parser.error("each edge count must be positive, divisible by degree, and <= 2^21")
    cases = []
    for edges in args.edges:
        for pattern in args.patterns:
            for operation in args.operations:
                print(f"E={edges} {pattern} {operation}", flush=True)
                cases.append(benchmark_case(
                    edges, args.degree, args.channels, pattern, operation,
                    args.warmup, args.repeat, args.seed,
                ))
    script = Path(__file__)
    report = {
        "schema": "mps-pointops.native-scatter-fanin.v1",
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": "PYTORCH_ENABLE_MPS_FALLBACK=0 "
        f"PYTORCH_MPS_FAST_MATH={os.environ['PYTORCH_MPS_FAST_MATH']} "
        + shlex.join(["python3", str(script.relative_to(ROOT)), *sys.argv[1:]]),
        "environment": {
            "chip": chip_name(),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyg": package_version("torch-geometric"),
            "fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
            "base_commit": command("git", "rev-parse", "HEAD"),
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        },
        "method": {
            "edges": args.edges,
            "degree": args.degree,
            "channels": args.channels,
            "patterns": args.patterns,
            "operations": args.operations,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "timing": "perf_counter_ns synchronized before and after each native PyTorch call",
            "output_reset_in_timed_forward": False,
            "forward_construction_in_timed_backward": False,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(report))
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

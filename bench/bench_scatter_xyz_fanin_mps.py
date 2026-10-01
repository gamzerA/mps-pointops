#!/usr/bin/env python3
"""Profile native scatter_add_ for batched 3D point-gradient contributions.

Input contributions have logical shape [B, N, 3] and are flattened to [B*N, 3].
Each contribution addresses one reference point in its own batch. Safe and
Fast Math must be run in separate processes with MPS CPU fallback disabled.
Times are synchronized host-wall call times, not GPU kernel measurements.
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
import sys
import time
from pathlib import Path

import bench_scatter_fanin_mps as base

torch = base.torch
ROOT = Path(__file__).resolve().parents[1]
PATTERNS = ("uniform", "hub", "single_hot")


def timed_cpu(call) -> float:
    start = time.perf_counter_ns()
    call()
    return (time.perf_counter_ns() - start) / 1e6


def make_workload(
    edges: int, batches: int, degree: int, pattern: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    edges_per_batch = edges // batches
    nodes_per_batch = edges_per_batch // degree
    nodes = nodes_per_batch * batches
    index_rows = []
    for batch in range(batches):
        local = base.destinations(edges_per_batch, nodes_per_batch, pattern, seed + batch * 10007)
        index_rows.append(local + batch * nodes_per_batch)
    index = torch.cat(index_rows)
    source_batched = base.make_source(edges, 3, "add_c32", seed).reshape(
        batches, edges_per_batch, 3
    )
    source = source_batched.reshape(edges, 3)
    counts = torch.bincount(index, minlength=nodes)
    shape = {
        "logical_source": [batches, edges_per_batch, 3],
        "flattened_source": [edges, 3],
        "flattened_output": [nodes, 3],
        "batches": batches,
        "contributions": edges,
        "destinations": nodes,
        "nominal_degree": degree,
        "pattern": pattern,
        "occupied_destinations": int((counts > 0).sum()),
        "max_in_degree": int(counts.max()),
        "p99_in_degree": float(torch.quantile(counts.to(torch.float32), 0.99)),
        "top_one_percent_contribution_fraction": float(
            counts.topk(max(1, nodes // 100)).values.sum() / edges
        ),
        "single_hot_destinations": batches if pattern == "single_hot" else None,
    }
    return index, source, counts, shape


def validate(
    nodes: int,
    index_cpu: torch.Tensor,
    source_cpu: torch.Tensor,
    index_mps: torch.Tensor,
    source_mps: torch.Tensor,
) -> dict:
    # Nonuniform upstream values verify the backward index mapping as well as
    # the 3-channel source gradient. Values are exact binary fractions.
    upstream_cpu = (
        (torch.arange(nodes * 3, dtype=torch.int64).reshape(nodes, 3) % 17) - 8
    ).to(torch.float32) / 16
    upstream_mps = upstream_cpu.to("mps")
    cpu_leaf = source_cpu.detach().clone().requires_grad_()
    mps_leaf = source_mps.detach().clone().requires_grad_()
    cpu_output = torch.zeros((nodes, 3), dtype=torch.float32)
    mps_output = torch.zeros((nodes, 3), dtype=torch.float32, device="mps")
    expanded_cpu = index_cpu[:, None].expand(-1, 3)
    expanded_mps = index_mps[:, None].expand(-1, 3)
    cpu_output.scatter_add_(0, expanded_cpu, cpu_leaf)
    mps_output.scatter_add_(0, expanded_mps, mps_leaf)
    cpu_output.backward(upstream_cpu)
    mps_output.backward(upstream_mps)
    base.synchronize()
    actual_output = mps_output.detach().cpu()
    actual_grad = mps_leaf.grad.detach().cpu()
    expected_output = cpu_output.detach()
    expected_grad = cpu_leaf.grad.detach()
    expected_mapping = upstream_cpu.index_select(0, index_cpu)
    output_equal = torch.equal(actual_output, expected_output)
    grad_equal = torch.equal(actual_grad, expected_grad)
    mapping_equal = torch.equal(actual_grad, expected_mapping)
    if not (output_equal and grad_equal and mapping_equal):
        raise AssertionError(
            "3-channel CPU/MPS exact fixture mismatch: "
            f"output={output_equal}, source_gradient={grad_equal}, "
            f"upstream_index_mapping={mapping_equal}"
        )
    return {
        "cpu_mps_output_bitwise_equal": output_equal,
        "cpu_mps_source_gradient_bitwise_equal": grad_equal,
        "source_gradient_matches_upstream_at_global_index": mapping_equal,
        "output_all_finite": bool(torch.isfinite(actual_output).all()),
        "gradient_all_finite": bool(torch.isfinite(actual_grad).all()),
        "output_max_abs_difference": float((actual_output - expected_output).abs().max()),
        "gradient_max_abs_difference": float((actual_grad - expected_grad).abs().max()),
    }


def benchmark_case(
    edges: int,
    batches: int,
    degree: int,
    pattern: str,
    seed: int,
    warmup: int,
    repeat: int,
    cpu_warmup: int,
    cpu_repeat: int,
) -> dict:
    index_cpu, source_cpu, _counts, shape = make_workload(edges, batches, degree, pattern, seed)
    nodes = shape["destinations"]
    index_mps = index_cpu.to("mps")
    source_mps = source_cpu.to("mps")
    validation = validate(nodes, index_cpu, source_cpu, index_mps, source_mps)
    expanded_cpu = index_cpu[:, None].expand(-1, 3)
    expanded_mps = index_mps[:, None].expand(-1, 3)

    forward_mps = []
    output_mps = torch.empty((nodes, 3), dtype=torch.float32, device="mps")
    for iteration in range(warmup + repeat):
        output_mps.zero_()
        base.synchronize()
        elapsed = base.timed(lambda: output_mps.scatter_add_(0, expanded_mps, source_mps))
        if iteration >= warmup:
            forward_mps.append(elapsed)
    del output_mps

    forward_cpu = []
    output_cpu = torch.empty((nodes, 3), dtype=torch.float32)
    for iteration in range(cpu_warmup + cpu_repeat):
        output_cpu.zero_()
        elapsed = timed_cpu(lambda: output_cpu.scatter_add_(0, expanded_cpu, source_cpu))
        if iteration >= cpu_warmup:
            forward_cpu.append(elapsed)
    del output_cpu

    upstream_cpu = torch.ones((nodes, 3), dtype=torch.float32)
    upstream_mps = upstream_cpu.to("mps")
    source_leaf_mps = source_mps.detach().requires_grad_()
    backward_mps = []
    for iteration in range(warmup + repeat):
        source_leaf_mps.grad = None
        output_mps = torch.zeros((nodes, 3), dtype=torch.float32, device="mps")
        output_mps.scatter_add_(0, expanded_mps, source_leaf_mps)
        base.synchronize()
        elapsed = base.timed(lambda: output_mps.backward(upstream_mps))
        if iteration >= warmup:
            backward_mps.append(elapsed)
        del output_mps

    source_leaf_cpu = source_cpu.detach().requires_grad_()
    backward_cpu = []
    for iteration in range(cpu_warmup + cpu_repeat):
        source_leaf_cpu.grad = None
        output_cpu = torch.zeros((nodes, 3), dtype=torch.float32)
        output_cpu.scatter_add_(0, expanded_cpu, source_leaf_cpu)
        elapsed = timed_cpu(lambda: output_cpu.backward(upstream_cpu))
        if iteration >= cpu_warmup:
            backward_cpu.append(elapsed)
        del output_cpu

    return {
        "shape": shape,
        "validation": validation,
        "mps_forward": base.summarize(forward_mps),
        "mps_backward": base.summarize(backward_mps),
        "cpu_forward": base.summarize(forward_cpu),
        "cpu_backward": base.summarize(backward_cpu),
    }


def render_markdown(report: dict) -> str:
    env = report["environment"]
    lines = [
        "# Three-channel point-gradient scatter fan-in probe",
        "",
        f"Captured `{report['captured_utc']}` on {env['chip']}, macOS {env['macos']}, "
        f"PyTorch {env['torch']}, Fast Math `{env['fast_math']}`.",
        f"Base commit `{env['base_commit']}`; script SHA-256 `{env['script_sha256']}`; "
        f"imported helper SHA-256 `{env['helper_sha256']}`.",
        f"Run: `{report['command']}`",
        "",
        "| Contributions | B | Pattern | Max fan-in | Top 1% share | MPS fwd ms | MPS bwd ms | CPU fwd ms | CPU bwd ms |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        shape = case["shape"]
        lines.append(
            f"| {shape['contributions']} | {shape['batches']} | {shape['pattern']} | "
            f"{shape['max_in_degree']} | {shape['top_one_percent_contribution_fraction']:.3f} | "
            f"{case['mps_forward']['median_ms']:.3f} | {case['mps_backward']['median_ms']:.3f} | "
            f"{case['cpu_forward']['median_ms']:.3f} | {case['cpu_backward']['median_ms']:.3f} |"
        )
    lines += [
        "",
        "The logical [B,N,3] contribution tensor is flattened to [B*N,3]. "
        "Each source addresses one reference point within its own batch; "
        "single_hot means one destination per batch. Sources are exact "
        "float32 binary-lattice values, so this fixture checks bitwise "
        "CPU/MPS output and a nonuniform-index source gradient. This does "
        "not imply general floating-point determinism.",
        "",
        "All MPS calls are timed as synchronized host-wall intervals. Output "
        "reset is outside forward timing; forward construction is outside "
        "backward timing. CPU times use the same call boundaries without "
        "MPS synchronization. Input generation and device transfers are "
        "excluded. The method cannot identify the PyTorch GPU primitive or "
        "attribute a latency difference to Metal atomics.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", type=int, nargs="+", default=[262144, 1048576])
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--patterns", choices=PATTERNS, nargs="+", default=list(PATTERNS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=12)
    parser.add_argument("--cpu-warmup", type=int, default=2)
    parser.add_argument("--cpu-repeat", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    if args.batches <= 0 or args.degree <= 0 or args.warmup < 0 or args.repeat < 1:
        parser.error("batches and degree must be positive; warmup >= 0; repeat >= 1")
    if args.cpu_warmup < 0 or args.cpu_repeat < 1:
        parser.error("CPU warmup >= 0 and repeat >= 1 are required")
    if any(edges <= 0 or edges > 2**21 or edges % (args.batches * args.degree) for edges in args.edges):
        parser.error("each edge count must be positive, <= 2^21, and divisible by batches*degree")

    cases = []
    for edges in args.edges:
        for pattern in args.patterns:
            print(f"contributions={edges} B={args.batches} pattern={pattern}", flush=True)
            cases.append(benchmark_case(
                edges, args.batches, args.degree, pattern, args.seed,
                args.warmup, args.repeat, args.cpu_warmup, args.cpu_repeat,
            ))

    script = Path(__file__)
    helper = Path(base.__file__)
    report = {
        "schema": "mps-pointops.xyz-scatter-fanin.v1",
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": "PYTORCH_ENABLE_MPS_FALLBACK=0 "
        f"PYTORCH_MPS_FAST_MATH={os.environ['PYTORCH_MPS_FAST_MATH']} "
        + shlex.join(["python3", str(script.relative_to(ROOT)), *sys.argv[1:]]),
        "environment": {
            "chip": base.chip_name(),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyg": base.package_version("torch-geometric"),
            "fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
            "cpu_threads": torch.get_num_threads(),
            "base_commit": base.command("git", "rev-parse", "HEAD"),
            "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        },
        "method": {
            "edges": args.edges,
            "batches": args.batches,
            "degree": args.degree,
            "channels": 3,
            "patterns": args.patterns,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "cpu_warmup": args.cpu_warmup,
            "cpu_repeat": args.cpu_repeat,
            "seed": args.seed,
            "mps_timing": "perf_counter_ns synchronized before and after each native PyTorch call",
            "cpu_timing": "perf_counter_ns around the same native PyTorch call",
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

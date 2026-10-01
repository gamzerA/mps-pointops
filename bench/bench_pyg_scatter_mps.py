#!/usr/bin/env python3
"""Profile native PyTorch scatter in representative PyG 2.8 MPS graphs.

Run in a fresh process with CPU fallback disabled::

    PYTORCH_ENABLE_MPS_FALLBACK=0 python bench/bench_pyg_scatter_mps.py \
        --output bench/results/local-pyg-scatter.json

This is a survey, not a custom-kernel speed comparison. Isolated operations
use preallocated source/index/output tensors; model timings include the entire
PyG two-layer forward or backward. Each timed call is synchronized on MPS.
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

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
    raise SystemExit("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch")

import torch  # noqa: E402
import torch_geometric  # noqa: E402
from torch_geometric.nn.models import GAT, GCN, GraphSAGE  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
REDUCTIONS = ("scatter_add", "sum", "mean", "amax", "amin", "scatter_add_scalar", "amax_scalar")
MODELS = ("GCN", "GraphSAGE", "GAT")


def command(*args: str) -> str:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chip_name() -> str:
    for line in command("system_profiler", "SPHardwareDataType").splitlines():
        if line.strip().startswith("Chip:"):
            return line.partition(":")[2].strip()
    return "unknown"


def synchronize() -> None:
    torch.mps.synchronize()


def timed(call) -> tuple[object, float]:
    synchronize()
    start = time.perf_counter_ns()
    value = call()
    synchronize()
    return value, (time.perf_counter_ns() - start) / 1e6


def summarized(samples: list[float]) -> dict:
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def make_graph(nodes: int, degree: int, channels: int, pattern: str, seed: int) -> dict:
    """Uniform destination or 80% of edges directed at the first 1% of nodes."""
    generator = torch.Generator().manual_seed(seed)
    edges = nodes * degree
    src_cpu = torch.randint(nodes, (edges,), generator=generator, dtype=torch.int64)
    dst_cpu = torch.randint(nodes, (edges,), generator=generator, dtype=torch.int64)
    if pattern == "hub":
        hub_count = max(1, nodes // 100)
        hub_mask = torch.rand((edges,), generator=generator) < 0.8
        dst_cpu[hub_mask] = torch.randint(
            hub_count, (int(hub_mask.sum()),), generator=generator, dtype=torch.int64
        )
    features_cpu = torch.randn((nodes, channels), generator=generator, dtype=torch.float32)
    source_cpu = features_cpu[src_cpu].contiguous()
    # GAT adds one self-loop per node and reduces one attention value per
    # edge/head; its amax path does not reduce the 32-channel message tensor.
    attention_dst_cpu = torch.cat((dst_cpu, torch.arange(nodes, dtype=torch.int64)))
    attention_source_cpu = torch.cat((source_cpu[:, :1].contiguous(), features_cpu[:, :1].contiguous()))
    counts = torch.bincount(dst_cpu, minlength=nodes)
    return {
        "nodes": nodes,
        "edges": edges,
        "channels": channels,
        "degree": degree,
        "pattern": pattern,
        "seed": seed,
        "src_cpu": src_cpu,
        "dst_cpu": dst_cpu,
        "features_cpu": features_cpu,
        "source_cpu": source_cpu,
        "attention_dst_cpu": attention_dst_cpu,
        "attention_source_cpu": attention_source_cpu,
        "edge_index": torch.stack((src_cpu, dst_cpu)).to("mps"),
        "features": features_cpu.to("mps"),
        "source": source_cpu.to("mps"),
        "index": dst_cpu.to("mps").unsqueeze(-1).expand(-1, channels),
        "attention_source": attention_source_cpu.to("mps"),
        "attention_index": attention_dst_cpu.to("mps").unsqueeze(-1),
        "destination_stats": {
            "occupied_nodes": int((counts > 0).sum().item()),
            "max_in_degree": int(counts.max().item()),
            "p99_in_degree": float(torch.quantile(counts.float(), 0.99).item()),
        },
    }


def scatter_call(output: torch.Tensor, index: torch.Tensor, source: torch.Tensor, operation: str) -> None:
    operation = operation.removesuffix("_scalar")
    if operation == "scatter_add":
        output.scatter_add_(0, index, source)
    else:
        output.scatter_reduce_(0, index, source, reduce=operation, include_self=False)


def scatter_correctness(graph: dict, operation: str) -> dict:
    nodes, channels = graph["nodes"], graph["channels"]
    if operation.endswith("_scalar"):
        channels = 1
        cpu_index = graph["attention_dst_cpu"].unsqueeze(-1)
        cpu_source = graph["attention_source_cpu"]
        mps_index = graph["attention_index"]
        mps_source = graph["attention_source"]
    else:
        cpu_index = graph["dst_cpu"].unsqueeze(-1).expand(-1, channels)
        cpu_source = graph["source_cpu"]
        mps_index = graph["index"]
        mps_source = graph["source"]
    expected = torch.zeros((nodes, channels), dtype=torch.float32)
    actual = torch.zeros((nodes, channels), device="mps", dtype=torch.float32)
    scatter_call(expected, cpu_index, cpu_source, operation)
    scatter_call(actual, mps_index, mps_source, operation)
    synchronize()
    actual_cpu = actual.cpu()
    # Floating-point atomic reduction order is not guaranteed across CPU/MPS.
    torch.testing.assert_close(actual_cpu, expected, rtol=1e-4, atol=1e-4)
    return {
        "status": "passed",
        "rtol": 1e-4,
        "atol": 1e-4,
        "max_abs_error": float((actual_cpu - expected).abs().max().item()),
        "all_finite": bool(torch.isfinite(actual_cpu).all().item()),
    }


def benchmark_scatter(graph: dict, warmup: int, repeat: int) -> dict:
    results = {}
    for operation in REDUCTIONS:
        scalar = operation.endswith("_scalar")
        output = torch.empty((graph["nodes"], 1 if scalar else graph["channels"]), device="mps")
        index = graph["attention_index"] if scalar else graph["index"]
        source = graph["attention_source"] if scalar else graph["source"]
        validation = scatter_correctness(graph, operation)
        samples = []
        for iteration in range(warmup + repeat):
            output.zero_()  # Reset is intentionally outside the measured region.
            synchronize()
            _, elapsed = timed(lambda: scatter_call(output, index, source, operation))
            if iteration >= warmup:
                samples.append(elapsed)
        results[operation] = {"validation": validation, "timing": summarized(samples)}
    return results


def make_model(name: str, channels: int):
    model_class = {"GCN": GCN, "GraphSAGE": GraphSAGE, "GAT": GAT}[name]
    return model_class(channels, channels, num_layers=2, out_channels=channels, dropout=0.0)


def model_correctness(model_cpu, graph: dict) -> dict:
    """Check one full forward/backward against CPU for the same graph and weights."""
    import copy

    model_mps = copy.deepcopy(model_cpu).to("mps")
    x_cpu = graph["features_cpu"].detach().clone().requires_grad_()
    x_mps = graph["features"].detach().clone().requires_grad_()
    edge_cpu = torch.stack((graph["src_cpu"], graph["dst_cpu"]))
    out_cpu = model_cpu(x_cpu, edge_cpu)
    out_mps = model_mps(x_mps, graph["edge_index"])
    loss_cpu = out_cpu.square().mean()
    loss_mps = out_mps.square().mean()
    loss_cpu.backward()
    loss_mps.backward()
    synchronize()
    out_mps_cpu = out_mps.detach().cpu()
    grad_mps_cpu = x_mps.grad.detach().cpu()
    torch.testing.assert_close(out_mps_cpu, out_cpu.detach(), rtol=5e-4, atol=5e-4)
    torch.testing.assert_close(grad_mps_cpu, x_cpu.grad, rtol=5e-4, atol=5e-4)
    return {
        "status": "passed",
        "rtol": 5e-4,
        "atol": 5e-4,
        "max_abs_output_error": float((out_mps_cpu - out_cpu.detach()).abs().max().item()),
        "max_abs_input_gradient_error": float((grad_mps_cpu - x_cpu.grad).abs().max().item()),
    }


def model_dispatch_operators(model, x: torch.Tensor, edge_index: torch.Tensor) -> dict:
    """CPU profiler names/counts only: these are not GPU timings."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profiler:
        y = model(x, edge_index)
        y.square().mean().backward()
        synchronize()
    keys = ("scatter_add_", "scatter_reduce_", "index_add_", "gather")
    counts = {
        event.key: event.count for event in profiler.key_averages()
        if any(key in event.key for key in keys)
    }
    shapes = {}
    for event in profiler.events():
        if any(key in event.name for key in keys):
            observed = event.input_shapes
            if observed not in shapes.setdefault(event.name, []):
                shapes[event.name].append(observed)
    return {"counts": counts, "input_shapes": shapes}


def benchmark_models(graph: dict, warmup: int, repeat: int, verify_cpu: bool) -> dict:
    results = {}
    for offset, name in enumerate(MODELS):
        torch.manual_seed(graph["seed"] + offset + 1000)
        model_cpu = make_model(name, graph["channels"])
        validation = model_correctness(model_cpu, graph) if verify_cpu else {
            "status": "not_run", "reason": "CPU parity limited to the smaller graph"
        }
        model = model_cpu.to("mps")
        x = graph["features"].detach().clone().requires_grad_()
        operators = model_dispatch_operators(model, x, graph["edge_index"])
        model.zero_grad(set_to_none=True)
        x.grad = None
        forward_samples, backward_samples = [], []
        for iteration in range(warmup + repeat):
            model.zero_grad(set_to_none=True)
            x.grad = None
            output, forward_ms = timed(lambda: model(x, graph["edge_index"]))
            loss = output.square().mean()
            _, backward_ms = timed(loss.backward)
            if iteration >= warmup:
                forward_samples.append(forward_ms)
                backward_samples.append(backward_ms)
        assert x.grad is not None and torch.isfinite(x.grad).all().item()
        results[name] = {
            "validation": validation,
            "dispatch_operators": operators,
            "forward": summarized(forward_samples),
            "backward": summarized(backward_samples),
        }
    return results


def render_markdown(report: dict) -> str:
    environment = report["environment"]
    method = report["method"]
    lines = [
        "# Native PyTorch scatter in PyG 2.8 graph workloads on MPS",
        "",
        f"Captured {report['captured_utc']}; code base `{environment['base_commit']}`; "
        f"benchmark SHA-256 `{environment['script_sha256']}`.",
        "",
        f"{environment['chip']}; macOS {environment['macos']}; Python {environment['python']}; "
        f"PyTorch {environment['torch']}; PyG {environment['pyg']}. "
        "`PYTORCH_ENABLE_MPS_FALLBACK=0` was fixed before importing PyTorch.",
        f"Reproduce: `{report['command']}`",
        "",
        f"{method['warmup']} warmups, {method['repeat']} timed calls per case. "
        "All values below are synchronized host-wall medians in milliseconds.",
        "",
        "## Native operation controls",
        "",
        "| Nodes | Edges | Channels | Destination | Max in-degree | `scatter_add_` | `scatter_reduce_` sum | mean | amax | amin | GAT scalar add | GAT scalar amax |",
        "| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:
        shape = case["shape"]
        scatter = case["scatter"]
        ms = lambda op: f"{scatter[op]['timing']['median_ms']:.3f}"
        lines.append(
            f"| {shape['nodes']} | {shape['edges']} | {shape['channels']} | {shape['pattern']} | "
            f"{shape['max_in_degree']} | {ms('scatter_add')} | {ms('sum')} | {ms('mean')} | "
            f"{ms('amax')} | {ms('amin')} | {ms('scatter_add_scalar')} | {ms('amax_scalar')} |"
        )
    lines += [
        "",
        "## Two-layer PyG models",
        "",
        "| Nodes | Edges | Destination | Model | Forward | Backward | Observed CPU-dispatch operators |",
        "| ---: | ---: | --- | --- | ---: | ---: | --- |",
    ]
    for case in report["cases"]:
        shape = case["shape"]
        for name in MODELS:
            model = case["models"][name]
            operators = ", ".join(
                f"{key}×{count}" for key, count in sorted(model["dispatch_operators"]["counts"].items())
            )
            lines.append(
                f"| {shape['nodes']} | {shape['edges']} | {shape['pattern']} | {name} | "
                f"{model['forward']['median_ms']:.3f} | {model['backward']['median_ms']:.3f} | {operators} |"
            )
    lines += [
        "",
        "## Scope and interpretation",
        "",
        "- Graphs are seeded synthetic directed edges. Uniform destinations are sampled across all nodes; "
        "hub destinations send 80% of edges to the first 1% of nodes. These are stress shapes, not a public dataset.",
        "- Channel-wide isolated operations use a precomputed contiguous float32 edge-feature source, "
        "int64 destination indices expanded across channels, and a preallocated output. The GAT scalar controls "
        "append one self-loop per node, matching its attention reduction shape. Output reset is outside timing. "
        "All seven MPS outputs are compared with CPU using rtol=atol=1e-4; reduction order is not bit-stable.",
        "- Models use PyG 2.8 GCN, GraphSAGE and GAT, two layers, dropout 0, float32. The smaller "
        "graph cases also compare model output and input gradient with CPU (rtol=atol=5e-4). "
        "Full model timings include normalization, gather, aggregation, dense layers, autograd and dispatch.",
        "- CPU profiler operator names/counts establish that a native scatter path was invoked, but "
        "its self CPU time is *not* GPU time. Input shapes are in the raw JSON. The isolated scatter median "
        "cannot be divided by the "
        "model median to assign a model bottleneck percentage: message shapes and dispatch counts differ.",
        "- `scatter_reduce_` uses `include_self=False` for sum, mean, amax and amin; this matches "
        "PyG's MPS min/max reduction path. PyG mean aggregation itself uses `scatter_add_` for "
        "sums and counts. The raw JSON retains every timing sample and correctness result.",
        "- A custom Metal segmented reduction or float-atomic path should be built only after "
        "an end-to-end bottleneck is established and compared on more hardware and graph distributions.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[4096, 32768])
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2201)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    if torch_geometric.__version__ != "2.8.0":
        parser.error(f"expected PyG 2.8.0, got {torch_geometric.__version__}")
    if min(args.nodes) <= 0 or min(args.degree, args.channels, args.repeat) <= 0 or args.warmup < 0:
        parser.error("nodes, degree, channels and repeat must be positive; warmup cannot be negative")
    cases = []
    for nodes in args.nodes:
        for pattern_index, pattern in enumerate(("uniform", "hub")):
            graph = make_graph(nodes, args.degree, args.channels, pattern, args.seed + nodes + pattern_index)
            print(f"Running N={nodes}, E={graph['edges']}, C={args.channels}, {pattern}", flush=True)
            scatter = benchmark_scatter(graph, args.warmup, args.repeat)
            models = benchmark_models(graph, args.warmup, args.repeat, verify_cpu=nodes == min(args.nodes))
            cases.append({
                "shape": {
                    "nodes": nodes,
                    "edges": graph["edges"],
                    "channels": args.channels,
                    "degree": args.degree,
                    "pattern": pattern,
                    "seed": graph["seed"],
                    **graph["destination_stats"],
                },
                "scatter": scatter,
                "models": models,
            })
    report = {
        "schema": "mps-pointops.pyg-scatter-profile.v1",
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": "PYTORCH_ENABLE_MPS_FALLBACK=0 "
        f"PYTORCH_MPS_FAST_MATH={os.environ.get('PYTORCH_MPS_FAST_MATH', 'unset')} " + shlex.join([
            "python3", str(Path(__file__).resolve().relative_to(ROOT)), *sys.argv[1:]
        ]),
        "environment": {
            "chip": chip_name(),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyg": torch_geometric.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "pytorch_enable_mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "pytorch_mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "unset"),
            "base_commit": command("git", "rev-parse", "HEAD"),
            "script_sha256": sha256(Path(__file__)),
            "pyg_scatter_source_sha256": sha256(Path(torch_geometric.__file__).parent / "utils" / "_scatter.py"),
        },
        "method": {
            "nodes": args.nodes,
            "degree": args.degree,
            "channels": args.channels,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "seed": args.seed,
            "timing": "perf_counter_ns host wall around each call, torch.mps.synchronize before and after",
            "scatter_output_reset_in_timer": False,
            "graph_patterns": "uniform destinations, or 80% of edges to first 1% of nodes",
            "model": "two-layer GCN, GraphSAGE, GAT; dropout 0; output squared-mean loss",
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

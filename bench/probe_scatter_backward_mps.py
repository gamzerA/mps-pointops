#!/usr/bin/env python3
"""Check native MPS scatter backward with ties and empty graph segments.

Run Safe and Fast Math in separate processes with CPU fallback disabled::

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python3 bench/probe_scatter_backward_mps.py --output /tmp/scatter-safe.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
      python3 bench/probe_scatter_backward_mps.py --output /tmp/scatter-fast.json

This small numerical probe does not time kernels or exercise a custom Metal
implementation. An independent Python oracle derives forward and first-order
source gradients for five native PyTorch operations.
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
import subprocess
import sys
from pathlib import Path

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
    raise SystemExit("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch")
if os.environ.get("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
    raise SystemExit("Set PYTORCH_MPS_FAST_MATH=0 or 1 before importing torch")

import torch  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ("scatter_add", "sum", "mean", "amax", "amin")
INDEX = (0, 0, 0, 2, 2, 2)
SOURCE = (
    (1.0, -4.0),
    (1.0, 3.0),
    (-2.0, 3.0),
    (4.0, -1.0),
    (4.0, -1.0),
    (0.0, 0.0),
)
UPSTREAM = (
    (2.0, 6.0),
    (7.0, 8.0),
    (3.0, 4.0),
    (9.0, 10.0),
)
NODES = len(UPSTREAM)
CHANNELS = len(SOURCE[0])


def command(*args: str) -> str:
    try:
        return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def chip_name() -> str:
    for line in command("system_profiler", "SPHardwareDataType").splitlines():
        if line.strip().startswith("Chip:"):
            return line.partition(":")[2].strip()
    return "unknown"


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def oracle(operation: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive each segment/channel independently from exact small integers."""
    out = [[0.0] * CHANNELS for _ in range(NODES)]
    grad = [[0.0] * CHANNELS for _ in INDEX]
    for destination in range(NODES):
        edges = [e for e, index in enumerate(INDEX) if index == destination]
        for channel in range(CHANNELS):
            if not edges:
                continue
            values = [SOURCE[e][channel] for e in edges]
            if operation in ("scatter_add", "sum", "mean"):
                total = sum(values)
                out[destination][channel] = total / len(edges) if operation == "mean" else total
                contribution = UPSTREAM[destination][channel] / (len(edges) if operation == "mean" else 1)
                for edge in edges:
                    grad[edge][channel] = contribution
            else:
                extreme = max(values) if operation == "amax" else min(values)
                out[destination][channel] = extreme
                tied = [edge for edge in edges if SOURCE[edge][channel] == extreme]
                contribution = UPSTREAM[destination][channel] / len(tied)
                for edge in tied:
                    grad[edge][channel] = contribution
    return torch.tensor(out, dtype=torch.float32), torch.tensor(grad, dtype=torch.float32)


def execute(operation: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    source = torch.tensor(SOURCE, dtype=torch.float32, device=device).requires_grad_()
    index = torch.tensor(INDEX, dtype=torch.int64, device=device).unsqueeze(-1).expand(-1, CHANNELS)
    upstream = torch.tensor(UPSTREAM, dtype=torch.float32, device=device)
    output = torch.zeros((NODES, CHANNELS), dtype=torch.float32, device=device)
    if operation == "scatter_add":
        output.scatter_add_(0, index, source)
    else:
        output.scatter_reduce_(0, index, source, reduce=operation, include_self=False)
    output.backward(upstream)
    if device == "mps":
        torch.mps.synchronize()
    assert source.grad is not None
    return output.detach().cpu(), source.grad.detach().cpu()


def verify(operation: str, repeats: int) -> dict:
    expected_output, expected_gradient = oracle(operation)
    cpu_output, cpu_gradient = execute(operation, "cpu")
    torch.testing.assert_close(cpu_output, expected_output, rtol=0, atol=1e-7)
    cpu_gradient_matches_math = bool(torch.allclose(cpu_gradient, expected_gradient, rtol=1e-6, atol=1e-6))

    baseline_output = baseline_gradient = None
    output_bitwise_stable = gradient_bitwise_stable = True
    max_output_error = max_gradient_error_vs_cpu = max_gradient_error_vs_math = 0.0
    mps_gradient_matches_cpu = True
    mps_gradient_matches_math = True
    for _ in range(repeats):
        output, gradient = execute(operation, "mps")
        torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
        max_output_error = max(max_output_error, float((output - expected_output).abs().max().item()))
        max_gradient_error_vs_cpu = max(
            max_gradient_error_vs_cpu, float((gradient - cpu_gradient).abs().max().item())
        )
        max_gradient_error_vs_math = max(
            max_gradient_error_vs_math, float((gradient - expected_gradient).abs().max().item())
        )
        mps_gradient_matches_cpu &= bool(torch.allclose(gradient, cpu_gradient, rtol=1e-6, atol=1e-6))
        mps_gradient_matches_math &= bool(
            torch.allclose(gradient, expected_gradient, rtol=1e-6, atol=1e-6)
        )
        if baseline_output is None:
            baseline_output, baseline_gradient = output, gradient
        else:
            output_bitwise_stable &= torch.equal(output, baseline_output)
            gradient_bitwise_stable &= torch.equal(gradient, baseline_gradient)
    assert baseline_output is not None and baseline_gradient is not None
    return {
        "status": "passed" if mps_gradient_matches_cpu else "mps_cpu_gradient_mismatch",
        "expected_output": expected_output.tolist(),
        "expected_source_gradient": expected_gradient.tolist(),
        "cpu_output": cpu_output.tolist(),
        "cpu_source_gradient": cpu_gradient.tolist(),
        "mps_output_first_repeat": baseline_output.tolist(),
        "mps_source_gradient_first_repeat": baseline_gradient.tolist(),
        "max_abs_mps_output_error": max_output_error,
        "max_abs_cpu_source_gradient_error_vs_math": float(
            (cpu_gradient - expected_gradient).abs().max().item()
        ),
        "max_abs_mps_source_gradient_error_vs_cpu": max_gradient_error_vs_cpu,
        "max_abs_mps_source_gradient_error_vs_math": max_gradient_error_vs_math,
        "cpu_gradient_matches_math": cpu_gradient_matches_math,
        "mps_gradient_matches_cpu": mps_gradient_matches_cpu,
        "mps_gradient_matches_math": mps_gradient_matches_math,
        "output_bitwise_equal_across_repeats": output_bitwise_stable,
        "gradient_bitwise_equal_across_repeats": gradient_bitwise_stable,
    }


def render_markdown(report: dict) -> str:
    env = report["environment"]
    rows = []
    for name, result in report["cases"].items():
        rows.append(
            f"| `{name}` | {result['status']} | {result['max_abs_mps_output_error']:.3g} | "
            f"{result['max_abs_mps_source_gradient_error_vs_cpu']:.3g} | "
            f"{result['max_abs_cpu_source_gradient_error_vs_math']:.3g} | "
            f"{result['cpu_gradient_matches_math']} | {result['mps_gradient_matches_math']} | "
            f"{result['output_bitwise_equal_across_repeats']} | "
            f"{result['gradient_bitwise_equal_across_repeats']} |"
        )
    return "\n".join([
        "# Native MPS scatter backward: tied extrema and empty segments",
        "",
        f"Captured {report['captured_utc']}; `{env['chip']}`, macOS {env['macos']}, "
        f"PyTorch {env['torch']}, Fast Math `{env['pytorch_mps_fast_math']}`.",
        f"Repository base `{env['base_commit']}`; script SHA-256 `{env['script_sha256']}`.",
        f"Reproduce: `{report['command']}`",
        "",
        f"CPU and MPS forward were checked against an independent segment-wise Python oracle; "
        f"MPS ran {report['method']['repeats']} complete forward/backward repetitions. "
        "The process disabled MPS CPU fallback before importing PyTorch.",
        "",
        "| Operation | MPS↔CPU grad | Max MPS output error vs math | Max MPS grad error vs CPU | Max CPU grad error vs math | CPU grad matches math | MPS grad matches math | Output repeat bits | Gradient repeat bits |",
        "| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        *rows,
        "",
        "The source has shape `[6, 2]`; destinations are `[0, 0, 0, 2, 2, 2]` "
        "within a four-node output. Nodes 1 and 3 are empty but receive nonzero "
        "upstream gradients, testing that those gradients do not leak to source "
        "edges. Both channels contain exact tied extrema; node 2, channel 1 has "
        "a maximum of zero from a source edge, testing `include_self=False`.",
        "",
        "The oracle splits `amax`/`amin` gradient among exact tied *source* "
        "extrema. PyTorch's observed native backward can differ when a "
        "nonempty segment's extremum equals the initial output zero, even with "
        "`include_self=False`; the table records that discrepancy rather than "
        "silently treating CPU as the mathematical oracle. Empty output "
        "segments stay zero. This establishes only this PyTorch version, "
        "dtype, fixture and device; it does not establish global bitwise "
        "determinism, NaN behavior, or `torch_scatter` arg-index compatibility. "
        "No kernel timing is included.",
        "",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be at least 1")
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    cases = {operation: verify(operation, args.repeats) for operation in OPERATIONS}
    report = {
        "schema": "mps-pointops.scatter-backward-probe.v1",
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": "PYTORCH_ENABLE_MPS_FALLBACK=0 "
        f"PYTORCH_MPS_FAST_MATH={os.environ['PYTORCH_MPS_FAST_MATH']} "
        + shlex.join(["python3", str(Path(__file__).resolve().relative_to(ROOT)), *sys.argv[1:]]),
        "environment": {
            "chip": chip_name(),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pyg": version("torch-geometric"),
            "pytorch_enable_mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "pytorch_mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
            "base_commit": command("git", "rev-parse", "HEAD"),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "method": {
            "source": SOURCE,
            "destinations": INDEX,
            "upstream_gradient": UPSTREAM,
            "dtype": "float32 source/output; int64 index",
            "include_self": False,
            "repeats": args.repeats,
            "timing": "none",
            "numerical_tolerance": "forward rtol=atol=1e-6 for MPS vs oracle; CPU forward atol=1e-7; "
            "gradient CPU/MPS and mathematical agreement recorded separately at rtol=atol=1e-6",
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(render_markdown(report))
    print(f"Wrote {args.output} and {args.output.with_suffix('.md')}")
    return 0 if all(result["status"] == "passed" for result in cases.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

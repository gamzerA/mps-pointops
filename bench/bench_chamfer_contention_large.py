"""Large bidirectional Chamfer fan-in probe with paired native scatter control.

Run Safe and Fast Math in separate processes, for example::

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_chamfer_contention_large.py --output large-safe.json
    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
      python bench/bench_chamfer_contention_large.py --output large-fast.json

The synthetic patterns have identical shape and dtype. ``uniform`` sends each
point to its same-numbered neighbor; ``random`` permutes the one-to-one map;
``concentrated`` sends every point in both directions to reference zero. This
isolates fan-in and order while holding the O(B*N*N) nearest-search size fixed.
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

# The setting must precede torch import. An enabled fallback invalidates this probe.
if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
    raise SystemExit("set PYTORCH_ENABLE_MPS_FALLBACK=0 before launching")

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops.chamfer import _nearest_mps, chamfer_distance  # noqa: E402

PATTERNS = ("uniform", "random", "concentrated")
BYTES_PER_POINT_PER_PATTERN = 512  # Conservative reserve for input, maps, gradients and control buffers.


def _command(*args: str) -> str:
    result = subprocess.run(args, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(f"{tensor.dtype}:{tuple(tensor.shape)}:".encode())
    digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def guard_case(batch: int, n: int, max_estimated_mib: int, max_pairs: int) -> dict[str, int]:
    """Reject oversized cases before allocating any case tensors on MPS."""
    if batch < 1 or n < 1 or max_estimated_mib < 1 or max_pairs < 1:
        raise ValueError("batch, N, and memory/pair limits must be positive")
    estimated_bytes = len(PATTERNS) * batch * n * BYTES_PER_POINT_PER_PATTERN
    distance_pairs = 2 * batch * n * n  # Both Chamfer search directions.
    if estimated_bytes > max_estimated_mib * 2**20:
        raise ValueError(
            f"estimated case buffers need {estimated_bytes / 2**20:.1f} MiB, "
            f"above --max-estimated-mib={max_estimated_mib}"
        )
    if distance_pairs > max_pairs:
        raise ValueError(
            f"bidirectional search needs {distance_pairs:,} pair checks, "
            f"above --max-pairs={max_pairs:,}"
        )
    return {"estimated_case_bytes": estimated_bytes, "bidirectional_distance_pairs": distance_pairs}


def make_case(batch: int, n: int, pattern: str, seed: int) -> dict[str, torch.Tensor]:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown pattern: {pattern}")
    rows = torch.arange(n, dtype=torch.float32)
    x_cpu = torch.zeros((batch, n, 3), dtype=torch.float32)
    y_cpu = torch.zeros_like(x_cpu)
    xy = torch.empty((batch, n), dtype=torch.long)
    yx = torch.empty_like(xy)
    if pattern == "concentrated":
        # Distinct, well-separated nearest minima: all x -> y[0], all y -> x[0].
        x_cpu[:, :, 0] = 0.5 + rows / 1_000_000
        y_cpu[:, :, 0] = -rows / 1_000_000
        xy.zero_()
        yx.zero_()
    else:
        y_cpu[:, :, 0] = 4 * rows
        for cloud in range(batch):
            permutation = (
                torch.arange(n)
                if pattern == "uniform"
                else torch.randperm(n, generator=torch.Generator().manual_seed(seed + cloud))
            )
            x_cpu[cloud, :, 0] = 4 * permutation.float() + 0.5
            xy[cloud] = permutation
            yx[cloud] = torch.argsort(permutation)
    return {
        "x_cpu": x_cpu, "y_cpu": y_cpu,
        "xy_cpu": xy, "yx_cpu": yx,
        "x": x_cpu.to("mps").requires_grad_(),
        "y": y_cpu.to("mps").requires_grad_(),
        "xy": xy.to("mps"), "yx": yx.to("mps"),
        "lengths": torch.full((batch,), n, dtype=torch.long, device="mps"),
    }


def verify_case(case: dict[str, torch.Tensor], batch: int, n: int) -> dict[str, object]:
    x, y, lengths = case["x"], case["y"], case["lengths"]
    _, actual_xy = _nearest_mps(x.detach(), y.detach(), lengths, lengths)
    _, actual_yx = _nearest_mps(y.detach(), x.detach(), lengths, lengths)
    torch.mps.synchronize()
    torch.testing.assert_close(actual_xy.cpu(), case["xy_cpu"], rtol=0, atol=0)
    torch.testing.assert_close(actual_yx.cpu(), case["yx_cpu"], rtol=0, atol=0)

    x.grad = y.grad = None
    loss, _ = chamfer_distance(x, y, single_directional=False)
    loss.backward()
    torch.mps.synchronize()

    # Independent double-precision, fixed-neighbor analytic oracle. The
    # input float32 values are converted exactly before the arithmetic.
    x64, y64 = case["x_cpu"].double(), case["y_cpu"].double()
    xy, yx = case["xy_cpu"], case["yx_cpu"]
    delta_xy = x64 - y64.gather(1, xy[..., None].expand(-1, -1, 3))
    delta_yx = y64 - x64.gather(1, yx[..., None].expand(-1, -1, 3))
    expected_loss = (delta_xy.square().sum(-1).mean() + delta_yx.square().sum(-1).mean()).float()
    gxy = 2 * delta_xy / (batch * n)
    gyx = 2 * delta_yx / (batch * n)
    expected_x = gxy.clone().scatter_add_(1, yx[..., None].expand(-1, -1, 3), -gyx).float()
    expected_y = gyx.clone().scatter_add_(1, xy[..., None].expand(-1, -1, 3), -gxy).float()
    actual_x, actual_y = x.grad.cpu(), y.grad.cpu()
    torch.testing.assert_close(loss.cpu(), expected_loss, rtol=5e-4, atol=2e-6)
    torch.testing.assert_close(actual_x, expected_x, rtol=5e-4, atol=2e-6)
    torch.testing.assert_close(actual_y, expected_y, rtol=5e-4, atol=2e-6)

    counts_xy = torch.bincount(actual_xy[0].cpu(), minlength=n)
    counts_yx = torch.bincount(actual_yx[0].cpu(), minlength=n)
    result = {
        "exact_xy_indices": True, "exact_yx_indices": True,
        "loss": float(loss.item()),
        "max_abs_loss_error": abs(float(loss.item()) - float(expected_loss.item())),
        "max_abs_x_gradient_error": float((actual_x - expected_x).abs().max().item()),
        "max_abs_y_gradient_error": float((actual_y - expected_y).abs().max().item()),
        "occupied_y_for_xy": int((counts_xy > 0).sum().item()),
        "occupied_x_for_yx": int((counts_yx > 0).sum().item()),
        "max_xy_fanin": int(counts_xy.max().item()),
        "max_yx_fanin": int(counts_yx.max().item()),
        "input_sha256": {
            "x_float32": _tensor_sha256(case["x_cpu"]),
            "y_float32": _tensor_sha256(case["y_cpu"]),
            "xy_int64": _tensor_sha256(xy),
            "yx_int64": _tensor_sha256(yx),
        },
    }
    x.grad = y.grad = None
    return result


def _sync() -> None:
    torch.mps.synchronize()


def measure_chamfer(case: dict[str, torch.Tensor]) -> tuple[float, float, float]:
    x, y = case["x"], case["y"]
    x.grad = y.grad = None
    _sync()
    start = time.perf_counter_ns()
    loss, _ = chamfer_distance(x, y, single_directional=False)
    _sync()
    forward_ms = (time.perf_counter_ns() - start) / 1e6
    start = time.perf_counter_ns()
    loss.backward()
    _sync()
    backward_ms = (time.perf_counter_ns() - start) / 1e6

    x.grad = y.grad = None
    _sync()
    start = time.perf_counter_ns()
    loss, _ = chamfer_distance(x, y, single_directional=False)
    loss.backward()
    _sync()
    e2e_ms = (time.perf_counter_ns() - start) / 1e6
    return forward_ms, backward_ms, e2e_ms


def measure_scatter(
    case: dict[str, torch.Tensor], source: torch.Tensor,
    destination_x: torch.Tensor, destination_y: torch.Tensor,
) -> float:
    destination_x.zero_()
    destination_y.zero_()
    _sync()
    start = time.perf_counter_ns()
    destination_y.scatter_add_(1, case["xy"][..., None].expand_as(source), source)
    destination_x.scatter_add_(1, case["yx"][..., None].expand_as(source), source)
    _sync()
    return (time.perf_counter_ns() - start) / 1e6


def _summary(samples: list[float]) -> dict[str, object]:
    return {"median_ms": statistics.median(samples), "samples_ms": samples}


def _driver_bytes() -> int | None:
    fn = getattr(torch.mps, "driver_allocated_memory", None)
    return int(fn()) if fn is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sizes", type=int, nargs="+", default=[32768, 65536])
    parser.add_argument("--seed", type=int, default=8042)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--max-estimated-mib", type=int, default=256)
    parser.add_argument("--max-driver-mib", type=int, default=4096)
    parser.add_argument("--max-pairs", type=int, default=10_000_000_000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
        parser.error("set PYTORCH_MPS_FAST_MATH=0 or 1 before launching")
    if not torch.backends.mps.is_available():
        parser.error("MPS is required")
    if not args.sizes or args.warmups < 0 or args.repeats < 1 or args.repeats % len(PATTERNS):
        parser.error("sizes required; warmups nonnegative; repeats a positive multiple of three")
    if args.max_driver_mib < 1 or args.output.suffix != ".json":
        parser.error("--max-driver-mib must be positive and --output must end in .json")
    guard = {
        n: guard_case(args.batch, n, args.max_estimated_mib, args.max_pairs)
        for n in args.sizes
    }

    results = []
    for n in args.sizes:
        cases = {name: make_case(args.batch, n, name, args.seed) for name in PATTERNS}
        checks = {name: verify_case(case, args.batch, n) for name, case in cases.items()}
        source = torch.ones((args.batch, n, 3), dtype=torch.float32, device="mps")
        destinations = {
            name: (torch.zeros_like(source), torch.zeros_like(source)) for name in PATTERNS
        }
        samples = {name: {key: [] for key in ("forward", "backward", "e2e", "scatter_pair")}
                   for name in PATTERNS}
        for iteration in range(args.warmups + args.repeats):
            rotated = PATTERNS[iteration % len(PATTERNS):] + PATTERNS[:iteration % len(PATTERNS)]
            for name in rotated:
                forward, backward, e2e = measure_chamfer(cases[name])
                scatter = measure_scatter(cases[name], source, *destinations[name])
                if iteration >= args.warmups:
                    for key, value in zip(("forward", "backward", "e2e", "scatter_pair"),
                                          (forward, backward, e2e, scatter)):
                        samples[name][key].append(value)

        for name, (destination_x, destination_y) in destinations.items():
            expected = torch.ones_like(source)
            if name == "concentrated":
                expected.zero_()
                expected[:, 0, :] = n
            torch.testing.assert_close(destination_x.cpu(), expected.cpu(), rtol=0, atol=0)
            torch.testing.assert_close(destination_y.cpu(), expected.cpu(), rtol=0, atol=0)

        driver_bytes = _driver_bytes()
        if driver_bytes is not None and driver_bytes > args.max_driver_mib * 2**20:
            raise MemoryError(
                f"MPS driver allocation {driver_bytes / 2**20:.1f} MiB exceeds "
                f"--max-driver-mib={args.max_driver_mib}"
            )
        row = {"batch": args.batch, "n": n, **guard[n], "driver_allocated_bytes_after": driver_bytes}
        for name in PATTERNS:
            row[name] = {key: _summary(values) for key, values in samples[name].items()}
            row[name]["correctness"] = checks[name]
        row["paired_ratios"] = {
            key + "_concentrated_over_uniform": statistics.median(
                c / u for c, u in zip(samples["concentrated"][key], samples["uniform"][key])
            ) for key in ("backward", "e2e", "scatter_pair")
        }
        results.append(row)
        print(json.dumps({"n": n, "paired_ratios": row["paired_ratios"]}), flush=True)
        del cases, source, destinations

    source_paths = (
        "bench/bench_chamfer_contention_large.py",
        "mps_pointops/chamfer.py",
        "mps_pointops/kernels/chamfer_nn.metal",
    )
    report = {
        "schema": "mps-pointops.chamfer-contention-large.v2",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "chip": _command("sysctl", "-n", "machdep.cpu.brand_string"),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
            "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "source_commit": _command("git", "rev-parse", "HEAD"),
            "source_dirty": bool(_command("git", "status", "--porcelain")),
            "source_sha256": {name: _sha256(ROOT / name) for name in source_paths},
        },
        "method": {
            "batch": args.batch, "sizes": args.sizes, "seed": args.seed,
            "warmups": args.warmups, "repeats": args.repeats,
            "patterns": list(PATTERNS), "max_estimated_mib": args.max_estimated_mib,
            "max_driver_mib": args.max_driver_mib, "max_pairs": args.max_pairs,
            "chamfer": "bidirectional squared L2; point and batch means; no padding or weights",
            "timing": "preloaded MPS float32; synchronized host wall; forward/backward separately and full loss+backward; no transfers",
            "scatter_control": "two native scatter_add_ calls into preallocated buffers, reset outside timer; sources are ones",
            "order": "uniform/random/concentrated rotated by iteration; each pattern occupies each position equally",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

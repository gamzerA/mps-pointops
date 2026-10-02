"""Synchronized private SubM forward/backward comparison for CPU and MPS rulebooks.

Run Safe and Fast Math in separate processes with MPS fallback disabled. The
reported forward time includes coordinate validation and rulebook construction;
backward time begins after the forward queue is synchronized.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mps_pointops._subm_conv_mps import subm_conv3d_forward_mps


def _coordinates(rows: int) -> torch.Tensor:
    shape = (64, 32, 16)
    flat = torch.randperm(
        shape[0] * shape[1] * shape[2],
        generator=torch.Generator().manual_seed(5419),
    )[:rows]
    axis0 = flat // (shape[1] * shape[2])
    axis1 = (flat // shape[2]) % shape[1]
    axis2 = flat % shape[2]
    return torch.stack((torch.zeros_like(flat), axis0, axis1, axis2), dim=1).to(torch.int32)


def _sample(indices: torch.Tensor, backend: str) -> tuple[float, float]:
    rows = len(indices)
    generator = torch.Generator().manual_seed(718)
    features = torch.randn((rows, 4), generator=generator).to("mps").requires_grad_()
    weights = torch.randn((4, 4, 3, 3, 3), generator=generator).to("mps").requires_grad_()
    upstream = torch.randn((rows, 4), generator=generator).to("mps")
    torch.mps.synchronize()
    start = perf_counter()
    output = subm_conv3d_forward_mps(
        indices, features, weights, (64, 32, 16), 1, rulebook_backend=backend
    )
    torch.mps.synchronize()
    forward_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    (output * upstream).sum().backward()
    torch.mps.synchronize()
    backward_ms = (perf_counter() - start) * 1000
    return forward_ms, backward_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1025, 10_000])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("an MPS device is required")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be >= 0 and repeats >= 1")
    if any(rows < 0 or rows > 64 * 32 * 16 for rows in args.rows):
        parser.error("each row count must be between 0 and 32768")

    cases = []
    for rows in args.rows:
        indices = _coordinates(rows)
        for backend in ("cpu", "mps"):
            for _ in range(args.warmup):
                _sample(indices, backend)
            samples = [_sample(indices, backend) for _ in range(args.repeats)]
            cases.append({
                "rows": rows,
                "rulebook_backend": backend,
                "forward_median_ms": statistics.median(item[0] for item in samples),
                "backward_median_ms": statistics.median(item[1] for item in samples),
                "samples_ms": [{"forward": f, "backward": b} for f, b in samples],
            })
    result = {
        "device": torch.backends.mps.get_name(),
        "macos": platform.mac_ver()[0],
        "torch": torch.__version__,
        "mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH"),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "warmup": args.warmup,
        "repeats": args.repeats,
        "cases": cases,
    }
    data = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(data)
    print(data, end="")


if __name__ == "__main__":
    main()

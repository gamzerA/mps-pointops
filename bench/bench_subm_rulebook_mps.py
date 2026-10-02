"""Synchronized, GPU-only timing for the private SubM rulebook builder.

Run Safe and Fast Math in separate processes, for example::

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/bench_subm_rulebook_mps.py
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
from mps_pointops._subm_rulebook_mps import generate_subm_rulebook_mps


def _coordinates(rows: int) -> torch.Tensor:
    shape = (64, 32, 16)
    per_batch = shape[0] * shape[1] * shape[2]
    flat = torch.randperm(2 * per_batch, generator=torch.Generator().manual_seed(5419))[:rows]
    batch = flat // per_batch
    rest = flat % per_batch
    axis0 = rest // (shape[1] * shape[2])
    rest = rest % (shape[1] * shape[2])
    axis1 = rest // shape[2]
    axis2 = rest % shape[2]
    return torch.stack((batch, axis0, axis1, axis2), dim=1).to(torch.int32).to("mps")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("an MPS device is required")
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be >= 0 and repeats must be >= 1")

    cases = []
    for rows in (1_024, 10_000):
        indices = _coordinates(rows)
        for _ in range(args.warmup):
            rulebook = generate_subm_rulebook_mps(indices, kernel_size=3)
        torch.mps.synchronize()
        samples = []
        for _ in range(args.repeats):
            start = perf_counter()
            rulebook = generate_subm_rulebook_mps(indices, kernel_size=3)
            torch.mps.synchronize()
            samples.append((perf_counter() - start) * 1000)
        cases.append({
            "rows": rows,
            "slots": rows * 27,
            "pairs": int(rulebook.pair_count.cpu()[0]),
            "median_ms": statistics.median(samples),
            "samples_ms": samples,
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

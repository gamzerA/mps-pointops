"""Synchronized PyG 2.8 graph avg_pool forward/backward CPU/MPS timing.

Example:
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
    python bench/bench_pyg28_avg_pool.py --output results-safe.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.nn import avg_pool


def _device_name() -> str:
    result = subprocess.run(
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() or "unavailable"


def _data(nodes: int, degree: int, device: str) -> tuple[torch.Tensor, Data]:
    if nodes % 16:
        raise ValueError("nodes must be divisible by 16")
    if degree < 1 or degree > 8:
        raise ValueError("degree must be between 1 and 8")
    generator = torch.Generator().manual_seed(2701)
    batch_lengths = (nodes // 8, nodes // 4, nodes - 3 * nodes // 8)
    offsets = (0, 1, 2, 3, 7, 11, 17, 29)[:degree]
    edge_parts = []
    start = 0
    for length in batch_lengths:
        source = torch.arange(length, dtype=torch.long).repeat_interleave(degree)
        target = (source + torch.tensor(offsets).repeat(length)) % length
        edge_parts.append(torch.stack((source + start, target + start)))
        start += length
    edge_index = torch.cat(edge_parts, dim=1)
    batch = torch.repeat_interleave(
        torch.arange(3, dtype=torch.long), torch.tensor(batch_lengths),
    )
    # Each group of four points maps to one coarse node, and group IDs do not
    # overlap between batches. Inputs are created on CPU once and transferred
    # before the timed region.
    cluster = torch.arange(nodes, dtype=torch.long) // 4
    x = torch.randn((nodes, 32), generator=generator)
    pos = torch.randn((nodes, 3), generator=generator)
    edge_attr = torch.randn((edge_index.shape[1], 4), generator=generator)
    return cluster.to(device), Data(
        x=x.to(device).detach().requires_grad_(True),
        pos=pos.to(device).detach().requires_grad_(True),
        edge_index=edge_index.to(device),
        edge_attr=edge_attr.to(device).detach().requires_grad_(True),
        batch=batch.to(device),
    )


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def _step(cluster: torch.Tensor, data: Data, device: str) -> tuple[float, float]:
    for value in (data.x, data.pos, data.edge_attr):
        value.grad = None
    _sync(device)
    began = time.perf_counter()
    pooled = avg_pool(cluster, data)
    _sync(device)
    forward = time.perf_counter()
    loss = pooled.x.square().mean() + pooled.pos.square().mean()
    if pooled.edge_attr.numel():
        loss = loss + pooled.edge_attr.square().mean()
    loss.backward()
    _sync(device)
    backward = time.perf_counter()
    return (forward - began) * 1000, (backward - forward) * 1000


def _summarize(samples: list[float]) -> dict[str, float | list[float]]:
    ordered = sorted(samples)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "samples_ms": [round(item, 3) for item in samples],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, nargs="+", default=[16384, 65536])
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 before launching")
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    if version("torch-geometric") != "2.8.0" or not version("pyg-lib").startswith("0.7.0"):
        raise RuntimeError("pin torch-geometric==2.8.0 and pyg-lib==0.7.0")
    if args.repeats < 1 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups nonnegative")

    cases = []
    for nodes in args.nodes:
        times = {}
        for device in ("cpu", "mps"):
            cluster, data = _data(nodes, args.degree, device)
            forward, backward = [], []
            for iteration in range(args.warmups + args.repeats):
                fwd, bwd = _step(cluster, data, device)
                if iteration >= args.warmups:
                    forward.append(fwd)
                    backward.append(bwd)
            times[device] = {
                "forward": _summarize(forward),
                "backward": _summarize(backward),
            }
        cases.append({
            "nodes": nodes,
            "edges": nodes * args.degree,
            "coarse_nodes": nodes // 4,
            "batch_lengths": [nodes // 8, nodes // 4, nodes - 3 * nodes // 8],
            "timings": times,
        })
    result = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "device": _device_name(),
        "macos": platform.mac_ver()[0],
        "torch": torch.__version__,
        "torch_geometric": version("torch-geometric"),
        "pyg_lib": version("pyg-lib"),
        "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
        "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH", "unset"),
        "warmups": args.warmups,
        "repeats": args.repeats,
        "degree": args.degree,
        "features": 32,
        "edge_attr_features": 4,
        "positions": 3,
        "cases": cases,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

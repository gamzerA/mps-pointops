#!/usr/bin/env python
"""Benchmark point cloud ops (FPS, kNN, ball query) on Apple Silicon.

Compares pure-PyTorch implementations on MPS and CPU with optimized CPU
libraries, at MulSen-AD scale: 20k-100k points per object, 1024 FPS centers,
k=128 neighbors (Point-MAE grouping).

    python bench/bench_pointops.py
    python bench/bench_pointops.py --sizes 20000 --ops fps --repeat 3
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import ops, reference as ref  # noqa: E402

try:
    import fpsample
except ImportError:
    fpsample = None
try:
    import scipy
    from scipy.spatial import cKDTree
except ImportError:
    scipy = None
    cKDTree = None

HAS_MPS = torch.backends.mps.is_available()


# ---------------------------------------------------------------- helpers


def sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def measure(fn, device: str, warmup: int, repeat: int):
    out = None
    for _ in range(warmup):
        out = fn()
        sync(device)
    times = []
    for _ in range(repeat):
        sync(device)
        t0 = time.perf_counter()
        out = fn()
        sync(device)
        times.append(time.perf_counter() - t0)
    return out, times


def make_cloud(n: int, batch: int, seed: int, order: str = "random") -> torch.Tensor:
    """Points near a unit sphere surface, like the shell of a scanned object.

    order="sorted" stores them sorted by x, like the spatially ordered vertices
    of a real scan. The points themselves are the same.
    """
    g = torch.Generator().manual_seed(seed)
    p = torch.randn(batch, n, 3, generator=g)
    p = p / p.norm(dim=-1, keepdim=True)
    p = p + 0.01 * torch.randn(batch, n, 3, generator=g)
    if order == "sorted":
        p = torch.stack([c[c[:, 0].argsort()] for c in p])
    return p.float().contiguous()


def pick_queries(pts: torch.Tensor, m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed + 1)
    sel = torch.randperm(pts.shape[1], generator=g)[:m]
    return pts[:, sel].contiguous()


def to_np(x) -> np.ndarray:
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def cmd(*args: str) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "?"


def environment() -> dict:
    mem = cmd("sysctl", "-n", "hw.memsize")
    return {
        "chip": cmd("sysctl", "-n", "machdep.cpu.brand_string"),
        "memory_gb": round(int(mem) / 2**30) if mem.isdigit() else "?",
        "macos": platform.mac_ver()[0],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": getattr(scipy, "__version__", None),
        "fpsample": getattr(fpsample, "__version__", "installed" if fpsample else None),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
    }


# ---------------------------------------------------------------- checks


def fps_check(idx, want) -> str:
    """Fraction of indices equal to the torch-cpu result."""
    a, b = to_np(idx), to_np(want)
    return f"idx match {100 * (a == b).mean():.1f}%"


def knn_check(idx, want) -> str:
    """Mean recall of neighbor sets against exact float64 search."""
    a, b = to_np(idx), to_np(want)
    k = a.shape[-1]
    hit = (a[..., :, None] == b[..., None, :]).any(-1).sum(-1) / k
    return f"recall {100 * hit.mean():.2f}%"


def ball_check(idx, want) -> str:
    a, b = to_np(idx), to_np(want)
    return f"idx match {100 * (a == b).mean():.2f}%"


# ---------------------------------------------------------------- cases


def fps_cases(pts, args):
    npoint = args.npoint
    cases = []
    if HAS_MPS:
        pm = pts.to("mps")
        cases.append(("torch (MPS)", "mps", lambda: ref.furthest_point_sample(pm, npoint)))
        cases.append(("mps-pointops Metal (MPS)", "mps", lambda: ops.furthest_point_sample(pm, npoint)))
    cases.append(("torch (CPU)", "cpu", lambda: ref.furthest_point_sample(pts, npoint)))
    if fpsample is not None:
        arr = pts.numpy()
        # QuickFPS (bucket_fps_kdline_sampling) is left out: in fpsample 1.0.2 it
        # ignores start_idx and returns a different, sorted sample set.
        cases.append((
            "fpsample vanilla (CPU)", "cpu",
            lambda: np.stack([fpsample.fps_sampling(a, npoint, start_idx=0) for a in arr]),
        ))
    want = ref.furthest_point_sample(pts, npoint)
    return cases, lambda out: fps_check(out, want)


def knn_cases(pts, q, args):
    k = args.k
    cases = []
    if HAS_MPS:
        pm, qm = pts.to("mps"), q.to("mps")
        cases.append(("torch cdist+topk (MPS)", "mps", lambda: ref.knn(qm, pm, k)[1]))
        cases.append(("mps-pointops Metal (MPS)", "mps", lambda: ops.knn(qm, pm, k)[1]))
    cases.append(("torch cdist+topk (CPU)", "cpu", lambda: ref.knn(q, pts, k)[1]))
    arr, qarr = pts.numpy(), q.numpy()
    if cKDTree is not None:
        cases.append((
            "scipy cKDTree build+query (CPU)", "cpu",
            lambda: np.stack([cKDTree(a).query(b, k=k, workers=-1)[1] for a, b in zip(arr, qarr)]),
        ))
        exact = np.stack([
            cKDTree(a.astype(np.float64)).query(b.astype(np.float64), k=k)[1]
            for a, b in zip(arr, qarr)
        ])
    else:
        exact = to_np(ref.knn(q.double(), pts.double(), k)[1])
    return cases, lambda out: knn_check(out, exact)


def ball_cases(pts, q, args):
    r, K = args.radius, args.ball_k
    cases = []
    if HAS_MPS:
        pm, qm = pts.to("mps"), q.to("mps")
        cases.append(("torch cdist+mask+topk (MPS)", "mps", lambda: ref.ball_query(qm, pm, r, K)[1]))
    cases.append(("torch cdist+mask+topk (CPU)", "cpu", lambda: ref.ball_query(q, pts, r, K)[1]))
    if cKDTree is not None:
        arr, qarr = pts.numpy(), q.numpy()

        def scipy_ball():
            out = np.full((len(arr), qarr.shape[1], K), -1, dtype=np.int64)
            for b, (a, qq) in enumerate(zip(arr, qarr)):
                hits = cKDTree(a).query_ball_point(qq, r, workers=-1, return_sorted=True)
                for i, h in enumerate(hits):
                    h = h[:K]
                    out[b, i, : len(h)] = h
            return out

        cases.append(("scipy cKDTree build+query (CPU)", "cpu", scipy_ball))
    want = ref.ball_query(q, pts, r, K)[1]
    return cases, lambda out: ball_check(out, want)


# ---------------------------------------------------------------- main


def run(args) -> list[dict]:
    rows = []
    for n in args.sizes:
        pts = make_cloud(n, args.batch, args.seed, args.order)
        q = pick_queries(pts, args.queries, args.seed)
        builders = {
            "fps": lambda: fps_cases(pts, args),
            "knn": lambda: knn_cases(pts, q, args),
            "ball_query": lambda: ball_cases(pts, q, args),
        }
        for op in args.ops:
            cases, check = builders[op]()
            for name, device, fn in cases:
                row = {"op": op, "n_points": n, "batch": args.batch, "impl": name, "device": device}
                try:
                    out, times = measure(fn, device, args.warmup, args.repeat)
                    row.update(
                        median_ms=1e3 * statistics.median(times),
                        min_ms=1e3 * min(times),
                        check=check(out),
                    )
                except Exception as e:  # keep going; unsupported ops are a result too
                    row.update(median_ms=None, min_ms=None, check=None, error=f"{type(e).__name__}: {e}"[:200])
                rows.append(row)
                ms = f"{row['median_ms']:10.1f} ms" if row["median_ms"] is not None else "     error"
                print(f"{op:10s} N={n:>7d}  {name:34s} {ms}  {row.get('check') or row.get('error')}", flush=True)
            if HAS_MPS:
                torch.mps.empty_cache()
    return rows


def markdown(rows: list[dict], env: dict, args) -> str:
    lines = [
        f"# Benchmark: {env['chip']}, {env['memory_gb']} GB",
        "",
        f"- macOS {env['macos']}, Python {env['python']}, torch {env['torch']}, "
        f"scipy {env['scipy']}, fpsample {env['fpsample']}",
        f"- batch={args.batch}, FPS npoint={args.npoint}, kNN queries={args.queries} k={args.k}, "
        f"ball query queries={args.queries} K={args.ball_k} radius={args.radius}, point order={args.order}",
        f"- median of {args.repeat} runs after {args.warmup} warmup, "
        f"PYTORCH_ENABLE_MPS_FALLBACK={env['mps_fallback']}",
        "- speedup: torch (MPS) time / this time (>1 means faster than pure PyTorch on MPS)",
        "",
    ]
    titles = {"fps": "Farthest point sampling", "knn": "k nearest neighbors", "ball_query": "Ball query"}
    for op in args.ops:
        lines += [f"## {titles[op]}", "", "| points | implementation | median (ms) | speedup | check |", "|---:|---|---:|---:|---|"]
        for n in args.sizes:
            group = [r for r in rows if r["op"] == op and r["n_points"] == n]
            base = next((r["median_ms"] for r in group if r["device"] == "mps" and r["median_ms"]), None)
            for r in group:
                if r["median_ms"] is None:
                    lines.append(f"| {n:,} | {r['impl']} | error | | {r['error']} |")
                    continue
                speed = f"{base / r['median_ms']:.2f}x" if base else ""
                lines.append(f"| {n:,} | {r['impl']} | {r['median_ms']:.1f} | {speed} | {r['check']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sizes", type=int, nargs="+", default=[20_000, 50_000, 100_000])
    p.add_argument("--ops", nargs="+", default=["fps", "knn", "ball_query"], choices=["fps", "knn", "ball_query"])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--npoint", type=int, default=1024, help="FPS samples (Point-MAE num_group)")
    p.add_argument("--queries", type=int, default=1024, help="query points for kNN and ball query")
    p.add_argument("--k", type=int, default=128, help="kNN neighbors (Point-MAE group_size)")
    p.add_argument("--ball-k", type=int, default=64)
    p.add_argument("--radius", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--order", choices=["random", "sorted"], default="random",
                   help="storage order of the points; sorted mimics real scans")
    p.add_argument("--out", type=Path, default=ROOT / "bench" / "results")
    args = p.parse_args()

    env = environment()
    print(json.dumps(env, indent=2))
    if not HAS_MPS:
        print("MPS is not available; running CPU cases only.")
    rows = run(args)

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{dt.date.today().isoformat()}-{env['chip'].replace(' ', '-').lower()}"
    if args.order != "random":
        stem += f"-{args.order}"
    (args.out / f"{stem}.json").write_text(
        json.dumps({"env": env, "args": {k: str(v) for k, v in vars(args).items()}, "rows": rows}, indent=2)
    )
    md = markdown(rows, env, args)
    (args.out / f"{stem}.md").write_text(md + "\n")
    print("\n" + md)
    print(f"\nsaved to {args.out / stem}.{{json,md}}")


if __name__ == "__main__":
    main()

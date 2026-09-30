#!/usr/bin/env python
"""Run MulSen-AD's own Point-MAE grouping on real MulSen point clouds, on MPS.

MulSen-AD's ``models.models.Group`` calls ``pointnet2_ops`` (FPS + gather)
and ``knn_cuda`` (kNN), which are CUDA only. With ``mps_pointops.compat``
installed, the unmodified MulSen code runs on the Mac GPU. This script:

1. loads each STL the way MulSen's dataset does (open3d, duplicate vertices
   removed, centered),
2. runs MulSen's ``Group(num_group=1024, group_size=128)`` on MPS,
3. checks the FPS centers against the pointnet2_ops-contract reference and
   the neighbors against an exact float32 oracle,
4. times it against the best CPU libraries (fpsample FPS + scipy cKDTree kNN)
   and against plain PyTorch on MPS.

    python examples/mulsen_grouping.py --mulsen-code path/to/MulSen-AD \\
        --data path/to/MulSen_AD --per-class 2

Needs open3d, timm, scipy and fpsample. No model weights are needed: this
covers only the grouping step, which is where the CUDA-only ops are.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
from bench_pointops import exact_knn  # noqa: E402
from mps_pointops import compat, reference  # noqa: E402

compat.install()

import fpsample  # noqa: E402
import open3d as o3d  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

NUM_GROUP, GROUP_SIZE = 1024, 128


def load_cloud(path: Path) -> np.ndarray:
    """Same steps as MulSen-AD's dataset.py."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    mesh = mesh.remove_duplicated_vertices()
    pc = np.asarray(mesh.vertices)
    return pc - pc.mean(axis=0, keepdims=True)


def timed(fn, device: str, repeat: int):
    sync = torch.mps.synchronize if device == "mps" else (lambda: None)
    out = fn()
    sync()
    times = []
    for _ in range(repeat):
        sync()
        t0 = time.perf_counter()
        out = fn()
        sync()
        times.append(time.perf_counter() - t0)
    return out, 1e3 * statistics.median(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mulsen-code", type=Path, required=True, help="MulSen-AD repository")
    p.add_argument("--data", type=Path, required=True, help="MulSen_AD dataset root")
    p.add_argument("--classes", nargs="*", help="default: all classes")
    p.add_argument("--per-class", type=int, default=1, help="train clouds per class")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--out", type=Path, default=ROOT / "examples" / "results")
    args = p.parse_args()

    sys.path.insert(0, str(args.mulsen_code))
    from models.models import Group  # MulSen-AD's own module, unmodified

    group = Group(num_group=NUM_GROUP, group_size=GROUP_SIZE)
    classes = args.classes or sorted(d.name for d in args.data.iterdir() if (d / "Pointcloud").is_dir())
    rows = []
    for cls in classes:
        files = sorted((args.data / cls / "Pointcloud" / "train").glob("*.stl"), key=lambda f: int(f.stem))
        for path in files[: args.per_class]:
            pc = load_cloud(path)
            xyz = torch.from_numpy(pc).float()[None]
            xyz_mps = xyz.to("mps")
            n = xyz.shape[1]

            (neighborhood, center, nbr_idx, center_idx), t_mps = timed(lambda: group(xyz_mps), "mps", args.repeat)

            # Correctness: FPS against the pointnet2_ops contract, kNN against an exact oracle.
            want_centers = reference.furthest_point_sample(xyz, NUM_GROUP, skip_near_origin=True)
            fps_bad = int((center_idx.cpu().long() != want_centers).sum())
            centers = xyz[:, want_centers[0]]
            _, want_nbr = exact_knn(centers.numpy(), xyz.numpy(), GROUP_SIZE)
            knn_bad = int((nbr_idx.cpu().numpy() != want_nbr).sum())
            near_origin = int((reference._sqdist(xyz, torch.zeros(3)) < torch.tensor(1e-3)).sum())

            # Same two steps with the best CPU libraries, and in plain PyTorch on MPS.
            arr = xyz[0].numpy()

            def cpu_best():
                c = fpsample.fps_sampling(arr, NUM_GROUP, start_idx=0)
                return cKDTree(arr).query(arr[c], k=GROUP_SIZE, workers=-1)[1]

            def torch_mps():
                c = reference.furthest_point_sample(xyz_mps, NUM_GROUP)
                return reference.knn(xyz_mps[:, c[0]], xyz_mps, GROUP_SIZE)[1]

            _, t_cpu = timed(cpu_best, "cpu", args.repeat)
            _, t_torch = timed(torch_mps, "mps", args.repeat)

            row = {
                "class": cls, "file": path.name, "points": n, "near_origin_points": near_origin,
                "mps_pointops_ms": t_mps, "cpu_best_ms": t_cpu, "torch_mps_ms": t_torch,
                "fps_mismatches": fps_bad, "knn_mismatches": knn_bad,
                "neighborhood_shape": list(neighborhood.shape),
            }
            rows.append(row)
            print(
                f"{cls:16s} {path.name:>7s} N={n:>7,d}  mps-pointops {t_mps:6.1f} ms  "
                f"CPU best {t_cpu:6.1f} ms  torch MPS {t_torch:6.1f} ms  "
                f"mismatches fps {fps_bad} knn {knn_bad}  near-origin {near_origin}",
                flush=True,
            )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "mulsen_grouping.json").write_text(json.dumps(rows, indent=2))
    print(f"\nsaved {len(rows)} clouds to {args.out / 'mulsen_grouping.json'}")


if __name__ == "__main__":
    main()

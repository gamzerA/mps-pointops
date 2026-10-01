#!/usr/bin/env python3
"""Compare the local CPU Graclus subset with a pinned official 1.6.3 build.

Build the original csrc/graclus.cpp and csrc/cpu/graclus_cpu.cpp outside this
repository, then pass the resulting shared library and original checkout.
No upstream source or binary is copied into this project.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import torch

from mps_pointops.graclus import graclus_cluster


PINNED_COMMIT = "29cd22bf1a5b82fc06b108d6573f81302c5d6b12"
UPSTREAM_FILES = (
    "torch_cluster/graclus.py",
    "csrc/graclus.cpp",
    "csrc/cpu/graclus_cpu.cpp",
    "csrc/cuda/graclus_cuda.cu",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_identity(upstream: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"upstream checkout {commit} != pinned {PINNED_COMMIT}")
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--", *UPSTREAM_FILES], text=True
    ).strip()
    if dirty:
        raise RuntimeError(f"upstream source is modified: {dirty}")
    return {
        "commit": commit,
        "git_blob_sha1": {
            name: subprocess.check_output(
                ["git", "-C", str(upstream), "rev-parse", f"HEAD:{name}"], text=True
            ).strip() for name in UPSTREAM_FILES
        },
        "source_sha256": {name: sha256(upstream / name) for name in UPSTREAM_FILES},
    }


def load_official(upstream: Path, library: Path):
    torch.ops.load_library(str(library.resolve()))
    file = upstream / "torch_cluster/graclus.py"
    spec = importlib.util.spec_from_file_location("pinned_graclus_163", file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.graclus_cluster


def run(upstream: Path, library: Path) -> dict:
    provenance = source_identity(upstream)
    official = load_official(upstream, library)
    cases = [
        ("disjoint_unweighted", [0, 1, 2, 3, 4], [1, 0, 3, 2, 4], None, 6),
        ("disjoint_weighted", [0, 1, 2, 3, 4], [1, 0, 3, 2, 4], [2., 2., 3., 3., 100.], 6),
        ("general_unweighted", [0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6],
         [1, 2, 0, 2, 3, 0, 1, 1, 5, 6, 4, 4], None, 8),
        ("general_weighted", [0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6],
         [1, 2, 0, 2, 3, 0, 1, 1, 5, 6, 4, 4],
         [0., 3., 0., 2., 1., 3., 2., 1., 4., 2., 4., 2.], 8),
        ("zero_and_negative", [0, 0, 1, 2], [1, 2, 0, 0], [-1., 0., -1., 0.], None),
        ("all_negative", [0, 1], [1, 0], [-2., -2.], None),
        ("weighted_equal_ties", [0, 0, 1, 2], [1, 2, 0, 0], [2., 2., 2., 2.], 3),
        ("empty_with_num_nodes", [], [], None, 4),
    ]
    observations = []
    for name, row_values, col_values, weight_values, num_nodes in cases:
        row = torch.tensor(row_values, dtype=torch.int64)
        col = torch.tensor(col_values, dtype=torch.int64)
        weight = torch.tensor(weight_values, dtype=torch.float32) if weight_values is not None else None
        for seed in (0, 1, 7, 19, 42):
            torch.manual_seed(seed)
            expected = official(row, col, weight, num_nodes)
            torch.manual_seed(seed)
            actual = graclus_cluster(row, col, weight, num_nodes)
            observations.append({
                "case": name, "seed": seed,
                "official": expected.tolist(), "project": actual.tolist(),
                "exact": bool(torch.equal(expected, actual)),
            })
    return {
        "upstream": provenance,
        "extension_filename": library.name,
        "extension_sha256": sha256(library),
        "project_source_sha256": sha256(Path(__file__).resolve().parents[1] / "mps_pointops/graclus.py"),
        "torch": torch.__version__, "platform": platform.platform(),
        "total": len(observations),
        "exact": sum(item["exact"] for item in observations),
        "observations": observations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.upstream_checkout, args.extension)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"total": result["total"], "exact": result["exact"], "output": str(args.output)}))
    if result["exact"] != result["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

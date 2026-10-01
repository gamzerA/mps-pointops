#!/usr/bin/env python3
"""Build only the unmodified official torch_cluster 1.6.3 Graclus CPU op.

The checkout and build directory are external to this repository. PyTorch's
C++ extension cache needs Ninja and a C++ compiler on PATH.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from torch.utils.cpp_extension import load


PINNED_COMMIT = "29cd22bf1a5b82fc06b108d6573f81302c5d6b12"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "-C", str(args.upstream_checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != PINNED_COMMIT:
        raise RuntimeError(f"upstream checkout {revision} != pinned {PINNED_COMMIT}")
    paths = ("csrc/graclus.cpp", "csrc/cpu/graclus_cpu.cpp")
    changed = subprocess.check_output(
        ["git", "-C", str(args.upstream_checkout), "status", "--porcelain", "--", *paths], text=True
    ).strip()
    if changed:
        raise RuntimeError(f"upstream source is modified: {changed}")
    args.build_dir.mkdir(parents=True, exist_ok=True)
    root = args.upstream_checkout
    load(
        name="graclus_163_cpu",
        sources=[str(root / "csrc/graclus.cpp"), str(root / "csrc/cpu/graclus_cpu.cpp")],
        extra_include_paths=[str(root / "csrc")],
        build_directory=str(args.build_dir),
        is_python_module=False,
        with_cuda=False,
        verbose=True,
    )
    libraries = list(args.build_dir.glob("graclus_163_cpu.*"))
    if len(libraries) != 1:
        raise RuntimeError(f"expected one built library, found {libraries}")
    print(libraries[0])


if __name__ == "__main__":
    main()

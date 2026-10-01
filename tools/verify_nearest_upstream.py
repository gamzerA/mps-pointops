"""Compare nearest with the unmodified torch_cluster 1.6.3 CPU function.

The external source is passed by path and is never copied into this project.
Its CPU function calls SciPy directly, so the extension binary is unnecessary.
Run separately for MPS Safe and Fast Math, with MPS fallback disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import scipy
import torch

from mps_pointops.nearest import nearest


_UPSTREAM_SHA256 = "047209f84999997d378d6a1bb96f228f1ac58e1bbefaef1899920a5b5c59c904"


def _upstream(path: Path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != _UPSTREAM_SHA256:
        raise ValueError("upstream nearest.py does not match pinned torch-cluster 1.6.3")
    spec = importlib.util.spec_from_file_location("upstream_torch_cluster_nearest_163", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load upstream nearest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.nearest


def _cases():
    x = torch.tensor([
        [-1., -1.], [-1., 1.], [1., 1.], [1., -1.],
        [-2., -2.], [-2., 2.], [2., 2.], [2., -2.],
    ])
    y = torch.tensor([[-1., 0.], [1., 0.], [-2., 0.], [2., 0.]])
    yield "upstream_fixture_with_empty_id", x, y, torch.tensor([0] * 4 + [2] * 4), torch.tensor([0] * 2 + [2] * 2)

    generator = torch.Generator().manual_seed(123)
    for dim in (1, 3, 64, 128):
        y = torch.randn(60, dim, generator=generator) * 5
        selected = torch.randint(0, 60, (100,), generator=generator)
        x = y[selected] + torch.randn(100, dim, generator=generator) * 0.001
        yield f"random_D{dim}", x, y, None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-file", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.device == "mps":
        if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
            raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing PyTorch")
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")

    upstream = _upstream(args.upstream_file)
    cases = []
    for name, x, y, bx, by in _cases():
        expected = upstream(x, y, bx, by)
        moved = [t.to(args.device) if t is not None else None for t in (x, y, bx, by)]
        actual = nearest(*moved).cpu()
        cases.append({"name": name, "rows": len(x), "same_indices": bool(torch.equal(actual, expected))})
    # The upstream CPU SciPy path takes the first global index on this tie.
    # The source-derived CUDA rule instead chooses the lower stride-1024 lane.
    tie_y = torch.full((1030, 1), 10.0)
    tie_y[1, 0] = tie_y[1024, 0] = 0.0
    tie_x = torch.zeros(1, 1)
    upstream_tie = upstream(tie_x, tie_y).item()
    project_tie = nearest(tie_x.to(args.device), tie_y.to(args.device)).cpu().item()
    record = {
        "upstream_tag": "torch-cluster 1.6.3",
        "upstream_commit": "29cd22bf1a5b82fc06b108d6573f81302c5d6b12",
        "upstream_source_sha256": _UPSTREAM_SHA256,
        "reference": "unmodified upstream Python/SciPy CPU nearest, not its CUDA binary",
        "device": args.device,
        "torch": torch.__version__,
        "scipy": scipy.__version__,
        "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH", "unset"),
        "cases": cases,
        "known_cpu_cuda_style_tie_difference": {
            "upstream_scipy_cpu_index": upstream_tie,
            "project_cuda_style_index": project_tie,
            "cuda_binary_executed": False,
        },
    }
    output = json.dumps(record, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    else:
        print(output, end="")
    if not all(row["same_indices"] for row in cases) or (upstream_tie, project_tie) != (1, 1024):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Opt-in direct PyTorch3D CPU-oracle parity for tensor-coordinate Chamfer.

The fixture covers D=1, 2, and 4 without normals. It compares the pinned
upstream's forward loss, first gradients, and valid-row nearest indices with
the package on CPU or MPS. Upstream source is imported, never copied here.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch

from verify_upstream_chamfer import (
    ATOL, RTOL, EXPECTED_UPSTREAM, error_metrics, git_head, scalarize,
    signature_metrics,
)


def make_clouds(device: str, dimension: int, fixture: str):
    generator = torch.Generator().manual_seed(131 + dimension)
    x = torch.randn((2, 7, dimension), generator=generator)
    y = torch.randn((2, 6, dimension), generator=generator)
    if fixture == "ties":
        # Two exactly coincident candidates expose the selected-index gradient.
        x[0, 0] = 0
        y[0, 0] = 0
        y[0, 1] = 0
        # In D>=2, L1 selects y[2] but squared L2 selects y[3].
        x[0, 2] = 10
        y[0, 2] = x[0, 2]
        y[0, 2, 0] += 2
        y[0, 3] = x[0, 2]
        y[0, 3, 0] += 1.1
        if dimension >= 2:
            y[0, 3, 1] += 1.1
    # Finite padding is deliberately far away; valid lengths must mask it.
    x[1, 4:] = 50
    y[1, 3:] = -50
    return (
        x.to(device).requires_grad_(), y.to(device).requires_grad_(),
        torch.tensor([7, 4], dtype=torch.long, device=device),
        torch.tensor([6, 3], dtype=torch.long, device=device),
    )


def run_one(fn, device: str, spec: dict, *, upstream: bool):
    from mps_pointops.chamfer import _nearest_indices
    from pytorch3d.ops.knn import knn_points

    x, y, lx, ly = make_clouds(device, spec["dimension"], spec["fixture"])
    weights = None
    if spec["weights"] != "none":
        values = {"positive": [0.3, 1.7], "mixed_zero": [0.0, 1.7],
                  "all_zero": [0.0, 0.0]}[spec["weights"]]
        weights = torch.tensor(values, device=device, requires_grad=True)
    loss, normals = fn(
        x, y, x_lengths=lx, y_lengths=ly, weights=weights,
        norm=spec["norm"], point_reduction=spec["point_reduction"],
        batch_reduction=spec["batch_reduction"],
        single_directional=spec["single_directional"],
    )
    inputs = (x, y) if weights is None else (x, y, weights)
    grads = torch.autograd.grad(scalarize(loss), inputs, allow_unused=True)
    if upstream:
        indices = knn_points(x, y, lengths1=lx, lengths2=ly,
                             norm=spec["norm"], K=1).idx[..., 0]
    else:
        indices = _nearest_indices(x, y, lx, ly, spec["norm"])
    if device == "mps":
        torch.mps.synchronize()
    valid = torch.arange(x.shape[1], device=device)[None] < lx[:, None]
    parts = loss if isinstance(loss, tuple) else (loss,)
    return {
        "loss_shape": [list(part.shape) for part in parts],
        "loss": [float(v) for part in parts for v in part.detach().cpu().flatten().tolist()],
        "normals_shape": None if normals is None else (
            [list(part.shape) for part in normals] if isinstance(normals, tuple)
            else list(normals.shape)
        ),
        "gradient_present": [g is not None for g in grads],
        "gradients": [
            (torch.zeros_like(t) if g is None else g).detach().cpu().flatten().tolist()
            for t, g in zip(inputs, grads)
        ],
        "selected_valid_indices": indices[valid].detach().cpu().tolist(),
    }


def specs():
    for dimension, norm, fixture, point, directional, weight in itertools.product(
        (1, 2, 4), (1, 2), ("random", "ties"),
        ("mean", "sum", "max", None), (False, True),
        ("none", "positive", "mixed_zero", "all_zero"),
    ):
        batch = {"mean": "mean", "sum": "sum", "max": None, None: None}[point]
        yield {"dimension": dimension, "norm": norm, "fixture": fixture,
               "point_reduction": point, "batch_reduction": batch,
               "single_directional": directional, "weights": weight}


def compare(actual: dict, expected: dict):
    checks = {}
    for field in ("loss_shape", "normals_shape", "gradient_present",
                  "selected_valid_indices"):
        checks[field] = signature_metrics(actual[field], expected[field])
    checks["loss"] = error_metrics(actual["loss"], expected["loss"])
    for i, (got, want) in enumerate(zip(actual["gradients"], expected["gradients"])):
        checks[f"gradient_{i}"] = error_metrics(got, want)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--upstream-build-note", required=True)
    parser.add_argument("--devices", choices=("cpu", "mps", "auto"), default="auto")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    upstream_root = args.upstream_root.resolve()
    if git_head(upstream_root) != EXPECTED_UPSTREAM:
        parser.error(f"expected PyTorch3D {EXPECTED_UPSTREAM}")
    sys.path.insert(0, str(upstream_root))
    from pytorch3d import __version__ as upstream_version
    from pytorch3d.loss import chamfer_distance as upstream_chamfer
    from mps_pointops.chamfer import chamfer_distance as port_chamfer

    targets = (["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"])
    if args.devices != "auto":
        targets = [args.devices]
    if "mps" in targets and not torch.backends.mps.is_available():
        parser.error("MPS requested but unavailable")
    if "mps" in targets and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("MPS comparison requires PYTORCH_ENABLE_MPS_FALLBACK=0")

    cases = []
    for spec in specs():
        expected = run_one(upstream_chamfer, "cpu", spec, upstream=True)
        case = {"spec": spec, "checks": {}}
        for target in targets:
            observed = run_one(port_chamfer, target, spec, upstream=False)
            case["checks"][target] = compare(observed, expected)
        cases.append(case)
    summary = {}
    for target in targets:
        checks = [check for case in cases for check in case["checks"][target].values()]
        summary[target] = {
            "cases": len(cases), "checks": len(checks),
            "failed_checks": sum(not c["pass"] for c in checks),
            "failed_elements": sum(c["mismatches"] for c in checks),
            "max_abs": max(c["max_abs"] for c in checks),
        }
    package_root = Path(__file__).resolve().parents[1]
    record = {
        "upstream": {"commit": EXPECTED_UPSTREAM, "version": upstream_version,
                     "execution_device": "cpu", "build_note": args.upstream_build_note},
        "port": {"commit": git_head(package_root),
                 "dirty": bool(subprocess.check_output(
                     ["git", "status", "--porcelain"], cwd=package_root).strip()),
                 "chamfer_sha256": hashlib.sha256((package_root / "mps_pointops/chamfer.py").read_bytes()).hexdigest(),
                 "kernel_sha256": hashlib.sha256((package_root / "mps_pointops/kernels/chamfer_nn.metal").read_bytes()).hexdigest(),
                 "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "platform": platform.platform(),
                        "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH"),
                        "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")},
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "summary": summary, "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if all(v["failed_checks"] == 0 for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

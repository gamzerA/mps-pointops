"""Opt-in direct PyTorch3D 0.7.9 parity for normals and Pointclouds inputs.

Run with the pinned PyTorch3D CPU extension and ``--upstream-root``. The
package's Metal path is checked separately from the upstream CPU oracle.
No upstream implementation is copied into this file.
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
    ATOL, RTOL, EXPECTED_UPSTREAM, error_metrics, git_head, make_clouds,
    scalarize, signature_metrics,
)


def _flatten(value):
    if value is None:
        return []
    parts = value if isinstance(value, tuple) else (value,)
    return [float(v) for part in parts for v in part.detach().cpu().flatten().tolist()]


def _shape(value):
    if value is None:
        return None
    parts = value if isinstance(value, tuple) else (value,)
    return [list(part.shape) for part in parts]


def _grad_record(objective, inputs):
    if objective is None:
        return None
    grads = torch.autograd.grad(objective, inputs, allow_unused=True, retain_graph=True)
    return {
        "present": [g is not None for g in grads],
        "values": [_flatten(g if g is not None else torch.zeros_like(t))
                   for t, g in zip(inputs, grads)],
    }


def run_case(fn, Pointclouds, device, spec):
    x, y, lx, ly = make_clouds(device, spec["seed"])
    empty = spec.get("empty")
    if empty in ("x_first", "both_first"):
        lx[0] = 0
    if empty in ("y_first", "both_first"):
        ly[0] = 0
    generator = torch.Generator().manual_seed(spec["seed"] + 900)
    nx = torch.randn(x.shape, generator=generator).to(device).requires_grad_()
    ny = torch.randn(y.shape, generator=generator).to(device).requires_grad_()
    # Exercise parallel/opposite/orthogonal and zero-vector cosine derivatives.
    nx.data[0, 0] = torch.tensor([1., 0., 0.], device=device)
    nx.data[0, 1] = torch.tensor([0., 1., 0.], device=device)
    nx.data[0, 2] = 0
    ny.data[0, 0] = torch.tensor([-1., 0., 0.], device=device)
    ny.data[0, 1] = torch.tensor([1., 0., 0.], device=device)
    ny.data[0, 2] = 0

    normal_mode = spec["normals"]
    x_arg, y_arg = x, y
    x_lengths, y_lengths = lx, ly
    x_normals = nx if normal_mode != "none" else None
    y_normals = ny if normal_mode == "both" else None
    if spec["representation"] in ("x_object", "both_objects"):
        x_arg = Pointclouds(
            points=[x[i, :int(lx[i])] for i in range(2)],
            normals=[nx[i, :int(lx[i])] for i in range(2)] if x_normals is not None else None,
        )
        # The object must replace both of these deliberately invalid values.
        x_lengths = torch.tensor([99, 99], device=device)
        x_normals = torch.zeros((1, 1, 3), device=device)
    if spec["representation"] == "both_objects":
        y_arg = Pointclouds(
            points=[y[i, :int(ly[i])] for i in range(2)],
            normals=[ny[i, :int(ly[i])] for i in range(2)] if y_normals is not None else None,
        )
        y_lengths = torch.tensor([99, 99], device=device)
        y_normals = torch.zeros((1, 1, 3), device=device)
    weights = None
    if spec["weights"] != "none":
        weights = torch.tensor({"positive": [0.3, 1.7], "mixed_zero": [0., 1.7],
                                "all_zero": [0., 0.]}[spec["weights"]],
                               device=device, requires_grad=True)
    loss, normal_loss = fn(
        x_arg, y_arg, x_lengths=x_lengths, y_lengths=y_lengths,
        x_normals=x_normals, y_normals=y_normals, weights=weights,
        norm=spec["norm"], abs_cosine=spec["abs_cosine"],
        point_reduction=spec["point"], batch_reduction=spec["batch"],
        single_directional=spec["directional"],
    )
    inputs = (x, y, nx, ny) if weights is None else (x, y, nx, ny, weights)
    both_objective = scalarize(loss)
    if normal_loss is not None:
        both_objective = both_objective + scalarize(normal_loss) * 0.71
    normal_objective = scalarize(normal_loss) if normal_loss is not None else None
    record = {
        "loss": _flatten(loss), "loss_shape": _shape(loss),
        "normals": _flatten(normal_loss), "normal_shape": _shape(normal_loss),
        "both_gradient": _grad_record(both_objective, inputs),
        "normal_only_gradient": _grad_record(normal_objective, inputs),
    }
    if device == "mps":
        torch.mps.synchronize()
    return record


def case_specs():
    # Tensor normals cover all meaningful reductions and weight branches.
    for norm, seed, abs_cosine, point, directional, weights in itertools.product(
        (1, 2), (11, 101), (False, True), ("mean", "sum", None),
        (False, True), ("none", "positive", "all_zero"),
    ):
        yield dict(norm=norm, seed=seed, abs_cosine=abs_cosine, point=point,
                   batch=None if point is None else "mean", directional=directional,
                   weights=weights, normals="both", representation="tensor")
    # Add batch sum and per-cloud outputs, partial-zero weights, one-sided
    # normals, and tensor/object mixtures without multiplying the full matrix.
    for representation, norm, abs_cosine, point, directional in itertools.product(
        ("tensor", "x_object", "both_objects"), (1, 2), (False, True),
        ("mean", None), (False, True),
    ):
        yield dict(norm=norm, seed=29, abs_cosine=abs_cosine, point=point,
                   batch=None if point is None else "sum", directional=directional,
                   weights="mixed_zero", normals="both", representation=representation)
    for representation, norm, point, directional in itertools.product(
        ("tensor", "x_object", "both_objects"), (1, 2), ("mean", None),
        (False, True),
    ):
        yield dict(norm=norm, seed=11, abs_cosine=True, point=point,
                   batch=None if point is None else "mean", directional=directional,
                   weights="none", normals="only_x", representation=representation)
    for empty, representation, norm, directional in itertools.product(
        ("x_first", "y_first", "both_first"), ("tensor", "x_object", "both_objects"),
        (1, 2), (False, True),
    ):
        yield dict(norm=norm, seed=11, abs_cosine=True, point="mean", batch="mean",
                   directional=directional, weights="none", normals="both",
                   representation=representation, empty=empty)


def _compare(actual, reference):
    checks = {}
    for name, expected in reference.items():
        observed = actual[name]
        if name.endswith("shape"):
            checks[name] = signature_metrics(observed, expected)
        elif name.endswith("gradient"):
            if expected is None or observed is None:
                checks[name] = signature_metrics(observed, expected)
            else:
                checks[name + "_present"] = signature_metrics(observed["present"], expected["present"])
                for i, (a, e) in enumerate(zip(observed["values"], expected["values"])):
                    checks[f"{name}_{i}"] = error_metrics(a, e)
        else:
            checks[name] = error_metrics(observed, expected)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", choices=("cpu", "mps", "auto"), default="auto")
    args = parser.parse_args()
    root = args.upstream_root.resolve()
    sha = git_head(root)
    if sha != EXPECTED_UPSTREAM:
        parser.error(f"expected pinned PyTorch3D {EXPECTED_UPSTREAM}, got {sha}")
    sys.path.insert(0, str(root))
    from pytorch3d.loss import chamfer_distance as upstream
    from pytorch3d.structures import Pointclouds
    from mps_pointops.chamfer import chamfer_distance as port

    targets = ["cpu", "mps"] if args.devices == "auto" and torch.backends.mps.is_available() else ["cpu"]
    if args.devices != "auto":
        targets = [args.devices]
    if "mps" in targets and not torch.backends.mps.is_available():
        parser.error("MPS requested but unavailable")
    if "mps" in targets and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("MPS parity requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    cases = []
    for spec in case_specs():
        expected = run_case(upstream, Pointclouds, "cpu", spec)
        case = {"spec": spec, "checks": {}}
        for target in targets:
            actual = run_case(port, Pointclouds, target, spec)
            case["checks"][target] = _compare(actual, expected)
        cases.append(case)
    summary = {}
    for target in targets:
        checks = [check for case in cases for check in case["checks"][target].values()]
        summary[target] = {"cases": len(cases), "checks": len(checks),
                           "failed_checks": sum(not check["pass"] for check in checks),
                           "failed_elements": sum(check["mismatches"] for check in checks),
                           "max_abs": max(check["max_abs"] for check in checks)}
    repo_root = Path(__file__).resolve().parents[1]
    record = {
        "upstream": {"commit": sha, "root": str(root)},
        "port": {"commit": git_head(repo_root),
                 "dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root).strip()),
                 "chamfer_sha256": hashlib.sha256((repo_root / "mps_pointops/chamfer.py").read_bytes()).hexdigest(),
                 "kernel_sha256": hashlib.sha256((repo_root / "mps_pointops/kernels/chamfer_nn.metal").read_bytes()).hexdigest(),
                 "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "platform": platform.platform(), "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH"),
                        "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK")},
        "tolerance": {"rtol": RTOL, "atol": ATOL}, "summary": summary, "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    header = {key: value for key, value in record.items() if key != "cases"}
    prefix = json.dumps(header, indent=2, sort_keys=True)
    assert prefix.endswith("\n}")
    case_lines = ["    " + json.dumps(case, sort_keys=True, separators=(",", ":"))
                  for case in cases]
    args.out.write_text(
        prefix[:-2] + ',\n  "cases": [\n' + ",\n".join(case_lines) + "\n  ]\n}\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if all(v["failed_checks"] == 0 for v in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

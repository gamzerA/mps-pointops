"""Compare the supported Chamfer subset with an installed upstream PyTorch3D.

This is an opt-in parity experiment, not a normal pytest test. Build the pinned
upstream CPU extension separately and supply its source checkout via
--upstream-root. The script imports and executes PyTorch3D itself; no upstream
implementation is embedded here.
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

EXPECTED_UPSTREAM = "88e182f989c80836f4bd744e0d9cb1852762ce01"
ATOL = 2e-5
RTOL = 2e-4


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def make_clouds(device: str, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(2, 7, 3, generator=generator, dtype=torch.float32)
    y = torch.randn(2, 6, 3, generator=generator, dtype=torch.float32)
    # Uneven lengths and faraway but finite padding exercise forward masking
    # and ensure padded coordinates have zero gradients.
    x[1, 4:] = torch.tensor([11.0, 12.0, 13.0])
    y[1, 3:] = torch.tensor([-10.0, -11.0, -12.0])
    return (
        x.to(device).requires_grad_(),
        y.to(device).requires_grad_(),
        torch.tensor([7, 4], dtype=torch.int64, device=device),
        torch.tensor([6, 3], dtype=torch.int64, device=device),
    )


def scalarize(loss: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    parts = loss if isinstance(loss, tuple) else (loss,)
    scalar = None
    for side, part in enumerate(parts):
        factors = torch.arange(1, part.numel() + 1, device=part.device, dtype=part.dtype)
        factors = (0.2 + factors * 0.037).reshape(part.shape)
        term = (part * factors * (1.0 if side == 0 else 0.71)).sum()
        scalar = term if scalar is None else scalar + term
    assert scalar is not None
    return scalar


def flatten_loss(loss: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> list[float]:
    parts = loss if isinstance(loss, tuple) else (loss,)
    return [float(v) for part in parts for v in part.detach().cpu().flatten().tolist()]


def run_one(fn, device: str, seed: int, point: str | None, batch: str | None,
            directional: bool, weight_mode: str) -> dict[str, object]:
    x, y, lx, ly = make_clouds(device, seed)
    weights = None
    if weight_mode != "none":
        values = {"positive": [0.3, 1.7], "mixed_zero": [0.0, 1.7],
                  "all_zero": [0.0, 0.0]}[weight_mode]
        weights = torch.tensor(values, dtype=x.dtype, device=device, requires_grad=True)
    loss, normals = fn(
        x, y, x_lengths=lx, y_lengths=ly, weights=weights,
        batch_reduction=batch, point_reduction=point, norm=2,
        single_directional=directional,
    )
    inputs = (x, y) if weights is None else (x, y, weights)
    grads = torch.autograd.grad(scalarize(loss), inputs, allow_unused=True)
    if device == "mps":
        torch.mps.synchronize()
    parts = loss if isinstance(loss, tuple) else (loss,)
    if normals is None:
        normals_signature = "none"
    elif isinstance(normals, tuple):
        normals_signature = [list(part.shape) for part in normals]
    else:
        normals_signature = list(normals.shape)
    normalized_grads = [torch.zeros_like(tensor) if grad is None else grad
                        for tensor, grad in zip(inputs, grads)]
    out = {"loss": flatten_loss(loss),
           "loss_shapes": [list(part.shape) for part in parts],
           "normals_signature": normals_signature,
           "gradient_present": [grad is not None for grad in grads],
           "x_grad": normalized_grads[0].detach().cpu().flatten().tolist(),
           "y_grad": normalized_grads[1].detach().cpu().flatten().tolist()}
    if weights is not None:
        out["weights_grad"] = normalized_grads[2].detach().cpu().flatten().tolist()
    return out


def error_metrics(actual: list[float], expected: list[float]) -> dict[str, float | int | bool]:
    a = torch.tensor(actual, dtype=torch.float64)
    e = torch.tensor(expected, dtype=torch.float64)
    assert a.shape == e.shape
    delta = (a - e).abs()
    allowed = ATOL + RTOL * e.abs()
    return {
        "max_abs": float(delta.max().item()) if delta.numel() else 0.0,
        "max_rel_nonzero": float((delta / e.abs().clamp_min(1e-7)).max().item()) if delta.numel() else 0.0,
        "mismatches": int((delta > allowed).sum().item()),
        "elements": a.numel(),
        "pass": bool((delta <= allowed).all().item()),
    }


def signature_metrics(actual, expected) -> dict[str, float | int | bool | object]:
    same = actual == expected
    return {"max_abs": 0.0, "max_rel_nonzero": 0.0,
            "mismatches": 0 if same else 1, "elements": 1, "pass": same,
            "expected": expected, "actual": actual}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--upstream-build-note", required=True)
    parser.add_argument("--allow-other-upstream", action="store_true")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    upstream_root = args.upstream_root.resolve()
    upstream_sha = git_head(upstream_root)
    if upstream_sha != EXPECTED_UPSTREAM and not args.allow_other_upstream:
        parser.error(f"expected upstream {EXPECTED_UPSTREAM}; got {upstream_sha}")
    sys.path.insert(0, str(upstream_root))
    from pytorch3d import __version__ as upstream_version
    from pytorch3d.loss import chamfer_distance as upstream_chamfer
    from mps_pointops.chamfer import chamfer_distance as port_chamfer

    if args.devices == "auto":
        targets = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    else:
        targets = [args.devices]
    if "mps" in targets and not torch.backends.mps.is_available():
        parser.error("MPS requested but unavailable")
    if "mps" in targets and os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("MPS parity requires PYTORCH_ENABLE_MPS_FALLBACK=0")
    cases = []
    modes = itertools.product(
        (11, 29), ("mean", "sum", "max", None),
        ("mean", "sum", None), (False, True),
        ("none", "positive", "mixed_zero", "all_zero"),
    )
    for seed, point, batch, directional, weight_mode in modes:
        if point is None and batch is not None:
            continue
        ref = run_one(upstream_chamfer, "cpu", seed, point, batch, directional, weight_mode)
        entry = {"seed": seed, "point_reduction": point,
                 "batch_reduction": batch, "single_directional": directional,
                 "weights": weight_mode, "checks": {}}
        for target in targets:
            actual = run_one(port_chamfer, target, seed, point, batch, directional, weight_mode)
            checks = {}
            for field in ref:
                if field in ("loss_shapes", "normals_signature", "gradient_present"):
                    checks[field] = signature_metrics(actual[field], ref[field])
                elif field == "loss" and actual["loss_shapes"] != ref["loss_shapes"]:
                    checks[field] = signature_metrics(actual[field], ref[field])
                else:
                    checks[field] = error_metrics(actual[field], ref[field])
            entry["checks"][target] = checks
        cases.append(entry)
    summary = {}
    for target in targets:
        checks = [case["checks"][target][field] for case in cases for field in case["checks"][target]]
        summary[target] = {
            "cases": len(cases), "checked_tensors": len(checks),
            "failed_tensors": sum(not check["pass"] for check in checks),
            "failed_elements": sum(check["mismatches"] for check in checks),
            "max_abs": max(check["max_abs"] for check in checks),
            "max_rel_nonzero": max(check["max_rel_nonzero"] for check in checks),
        }
    result = {
        "experiment": "PyTorch3D upstream Chamfer direct forward and first-order gradient parity",
        "upstream": {"repository": "https://github.com/facebookresearch/pytorch3d",
                     "commit": upstream_sha, "package_version": upstream_version,
                     "execution_device": "cpu", "build_note": args.upstream_build_note},
        "port": {"repository": "https://github.com/gamzerA/mps-pointops",
                 "commit": git_head(repo_root),
                 "chamfer_py_sha256": hashlib.sha256((repo_root / "mps_pointops/chamfer.py").read_bytes()).hexdigest(),
                 "metal_kernel_sha256": hashlib.sha256((repo_root / "mps_pointops/kernels/chamfer_nn.metal").read_bytes()).hexdigest(),
                 "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "platform": platform.platform(),
                        "mps_available": torch.backends.mps.is_available(),
                        "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH", "0"),
                        "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "unset")},
        "test_design": {"seeds": [11, 29], "batch_size": 2,
                        "x_shape": [2, 7, 3], "y_shape": [2, 6, 3],
                        "x_lengths": [7, 4], "y_lengths": [6, 3],
                        "point_reduction": ["mean", "sum", "max", None],
                        "batch_reduction": ["mean", "sum", None],
                        "single_directional": [False, True],
                        "weights": ["none", "positive", "mixed_zero", "all_zero"],
                        "atol": ATOL, "rtol": RTOL,
                        "scope": "finite float32 point clouds, forward loss and x/y/weight first derivatives; no normals or norm=1"},
        "summary": summary, "cases": cases,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Keep provenance readable while retaining one compact, inspectable line
    # per case. A fully indented matrix adds tens of thousands of diff lines.
    header = {key: value for key, value in result.items() if key != "cases"}
    prefix = json.dumps(header, indent=2, sort_keys=True)
    assert prefix.endswith("\n}")
    case_lines = ["    " + json.dumps(case, sort_keys=True, separators=(",", ":"))
                  for case in cases]
    args.out.write_text(
        prefix[:-2] + ',\n  "cases": [\n' + ",\n".join(case_lines) + "\n  ]\n}\n"
    )
    print(json.dumps({"upstream": upstream_sha, "port": result["port"]["commit"], "summary": summary}, indent=2))
    return 0 if all(item["failed_tensors"] == 0 for item in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

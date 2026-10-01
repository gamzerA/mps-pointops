#!/usr/bin/env python3
"""Run an architecture-matched PointNet++ SSG segmentation comparison.

This harness imports the original Python ``pointnet2_modules`` file at a
pinned upstream checkout. On CUDA it uses the installed original C++/CUDA
extension. On MPS it replaces only the low-level ``pointnet2_utils`` calls
with mps-pointops and an independently written grouping composition. The
model's SA/FP dimensions and forward graph match the upstream SSG semantic
segmentation model; no upstream source is included in this repository.

The deterministic fixture contains finite float32 inputs, labels, and every
parameter/buffer. Run the two backends against the same fixture bytes, then
compare their saved full arrays. This is a numerical validation, not a model
accuracy or training convergence experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


UPSTREAM_COMMIT = "b5ceb6d9ca0467ea34beb81023f96ee82228f626"
FIXTURE_ID = "pointnet2-ssg-segmentation-synthetic-v1"
N_CLASSES = 13


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upstream_identity(root: Path) -> dict:
    root = root.resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream revision {revision} != {UPSTREAM_COMMIT}")
    relpaths = [
        "pointnet2/models/pointnet2_ssg_sem.py",
        "pointnet2_ops_lib/pointnet2_ops/pointnet2_modules.py",
        "pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py",
        "pointnet2_ops_lib/pointnet2_ops/_ext-src/src/interpolate_gpu.cu",
        "pointnet2_ops_lib/pointnet2_ops/_ext-src/src/ball_query_gpu.cu",
        "pointnet2_ops_lib/pointnet2_ops/_ext-src/src/sampling_gpu.cu",
    ]
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--", *relpaths],
        text=True,
    ).strip()
    if dirty:
        raise RuntimeError(f"upstream model/extension source changed: {dirty}")
    return {
        "commit": revision,
        # The Git blobs are identical on Windows and macOS even when the
        # checked-out source files have different CRLF/LF line endings.
        "source_git_blob_sha1": {
            p: subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", f"HEAD:{p}"], text=True
            ).strip() for p in relpaths
        },
        "source_sha256": {p: digest(root / p) for p in relpaths},
        "build_only_setup_diff": subprocess.check_output(
            ["git", "-C", str(root), "diff", "--", "pointnet2_ops_lib/setup.py"],
            text=True,
        ),
    }


def _install_project_backend(upstream: Path) -> None:
    from mps_pointops import compat, pointnet2

    compat.install(force=True)
    package = sys.modules["pointnet2_ops"]
    utils = sys.modules["pointnet2_ops.pointnet2_utils"]
    package.__path__ = [str(upstream / "pointnet2_ops_lib" / "pointnet2_ops")]
    utils.three_nn = pointnet2.three_nn
    utils.three_interpolate = pointnet2.three_interpolate

    # Compose the original model's grouping contract from the public primitive
    # calls. These two class bodies are independent of the upstream source.
    class QueryAndGroup(nn.Module):
        def __init__(self, radius: float, nsample: int, use_xyz: bool = True):
            super().__init__()
            self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

        def forward(self, xyz, new_xyz, features=None):
            idx = utils.ball_query(self.radius, self.nsample, xyz, new_xyz)
            grouped_xyz = utils.grouping_operation(
                xyz.transpose(1, 2).contiguous(), idx
            ) - new_xyz.transpose(1, 2).unsqueeze(-1)
            if features is None:
                if not self.use_xyz:
                    raise ValueError("features or use_xyz is required")
                return grouped_xyz
            grouped_features = utils.grouping_operation(features, idx)
            return (
                torch.cat([grouped_xyz, grouped_features], dim=1)
                if self.use_xyz else grouped_features
            )

    class GroupAll(nn.Module):
        def __init__(self, use_xyz: bool = True):
            super().__init__()
            self.use_xyz = use_xyz

        def forward(self, xyz, new_xyz, features=None):
            grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
            if features is None:
                return grouped_xyz
            grouped_features = features.unsqueeze(2)
            return (
                torch.cat([grouped_xyz, grouped_features], dim=1)
                if self.use_xyz else grouped_features
            )

    utils.QueryAndGroup = QueryAndGroup
    utils.GroupAll = GroupAll


def load_modules(backend: str, upstream: Path):
    if backend == "project":
        _install_project_backend(upstream)
    else:
        # The installed pointnet2_ops package must be the compiled upstream
        # build. Do not silently import mps-pointops' stand-in here.
        import pointnet2_ops._ext  # noqa: F401

    from pointnet2_ops import pointnet2_utils as utils
    from pointnet2_ops import pointnet2_modules as modules

    module_file = Path(modules.__file__).resolve()
    wanted = upstream / "pointnet2_ops_lib" / "pointnet2_ops" / "pointnet2_modules.py"
    if module_file.read_bytes() != wanted.read_bytes():
        raise RuntimeError(f"loaded pointnet2_modules differs from {wanted}")
    if backend == "upstream":
        wrapper = Path(utils.__file__).resolve()
        wanted_wrapper = upstream / "pointnet2_ops_lib" / "pointnet2_ops" / "pointnet2_utils.py"
        if wrapper.read_bytes() != wanted_wrapper.read_bytes():
            raise RuntimeError(f"installed pointnet2_utils differs from {wanted_wrapper}")
    return modules, utils


class SegmentationSSG(nn.Module):
    """Architecture-matched Wijmans PointNet++ semantic segmentation graph.

    The module classes themselves are imported from the original checkout;
    these dimensions mirror ``pointnet2_ssg_sem.py`` at UPSTREAM_COMMIT.
    """

    def __init__(self, modules):
        super().__init__()
        self.sa = nn.ModuleList([
            modules.PointnetSAModule(npoint=1024, radius=0.1, nsample=32, mlp=[6, 32, 32, 64], use_xyz=True),
            modules.PointnetSAModule(npoint=256, radius=0.2, nsample=32, mlp=[64, 64, 64, 128], use_xyz=True),
            modules.PointnetSAModule(npoint=64, radius=0.4, nsample=32, mlp=[128, 128, 128, 256], use_xyz=True),
            modules.PointnetSAModule(npoint=16, radius=0.8, nsample=32, mlp=[256, 256, 256, 512], use_xyz=True),
        ])
        self.fp = nn.ModuleList([
            modules.PointnetFPModule(mlp=[128 + 6, 128, 128, 128]),
            modules.PointnetFPModule(mlp=[256 + 64, 256, 128]),
            modules.PointnetFPModule(mlp=[256 + 128, 256, 256]),
            modules.PointnetFPModule(mlp=[512 + 256, 256, 256]),
        ])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(True), nn.Dropout(0.5),
            nn.Conv1d(128, N_CLASSES, kernel_size=1),
        )

    def forward(self, points):
        xyz = points[..., :3].contiguous()
        features = points[..., 3:].transpose(1, 2).contiguous()
        xyz_levels, feature_levels = [xyz], [features]
        for i, layer in enumerate(self.sa):
            xyz_i, features_i = layer(xyz_levels[i], feature_levels[i])
            xyz_levels.append(xyz_i)
            feature_levels.append(features_i)
        for i in range(-1, -len(self.fp) - 1, -1):
            feature_levels[i - 1] = self.fp[i](
                xyz_levels[i - 1], xyz_levels[i],
                feature_levels[i - 1], feature_levels[i],
            )
        return self.head(feature_levels[0])


def make_fixture(path: Path, modules, n: int = 1152) -> None:
    if n < 1024:
        raise ValueError("N must be >= the first set-abstraction sample count 1024")
    rng = np.random.default_rng(20261002)
    model = SegmentationSSG(modules)
    payload: dict[str, np.ndarray] = {}
    # Binary-fraction coordinates avoid floating-point conversion surprises.
    # The cloud is a three-component mixture with positive translation so the
    # upstream FPS origin filter never suppresses any point.
    xyz = (rng.integers(-160, 161, size=(1, n, 3)) / 1024).astype("f4")
    cluster = rng.integers(0, 3, size=(1, n, 1))
    offsets = np.array([[1.5, 1.5, 1.5], [1.9, 1.5, 1.5], [1.5, 1.9, 1.5]], dtype="f4")
    xyz += offsets[cluster[..., 0]]
    features = (rng.integers(-128, 129, size=(1, n, 6)) / 256).astype("f4")
    payload["points"] = np.concatenate([xyz, features], axis=-1)
    payload["target"] = rng.integers(0, N_CLASSES, size=(1, n), dtype=np.int64)
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point():
            if name.endswith("running_mean"):
                arr = np.zeros(tuple(tensor.shape), dtype="f4")
            elif name.endswith("running_var"):
                arr = np.ones(tuple(tensor.shape), dtype="f4")
            elif name.endswith("weight") and tensor.ndim == 1:
                arr = np.ones(tuple(tensor.shape), dtype="f4")
            elif tensor.ndim == 1:
                arr = np.full(tuple(tensor.shape), 0.05, dtype="f4")
            else:
                fan_in = int(np.prod(tensor.shape[1:]))
                arr = (rng.standard_normal(tuple(tensor.shape)) * np.sqrt(2.0 / fan_in)).astype("f4")
        else:
            arr = np.zeros(tuple(tensor.shape), dtype="i8")
        payload[f"state::{name}"] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    print(f"fixture {path}: SHA256 {digest(path)}, {n} points, {len(payload)-2} state tensors")


def run(backend: str, device: str, upstream: Path, fixture_path: Path, output_path: Path) -> None:
    identity = upstream_identity(upstream)
    modules, utils = load_modules(backend, upstream)
    fixture = np.load(fixture_path, allow_pickle=False)
    model = SegmentationSSG(modules)
    state = {k[7:]: torch.from_numpy(fixture[k].copy()) for k in fixture.files if k.startswith("state::")}
    model.load_state_dict(state, strict=True)
    model.eval()  # Cross-device deterministic dropout and BatchNorm policy.
    model.to(device)
    selections = []
    for opname in ("furthest_point_sample", "ball_query", "three_nn"):
        original = getattr(utils, opname)

        def capture(*args, _op=opname, _fn=original, **kwargs):
            value = _fn(*args, **kwargs)
            selected = value[1] if _op == "three_nn" else value
            selections.append((_op, selected.detach().cpu().numpy().copy()))
            return value

        setattr(utils, opname, capture)

    points = torch.from_numpy(fixture["points"].copy()).to(device).requires_grad_(True)
    target = torch.from_numpy(fixture["target"].copy()).to(device)
    logits = model(points)
    loss = F.cross_entropy(logits, target)
    loss.backward()
    if device == "mps":
        torch.mps.synchronize()
    arrays = {
        "logits": logits.detach().cpu().numpy(),
        "loss": np.array([loss.item()], dtype="f4"),
        "input_grad": points.grad.detach().cpu().numpy(),
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise RuntimeError(f"missing parameter gradient: {name}")
        arrays[f"grad::{name}"] = parameter.grad.detach().cpu().numpy()
    for i, (opname, indices) in enumerate(selections):
        arrays[f"selection::{i:02d}::{opname}"] = indices
    if [kind for kind, _ in selections] != ["furthest_point_sample", "ball_query"] * 4 + ["three_nn"] * 4:
        raise RuntimeError("the full FPS/Ball Query/propagation path was not exercised")
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise RuntimeError(f"non-finite values in {name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    meta = {
        "backend": backend,
        "device": device,
        "fixture_id": FIXTURE_ID,
        "fixture_sha256": digest(fixture_path),
        "result_sha256": digest(output_path),
        "upstream": identity,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH"),
        "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
        "model_mode": "eval, with cross-entropy backward",
        "points": list(points.shape),
        "logits": list(logits.shape),
        "selection_kinds": [kind for kind, _ in selections],
        "num_parameter_gradients": sum(k.startswith("grad::") for k in arrays),
        "harness_sha256": digest(Path(__file__)),
        "project_source_sha256": {
            p: digest(PROJECT_ROOT / p) for p in (
                "mps_pointops/compat.py", "mps_pointops/ops.py",
                "mps_pointops/pointnet2.py", "mps_pointops/kernels/pointnet2.metal",
                "mps_pointops/kernels/fps.metal", "mps_pointops/kernels/ball_query.metal",
            )
        } if backend == "project" else None,
        "gpu_name": (
            torch.cuda.get_device_name(0) if device == "cuda" else
            subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if device == "mps" else None
        ),
        # The binary hash and basename identify the loaded extension without
        # publishing a host-local home-directory path in the evidence JSON.
        "extension_filename": Path(importlib.import_module("pointnet2_ops._ext").__file__).name if backend == "upstream" else None,
        "extension_sha256": digest(Path(importlib.import_module("pointnet2_ops._ext").__file__).resolve()) if backend == "upstream" else None,
    }
    meta_path = output_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"loss": float(loss.detach()), "logits_shape": list(logits.shape), "selections": len(selections), "result": str(output_path), "metadata": str(meta_path)}, indent=2))


def compare(left: Path, right: Path, *, atol: float, rtol: float) -> dict:
    m_left = json.loads(left.with_suffix(".json").read_text())
    m_right = json.loads(right.with_suffix(".json").read_text())
    if m_left["fixture_sha256"] != m_right["fixture_sha256"]:
        raise RuntimeError("fixture bytes differ")
    if m_left["upstream"]["commit"] != m_right["upstream"]["commit"]:
        raise RuntimeError("upstream commits differ")
    if ("source_git_blob_sha1" in m_left["upstream"] and
            "source_git_blob_sha1" in m_right["upstream"] and
            m_left["upstream"]["source_git_blob_sha1"] != m_right["upstream"]["source_git_blob_sha1"]):
        raise RuntimeError("upstream source Git blobs differ")
    with np.load(left, allow_pickle=False) as a, np.load(right, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise RuntimeError(f"array names differ: {set(a.files) ^ set(b.files)}")
        checks = {}
        for key in sorted(a.files):
            x, y = a[key], b[key]
            if x.shape != y.shape:
                raise RuntimeError(f"shape differs for {key}: {x.shape} vs {y.shape}")
            if key.startswith("selection::"):
                mismatch = int(np.count_nonzero(x != y))
                checks[key] = {"elements": int(x.size), "mismatches": mismatch, "passed": mismatch == 0}
            else:
                delta = np.abs(x.astype("f8") - y.astype("f8"))
                passed = bool(np.isfinite(x).all() and np.isfinite(y).all() and np.allclose(x, y, atol=atol, rtol=rtol))
                checks[key] = {"elements": int(x.size), "max_abs_error": float(delta.max(initial=0)), "passed": passed}
    result = {
        "left": str(left), "right": str(right),
        "left_sha256": digest(left), "right_sha256": digest(right),
        "fixture_sha256": m_left["fixture_sha256"],
        "atol": atol, "rtol": rtol, "passed": all(v["passed"] for v in checks.values()),
        "checks": checks,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--backend", choices=("project", "upstream"))
    parser.add_argument("--device", choices=("mps", "cuda", "cpu"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare", type=Path, nargs=2, metavar=("MPS", "CUDA"))
    parser.add_argument("--compare-output", type=Path)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    # Windows RDP keyboard layouts can make an underscore in the checkout
    # directory awkward to enter. A unique sibling glob is unambiguous and
    # avoids relying on shell-specific wildcard expansion.
    if "*" in args.upstream_checkout.name or "?" in args.upstream_checkout.name:
        matches = [p for p in args.upstream_checkout.parent.glob(args.upstream_checkout.name) if p.is_dir()]
        if len(matches) != 1:
            parser.error(f"upstream checkout pattern must match one directory, got {matches}")
        args.upstream_checkout = matches[0]
    if args.compare:
        result = compare(*args.compare, atol=args.atol, rtol=args.rtol)
        if args.compare_output:
            args.compare_output.parent.mkdir(parents=True, exist_ok=True)
            args.compare_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(json.dumps({
                "passed": result["passed"],
                "checks": len(result["checks"]),
                "failed": [key for key, value in result["checks"].items() if not value["passed"]],
                "report": str(args.compare_output),
            }, indent=2))
        else:
            print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit(1)
    elif args.make_fixture:
        upstream_identity(args.upstream_checkout)
        modules, _ = load_modules("project", args.upstream_checkout)
        make_fixture(args.fixture, modules)
    elif args.backend and args.device and args.output:
        run(args.backend, args.device, args.upstream_checkout, args.fixture, args.output)
    else:
        parser.error("choose --make-fixture, --compare, or --backend/--device/--output")


if __name__ == "__main__":
    main()

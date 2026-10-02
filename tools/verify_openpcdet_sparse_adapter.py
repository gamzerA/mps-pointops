"""Opt-in local check of the private sparse adapter in pinned OpenPCDet models.

This script loads unmodified upstream source files supplied by the caller. It
substitutes only the two imported utility modules in this test process; the
package never installs or impersonates ``spconv`` globally. This is not a
direct comparison with upstream CUDA spconv.

Example:
  python tools/verify_openpcdet_sparse_adapter.py \
    --upstream-dir ../research/sparse-upstream --device both \
    --json-out /tmp/openpcdet-sparse-adapter.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import types

import numpy as np
import torch
from torch.nn import functional as F

from mps_pointops import _spconv_compat as spconv


UPSTREAM_COMMIT = "233f849829b6ac19afb8af8837a0246890908755"
UPSTREAM_BLOBS = {
    "spconv_backbone.py": "f0c231a6668977fa45e63b2af8767cf6ef1d74bf",
    "spconv_unet.py": "a5e7c4b369ed7694cbc97970411a892da2e6ed12",
}
RTOL = 1e-4
ATOL = 1e-5


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def _verify_upstream(path: Path) -> None:
    for filename, expected in UPSTREAM_BLOBS.items():
        found = _git_blob_sha1((path / filename).read_bytes())
        if found != expected:
            raise ValueError(f"{filename}: expected upstream Git blob {expected}, got {found}")


def _register_package(name: str) -> None:
    module = types.ModuleType(name)
    module.__path__ = []
    sys.modules[name] = module


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_official_models(path: Path):
    _verify_upstream(path)
    for name in ("pcdet", "pcdet.utils", "pcdet.models", "pcdet.models.backbones_3d"):
        _register_package(name)

    sparse_utils = types.ModuleType("pcdet.utils.spconv_utils")
    sparse_utils.spconv = spconv
    sparse_utils.replace_feature = lambda value, features: value.replace_feature(features)
    sys.modules[sparse_utils.__name__] = sparse_utils

    common_utils = types.ModuleType("pcdet.utils.common_utils")

    def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
        # The official UNet calls this only to report decoded point positions.
        # The independent expression makes no claim about other common_utils APIs.
        size = torch.as_tensor(voxel_size, dtype=torch.float32, device=voxel_coords.device)
        origin = torch.as_tensor(point_cloud_range[:3], dtype=torch.float32, device=voxel_coords.device)
        return (voxel_coords[:, [2, 1, 0]].float() + 0.5) * size * downsample_times + origin

    common_utils.get_voxel_centers = get_voxel_centers
    sys.modules[common_utils.__name__] = common_utils
    backbone = _load_module("pcdet.models.backbones_3d.spconv_backbone", path / "spconv_backbone.py")
    unet = _load_module("pcdet.models.backbones_3d.spconv_unet", path / "spconv_unet.py")
    return backbone.VoxelBackBone8x, unet.UNetV2


def _fixture():
    # Unique active voxels, distributed over all four downsampling levels.
    z = (1, 2, 4, 7, 9, 12, 14, 16, 19, 22, 25, 28, 30)
    coords = torch.tensor(
        [[0, zi, 1 + (i * 3) % 6, 1 + (i * 2) % 6] for i, zi in enumerate(z)],
        dtype=torch.int32,
    )
    features = torch.stack(
        (torch.linspace(0.65, 1.25, len(z)), 0.85 + 0.1 * torch.cos(torch.arange(len(z)).float() * 0.4)),
        dim=1,
    )
    return coords, features


def _diagnostic_weights(model):
    """Keep positive sparse paths alive through the deep synthetic network.

    This fixed fixture exercises gradients through every encoder/decoder
    convolution. It is intentionally not a learned OpenPCDet checkpoint.
    """
    with torch.no_grad():
        for module in model.modules():
            if not isinstance(module, spconv.SparseConvolution):
                continue
            module.weight.zero_()
            offsets = (tuple(size // 2 for size in module.kernel_size),) if module.subm else tuple(
                (i, j, k)
                for i in range(module.kernel_size[0])
                for j in range(module.kernel_size[1])
                for k in range(module.kernel_size[2])
            )
            scale = 0.5 if not module.subm else 0.9
            for out_channel in range(module.out_channels):
                in_channel = out_channel % module.in_channels
                for offset in offsets:
                    module.weight[(out_channel, *offset, in_channel)] = scale
            if module.bias is not None:
                module.bias.zero_()


def _new_model(model_cls, weights):
    torch.manual_seed(741)
    grid_size = np.array([8, 8, 32], dtype=np.int64)
    common = (dict(RETURN_ENCODED_TENSOR=True), 2, grid_size)
    if model_cls.__name__ == "UNetV2":
        model = model_cls(*common, voxel_size=[0.2, 0.2, 0.2], point_cloud_range=[0, 0, 0, 8, 8, 8])
    else:
        model = model_cls(*common)
    if weights == "diagnostic":
        _diagnostic_weights(model)
    return model.eval()


def _dense_first_layer_check(model, coords, features):
    first = model.conv_input[0]
    input_features = features.detach().clone().requires_grad_()
    sparse = spconv.SparseConvTensor(input_features, coords, model.sparse_shape, 1)
    got = first(sparse)
    dense = F.conv3d(
        sparse.dense(), first.weight.permute(0, 4, 1, 2, 3), first.bias,
        stride=1, padding=1,
    )
    rows = coords.long()
    expected = dense.permute(0, 2, 3, 4, 1)[rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]]
    torch.testing.assert_close(got.features, expected, rtol=2e-5, atol=2e-6)
    upstream = torch.cos(torch.arange(expected.numel(), dtype=expected.dtype).reshape_as(expected) * 0.31)
    sparse_dx, sparse_dw = torch.autograd.grad(
        got.features, (input_features, first.weight), grad_outputs=upstream
    )
    dense_dx, dense_dw = torch.autograd.grad(
        expected, (input_features, first.weight), grad_outputs=upstream
    )
    torch.testing.assert_close(sparse_dx, dense_dx, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(sparse_dw, dense_dw, rtol=2e-5, atol=2e-6)
    return {
        "output": float((got.features - expected).abs().max().detach()),
        "input_grad": float((sparse_dx - dense_dx).abs().max().detach()),
        "weight_grad": float((sparse_dw - dense_dw).abs().max().detach()),
    }


def _run(model, coords, features, device):
    model = model.to(device)
    data = features.to(device).detach().requires_grad_()
    stages = {}
    handles = []
    for name, layer in model.named_modules():
        if isinstance(layer, spconv.SparseModule):
            handles.append(layer.register_forward_hook(
                lambda _layer, _args, value, key=name: stages.__setitem__(key, value)
            ))
    start = time.perf_counter()
    try:
        result = model({"voxel_features": data, "voxel_coords": coords, "batch_size": 1})
    finally:
        for handle in handles:
            handle.remove()
    encoded = result["encoded_spconv_tensor"]
    loss = 0.03 * (encoded.features * torch.sin(
        torch.arange(encoded.features.numel(), device=device, dtype=data.dtype).reshape_as(encoded.features) * 0.37
    )).sum()
    if "point_features" in result:
        point = result["point_features"]
        loss = loss + 0.07 * (point * torch.cos(
            torch.arange(point.numel(), device=device, dtype=data.dtype).reshape_as(point) * 0.19
        )).sum()
    # Every sparse layer receives a direct diagnostic term so that a deep
    # ReLU path cannot make a layer-gradient parity check vacuous.
    loss = loss + 0.03 * sum(value.features.square().mean() for value in stages.values())
    loss.backward()
    if device == "mps":
        torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1e3
    if data.grad is None:
        raise AssertionError("model produced no input gradient")
    return {
        "stages": stages,
        "encoded": encoded,
        "point": result.get("point_features"),
        "point_coords": result.get("point_coords"),
        "input_grad": data.grad,
        "param_grads": {name: param.grad for name, param in model.named_parameters()},
        "loss": loss,
        "elapsed_ms": elapsed_ms,
    }


def _compare_tensor(label, actual, expected):
    a, b = actual.detach().cpu(), expected.detach().cpu()
    if a.shape != b.shape:
        raise AssertionError(f"{label}: shape {a.shape} != {b.shape}")
    torch.testing.assert_close(a, b, rtol=RTOL, atol=ATOL, msg=lambda _: label)
    if not a.numel():
        return 0.0, 0.0
    abs_diff = (a - b).abs()
    normalized = abs_diff / (ATOL + RTOL * b.abs())
    return float(abs_diff.max()), float(normalized.max())


def _compare_sparse(label, actual, expected):
    if not torch.equal(actual.indices, expected.indices):
        raise AssertionError(f"{label}: coordinate or row-order mismatch")
    if actual.spatial_shape != expected.spatial_shape:
        raise AssertionError(f"{label}: spatial shape mismatch")
    return _compare_tensor(label + ".features", actual.features, expected.features)


def _compare_runs(mps, cpu):
    if mps["stages"].keys() != cpu["stages"].keys():
        raise AssertionError("model did not execute the same sparse layers")
    errors = {name: _compare_sparse(name, mps["stages"][name], cpu["stages"][name])
              for name in cpu["stages"]}
    errors["encoded"] = _compare_sparse("encoded", mps["encoded"], cpu["encoded"])
    if cpu["point"] is not None:
        errors["point_features"] = _compare_tensor("point_features", mps["point"], cpu["point"])
        errors["point_coords"] = _compare_tensor("point_coords", mps["point_coords"], cpu["point_coords"])
    errors["input_grad"] = _compare_tensor("input_grad", mps["input_grad"], cpu["input_grad"])
    for name, expected in cpu["param_grads"].items():
        actual = mps["param_grads"][name]
        if actual is None or expected is None:
            if actual is not None or expected is not None:
                raise AssertionError(f"{name}: missing parameter gradient")
            continue
        errors["grad." + name] = _compare_tensor("grad." + name, actual, expected)
    return {
        "max_abs_error": max(abs_error for abs_error, _ in errors.values()),
        "max_normalized_error": max(normalized for _, normalized in errors.values()),
        "max_abs_error_by_tensor": {name: abs_error for name, (abs_error, _) in errors.items()},
    }


def _environment():
    environment = {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "mps_available": torch.backends.mps.is_available(),
        "pytorch_mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH", "unset"),
        "pytorch_enable_mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
    }
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            text=True, capture_output=True, check=False,
        )
        environment["chip"] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    return environment


def _implementation_identity():
    root = Path(__file__).resolve().parents[1]
    tracked_files = (
        "mps_pointops/_spconv_compat.py",
        "mps_pointops/_ball_query_mps.py",
        "mps_pointops/_sparse_conv_cpu.py",
        "mps_pointops/_sparse_conv_mps.py",
        "mps_pointops/_subm_conv_mps.py",
        "mps_pointops/_subm_rulebook_mps.py",
        "mps_pointops/_sparse_rulebook.py",
        "mps_pointops/kernels/sparse_conv_grad.metal",
        "mps_pointops/kernels/subm_conv.metal",
        "mps_pointops/kernels/subm_rulebook.metal",
        "tools/verify_openpcdet_sparse_adapter.py",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", *tracked_files],
        cwd=root, check=False,
    )
    if clean.returncode != 0:
        raise RuntimeError("implementation sources differ from HEAD; commit them before recording evidence")
    return {
        "git_commit": commit,
        "implementation_sources_match_commit": True,
        "source_sha256": {
            relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
            for relative in tracked_files
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "both"), default="both")
    parser.add_argument("--model", choices=("backbone", "unet", "both"), default="both")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--weights", choices=("diagnostic", "random"), default="diagnostic")
    args = parser.parse_args()
    if args.device == "both" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; pass --device cpu for the independent CPU checks")
    backbone, unet = _load_official_models(args.upstream_dir)
    coords, features = _fixture()
    report = {
        "upstream": {"repository": "open-mmlab/OpenPCDet", "commit": UPSTREAM_COMMIT,
                     "blobs": UPSTREAM_BLOBS},
        "torch": torch.__version__, "device": args.device, "weights": args.weights,
        "environment": _environment(),
        "implementation": _implementation_identity(),
        "fixture": {"active_voxels": len(coords), "grid_xyz": [8, 8, 32], "batch_size": 1},
        "model_results": {},
        "scope": "Opt-in local model integration; not direct spconv CUDA parity or complete OpenPCDet support.",
        "timing_scope": "Single smoke-run wall time, includes first compilation and Python overhead; not a benchmark.",
    }
    selected = {"backbone": backbone, "unet": unet}
    for name, cls in selected.items():
        if args.model != "both" and args.model != name:
            continue
        cpu_model = _new_model(cls, args.weights)
        dense_errors = _dense_first_layer_check(cpu_model, coords, features)
        if args.device == "both":
            mps_model = _new_model(cls, args.weights)
            mps_model.load_state_dict(cpu_model.state_dict())
        cpu = _run(cpu_model, coords, features, "cpu")
        zero_gradients = [
            param_name for param_name, gradient in cpu["param_grads"].items()
            if gradient is not None and not bool(torch.count_nonzero(gradient))
        ]
        if args.weights == "diagnostic" and zero_gradients:
            raise AssertionError(f"diagnostic loss left parameter gradients at zero: {zero_gradients}")
        item = {
            "cpu_elapsed_ms": cpu["elapsed_ms"],
            "dense_first_layer_max_abs_error": dense_errors,
            "sparse_layers": sum(isinstance(module, spconv.SparseConvolution)
                                 for module in cpu_model.modules()),
            "sparse_module_outputs_checked": len(cpu["stages"]),
            "encoded_rows": len(cpu["encoded"].indices),
            "point_rows": None if cpu["point"] is None else len(cpu["point"]),
            "nonzero_parameter_gradients": sum(
                gradient is not None and bool(torch.count_nonzero(gradient))
                for gradient in cpu["param_grads"].values()
            ),
            "total_parameters_with_gradients": sum(
                gradient is not None for gradient in cpu["param_grads"].values()
            ),
            "zero_parameter_gradients": zero_gradients,
        }
        if args.device == "both":
            mps = _run(mps_model, coords, features, "mps")
            mps_nonzero = sum(
                gradient is not None and bool(torch.count_nonzero(gradient))
                for gradient in mps["param_grads"].values()
            )
            if args.weights == "diagnostic" and mps_nonzero != item["total_parameters_with_gradients"]:
                raise AssertionError("diagnostic loss left an MPS parameter gradient at zero")
            item["mps_nonzero_parameter_gradients"] = mps_nonzero
            item.update({"mps_elapsed_ms": mps["elapsed_ms"], **_compare_runs(mps, cpu)})
        report["model_results"][name] = item
        print(f"{name}: CPU {item['cpu_elapsed_ms']:.1f} ms, "
              f"{item['sparse_layers']} sparse layers, encoded rows {item['encoded_rows']}")
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json_out is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"Wrote {args.json_out}")


if __name__ == "__main__":
    main()

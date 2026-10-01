"""Compare native PointNet++ CUDA with the independent MPS implementation.

Run the same deterministic fixtures in two environments and compare JSON:

    python tools/pointnet2parity.py --backend project --device mps --output mps.json
    python tools/pointnet2parity.py --backend upstream --device cuda \
        --upstream-checkout PATH --compare mps.json --output cuda.json

The upstream checkout must be the pinned official commit. Its CUDA kernel and
Python wrapper must be unmodified; build-only changes to setup.py are allowed.
No upstream code is bundled in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
from pathlib import Path

import torch


FIXTURE_ID = "pointnet2-propagation-v2"
UPSTREAM_COMMIT = "b5ceb6d9ca0467ea34beb81023f96ee82228f626"
UPSTREAM_KERNEL_SHA256 = "e4a327f15d49bcefe62bd72adc407157f97039c3b34037f8fc767b9a15936173"
ATOL = 1e-5
RTOL = 2e-6
ROOT = Path(__file__).resolve().parent.parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(checkout), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def _git_blob_sha256(checkout: Path, path: str) -> str:
    blob = subprocess.check_output(
        ["git", "-C", str(checkout), "show", f"HEAD:{path}"], stderr=subprocess.PIPE
    )
    return hashlib.sha256(blob).hexdigest()


def _provenance(backend: str, upstream_checkout: Path | None) -> dict:
    if backend == "project":
        from mps_pointops import pointnet2
        expected_module = (ROOT / "mps_pointops/pointnet2.py").resolve()
        actual_module = Path(pointnet2.__file__).resolve()
        if actual_module != expected_module:
            raise RuntimeError(
                f"imported project module {actual_module} differs from checkout {expected_module}"
            )

        return {
            "project_commit": _git(ROOT, "rev-parse", "HEAD"),
            "project_python_sha256": _sha256(actual_module),
            "project_metal_sha256": _sha256(ROOT / "mps_pointops/kernels/pointnet2.metal"),
            "python_module": "mps_pointops/pointnet2.py",
            "metal_source": "mps_pointops/kernels/pointnet2.metal",
        }

    if upstream_checkout is None:
        raise ValueError("--upstream-checkout is required for the upstream backend")
    checkout = upstream_checkout.resolve()
    head = _git(checkout, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise RuntimeError(f"upstream checkout HEAD {head} != pinned {UPSTREAM_COMMIT}")
    prefix = checkout / "pointnet2_ops_lib"
    wrapper = prefix / "pointnet2_ops/pointnet2_utils.py"
    kernel = prefix / "pointnet2_ops/_ext-src/src/interpolate_gpu.cu"
    kernel_rel = "pointnet2_ops_lib/pointnet2_ops/_ext-src/src/interpolate_gpu.cu"
    if _git_blob_sha256(checkout, kernel_rel) != UPSTREAM_KERNEL_SHA256:
        raise RuntimeError("upstream CUDA kernel differs from pinned official source")
    if _git(checkout, "diff", "--", "pointnet2_ops_lib/pointnet2_ops"):
        raise RuntimeError("upstream PointNet++ Python/CUDA implementation was modified")
    implementation_status = _git(
        checkout, "status", "--porcelain", "--", "pointnet2_ops_lib/pointnet2_ops"
    )
    if implementation_status:
        raise RuntimeError("upstream implementation contains staged or untracked changes")

    import pointnet2_ops._ext as extension
    import pointnet2_ops.pointnet2_utils as utils

    installed_wrapper = Path(utils.__file__).resolve()
    if _sha256(installed_wrapper) != _sha256(wrapper):
        raise RuntimeError("imported upstream wrapper differs from pinned checkout")
    installed_extension = Path(extension.__file__).resolve()
    return {
        "upstream_commit": head,
        "upstream_checkout_wrapper_sha256": _sha256(wrapper),
        "upstream_imported_wrapper_sha256": _sha256(installed_wrapper),
        "upstream_cuda_kernel_blob_sha256": _git_blob_sha256(checkout, kernel_rel),
        "upstream_cuda_kernel_worktree_sha256": _sha256(kernel),
        "upstream_extension_sha256": _sha256(installed_extension),
        "upstream_extension_filename": installed_extension.name,
        "upstream_setup_diff": _git(checkout, "diff", "--", "pointnet2_ops_lib/setup.py"),
        "upstream_imported_from": "installed pointnet2_ops package; wrapper hash verified against checkout",
    }


def _ops(backend: str):
    if backend == "upstream":
        from pointnet2_ops.pointnet2_utils import three_interpolate, three_nn
    else:
        from mps_pointops.pointnet2 import three_interpolate, three_nn
    return three_nn, three_interpolate


def _tensor_data(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _small_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    # Binary fractions make host serialization identical across machines.
    known = torch.tensor([
        [[-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
        [[2, 0, 0], [2, 1, 0], [2, -1, 0], [3, 0, 0], [1, 0, 0], [2, 0, 1]],
    ], dtype=torch.float32, device=device, requires_grad=True)
    unknown = torch.tensor([
        [[0, 0, 0], [0.25, 0.125, 0], [-0.5, 0.25, 0.125], [0, 0, 0.75]],
        [[2, 0, 0], [2.25, 0.125, 0], [1.5, -0.25, 0.125], [2, 0, 0.75]],
    ], dtype=torch.float32, device=device, requires_grad=True)
    features = torch.tensor([
        [[1, 2, 3, 4, 5, 6], [-0.5, 0.25, 1.5, -2, 2.5, 3.25]],
        [[3, 1, -1, 2, 4, -2], [0.125, -0.25, 0.5, 1, -1.5, 2]],
    ], dtype=torch.float32, device=device, requires_grad=True)
    weights = torch.tensor([
        [[0.5, 0.25, 0.25], [0.125, 0.375, 0.5], [0.25, 0.5, 0.25], [0.75, 0.125, 0.125]],
        [[0.25, 0.5, 0.25], [0.5, 0.25, 0.25], [0.375, 0.125, 0.5], [0.125, 0.75, 0.125]],
    ], dtype=torch.float32, device=device, requires_grad=True)
    gradient = torch.tensor([
        [[1, 0.25, -0.5, 0.125], [0.5, -1, 0.25, 0.75]],
        [[-0.25, 1.5, 0.5, -1], [1, 0.125, -0.75, 0.25]],
    ], dtype=torch.float32, device=device)
    return unknown, known, features, weights, gradient


def _large_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    # Local Python RNG yields identical binary-fraction values across hosts.
    rng = random.Random(20261001)

    def point() -> list[float]:
        return [rng.randint(-10000, 10000) / 1024 for _ in range(3)]

    known = torch.tensor([[point() for _ in range(257)]], dtype=torch.float32,
                         device=device, requires_grad=True)
    unknown = torch.tensor([[point() for _ in range(73)]], dtype=torch.float32,
                           device=device, requires_grad=True)
    features = torch.tensor([[
        [rng.randint(-2048, 2048) / 1024 for _ in range(257)] for _ in range(5)
    ]], dtype=torch.float32, device=device, requires_grad=True)
    weights = torch.tensor([[
        [rng.randint(0, 1024) / 1024 for _ in range(3)] for _ in range(73)
    ]], dtype=torch.float32, device=device, requires_grad=True)
    gradient = torch.tensor([[
        [rng.randint(-1024, 1024) / 1024 for _ in range(73)] for _ in range(5)
    ]], dtype=torch.float32, device=device)
    return unknown, known, features, weights, gradient


def _run_case(name: str, inputs: tuple[torch.Tensor, ...], backend: str) -> dict:
    unknown, known, features, weights, gradient = inputs
    three_nn, three_interpolate = _ops(backend)
    distances, nearest = three_nn(unknown, known)
    indices = nearest.clone()
    if name == "ties-and-duplicates":
        indices[0, 0] = torch.tensor([1, 1, 2], dtype=torch.int32, device=indices.device)
        indices[1, 0] = torch.tensor([3, 3, 3], dtype=torch.int32, device=indices.device)
    interpolated = three_interpolate(features, indices, weights)
    interpolated.backward(gradient)
    if features.device.type == "cuda":
        torch.cuda.synchronize()
    elif features.device.type == "mps":
        torch.mps.synchronize()
    return {
        "nearest_indices": _tensor_data(nearest),
        "distances": _tensor_data(distances),
        "interpolate_indices": _tensor_data(indices),
        "interpolated": _tensor_data(interpolated),
        "feature_gradient": _tensor_data(features.grad),
        "weight_gradient": _tensor_data(weights.grad),
        "coordinate_gradients": {
            "unknown": unknown.grad is not None,
            "known": known.grad is not None,
        },
    }


def run(backend: str, device_name: str, upstream_checkout: Path | None) -> dict:
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA backend unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend unavailable")
    if backend == "upstream" and device_name != "cuda":
        raise ValueError("official PointNet++ backend requires CUDA")
    if backend == "project":
        sys.path.insert(0, str(ROOT))
    provenance = _provenance(backend, upstream_checkout)
    device = torch.device(device_name)
    return {
        "fixture": FIXTURE_ID,
        "backend": backend,
        "device": device_name,
        "gpu": torch.cuda.get_device_name(device) if device_name == "cuda" else
               ("Apple Silicon MPS" if device_name == "mps" else "CPU"),
        "host_os": platform.system(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH", "0"),
        "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK"),
        "provenance": provenance,
        "cases": {
            "ties-and-duplicates": _run_case(
                "ties-and-duplicates", _small_inputs(device), backend),
            "seeded-stride": _run_case("seeded-stride", _large_inputs(device), backend),
        },
    }


def _shape(value):
    if not isinstance(value, list):
        return ()
    children = [_shape(item) for item in value]
    if children and any(child != children[0] for child in children):
        raise ValueError("ragged array in result JSON")
    return (len(value),) + (children[0] if children else ())


def _flatten(value):
    if isinstance(value, list):
        for item in value:
            yield from _flatten(item)
    else:
        yield value


def compare(left: dict, right: dict) -> dict:
    if left["fixture"] != right["fixture"]:
        raise ValueError("fixture IDs differ")
    if set(left["cases"]) != set(right["cases"]):
        raise ValueError("case sets differ")
    report = {"atol": ATOL, "rtol": RTOL, "cases": {}, "passed": True}
    exact_names = {"nearest_indices", "interpolate_indices"}
    numeric_names = {"distances", "interpolated", "feature_gradient", "weight_gradient"}
    for name in left["cases"]:
        a_case, b_case = left["cases"][name], right["cases"][name]
        case = {"arrays": {}, "passed": True}
        for key in sorted(exact_names | numeric_names):
            a_data, b_data = a_case[key], b_case[key]
            a_shape, b_shape = _shape(a_data), _shape(b_data)
            item = {"shape": list(a_shape), "shape_equal": a_shape == b_shape}
            if item["shape_equal"]:
                a = list(_flatten(a_data))
                b = list(_flatten(b_data))
                errors = [abs(float(x) - float(y)) for x, y in zip(a, b)]
                item["elements"] = len(a)
                item["exact_equal"] = a == b
                item["max_absolute_error"] = max(errors, default=0.0)
                item["passed"] = (
                    a == b if key in exact_names else
                    all(math.isfinite(float(x)) and math.isfinite(float(y)) and
                        abs(float(x) - float(y)) <= ATOL + RTOL * abs(float(y))
                        for x, y in zip(a, b))
                )
            else:
                item["passed"] = False
            case["arrays"][key] = item
            case["passed"] &= item["passed"]
        grads = a_case["coordinate_gradients"] == b_case["coordinate_gradients"] == {
            "unknown": False, "known": False
        }
        case["coordinate_gradients_match_and_absent"] = grads
        case["passed"] &= grads
        report["cases"][name] = case
        report["passed"] &= case["passed"]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("upstream", "project"), required=True)
    parser.add_argument("--device", choices=("cuda", "mps", "cpu"), required=True)
    parser.add_argument("--upstream-checkout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    result = run(args.backend, args.device, args.upstream_checkout)
    if args.compare:
        reference = json.loads(args.compare.read_text())
        result["comparison"] = compare(result, reference)
        print(json.dumps(result["comparison"], sort_keys=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.output}")
    if args.compare and not result["comparison"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Compare a pinned original DGCNN classifier with MPS feature-space kNN.

The upstream source stays in an external checkout and is never redistributed.
Its single hard-coded CUDA device line is adapted in memory to ``x.device``.
On CPU, upstream's original GEMM/topk kNN runs unchanged. On MPS, only that
module-level kNN function is replaced by mps_pointops' direct-accumulation
Metal kNN. All model layers and parameters come from upstream code.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import knn as mps_knn  # noqa: E402

UPSTREAM_COMMIT = "f765b469a67730658ba554e97dc11723a7bab628"
UPSTREAM_MODEL_SHA256 = "9be404728fa66eb5f9fc9a47d8553a9a75423f6e7d7b235496be354cfeb9b6e5"
UPSTREAM_LICENSE_SHA256 = "288c5357e9620f022174625a153eb2423d5c14a8d9838bb4a9ef3deadf10549d"
CUDA_DEVICE_LINE = "device = torch.device('cuda')"


def _git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=checkout, text=True).strip()


def _upstream(checkout: Path):
    if _git(checkout, "rev-parse", "HEAD") != UPSTREAM_COMMIT:
        raise ValueError(f"upstream checkout must be {UPSTREAM_COMMIT}")
    if _git(checkout, "status", "--porcelain"):
        raise ValueError("upstream checkout must be clean")
    source = (checkout / "pytorch" / "model.py").read_text()
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != UPSTREAM_MODEL_SHA256:
        raise ValueError(f"upstream model.py SHA-256 mismatch: {digest}")
    license_digest = hashlib.sha256((checkout / "LICENSE").read_bytes()).hexdigest()
    if license_digest != UPSTREAM_LICENSE_SHA256:
        raise ValueError(f"upstream LICENSE SHA-256 mismatch: {license_digest}")
    if source.count(CUDA_DEVICE_LINE) != 1:
        raise ValueError("upstream CUDA device line changed unexpectedly")
    adapted = source.replace(CUDA_DEVICE_LINE, "device = x.device")
    module = types.ModuleType("pinned_dgcnn_model")
    exec(compile(adapted, str(checkout / "pytorch" / "model.py"), "exec"), module.__dict__)
    return module


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a - b)).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4903)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if not 1 <= args.k <= min(args.points, 256):
        parser.error("k must be in [1, min(points, 256)]")
    upstream = _upstream(args.upstream)
    upstream_knn = upstream.knn
    torch.manual_seed(args.seed)
    settings = SimpleNamespace(k=args.k, emb_dims=128, dropout=0.0)
    cpu_model = upstream.DGCNN(settings, output_channels=10).eval()
    mps_model = copy.deepcopy(cpu_model).to("mps").eval()
    input_cpu = torch.randn((2, 3, args.points), requires_grad=True)
    input_mps = input_cpu.detach().to("mps").requires_grad_()
    cpu_indices = []
    mps_indices = []

    def cpu_search(features, k):
        result = upstream_knn(features, k)
        cpu_indices.append(result.detach().cpu())
        return result

    def metal_search(features, k):
        _, result = mps_knn(features.transpose(1, 2), features.transpose(1, 2), k)
        mps_indices.append(result.detach().cpu())
        return result

    upstream.knn = cpu_search
    cpu_logits = cpu_model(input_cpu)
    cpu_logits.square().mean().backward()
    upstream.knn = metal_search
    mps_logits = mps_model(input_mps)
    mps_logits.square().mean().backward()
    torch.mps.synchronize()

    rows = []
    for stage, (a, b) in enumerate(zip(cpu_indices, mps_indices), start=1):
        rows.append({
            "stage": stage,
            "feature_dimension": (3, 64, 64, 128)[stage - 1],
            "compared_indices": a.numel(),
            "index_mismatch_count": int((a != b).sum().item()),
        })
    cpu_grads = dict(cpu_model.named_parameters())
    mps_grads = dict(mps_model.named_parameters())
    grad_errors = {
        name: _max_abs(param.grad, mps_grads[name].grad.cpu())
        for name, param in cpu_grads.items()
    }
    max_logit_error = _max_abs(cpu_logits.detach(), mps_logits.detach().cpu())
    input_grad_error = _max_abs(input_cpu.grad, input_mps.grad.cpu())
    result = {
        "upstream": {
            "url": "https://github.com/WangYueFt/dgcnn",
            "commit": UPSTREAM_COMMIT,
            "model_sha256": UPSTREAM_MODEL_SHA256,
            "license": "MIT",
            "license_sha256": UPSTREAM_LICENSE_SHA256,
            "in_memory_adaptation": "device = torch.device('cuda') -> device = x.device",
            "search_change": "CPU original GEMM/topk; MPS only knn replaced with mps_pointops.knn",
        },
        "environment": {
            "os": platform.platform(), "torch": torch.__version__,
            "mps_device": torch.backends.mps.get_name(),
            "mps_fast_math": os.getenv("PYTORCH_MPS_FAST_MATH", "0"),
            "mps_fallback": os.getenv("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        },
        "fixture": {
            "seed": args.seed, "batch": 2, "points": args.points,
            "k": args.k, "embedding_dimension": 128, "dropout": 0.0,
            "model_mode": "eval, with gradients from mean(square(logits))",
        },
        "neighbor_indices": rows,
        "max_abs_logit_error": max_logit_error,
        "max_abs_input_gradient_error": input_grad_error,
        "max_abs_parameter_gradient_error": max(grad_errors.values()),
        "parameter_gradient_errors": grad_errors,
        "passed": (
            len(rows) == 4 and all(row["index_mismatch_count"] == 0 for row in rows)
            and max_logit_error <= 1e-3
            and input_grad_error <= 1e-3
            and max(grad_errors.values()) <= 1e-3
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in (
        "neighbor_indices", "max_abs_logit_error", "max_abs_input_gradient_error",
        "max_abs_parameter_gradient_error", "passed",
    )}, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

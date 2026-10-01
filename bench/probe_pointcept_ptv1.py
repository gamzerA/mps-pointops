"""Pinned Pointcept v1.2.1 PTv1 Seg26 CPU/MPS forward/backward probe.

This loads the official source from a local tag checkout. The only model
source substitution is ``torch.cuda.IntTensor(n_o)`` to an int32 tensor on the
input device. It runs in a temporary package namespace so Pointcept's broad
package imports and unrelated optional dependencies are not loaded.

Example:
  PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
    PYTHONPATH=/path/to/einops-target \
    python bench/probe_pointcept_ptv1.py \
      --pointcept-root /private/tmp/pointcept-v121 --output /private/tmp/ptv1.json
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import compat  # noqa: E402

POINTCEPT_COMMIT = "21fb5c51d8c622550b8e22c8ead984511e11d54a"
ORIGINAL = "torch.cuda.IntTensor(n_o)"
REPLACEMENT = "torch.tensor(n_o, dtype=torch.int32, device=p.device)"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
        text=True, check=True,
    ).stdout.strip()


@contextmanager
def _seg26_class(pointcept_root: Path):
    source_file = pointcept_root / "pointcept/models/point_transformer/point_transformer_seg.py"
    source = source_file.read_text()
    if source.count(ORIGINAL) != 1:
        raise RuntimeError("expected exactly one Seg26 CUDA offset constructor")
    patched = source.replace(ORIGINAL, REPLACEMENT)
    # Only this file is rewritten; the three dependency source files are
    # symlinked unchanged from the pinned checkout. Empty package initializers
    # prevent importing Pointcept's unrelated model families.
    with tempfile.TemporaryDirectory(prefix="ptv1-probe-") as temporary:
        temporary_root = Path(temporary)
        package = temporary_root / "pointcept"
        for relative in ("", "models", "models/point_transformer", "utils"):
            directory = package / relative
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").write_text("")
        links = {
            "models/builder.py": "pointcept/models/builder.py",
            "models/point_transformer/utils.py": "pointcept/models/point_transformer/utils.py",
            "utils/registry.py": "pointcept/utils/registry.py",
            "utils/misc.py": "pointcept/utils/misc.py",
        }
        for target, official in links.items():
            (package / target).symlink_to(pointcept_root / official)
        (package / "models/point_transformer/point_transformer_seg.py").write_text(patched)
        sys.path.insert(0, str(temporary_root))
        try:
            module = importlib.import_module(
                "pointcept.models.point_transformer.point_transformer_seg"
            )
            yield module.PointTransformerSeg26, {
                "official_seg_sha256": _sha256(source_file),
                "temporary_seg_sha256": _sha256_bytes(patched.encode()),
                "replacement_count": 1,
                "replacement_original": ORIGINAL,
                "replacement_applied": REPLACEMENT,
            }
        finally:
            sys.path.remove(str(temporary_root))
            for name in list(sys.modules):
                if name == "pointcept" or name.startswith("pointcept."):
                    del sys.modules[name]


def _input(points_per_batch: int, fixture: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if points_per_batch < 256 or points_per_batch % 256:
        raise ValueError("points per batch must be a positive multiple of 256")
    generator = torch.Generator(device="cpu").manual_seed(2701)
    coordinates = []
    for batch in range(2):
        if fixture == "random3d":
            coordinates.append(torch.rand((points_per_batch, 3), generator=generator) + batch * 4.0)
        elif fixture == "line":
            x = torch.arange(points_per_batch, dtype=torch.float32) * 0.25 + batch * 100.0
            coordinates.append(torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=1))
        else:
            raise ValueError(f"unknown fixture: {fixture}")
    coord = torch.cat(coordinates)
    feat = torch.randn((len(coord), 6), generator=generator)
    offset = torch.tensor([points_per_batch, len(coord)], dtype=torch.int32)
    return coord, feat, offset


def _run(model: torch.nn.Module, device: str, coord: torch.Tensor, feat: torch.Tensor, offset: torch.Tensor) -> dict[str, torch.Tensor | float]:
    model = model.to(device).eval()
    p = coord.to(device).detach().requires_grad_(True)
    x = feat.to(device).detach().requires_grad_(True)
    o = offset.to(device)
    if device == "mps":
        torch.mps.synchronize()
    start = time.perf_counter()
    result = model({"coord": p, "feat": x, "offset": o})
    loss = result.square().mean()
    loss.backward()
    if device == "mps":
        torch.mps.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    first_weight = model.enc1[0].linear.weight
    if p.grad is None or x.grad is None or first_weight.grad is None:
        raise AssertionError("missing model/input gradient")
    tensors = {
        "output": result.detach().cpu(),
        "coordinate_gradient": p.grad.detach().cpu(),
        "feature_gradient": x.grad.detach().cpu(),
        "first_weight_gradient": first_weight.grad.detach().cpu(),
    }
    if any(not bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise AssertionError("nonfinite output or gradient")
    return {**tensors, "elapsed_ms": elapsed, "loss": float(loss.detach().cpu())}


def _comparison(cpu: torch.Tensor, mps: torch.Tensor, atol: float, rtol: float) -> dict[str, object]:
    error = (cpu - mps).abs()
    permitted = atol + rtol * cpu.abs()
    return {
        "shape": list(cpu.shape),
        "max_abs": float(error.max()) if error.numel() else 0.0,
        "mean_abs": float(error.mean()) if error.numel() else 0.0,
        "max_excess_over_atol_rtol": float((error - permitted).clamp_min(0).max()) if error.numel() else 0.0,
        "allclose": bool(torch.all(error <= permitted)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcept-root", type=Path, required=True)
    parser.add_argument("--points-per-batch", type=int, default=256)
    parser.add_argument("--fixture", choices=["random3d", "line"], default="random3d")
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--rtol", type=float, default=5e-3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.getenv("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("set PYTORCH_ENABLE_MPS_FALLBACK=0")
    if os.getenv("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
        parser.error("set PYTORCH_MPS_FAST_MATH=0 or 1")
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    pointcept_root = args.pointcept_root.resolve()
    if _commit(pointcept_root) != POINTCEPT_COMMIT:
        parser.error(f"Pointcept checkout must be at {POINTCEPT_COMMIT}")
    import einops

    installed = compat.install(pointcept=True)
    if "pointops" not in installed:
        parser.error("pointops is already importable; run in an environment without upstream pointops")
    coord, feat, offset = _input(args.points_per_batch, args.fixture)
    with _seg26_class(pointcept_root) as (seg26_class, source):
        torch.manual_seed(2701)
        cpu_model = seg26_class(in_channels=6, num_classes=13)
        mps_model = copy.deepcopy(cpu_model)
        cpu = _run(cpu_model, "cpu", coord, feat, offset)
        mps = _run(mps_model, "mps", coord, feat, offset)
    comparisons = {
        name: _comparison(cpu[name], mps[name], args.atol, args.rtol)
        for name in ("output", "coordinate_gradient", "feature_gradient", "first_weight_gradient")
    }
    result = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "pointcept_commit": POINTCEPT_COMMIT,
        "pointcept_source": source,
        "mps_pointops_commit": _commit(ROOT),
        "pointcept_shim_sha256": _sha256(ROOT / "mps_pointops/pointcept.py"),
        "probe_sha256": _sha256(Path(__file__)),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "einops": einops.__version__,
        "macos": platform.mac_ver()[0],
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
        "points_per_batch": args.points_per_batch,
        "fixture": args.fixture,
        "batches": 2,
        "input_channels": 6,
        "classes": 13,
        "model_mode": "eval with autograd enabled",
        "loss": "mean(square(logits))",
        "atol": args.atol,
        "rtol": args.rtol,
        "cpu_elapsed_ms": cpu["elapsed_ms"],
        "mps_elapsed_ms": mps["elapsed_ms"],
        "cpu_loss": cpu["loss"],
        "mps_loss": mps["loss"],
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not all(item["allclose"] for item in comparisons.values()):
        raise SystemExit("CPU/MPS output or gradient exceeded tolerance")


if __name__ == "__main__":
    main()

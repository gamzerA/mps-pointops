"""Print reproducible MPS fused-vs-index_add cancellation evidence as JSON."""

from __future__ import annotations

import json
import os

import torch

from mps_pointops.voxel import voxel_downsample


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required")
    rtol, atol = 1e-4, 1e-5
    cases = []
    for amplitude in (1e4, 1e6):
        features_cpu = torch.tensor([[amplitude], [-amplitude], [0.1]]).repeat(341, 1)
        features = features_cpu.to("mps")
        pos = torch.full((len(features), 1), 0.25, device="mps")
        fused = voxel_downsample(pos, 1.0, features=features, feature_reduce="sum",
                                 pool_backend="fused_csr")
        baseline = voxel_downsample(pos, 1.0, features=features, feature_reduce="sum")
        assert fused.features is not None and baseline.features is not None
        torch.mps.synchronize()
        fused_value = float(fused.features.item())
        baseline_value = float(baseline.features.item())
        oracle = float(features_cpu.double().sum().item())
        allowed = atol + rtol * abs(baseline_value)
        cases.append({
            "features": f"341 repeats of [{amplitude:g}, {-amplitude:g}, float32(0.1)]",
            "float32_residue": float(features_cpu[2, 0].item()),
            "float64_oracle": oracle,
            "fused_csr": fused_value,
            "index_add": baseline_value,
            "absolute_backend_difference": abs(fused_value - baseline_value),
            "allowed_backend_difference": allowed,
            "backend_parity": abs(fused_value - baseline_value) <= allowed,
            "fused_oracle_error": abs(fused_value - oracle),
            "index_add_oracle_error": abs(baseline_value - oracle),
            "integer_maps_equal": all(
                torch.equal(getattr(fused.voxels, key), getattr(baseline.voxels, key))
                for key in ("voxel_coords", "batch", "inverse", "counts", "point_order", "ptr")
            ),
        })
    print(json.dumps({
        "torch_version": torch.__version__,
        "mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH"),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "rtol": rtol,
        "atol": atol,
        "cases": cases,
    }, indent=2))


if __name__ == "__main__":
    main()

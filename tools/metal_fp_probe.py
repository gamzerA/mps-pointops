# SPDX-License-Identifier: MIT
"""Print observed Metal float behavior in a fresh PyTorch process.

Run separately with PYTORCH_MPS_FAST_MATH=0 and =1. This is a diagnostic,
not a portability assertion: the Metal specification permits device-dependent
subnormal handling and compiler-dependent contraction.
"""

from __future__ import annotations

import json
import math
import os
import platform
import struct

import torch


KERNEL = r"""
#include <metal_stdlib>
using namespace metal;
kernel void probe(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float a = input[0], b = input[1], c = input[2];
    output[0] = a * b + c;
    output[1] = fma(a, b, c);
    output[2] = input[3] * input[4];
    output[3] = (input[5] + input[6]) - input[5];
    output[4] = input[7] / input[7];
}
"""


RADIUS_KERNEL = r"""
#pragma METAL fp contract(off)
#include <metal_stdlib>
using namespace metal;

inline float squared_norm(float3 delta) {
    float sum = delta.x * delta.x;
    sum = fma(delta.y, delta.y, sum);
    return fma(delta.z, delta.z, sum);
}

kernel void radius_probe(
    device const float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const float radius = input[0];
    const float3 delta = float3(input[1], input[2], input[3]);
    const float radius_sq = radius * radius;
    const float direct_sq = squared_norm(delta);
    const float3 scaled = delta / radius;
    const float normalized_sq = squared_norm(scaled);
    output[0] = radius;
    output[1] = radius_sq;
    output[2] = direct_sq;
    output[3] = normalized_sq;
    output[4] = scaled.x;
    output[5] = scaled.y;
    output[6] = scaled.z;
    output[7] = direct_sq < radius_sq ? 1.0f : 0.0f;
    output[8] = normalized_sq < 1.0f ? 1.0f : 0.0f;
    output[9] = 1.0f / radius;
    output[10] = 0.0f / radius;
    output[11] = normalized_sq * radius_sq;
}
"""


def encode(value: float) -> dict[str, object]:
    return {
        "float": value if math.isfinite(value) else str(value),
        "bits": f"0x{struct.unpack('<I', struct.pack('<f', value))[0]:08x}",
    }


def run(source: str, values: torch.Tensor) -> list[dict[str, object]]:
    output = torch.empty(5, dtype=torch.float32, device="mps")
    torch.mps.compile_shader(source).probe(values, output, threads=[1, 1, 1])
    torch.mps.synchronize()
    names = ["a*b+c", "fma(a,b,c)", "min_normal*0.5", "(2^24+1)-2^24", "0/0"]
    return [
        {"expression": name, **encode(value)}
        for name, value in zip(names, output.cpu().tolist())
    ]


def radius_case(name: str, radius: float, delta: tuple[float, float, float]) -> dict[str, object]:
    values = torch.tensor((radius, *delta), dtype=torch.float32, device="mps")
    output = torch.empty(12, dtype=torch.float32, device="mps")
    torch.mps.compile_shader(RADIUS_KERNEL).radius_probe(values, output, threads=[1, 1, 1])
    torch.mps.synchronize()
    names = (
        "radius",
        "radius_sq",
        "direct_sq",
        "normalized_sq",
        "scaled_x",
        "scaled_y",
        "scaled_z",
        "direct_inside",
        "normalized_inside",
        "reciprocal_radius",
        "zero_div_radius",
        "reconstructed_sq",
    )
    return {
        "case": name,
        "input": {"radius": radius, "delta": list(delta)},
        "output": {key: encode(value) for key, value in zip(names, output.cpu().tolist())},
    }


def permuted_delta(axis: int, radius: float, large_scale: float) -> tuple[float, float, float]:
    return tuple(
        0.25 * radius if i == axis
        else large_scale * radius if i == (axis + 1) % 3
        else 0.0
        for i in range(3)
    )


def main() -> None:
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; this probe must run on an Apple GPU")
    values = torch.tensor(
        [1 + 2**-23, 1 - 2**-23, -1, 2**-126, 0.5, 2**24, 1, 0],
        dtype=torch.float32,
        device="mps",
    )
    result = {
        "torch": torch.__version__,
        "macos": platform.mac_ver()[0],
        "device": torch.mps.get_name() if hasattr(torch.mps, "get_name") else "MPS",
        "PYTORCH_MPS_FAST_MATH": os.getenv("PYTORCH_MPS_FAST_MATH"),
        "default_source": run(KERNEL, values),
        "contract_off_source": run("#pragma METAL fp contract(off)\n" + KERNEL, values),
        "radius_cases": [
            radius_case(
                f"small_component_{axis}", 2.0**-62, permuted_delta(axis, 2.0**-62, 0.98)
            )
            for axis in range(3)
        ] + [
            radius_case(
                f"inside_small_component_{axis}", 2.0**-62, permuted_delta(axis, 2.0**-62, 0.8)
            )
            for axis in range(3)
        ] + [
            # Distinguish a fused y term with a normal accumulator from the
            # earlier y case, where fma(y, y, 0) itself is subnormal.
            radius_case(
                "small_y_after_normal_accumulator",
                2.0**-62,
                (0.98 * 2.0**-62, 0.25 * 2.0**-62, 0.0),
            ),
            radius_case("identical_subnormal_radius", 1e-40, (0.0, 0.0, 0.0)),
        ],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

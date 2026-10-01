#!/usr/bin/env python3
"""Verify the dense Ball Query SIMD output against its previous Metal kernel.

The old shader is read from a fixed Git commit. Both shaders run in the same
process on identical inputs with *poisoned* preallocated output buffers. The
CPU oracle checks first-K indices only: its wider floating-point arithmetic
is not a general bitwise oracle for Metal squared distances near boundaries.

Run Safe and Fast Math in separate processes because PyTorch caches the Metal
compiler configuration::

    PYTORCH_ENABLE_MPS_FALLBACK=0 python bench/verify_ball_query_simd_contract.py
    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
        python bench/verify_ball_query_simd_contract.py

The fixed baseline commit must be present locally (fetch its history if this
checkout is shallow). No package API, gradients, or performance is measured.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SHADER_PATH = "mps_pointops/kernels/ball_query.metal"
BASELINE_COMMIT = "ea07179dd13bd8545753f9275e628ba4974c6639"
BASELINE_SHA256 = "331d385610346131ef73f306411bf2a9bfe3f532bab6bd2e6cf41dcd48d3816f"
SIMD_SHA256 = "2856fa046cbbafb71ec0cd688f071c7dc5d2280e9f04d00471241e24d6f85ba3"
GROUP_SIZE = 256
QUERIES_PER_GROUP = 8
KS = (1, 31, 33, 65)
SCENARIOS = (
    "sorted", "permuted_finite", "permuted_nonfinite", "zero_radius",
    "empty_reference", "tiny_radius", "subnormal_component_r2m62",
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.contiguous().numpy().tobytes()


def command(*args: str) -> str:
    try:
        return subprocess.run(
            args, cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def sources() -> tuple[str, str]:
    try:
        baseline = subprocess.run(
            ["git", "show", f"{BASELINE_COMMIT}:{SHADER_PATH}"],
            cwd=ROOT, capture_output=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"baseline commit {BASELINE_COMMIT} is unavailable; fetch repository history"
        ) from exc
    simd = (ROOT / SHADER_PATH).read_bytes()
    for label, source, expected in (
        ("baseline", baseline, BASELINE_SHA256),
        ("SIMD", simd, SIMD_SHA256),
    ):
        actual = sha256(source)
        if actual != expected:
            raise RuntimeError(f"{label} shader SHA-256 changed: {actual} != {expected}")
    return baseline.decode(), simd.decode()


def case_inputs(scenario: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], float]:
    """Use exactly representable binary fractions for independent CPU decisions."""
    point_count = 97
    x = torch.arange(point_count, dtype=torch.float64) / 16
    y = (torch.arange(point_count, dtype=torch.int64) % 7 - 3).double() / 16
    z = (torch.arange(point_count, dtype=torch.int64) % 5 - 2).double() / 16
    first = torch.stack((x, y, z), dim=-1)
    first[72, 1:] = 0  # For query x=0, point 72 lies exactly at radius 4.5.
    second = first.clone()
    second[:, 0] += 0.25
    second[:, 1] *= -1
    points = torch.stack((first, second))

    query_x = torch.tensor([0, 0.5, 2, 3, 20, 6, 4.5, 5, 1], dtype=torch.float64)
    queries = torch.zeros((2, 9, 3), dtype=torch.float64)
    queries[:, :, 0] = query_x
    queries[1, :, 0] += 0.25
    queries[:, 3, 1] = 0.25
    queries[:, 6, 2] = -0.125
    query_lengths = [9, 7]  # Batch 1 has two inactive queries.
    point_lengths = [97, 63]  # Includes a partial SIMD block and K=65 padding.
    radius = 4.5

    if scenario in ("permuted_finite", "permuted_nonfinite"):
        generator = torch.Generator().manual_seed(20261001)
        permutation = torch.randperm(point_count, generator=generator)
        points = points[:, permutation].contiguous()
        if scenario == "permuted_nonfinite":
            points[:, 31, 0] = float("nan")
            points[:, 32, 1] = float("inf")
            points[:, 64, 2] = -float("inf")
            queries[:, 7, 0] = float("nan")
    elif scenario == "zero_radius":
        radius = 0.0
    elif scenario == "empty_reference":
        points = points[:, :0].contiguous()
        point_lengths = [0, 0]
    elif scenario == "tiny_radius":
        scale = 2.0**-60
        points = points * scale
        queries = queries * scale
        radius *= scale
    elif scenario == "subnormal_component_r2m62":
        radius = 2.0**-62
        j = torch.arange(point_count, dtype=torch.float64)
        first = torch.stack((
            j * (radius / 128),
            (j.remainder(7) - 3) * (radius / 256),
            (j.remainder(5) - 2) * (radius / 256),
        ), dim=-1)
        first[96] = torch.tensor([radius, 0, 0], dtype=torch.float64)
        second = first.clone()
        second[:, 0] += radius / 8
        points = torch.stack((first, second))
        query_x = torch.tensor(
            [0, 0.25, 0.5, 0.75, 2, 1, 0, 0.625, 0.125], dtype=torch.float64
        ) * radius
        queries = torch.zeros((2, 9, 3), dtype=torch.float64)
        queries[:, :, 0] = query_x
        queries[1, :, 0] += radius / 8
    elif scenario != "sorted":
        raise ValueError(scenario)

    return queries.to(dtype), points.to(dtype), query_lengths, point_lengths, radius


def cpu_first_k(
    queries: torch.Tensor, points: torch.Tensor,
    query_lengths: list[int], point_lengths: list[int], radius: float, k: int,
) -> torch.Tensor:
    """Independent scalar CPU first-K oracle for these exact binary inputs."""
    batch, query_count, _ = queries.shape
    answer = torch.full((batch, query_count, k), -1, dtype=torch.int64)
    radius_sq = radius * radius
    for b in range(batch):
        for i in range(query_lengths[b]):
            q = queries[b, i].tolist()
            if not all(math.isfinite(value) for value in q):
                continue
            found = 0
            for j in range(point_lengths[b]):
                p = points[b, j].tolist()
                if not all(math.isfinite(value) for value in p):
                    continue
                distance_sq = sum((a - c) ** 2 for a, c in zip(q, p))
                if distance_sq < radius_sq:
                    answer[b, i, found] = j
                    found += 1
                    if found == k:
                        break
    return answer


def radius_constants(radius: float) -> tuple[float, float]:
    radius_f32 = struct.unpack("f", struct.pack("f", radius))[0]
    radius_sq = struct.unpack("f", struct.pack("f", radius_f32 * radius_f32))[0]
    return radius_f32, radius_sq


def run_kernel(kernel, *, simd: bool, queries: torch.Tensor, points: torch.Tensor,
               query_lengths: list[int], point_lengths: list[int], radius: float,
               k: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch, query_count, _ = queries.shape
    point_count = points.shape[1]
    q_mps = queries.to("mps")
    p_mps = points.to("mps")
    qlen_mps = torch.tensor(query_lengths, dtype=torch.int64, device="mps")
    plen_mps = torch.tensor(point_lengths, dtype=torch.int64, device="mps")
    indices = torch.full((batch, query_count, k), -777, dtype=torch.int64, device="mps")
    distances = torch.full((batch, query_count, k), float("nan"), dtype=torch.float32, device="mps")
    radius_f32, radius_sq = radius_constants(radius)
    threads = (
        ((batch * query_count + QUERIES_PER_GROUP - 1) // QUERIES_PER_GROUP) * GROUP_SIZE
        if simd else batch * query_count
    )
    kernel(
        q_mps, p_mps, qlen_mps, plen_mps, indices, distances,
        batch, query_count, point_count, k, radius_sq, radius_f32,
        threads=[threads, 1, 1], group_size=[GROUP_SIZE, 1, 1],
    )
    torch.mps.synchronize()
    return indices.cpu(), distances.cpu()


def verify_case(old_library, simd_library, scenario: str, dtype: torch.dtype, k: int) -> dict:
    queries, points, query_lengths, point_lengths, radius = case_inputs(scenario, dtype)
    expected_indices = cpu_first_k(queries, points, query_lengths, point_lengths, radius, k)
    kernel_name = "ball_query_f32" if dtype == torch.float32 else "ball_query_f16"
    old_indices, old_distances = run_kernel(
        getattr(old_library, kernel_name), simd=False,
        queries=queries, points=points, query_lengths=query_lengths,
        point_lengths=point_lengths, radius=radius, k=k,
    )
    simd_indices, simd_distances = run_kernel(
        getattr(simd_library, kernel_name), simd=True,
        queries=queries, points=points, query_lengths=query_lengths,
        point_lengths=point_lengths, radius=radius, k=k,
    )
    shape = (2, 9, k)
    for label, indices, distances in (
        ("baseline", old_indices, old_distances),
        ("SIMD", simd_indices, simd_distances),
    ):
        if indices.shape != shape or distances.shape != shape:
            raise AssertionError(f"{label} {scenario} {dtype} K={k}: wrong output shape")
        if indices.dtype != torch.int64 or distances.dtype != torch.float32:
            raise AssertionError(f"{label} {scenario} {dtype} K={k}: wrong output dtype")
        if (indices == -777).any() or torch.isnan(distances).any():
            raise AssertionError(f"{label} {scenario} {dtype} K={k}: unwritten output slot")
    old_index_bytes = tensor_bytes(old_indices)
    simd_index_bytes = tensor_bytes(simd_indices)
    cpu_index_bytes = tensor_bytes(expected_indices)
    old_distance_bytes = tensor_bytes(old_distances)
    simd_distance_bytes = tensor_bytes(simd_distances)
    if old_index_bytes != simd_index_bytes:
        raise AssertionError(f"{scenario} {dtype} K={k}: SIMD indices differ from baseline")
    if simd_index_bytes != cpu_index_bytes:
        raise AssertionError(f"{scenario} {dtype} K={k}: SIMD indices differ from CPU first-K oracle")
    if old_distance_bytes != simd_distance_bytes:
        raise AssertionError(f"{scenario} {dtype} K={k}: SIMD squared-distance bits differ from baseline")
    return {
        "scenario": scenario,
        "coordinate_dtype": str(dtype),
        "batch": 2,
        "queries_per_batch": 9,
        "points_per_batch": points.shape[1],
        "query_lengths": query_lengths,
        "point_lengths": point_lengths,
        "k": k,
        "radius": radius,
        "input_sha256": {
            "queries": sha256(tensor_bytes(queries)),
            "points": sha256(tensor_bytes(points)),
        },
        "output_shape": list(shape),
        "indices_dtype": str(simd_indices.dtype),
        "squared_distances_dtype": str(simd_distances.dtype),
        "all_output_slots_written": True,
        "simd_indices_match_baseline_bytes": True,
        "simd_indices_match_cpu_bytes": True,
        "simd_squared_distances_match_baseline_bytes": True,
        "indices_sha256": sha256(simd_index_bytes),
        "squared_distances_sha256": sha256(simd_distance_bytes),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if not torch.backends.mps.is_available() or not hasattr(torch.mps, "compile_shader"):
        raise RuntimeError("Apple Silicon MPS and torch.mps.compile_shader are required")
    baseline_source, simd_source = sources()
    baseline_library = torch.mps.compile_shader(baseline_source)
    simd_library = torch.mps.compile_shader(simd_source)
    mode = "fast" if os.environ.get("PYTORCH_MPS_FAST_MATH") == "1" else "safe"
    result = {
        "environment": {
            "chip": command("sysctl", "-n", "machdep.cpu.brand_string"),
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "math_mode": mode,
            "pytorch_mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH", "unset"),
            "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "unset"),
        },
        "sources": {
            "baseline_commit": BASELINE_COMMIT,
            "baseline_shader_path": SHADER_PATH,
            "baseline_shader_sha256": BASELINE_SHA256,
            "simd_shader_path": SHADER_PATH,
            "simd_shader_sha256": SIMD_SHA256,
            "verification_script_path": "bench/verify_ball_query_simd_contract.py",
            "verification_script_sha256": sha256(Path(__file__).read_bytes()),
            "checkout_head": command("git", "rev-parse", "HEAD"),
        },
        "method": {
            "shader_invocation": "preallocated buffers poisoned with int64 -777 and float32 NaN; explicit torch.mps.synchronize after each launch",
            "index_oracle": "independent scalar CPU first-K scan on exact binary-fraction coordinates",
            "distance_oracle": "previous Metal shader bytes; CPU floating-point distances are not a general bitwise oracle near boundaries",
            "test_dimensions": "float32/float16; B=2, Q=9; K=1,31,33,65; P=97 or 0; shorter lengths, no hits, nonfinite values in Safe Math, spatial sorting, normalized small-radius path including r=2^-62",
            "fast_math_nonfinite_policy": "The nonfinite-coordinate case is skipped in Fast Math because that mode does not guarantee NaN/Inf exclusion.",
            "limits": "No performance, gradients, arbitrary near-boundary floating-point parity, or public-API allocation test",
        },
        "cases": [],
    }
    for scenario in SCENARIOS:
        if mode == "fast" and scenario == "permuted_nonfinite":
            continue
        for dtype in (torch.float32, torch.float16):
            if scenario in ("tiny_radius", "subnormal_component_r2m62") and dtype == torch.float16:
                continue
            for k in KS:
                result["cases"].append(
                    verify_case(baseline_library, simd_library, scenario, dtype, k)
                )
    result["cases_passed"] = len(result["cases"])
    chip_slug = "apple-m5-pro"
    out = args.out or ROOT / "bench" / "results" / (
        f"{dt.date.today().isoformat()}-{chip_slug}-ball-query-simd-contract-{mode}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['cases'])} differential cases passed ({mode}); saved {out}")


if __name__ == "__main__":
    main()

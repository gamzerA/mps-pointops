#!/usr/bin/env python3
"""Validate and merge one-case compact-voxel benchmark fragments.

The base must contain a prefix of the canonical 24-case matrix. Each
fragment must be a successful one-case invocation of bench_voxel_api.py for
the next missing case. Inputs are never modified; the complete publish copy
is written to a new path with source checksums and an executable-path note.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import tempfile
from datetime import datetime, timezone
from pathlib import Path

NODES = (20_000, 100_000, 500_000)
SHAPES = ("uniform", "ragged")
OCCUPANCIES = ("dense", "sparse")
DEVICES = ("cpu", "mps")
EXPECTED = list(itertools.product(NODES, SHAPES, OCCUPANCIES, DEVICES))
MATRIX = {
    "nodes": list(NODES),
    "batch_shapes": list(SHAPES),
    "occupancies": list(OCCUPANCIES),
    "devices": list(DEVICES),
}
TIMINGS = (
    "stage_voxelize_forward",
    "stage_aggregate_forward",
    "stage_aggregate_backward",
    "e2e_forward",
    "e2e_backward",
    "e2e_forward_backward",
)
VARIABLE_TOP_LEVEL = {"date_utc", "requested_matrix", "cases", "incomplete_case"}


class MergeError(ValueError):
    """An input cannot support a homogeneous complete matrix."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MergeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: JSON root must be an object")
    return value


def case_key(case: dict) -> tuple[int, str, str, str]:
    try:
        return (
            case["nodes"], case["batch_shape"], case["occupancy"],
            case["device"],
        )
    except (KeyError, TypeError) as error:
        raise MergeError(f"case lacks its matrix key: {error}") from error


def validate_case(case: dict, repeats: int) -> None:
    require(isinstance(case, dict), "case must be an object")
    key = case_key(case)
    require(key in EXPECTED, f"unexpected case key {key}")
    nodes, _shape, occupancy, device = key
    lengths = case.get("batch_lengths")
    require(
        isinstance(lengths, list) and len(lengths) == 4
        and all(isinstance(value, int) and value >= 0 for value in lengths)
        and sum(lengths) == nodes,
        f"{key}: invalid batch lengths",
    )
    require(case.get("batch_ids") == [0, 3, 11, 29], f"{key}: batch IDs changed")
    points_per_cell = 16 if occupancy == "dense" else 1
    require(case.get("points_per_cell") == points_per_cell, f"{key}: occupancy changed")
    voxels = sum((length + points_per_cell - 1) // points_per_cell for length in lengths)
    require(case.get("voxels") == voxels, f"{key}: voxel count changed")
    digest = case.get("input_sha256")
    require(
        isinstance(digest, str) and len(digest) == 64
        and all(char in "0123456789abcdef" for char in digest),
        f"{key}: invalid input SHA-256",
    )
    timings = case.get("timings")
    require(isinstance(timings, dict) and set(timings) == set(TIMINGS), f"{key}: timings missing")
    for name in TIMINGS:
        entry = timings[name]
        samples = entry.get("samples_ms") if isinstance(entry, dict) else None
        require(
            isinstance(samples, list) and len(samples) == repeats
            and all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in samples),
            f"{key}/{name}: invalid samples",
        )
        for field, expected in (
            ("median_ms", statistics.median(samples)),
            ("min_ms", min(samples)),
            ("max_ms", max(samples)),
        ):
            require(
                isinstance(entry.get(field), (int, float))
                and math.isclose(entry[field], expected, rel_tol=1e-12, abs_tol=1e-12),
                f"{key}/{name}: {field} does not match samples",
            )
    memory = case.get("memory")
    require(isinstance(memory, dict), f"{key}: missing memory checkpoints")
    for checkpoint in ("baseline", "after_inputs", "after_timing", "after_cleanup"):
        require(
            isinstance(memory.get(checkpoint), dict)
            and "rss_current_bytes" in memory[checkpoint]
            and "rss_process_peak_bytes" in memory[checkpoint],
            f"{key}: missing {checkpoint} RSS checkpoint",
        )
    require(
        isinstance(memory.get("observed_mps_boundary_max_bytes"), dict),
        f"{key}: missing MPS boundary maximum record",
    )
    if device == "mps":
        for checkpoint in ("baseline", "after_inputs", "after_timing", "after_cleanup"):
            require(
                "mps_current_allocated_bytes" in memory[checkpoint]
                and "mps_driver_allocated_bytes" in memory[checkpoint],
                f"{key}: missing {checkpoint} MPS allocation checkpoint",
            )


def one_case_matrix(key: tuple[int, str, str, str]) -> dict:
    nodes, shape, occupancy, device = key
    return {
        "nodes": [nodes], "batch_shapes": [shape],
        "occupancies": [occupancy], "devices": [device],
    }


def merge(base_path: Path, fragment_paths: list[Path], output_path: Path, expected_base_cases: int | None) -> dict:
    input_paths = [base_path, *fragment_paths]
    require(len(set(path.resolve() for path in input_paths)) == len(input_paths), "input paths must be distinct")
    require(output_path.resolve() not in {path.resolve() for path in input_paths}, "output would overwrite an input")
    require(not output_path.exists(), f"output already exists: {output_path}")
    base = read_json(base_path)
    require(base.get("requested_matrix") == MATRIX, "base requested_matrix is not the canonical 24-case matrix")
    for field in (
        "source_commit", "benchmark_script_sha256", "voxel_source_sha256",
        "python_executable", "python_version", "torch_version",
        "installed_distributions_sha256", "macos_version", "machine", "cpu_brand",
    ):
        require(isinstance(base.get(field), str) and base[field], f"base {field} missing")
    for field, length in (
        ("source_commit", 40), ("benchmark_script_sha256", 64),
        ("voxel_source_sha256", 64), ("installed_distributions_sha256", 64),
    ):
        value = base[field]
        require(
            len(value) == length and all(char in "0123456789abcdef" for char in value),
            f"base {field} has invalid digest",
        )
    require(base.get("mps_available") is True, "base did not have MPS available")
    require(base.get("mps_fallback") == "0", "base permits MPS fallback")
    require(base.get("mps_fast_math") in ("0", "1"), "base math mode is missing")
    require(base.get("warmups") == 2 and base.get("repeats") == 5, "base does not use 2 warmups and 5 repeats")
    require(
        base.get("seed") == 2701 and base.get("position_dimensions") == 3
        and base.get("feature_channels") == 32 and base.get("cell_size") == 0.5,
        "base voxel fixture metadata changed",
    )
    require("resume_provenance" not in base, "base is already merged; use its original partial")
    cases = base.get("cases")
    require(isinstance(cases, list) and len(cases) < 24, "base must be an incomplete case list")
    if expected_base_cases is not None:
        require(len(cases) == expected_base_cases, f"base has {len(cases)} cases, expected {expected_base_cases}")
    keys = [case_key(case) for case in cases]
    require(len(keys) == len(set(keys)), "base contains duplicate case keys")
    require(keys == EXPECTED[:len(keys)], "base cases are not a canonical prefix")
    for case in cases:
        validate_case(case, base["repeats"])
    if "incomplete_case" in base:
        failed = base["incomplete_case"]
        require(isinstance(failed, dict), "incomplete_case must be an object")
        require(case_key(failed) == EXPECTED[len(keys)], "incomplete_case is not the next case")

    metadata = {key: value for key, value in base.items() if key not in VARIABLE_TOP_LEVEL}
    source_records = [{"role": "base", "file": base_path.name, "sha256": sha256(base_path), "case_count": len(cases)}]
    for fragment_path in fragment_paths:
        require(len(cases) < len(EXPECTED), f"extra fragment after all {len(EXPECTED)} cases")
        fragment = read_json(fragment_path)
        fragment_metadata = {
            key: value for key, value in fragment.items() if key not in VARIABLE_TOP_LEVEL
        }
        require(fragment_metadata == metadata, f"{fragment_path}: metadata differs from base")
        fragment_cases = fragment.get("cases")
        require(isinstance(fragment_cases, list) and len(fragment_cases) == 1, f"{fragment_path}: expected one complete case")
        require("incomplete_case" not in fragment, f"{fragment_path}: worker failure marker present")
        case = fragment_cases[0]
        key = case_key(case)
        require(fragment.get("requested_matrix") == one_case_matrix(key), f"{fragment_path}: requested matrix differs from case")
        require(key == EXPECTED[len(cases)], f"{fragment_path}: expected {EXPECTED[len(cases)]}, found {key}")
        validate_case(case, base["repeats"])
        cases.append(case)
        source_records.append({"role": "fragment", "file": fragment_path.name, "sha256": sha256(fragment_path), "case_key": list(key)})
    require(len(cases) == 24, f"merge has {len(cases)} cases, expected 24")
    require([case_key(case) for case in cases] == EXPECTED, "merged case order is not canonical")

    for nodes, shape, occupancy in itertools.product(NODES, SHAPES, OCCUPANCIES):
        cpu = next(case for case in cases if case_key(case) == (nodes, shape, occupancy, "cpu"))
        mps = next(case for case in cases if case_key(case) == (nodes, shape, occupancy, "mps"))
        require(cpu["input_sha256"] == mps["input_sha256"], f"{nodes}/{shape}/{occupancy}: CPU/MPS inputs differ")

    result = dict(base)
    result["cases"] = cases
    failed = result.pop("incomplete_case", None)
    provenance = {
        "merged_utc": datetime.now(timezone.utc).isoformat(),
        "merge_tool_sha256": sha256(Path(__file__)),
        "source_files": source_records,
        "source_commit": base.get("source_commit"),
        "benchmark_script_sha256": base.get("benchmark_script_sha256"),
        "voxel_source_sha256": base.get("voxel_source_sha256"),
    }
    if failed is not None:
        provenance["prior_incomplete_case"] = {
            "case_key": list(case_key(failed)),
            "returncode": failed.get("returncode"),
            "raw_marker_sha256": hashlib.sha256(json.dumps(failed, sort_keys=True).encode()).hexdigest(),
        }
    executable = result.get("python_executable")
    require(isinstance(executable, str) and executable, "python_executable missing")
    basename = Path(executable).name
    require(basename not in ("", ".", ".."), "python_executable basename invalid")
    if executable != basename:
        result["python_executable"] = basename
        result["python_executable_path_policy"] = "redacted_to_basename_by_merge"
        provenance["metadata_redaction"] = {
            "field": "python_executable", "operation": "basename_only",
            "original_value_sha256": hashlib.sha256(executable.encode()).hexdigest(),
        }
    else:
        provenance["metadata_redaction"] = {"field": "python_executable", "operation": "none_already_basename"}
    result["resume_provenance"] = provenance

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", dir=output_path.parent, prefix=".voxel-merge-", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        file.write(payload)
    try:
        require(not output_path.exists(), f"output appeared during merge: {output_path}")
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="untouched partial 24-case JSON")
    parser.add_argument("--fragment", type=Path, action="append", required=True, help="next one-case JSON; repeat in matrix order")
    parser.add_argument("--output", type=Path, required=True, help="new complete publish JSON")
    parser.add_argument("--expect-base-cases", type=int, help="e.g. 21 for the M1 Safe partial")
    args = parser.parse_args()
    try:
        result = merge(args.base, args.fragment, args.output, args.expect_base_cases)
    except (MergeError, OSError, json.JSONDecodeError) as error:
        parser.exit(2, f"merge rejected: {error}\n")
    print(f"wrote {args.output} ({len(result['cases'])} validated cases, mode {result['mps_fast_math']})")


if __name__ == "__main__":
    main()

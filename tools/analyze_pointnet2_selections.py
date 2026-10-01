#!/usr/bin/env python3
"""Trace model-level local-index differences back to original point IDs.

The input NPZ files are produced by pointnet2_segmentation_parity.py. This
inspector is pure NumPy; it does not rerun either GPU backend or import any
upstream source. A numerical tie here is a mathematical float64 observation,
not a measurement of either backend's internal distance register.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection(run: np.lib.npyio.NpzFile, index: int, kind: str) -> np.ndarray:
    value = run[f"selection::{index:02d}::{kind}"]
    if value.shape[0] != 1:
        raise ValueError("this evidence fixture must have one cloud")
    return value[0]


def levels(run: np.lib.npyio.NpzFile, n: int) -> list[np.ndarray]:
    maps = [np.arange(n)]
    for i in (0, 2, 4, 6):
        maps.append(maps[-1][selection(run, i, "furthest_point_sample")])
    for mapping in maps:
        if len(set(mapping.tolist())) != len(mapping):
            raise ValueError("duplicate sampled point IDs prevent row mapping")
    return maps


def rows_by_point(
    run: np.lib.npyio.NpzFile,
    maps: list[np.ndarray],
    selection_index: int,
    kind: str,
    query_level: int,
    source_level: int,
) -> dict[int, tuple[int, ...]]:
    query_ids = maps[query_level]
    source_ids = maps[source_level]
    local = selection(run, selection_index, kind)
    return {
        int(query): tuple(int(x) for x in source_ids[row])
        for query, row in zip(query_ids, local, strict=True)
    }


def compare_rows(a: dict[int, tuple[int, ...]], b: dict[int, tuple[int, ...]]) -> dict:
    common = sorted(set(a) & set(b))
    different = [point for point in common if a[point] != b[point]]
    result = {
        "common_query_points": len(common),
        "query_set_symmetric_difference": len(set(a) ^ set(b)),
        "different_neighbor_rows": len(different),
    }
    if different:
        q = different[0]
        result["first_different_row"] = {
            "query_original_id": q,
            "mps_neighbor_original_ids": list(a[q]),
            "cuda_neighbor_original_ids": list(b[q]),
        }
    return result


def inspect(fixture: Path, mps: Path, cuda: Path) -> dict:
    with np.load(fixture, allow_pickle=False) as f, np.load(mps, allow_pickle=False) as a, np.load(cuda, allow_pickle=False) as b:
        xyz = f["points"][0, :, :3]
        amaps, bmaps = levels(a, len(xyz)), levels(b, len(xyz))
        fps_stages = []
        for level in range(1, 5):
            x, y = amaps[level], bmaps[level]
            diff = np.flatnonzero(x != y)
            fps_stages.append({
                "level": level,
                "ordered_index_mismatches": len(diff),
                "selected_set_symmetric_difference": len(set(x) ^ set(y)),
                "first_mismatch_position": int(diff[0]) if len(diff) else None,
            })

        first = fps_stages[0]["first_mismatch_position"]
        tie = None
        if first is not None:
            old_a, old_b = amaps[1], bmaps[1]
            common_prefix = bool(np.array_equal(old_a[:first], old_b[:first]))
            if common_prefix:
                squared = np.square(xyz.astype("f8")[:, None, :] - xyz[old_a[:first]].astype("f8")[None, :, :]).sum(axis=2).min(axis=1)
                ia, ib = int(old_a[first]), int(old_b[first])
                tie = {
                    "step": first,
                    "prefix_identical": True,
                    "mps_original_id": ia,
                    "cuda_original_id": ib,
                    "mps_candidate_min_squared_distance_float64": float(squared[ia]),
                    "cuda_candidate_min_squared_distance_float64": float(squared[ib]),
                    "global_max_min_squared_distance_float64": float(squared.max()),
                    "maximizing_candidate_original_ids_float64": [int(i) for i in np.flatnonzero(squared == squared.max())],
                    "gpu_internal_distance_bits_observed": False,
                }

        ball_stages = []
        for level, index in enumerate((1, 3, 5, 7), start=1):
            ar = rows_by_point(a, amaps, index, "ball_query", level, level - 1)
            br = rows_by_point(b, bmaps, index, "ball_query", level, level - 1)
            ball_stages.append({"level": level, **compare_rows(ar, br)})

        three_nn_stages = []
        for index, query_level, source_level in ((8, 3, 4), (9, 2, 3), (10, 1, 2), (11, 0, 1)):
            ar = rows_by_point(a, amaps, index, "three_nn", query_level, source_level)
            br = rows_by_point(b, bmaps, index, "three_nn", query_level, source_level)
            three_nn_stages.append({"index": index, **compare_rows(ar, br)})

    return {
        "fixture_sha256": digest(fixture),
        "mps_result_sha256": digest(mps),
        "cuda_result_sha256": digest(cuda),
        "meaning": "Original-ID normalization removes local-index changes caused by FPS ordering; it does not prove bitwise model parity.",
        "fps": fps_stages,
        "first_fps_mismatch": tie,
        "ball_query": ball_stages,
        "three_nn": three_nn_stages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--mps", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = inspect(args.fixture, args.mps, args.cuda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "fps_first": report["first_fps_mismatch"], "ball_different_rows": [x["different_neighbor_rows"] for x in report["ball_query"]], "three_nn_different_rows": [x["different_neighbor_rows"] for x in report["three_nn"]]}, indent=2))


if __name__ == "__main__":
    main()

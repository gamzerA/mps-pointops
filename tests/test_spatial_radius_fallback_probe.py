"""CPU checks for the archived radius fixture matrix and diagnostic contract."""

from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch

from bench.probe_spatial_radius_fallback import (
    _counts_and_parity,
    _expected_signatures,
    _sha256,
    _validate_legacy_fixture,
    cases,
    ROOT,
)


@pytest.mark.parametrize("matrix,count", [("m5", 9), ("m1", 12)])
def test_archived_radius_matrix_is_exact_and_staged(matrix: str, count: int) -> None:
    selected = cases(matrix)
    assert len(selected) == count
    assert len({case.name for case in selected}) == count
    assert {(case.fixture["distribution"], case.fixture["points"],
             case.fixture["queries"],
             float(case.fixture["radius_python_float_repr"]))
            for case in selected} == _expected_signatures(matrix)
    assert [(case.fixture["points"], case.fixture["queries"]) for case in selected] == sorted(
        (case.fixture["points"], case.fixture["queries"]) for case in selected
    )
    assert all(case.legacy_sha256 == _sha256(case.path.read_bytes())
               for case in selected)


def test_archived_fixture_drift_is_rejected_before_gpu_work() -> None:
    original = cases("m1")[0]
    record = json.loads(original.path.read_text())
    generator_hash = _sha256((ROOT / "bench/bench_spatial_bvh.py").read_bytes())
    drifted = deepcopy(record)
    drifted["fixture"]["cell_size"] = 1.0
    with pytest.raises(ValueError, match="constants"):
        _validate_legacy_fixture("m1", drifted, generator_hash)
    drifted = deepcopy(record)
    drifted["source_sha256"]["bench/bench_spatial_bvh.py"] = "0" * 64
    with pytest.raises(ValueError, match="generator"):
        _validate_legacy_fixture("m1", drifted, generator_hash)


def test_per_query_overflow_and_bit_parity_are_independent() -> None:
    indices = torch.tensor([[0, 3], [-1, -1], [2, -1]], dtype=torch.int64)
    distances = torch.tensor([[1.0, 4.0], [0.0, 0.0], [9.0, 0.0]])
    stats = torch.zeros((3, 5), dtype=torch.int32)
    stats[1, 4] = 1
    legacy_rows = {"zero_neighbor_rows": 1, "partially_filled_rows": 1,
                   "first_k_full_rows": 1, "total_valid_slots": 3}
    result = _counts_and_parity(distances, indices, stats, distances.clone(),
                                indices.clone(), legacy_rows)
    assert result["bvh_stack_fallback"] == {
        "rows": 1, "row_indices": [1], "query_count": 3, "fraction": 1 / 3,
    }
    assert result["parity"] == {
        "index_mismatches": 0, "squared_distance_bit_mismatches": 0,
        "original_index_order_violations": 0, "row_counts_match_legacy": True,
    }

    changed = distances.clone()
    changed[0, 0] = torch.nextafter(changed[0, 0], torch.tensor(2.0))
    assert _counts_and_parity(changed, indices, stats, distances, indices,
                              legacy_rows)["parity"]["squared_distance_bit_mismatches"] == 1
    invalid_stats = stats.clone()
    invalid_stats[1, 4] = 2
    with pytest.raises(ValueError, match="0/1"):
        _counts_and_parity(distances, indices, invalid_stats, distances,
                           indices, legacy_rows)


def test_m5_evidence_manifest_and_reported_fallbacks_are_integral() -> None:
    directory = ROOT / "bench/results/spatial-radius-overflow-m5pro-20261003"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "pass"
    assert manifest["expected_cases"] == len(manifest["completed_cases"]) == 9
    total_queries = 0
    for entry in manifest["completed_cases"]:
        path = directory / entry["file"]
        assert _sha256(path.read_bytes()) == entry["sha256"]
        report = json.loads(path.read_text())
        assert report["status"] == entry["status"] == "pass"
        assert report["source_commit"] == "4bfcb9192c468f0928842a7332fa84bde465e29c"
        assert report["hardware"] == "Apple M5 Pro"
        assert report["mps_fast_math"] == report["mps_fallback"] == "0"
        q = report["fixture"]["queries"]
        assert report["bvh_stack_fallback"] == {
            "rows": 0, "row_indices": [], "query_count": q, "fraction": 0.0,
        }
        assert report["parity"] == {
            "index_mismatches": 0,
            "squared_distance_bit_mismatches": 0,
            "original_index_order_violations": 0,
            "row_counts_match_legacy": True,
        }
        total_queries += q
    assert total_queries == 233_472

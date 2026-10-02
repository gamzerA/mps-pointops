"""Check the archived M1 spatial records against their source commit.

This uses only the standard library. It checks archival bytes, the clean
source revision recorded by each benchmark, and the tested parity counters.
It does not replay GPU measurements or establish a physical GPU-memory peak.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench/results/spatial-v090-m1"
EVIDENCE = ROOT / "docs/evidence/spatial-v090-m1"
CHAMFER_EVIDENCE = ROOT / "docs/evidence/chamfer-v090-m1"
SOURCE_COMMIT = "ef07a9337b40065c18a6c950bda4240c789f31f1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    manifest = EVIDENCE / "SHA256SUMS.txt"
    listed: set[Path] = set()
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        assert path.is_relative_to(ROOT) and path.is_file(), relative
        assert _sha256(path.read_bytes()) == digest, relative
        listed.add(path)

    records = sorted(RESULTS.glob("*.json"))
    logs = sorted(EVIDENCE.glob("*.log")) + sorted(CHAMFER_EVIDENCE.glob("*.log"))
    assert len(records) == 34, len(records)
    assert len(logs) == 4, len(logs)
    assert listed == set(records) | set(logs)
    assert "31 passed, 1 skipped" in (EVIDENCE / "pytest-spatial-safe-torch212-ef07a933.log").read_text()
    assert "5 passed, 27 skipped" in (EVIDENCE / "pytest-spatial-fast-torch212-ef07a933.log").read_text()
    for mode in ("safe", "fast"):
        log = (CHAMFER_EVIDENCE / f"pytest-chamfer-{mode}-torch212-ef07a933.log").read_text()
        assert "76 passed, 2 skipped" in log
        assert "No module named 'pytorch3d'" in log

    source_blobs: dict[str, bytes] = {}
    kinds: dict[str, int] = {}
    for path in records:
        record = json.loads(path.read_text())
        assert record["source_commit"] == SOURCE_COMMIT, path
        assert record["source_dirty"] is False, path
        assert record.get("source_status", []) == [], path
        assert record["hardware"] == "Apple M1", path
        assert record["mps_fast_math"] == "0", path
        assert record["mps_fallback"] == "0", path
        assert record["torch"] == "2.14.1", path
        for relative, expected in record["source_sha256"].items():
            if relative not in source_blobs:
                source_blobs[relative] = subprocess.check_output(
                    ["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=ROOT
                )
            assert _sha256(source_blobs[relative]) == expected, (path, relative)

        kind = path.name.split("-", 1)[0]
        kinds[kind] = kinds.get(kind, 0) + 1
        if kind == "knn":
            assert all(value == 0 for value in record["index_mismatch"].values()), path
            assert all(value == 0 for value in record["euclidean_distance_max_abs_error"].values()), path
            assert record["adaptive_decision"]["selected_backend"] == "scan", path
        elif kind == "radius":
            assert all(value == 0 for value in record["mismatch"].values()), path
            assert record["original_index_order_violations"] == 0, path
            assert record["auto_selected_backend"] == "scan", path
        elif kind == "bvh":
            assert record["bvh_index_mismatch_vs_native"] == 0, path
            assert record["brick_index_mismatch_vs_native"] == 0, path
            assert record["bvh_brick_squared_distance_bit_mismatch"] == 0, path
            assert record["bvh_total_fallbacks"] == 0, path
        elif kind == "memory":
            assert record["allocator_high_water_is_total_gpu_peak"] is False, path
            assert record["sampled_driver_high_water_is_true_driver_peak"] is False, path
        else:
            raise AssertionError(path)

    assert kinds == {"knn": 13, "radius": 12, "bvh": 5, "memory": 4}, kinds
    print(
        f"Verified {len(records)} M1 JSON records, {len(logs)} pytest logs, "
        f"and {len(source_blobs)} source blobs at {SOURCE_COMMIT[:8]}."
    )


if __name__ == "__main__":
    main()

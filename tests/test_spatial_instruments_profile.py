"""CPU-only integrity checks for the source-pinned Instruments case matrix."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.profile_spatial_instruments import (
    ARCHIVED_CASES,
    ARCHIVE_DIR,
    load_cases,
    validate_intervals,
)


def test_archived_six_cases_are_clean_and_source_pinned() -> None:
    cases = load_cases()
    assert len(cases) == len(ARCHIVED_CASES) == 6
    assert {case.name for case in cases} == {row[0] for row in ARCHIVED_CASES}
    assert all(not case.derived_from_archive for case in cases)
    assert all(case.fixture["points"] == 1_000_000 and case.fixture["k"] == 16
               for case in cases)


def test_uniform_scan_is_explicit_derived_pair_with_identical_input() -> None:
    cases = load_cases(include_uniform_scan=True)
    serial, native = cases[-2:]
    assert serial.name == "uniform-q65536-bvh-serial"
    assert native.name == "uniform-q65536-native-full-scan"
    assert native.derived_from_archive
    assert native.archive_sha256 == serial.archive_sha256
    assert native.archived_memory is None
    assert native.fixture == {**serial.fixture, "path": "native-full-scan"}


def test_uniform_pair_fixture_generator_produces_identical_inputs() -> None:
    pytest.importorskip("scipy")
    import argparse

    import numpy as np

    from bench.bench_spatial_bvh import _fixture

    serial = load_cases()[-1]
    # The actual generator ignores the search path. A bounded CPU fixture
    # checks that the paired native-scan option cannot change input points.
    fixture = {**serial.fixture, "points": 64, "queries": 16}
    a_points, a_query, _ = _fixture(argparse.Namespace(**fixture))
    fixture["path"] = "native-full-scan"
    b_points, b_query, _ = _fixture(argparse.Namespace(**fixture))
    np.testing.assert_array_equal(a_points, b_points)
    np.testing.assert_array_equal(a_query, b_query)


@pytest.mark.parametrize("mutation", ["source_commit", "fixture", "source_hash"])
def test_archive_tampering_fails_before_device_work(tmp_path: Path, mutation: str) -> None:
    for name, *_ in ARCHIVED_CASES:
        source = ARCHIVE_DIR / f"{name}.json"
        (tmp_path / source.name).write_bytes(source.read_bytes())
    target = tmp_path / f"{ARCHIVED_CASES[0][0]}.json"
    record = json.loads(target.read_text())
    if mutation == "source_commit":
        record["source_commit"] = "0" * 40
    elif mutation == "fixture":
        record["fixture"]["queries"] += 1
    else:
        record["source_sha256"]["bench/spatial_bvh.metal"] = "0" * 64
    target.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="provenance changed|fixed six-case matrix|current source differs"):
        load_cases(archive_dir=tmp_path)


def test_host_intervals_require_order_and_derived_synchronized_duration() -> None:
    good = [
        {"host_perf_start_ns": 100, "host_perf_end_ns": 200,
         "host_unix_start_ns": 1_000, "host_unix_end_ns": 1_100,
         "host_synchronized_ns": 100},
        {"host_perf_start_ns": 220, "host_perf_end_ns": 280,
         "host_unix_start_ns": 1_120, "host_unix_end_ns": 1_180,
         "host_synchronized_ns": 60},
    ]
    validate_intervals(good)
    with pytest.raises(ValueError, match="overlap"):
        validate_intervals([good[0], {**good[1], "host_perf_start_ns": 190,
                                       "host_synchronized_ns": 90}])
    with pytest.raises(ValueError, match="not derived"):
        validate_intervals([good[0], {**good[1], "host_synchronized_ns": 61}])

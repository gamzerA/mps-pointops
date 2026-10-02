"""CPU-only integrity checks for the source-pinned Instruments case matrix."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from bench import profile_spatial_instruments as profile
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
    with pytest.raises(ValueError, match="provenance changed|fixed six-case matrix|historical source hash map changed"):
        load_cases(archive_dir=tmp_path)


@pytest.fixture
def replay_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Isolate real file mutations from the repository and archived evidence."""
    for name in (*profile.PINNED_SOURCE_FILES, "bench/profile_spatial_instruments.py"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(profile.ROOT / name, target)
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    for name, *_ in ARCHIVED_CASES:
        shutil.copyfile(ARCHIVE_DIR / f"{name}.json", archive_dir / f"{name}.json")
    monkeypatch.setattr(profile, "ROOT", tmp_path)
    return archive_dir


def test_measurement_instrumentation_can_change_without_changing_replay(
    replay_checkout: Path,
) -> None:
    name = "bench/measure_spatial_memory.py"
    path = profile.ROOT / name
    previous = profile._current_source_hashes()[name]
    path.write_text(path.read_text() + "\n# Additional measurement-only instrumentation.\n")
    assert len(load_cases(archive_dir=replay_checkout)) == 6
    current = profile._current_source_hashes()
    assert current[name] != previous
    assert set(current) == {*profile.PINNED_SOURCE_FILES, "bench/profile_spatial_instruments.py"}


@pytest.mark.parametrize("name", profile.REPLAY_SOURCE_FILES)
def test_fixture_and_search_source_drift_still_prevents_replay(
    replay_checkout: Path, name: str,
) -> None:
    path = profile.ROOT / name
    path.write_bytes(path.read_bytes() + b"\n// source drift\n")
    with pytest.raises(ValueError, match="current source differs"):
        load_cases(archive_dir=replay_checkout)


@pytest.mark.parametrize("all_cases", [False, True])
def test_historical_measurement_hash_cannot_be_rewritten(
    replay_checkout: Path, all_cases: bool,
) -> None:
    # Even rewriting every record to agree cannot replace the historical map
    # with the new driver's hash or an arbitrary digest.
    paths = sorted(replay_checkout.glob("*.json"))
    for path in paths if all_cases else paths[:1]:
        record = json.loads(path.read_text())
        record["source_sha256"]["bench/measure_spatial_memory.py"] = "0" * 64
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="historical source hash map changed|disagree about source hashes"):
        load_cases(archive_dir=replay_checkout)


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

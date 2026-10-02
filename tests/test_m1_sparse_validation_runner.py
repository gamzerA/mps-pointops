"""Offline integrity checks for the physical-M1 evidence runner."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import tools.run_m1_sparse_validation as runner
from tools.run_m1_sparse_validation import (
    GateError, parse_pytest_summary, validate_hardware, validate_torch_version,
    verify_checkout,
)


def test_pytest_summary_requires_passes_and_rejects_skips():
    assert parse_pytest_summary("42 passed in 2.10s\n", 0) == 42
    with pytest.raises(GateError, match="non-passing"):
        parse_pytest_summary("42 passed, 1 skipped in 2.10s\n", 0)
    with pytest.raises(GateError, match="at least one"):
        parse_pytest_summary("1 skipped in 0.02s\n", 0)
    with pytest.raises(GateError, match="status 1"):
        parse_pytest_summary("41 passed, 1 failed in 2.10s\n", 1)


def test_m1_family_guard_does_not_accept_other_silicon():
    validate_hardware("darwin", "arm64", "Apple M1")
    validate_hardware("darwin", "arm64", "Apple M1 Pro")
    for system, machine, brand in (
        ("darwin", "arm64", "Apple M5 Pro"),
        ("darwin", "arm64", "Apple M10"),
        ("darwin", "x86_64", "Intel Core i7"),
        ("linux", "aarch64", "Apple M1"),
    ):
        with pytest.raises(GateError, match="M1-family"):
            validate_hardware(system, machine, brand)


def test_torch_guard_follows_declared_minimum_and_records_actual_version():
    validate_torch_version("2.7.0")
    validate_torch_version("2.14.1+cpu")
    with pytest.raises(GateError, match="2.7 or later"):
        validate_torch_version("2.6.1")
    with pytest.raises(GateError, match="2.7 or later"):
        validate_torch_version("invalid")


def test_source_gate_requires_exact_clean_full_commit(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked.txt").write_text("fixture\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "commit", "-qm", "fixture"], cwd=tmp_path, check=True,
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path,
                                   text=True).strip()
    verify_checkout(tmp_path, head)
    with pytest.raises(GateError, match="40-character"):
        verify_checkout(tmp_path, head[:8])
    with pytest.raises(GateError, match="mismatch"):
        verify_checkout(tmp_path, "0" * 40 if head != "0" * 40 else "1" * 40)
    (tmp_path / "untracked.txt").write_text("new\n")
    with pytest.raises(GateError, match="not clean"):
        verify_checkout(tmp_path, head)


def test_failed_pytest_keeps_log_and_failed_manifest(tmp_path, monkeypatch):
    output = tmp_path / "evidence"
    monkeypatch.setattr(runner, "verify_checkout", lambda *_: None)
    monkeypatch.setattr(runner, "_source_hashes", lambda *_: {"fixture.py": "abc"})
    monkeypatch.setattr(runner, "SUITES", ("tests/test_sparse_rulebook.py",))
    monkeypatch.setattr(runner, "_environment", lambda *_args, **_kw: {
        "torch": "2.14.1", "mps_device": "Apple M1", "mps_available": True,
    })

    def fake_platform_command(command, **_kwargs):
        if command[:2] == ["sysctl", "-n"]:
            return SimpleNamespace(stdout="Apple M1\n")
        if command[:2] == ["sw_vers", "-productVersion"]:
            return SimpleNamespace(stdout="26.5.2\n")
        raise AssertionError(command)

    def fake_pytest(command, *, destination, **_kwargs):
        assert command[1:3] == ["-m", "pytest"]
        destination.write_text("1 failed in 0.01s\n")
        return SimpleNamespace(stdout="1 failed in 0.01s\n", returncode=1)

    monkeypatch.setattr(runner.subprocess, "run", fake_platform_command)
    monkeypatch.setattr(runner, "_run_logged", fake_pytest)
    assert runner.main(["--commit", "0" * 40, "--output", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert "status 1" in manifest["failure"]
    assert (output / "pytest-safe.log").read_text() == "1 failed in 0.01s\n"
    assert manifest["artifacts_sha256"]["pytest-safe.log"] == runner._sha256(
        output / "pytest-safe.log"
    )

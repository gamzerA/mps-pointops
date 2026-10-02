"""Verify archived physical-M1 PyTorch3D Chamfer parity evidence.

This checks file integrity, measured source hashes, and every recorded case
without importing PyTorch or requiring MPS hardware. It does not rerun the
numerical experiment.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "docs/evidence/chamfer-v090-m1-upstream"
MEASURED = "ef07a9337b40065c18a6c950bda4240c789f31f1"
UPSTREAM = "88e182f989c80836f4bd744e0d9cb1852762ce01"
CASES = {"chamfer-safe.json": 480, "chamfer-fast.json": 480,
         "chamfer-extended-safe.json": 252, "chamfer-extended-fast.json": 252}
CHECKS = {480: 3240, 252: 3720}


def source_sha(path: str) -> str:
    content = subprocess.check_output(["git", "show", f"{MEASURED}:{path}"], cwd=ROOT)
    return hashlib.sha256(content).hexdigest()


def main() -> None:
    manifest = ARCHIVE / "SHA256SUMS"
    lines = manifest.read_text().splitlines()
    if len(lines) != 10:
        raise AssertionError(f"expected ten archived files, got {len(lines)}")
    names: set[str] = set()
    for line in lines:
        expected, name = line.split("  ", 1)
        if name in names or "/" in name or name in (".", ".."):
            raise AssertionError(f"unsafe or duplicate manifest path: {name!r}")
        names.add(name)
        actual = hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"SHA-256 mismatch: {name}")

    source_hashes = {
        "chamfer": source_sha("mps_pointops/chamfer.py"),
        "kernel": source_sha("mps_pointops/kernels/chamfer_nn.metal"),
        "base_verifier": source_sha("tests/verify_upstream_chamfer.py"),
        "extended_verifier": source_sha("tests/verify_upstream_chamfer_extended.py"),
    }
    for name, expected_cases in CASES.items():
        record = json.loads((ARCHIVE / name).read_text())
        summary = record["summary"]["mps"]
        if record["upstream"]["commit"] != UPSTREAM or record["port"]["commit"] != MEASURED:
            raise AssertionError(f"source commit mismatch: {name}")
        env = record["environment"]
        if env["mps_fallback"] != "0" or env["torch"] != "2.14.1" or "macOS-26.5.2-arm64" not in env["platform"]:
            raise AssertionError(f"environment mismatch: {name}")
        fast = name.endswith("-fast.json")
        if env["mps_fast_math"] != str(int(fast)):
            raise AssertionError(f"math mode mismatch: {name}")
        if summary["cases"] != expected_cases or len(record["cases"]) != expected_cases:
            raise AssertionError(f"case count mismatch: {name}")
        count_key = "checked_tensors" if expected_cases == 480 else "checks"
        failure_key = "failed_tensors" if expected_cases == 480 else "failed_checks"
        if summary[count_key] != CHECKS[expected_cases] or summary[failure_key] or summary["failed_elements"]:
            raise AssertionError(f"summary mismatch: {name}")
        check_count = 0
        for case in record["cases"]:
            checks = case["checks"]["mps"]
            for check in checks.values():
                check_count += 1
                if not check["pass"] or check["mismatches"]:
                    raise AssertionError(f"failed case in {name}")
        if check_count != CHECKS[expected_cases]:
            raise AssertionError(f"check count mismatch: {name}")
        port = record["port"]
        if expected_cases == 480:
            observed = (port["chamfer_py_sha256"], port["metal_kernel_sha256"], port["verifier_sha256"])
            expected = (source_hashes["chamfer"], source_hashes["kernel"], source_hashes["base_verifier"])
        else:
            if port["dirty"]:
                raise AssertionError(f"dirty measured source: {name}")
            observed = (port["chamfer_sha256"], port["kernel_sha256"], port["verifier_sha256"])
            expected = (source_hashes["chamfer"], source_hashes["kernel"], source_hashes["extended_verifier"])
        if observed != expected:
            raise AssertionError(f"source file hash mismatch: {name}")
    print("M1 Chamfer upstream archive valid: 10 blobs, 4 JSON matrices, 1,464 cases, 13,920 checks")


if __name__ == "__main__":
    main()

"""Untimed, source-pinned BVH radius fallback audit for archived v0.9 fixtures.

Run ``--matrix m1`` or ``--matrix m5`` from a clean checkout on the named
physical chip. Each fixture runs in a fresh child process, constructs one BVH,
issues one private radius query and one native Metal scan, then archives the
per-query overflow count and exact output parity. No latency is measured.
Older performance JSONs are read as fixture specifications, never rewritten.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LEGACY = {
    "m5": (ROOT / "bench/results/spatial-radius-dispatch-v090-m5pro-clean",
           "*.json", "62fcc0f295e127be47a475fbc122d4de686b4c58"),
    "m1": (ROOT / "bench/results/spatial-v090-m1",
           "radius-*.json", "ef07a9337b40065c18a6c950bda4240c789f31f1"),
}
EXPECTED_HARDWARE = {"m5": "Apple M5 Pro", "m1": "Apple M1"}
SOURCE_PATHS = (
    "bench/probe_spatial_radius_fallback.py",
    "bench/bench_spatial_bvh.py",
    "bench/bench_spatial_radius_dispatch.py",
    "mps_pointops/spatial.py",
    "mps_pointops/_spatial_bvh.py",
    "mps_pointops/_spatial_keys.py",
    "mps_pointops/_ball_query_mps.py",
    "mps_pointops/ops.py",
    "mps_pointops/kernels/spatial_bvh.metal",
    "mps_pointops/kernels/spatial_keys.metal",
    "mps_pointops/kernels/ball_query.metal",
)


@dataclass(frozen=True)
class Case:
    name: str
    path: Path
    legacy_sha256: str
    fixture: dict
    legacy_source_commit: str


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _expected_signatures(matrix: str) -> set[tuple[str, int, int, float]]:
    q_values = (4096, 8192, 65536)
    expected = {(distribution, 1_000_000, q, radius)
                for distribution, radius in (("uniform", 12.0),
                                             ("cluster-sparse", 0.03),
                                             ("collapsed", 0.5))
                for q in q_values}
    if matrix == "m1":
        expected.update({("collapsed", n, 256, 0.5)
                         for n in (20_000, 100_000, 1_000_000)})
    return expected


def _validate_legacy_fixture(matrix: str, record: dict, generator_hash: str) -> dict:
    source_commit = LEGACY[matrix][2]
    if record.get("source_commit") != source_commit or record.get("source_dirty") is not False:
        raise ValueError("legacy record has unexpected or dirty source commit")
    if record.get("source_sha256", {}).get("bench/bench_spatial_bvh.py") != generator_hash:
        raise ValueError("the fixture generator differs from the archived source")
    fixture = record["fixture"]
    signature = (fixture["distribution"], fixture["points"], fixture["queries"],
                 float(fixture["radius_python_float_repr"]))
    if signature not in _expected_signatures(matrix):
        raise ValueError(f"unexpected {matrix} fixture {signature}")
    if (fixture["limit"] != 16 or fixture["seed"] != 20261002
            or fixture["extent"] != 1024.0
            or fixture["cell_size"] != (1 / 64 if signature[0] == "cluster-sparse" else 16.0)):
        raise ValueError("legacy fixture constants differ from the archived matrix")
    radius32 = _float32(signature[3])
    if (fixture["radius_float32"] != radius32
            or fixture["radius_squared_float32"] != _float32(radius32 * radius32)):
        raise ValueError("legacy float32 radius arithmetic differs")
    expected_source = ("first half coincident, second half displaced +1 on x"
                       if signature[0] == "collapsed" else "independent")
    if fixture.get("query_source") != expected_source:
        raise ValueError("legacy query source differs from the expected construction")
    return fixture


def cases(matrix: str, root: Path = ROOT) -> list[Case]:
    if matrix not in LEGACY:
        raise ValueError(f"unknown matrix {matrix}")
    directory, pattern, _ = LEGACY[matrix]
    # The optional root makes CPU integrity tests independent of cwd.
    directory = root / directory.relative_to(ROOT)
    generator_hash = _sha256((root / "bench/bench_spatial_bvh.py").read_bytes())
    found: list[Case] = []
    signatures: set[tuple[str, int, int, float]] = set()
    for path in sorted(directory.glob(pattern)):
        raw = path.read_bytes()
        record = json.loads(raw)
        fixture = _validate_legacy_fixture(matrix, record, generator_hash)
        signature = (fixture["distribution"], fixture["points"], fixture["queries"],
                     float(fixture["radius_python_float_repr"]))
        if signature in signatures:
            raise ValueError(f"duplicate fixture {signature}")
        signatures.add(signature)
        found.append(Case(path.stem, path, _sha256(raw), fixture,
                          record["source_commit"]))
    if signatures != _expected_signatures(matrix):
        raise ValueError(f"{matrix} fixture matrix incomplete: missing={_expected_signatures(matrix) - signatures}, extra={signatures - _expected_signatures(matrix)}")
    return sorted(found, key=lambda c: (c.fixture["points"], c.fixture["queries"],
                                         c.fixture["distribution"]))


def _source_state() -> tuple[str, dict[str, str]]:
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).strip()
    if status:
        raise RuntimeError(f"a clean source checkout is required: {status}")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     cwd=ROOT, text=True).strip()
    return commit, {name: _sha256((ROOT / name).read_bytes()) for name in SOURCE_PATHS}


def _environment() -> None:
    if (os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0"
            or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0"):
        raise RuntimeError("set PYTORCH_ENABLE_MPS_FALLBACK=0 and PYTORCH_MPS_FAST_MATH=0 before Python starts")


def _counts_and_parity(diagnostic_dist, diagnostic_idx, stats,
                       scan_dist, scan_idx, legacy_rows: dict) -> dict:
    import torch

    q, k = scan_idx.shape
    if (tuple(diagnostic_idx.shape) != (q, k)
            or tuple(diagnostic_dist.shape) != (q, k)
            or tuple(stats.shape) != (q, 5)
            or stats.dtype != torch.int32):
        raise ValueError("private radius output or counter layout differs")
    # Copy only Q int32 flags (at most 256 KiB), preserving the exact rows
    # that took the full-scan recovery path without an MPS nonzero operation.
    flags = stats[:, 4].to("cpu").tolist()
    if any(flag not in (0, 1) for flag in flags):
        raise ValueError("overflow status must be a per-query 0/1 flag")
    fallback_rows = [row for row, flag in enumerate(flags) if flag == 1]
    valid = (scan_idx >= 0).sum(dim=1)
    row_counts = {
        "zero_neighbor_rows": int((valid == 0).sum().item()),
        "partially_filled_rows": int(((valid > 0) & (valid < k)).sum().item()),
        "first_k_full_rows": int((valid == k).sum().item()),
        "total_valid_slots": int(valid.sum().item()),
    }
    order_violations = 0 if k < 2 else int(
        ((scan_idx[:, 1:] >= 0) & (scan_idx[:, :-1] >= scan_idx[:, 1:])).sum().item()
    )
    return {
        "bvh_stack_fallback": {"rows": len(fallback_rows),
                               "row_indices": fallback_rows,
                               "query_count": q,
                               "fraction": len(fallback_rows) / q},
        "parity": {
            "index_mismatches": int(torch.count_nonzero(diagnostic_idx != scan_idx).item()),
            "squared_distance_bit_mismatches": int(torch.count_nonzero(
                diagnostic_dist.view(torch.int32) != scan_dist.view(torch.int32)).item()),
            "original_index_order_violations": order_violations,
            "row_counts_match_legacy": row_counts == legacy_rows,
        },
        "row_counts": row_counts,
    }


def _run_case(matrix: str, selected: Case, output: Path) -> int:
    _environment()
    if output.exists():
        raise FileExistsError(output)
    commit, source_hashes = _source_state()
    import numpy as np
    import torch
    from bench.bench_spatial_bvh import _fixture
    from mps_pointops.spatial import SpatialIndex

    if not torch.backends.mps.is_available():
        raise RuntimeError("requires a physical MPS device")
    hardware = subprocess.check_output(
        ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
    ).strip()
    if hardware != EXPECTED_HARDWARE[matrix]:
        raise RuntimeError(f"{matrix} matrix requires {EXPECTED_HARDWARE[matrix]}, found {hardware}")
    fixture = selected.fixture
    args = argparse.Namespace(distribution=fixture["distribution"],
                              points=fixture["points"], queries=fixture["queries"],
                              extent=fixture["extent"], seed=fixture["seed"],
                              query_source="independent")
    points_cpu, query_cpu, detail = _fixture(args)
    if fixture["distribution"] == "collapsed":
        query_cpu[fixture["queries"] // 2:, 0] = np.float32(fixture["extent"] / 2 + 1)
        detail = {"query_source": "first half coincident, second half displaced +1 on x",
                  "cluster_fraction": 1.0, "cluster_side": 0.0}
    query_cpu = np.ascontiguousarray(query_cpu)
    for key, value in detail.items():
        if fixture.get(key) != value:
            raise ValueError(f"generated fixture metadata differs at {key}")
    report = {
        "schema": "mps-pointops-spatial-radius-overflow-diagnostic-v1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "matrix": matrix, "case": selected.name,
        "legacy_record": selected.path.relative_to(ROOT).as_posix(),
        "legacy_record_sha256": selected.legacy_sha256,
        "legacy_source_commit": selected.legacy_source_commit,
        "source_commit": commit, "source_dirty": False,
        "source_sha256": source_hashes,
        "hardware": hardware,
        "physical_ram_bytes": int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()),
        "macos": platform.mac_ver()[0], "platform": platform.platform(),
        "python": sys.version.split()[0], "torch": torch.__version__,
        "numpy": np.__version__,
        "mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
        "mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
        "fixture": fixture,
        "input_sha256": {"points_float32_xyz": _sha256(points_cpu.tobytes()),
                         "query_float32_xyz": _sha256(query_cpu.tobytes())},
        "scope": "one untimed private BVH radius query and one untimed native Metal scan; no latency claim",
    }
    try:
        points = torch.from_numpy(points_cpu).to("mps")
        query = torch.from_numpy(query_cpu).to("mps")
        origin = torch.zeros(3, device="mps", dtype=torch.float32)
        radius = float(fixture["radius_python_float_repr"])
        index = SpatialIndex(points, backend="bvh", origin=origin,
                             cell_size=fixture["cell_size"])
        diagnostic_dist, diagnostic_idx, stats = index._get_bvh().radius(
            query, radius, fixture["limit"],
        )
        torch.mps.synchronize()
        scan_dist, scan_idx = SpatialIndex(points, backend="scan").ball_query(
            query, radius, fixture["limit"],
        )
        torch.mps.synchronize()
        legacy_rows = json.loads(selected.path.read_text())["row_counts"]
        report.update(_counts_and_parity(diagnostic_dist, diagnostic_idx, stats,
                                         scan_dist, scan_idx, legacy_rows))
        parity = report["parity"]
        if (parity["index_mismatches"] or parity["squared_distance_bit_mismatches"]
                or parity["original_index_order_violations"]
                or not parity["row_counts_match_legacy"]):
            raise AssertionError("diagnostic BVH, scan, or legacy row counts differ")
        report["status"] = "pass"
    except Exception as exc:
        report["status"] = "fail"
        report["error"] = f"{type(exc).__name__}: {exc}"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{selected.name}: {report['status']} "
          f"fallback={report.get('bvh_stack_fallback', {}).get('rows', 'unavailable')}/{fixture['queries']}")
    return 0 if report["status"] == "pass" else 1


def _run_matrix(matrix: str, output_dir: Path, max_seconds: int) -> int:
    _environment()
    selected = cases(matrix)
    _source_state()
    output_dir = output_dir.resolve()
    if output_dir.is_relative_to(ROOT.resolve()):
        raise ValueError("output directory must be outside the repository checkout")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    completed: list[dict] = []
    status = "pass"
    for case in selected:
        target = output_dir / f"{case.name}-overflow.json"
        cmd = [sys.executable, "-m", "bench.probe_spatial_radius_fallback",
               "--matrix", matrix, "--case", case.name, "--output", str(target)]
        try:
            child = subprocess.run(cmd, cwd=ROOT, text=True,
                                   capture_output=True, timeout=max_seconds,
                                   check=False)
            print(child.stdout, end="", flush=True)
            if child.stderr:
                print(child.stderr, file=sys.stderr, end="", flush=True)
            if target.exists():
                result = json.loads(target.read_text())
                completed.append({"case": case.name, "file": target.name,
                                  "sha256": _sha256(target.read_bytes()),
                                  "status": result.get("status")})
            if child.returncode != 0 or not target.exists():
                status = "fail"
                break
        except subprocess.TimeoutExpired:
            status = "timeout"
            completed.append({"case": case.name, "status": "timeout",
                              "max_seconds": max_seconds})
            break
    manifest = {"schema": "mps-pointops-spatial-radius-overflow-matrix-v1",
                "matrix": matrix, "status": status,
                "expected_cases": len(selected), "completed_cases": completed}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0 if status == "pass" and len(completed) == len(selected) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", choices=sorted(LEGACY), required=True)
    parser.add_argument("--output-dir", type=Path,
                        help="new directory outside the checkout; one fresh process per case")
    parser.add_argument("--max-seconds", type=int, default=180,
                        help="wall-time limit for each child process (default: 180)")
    parser.add_argument("--case", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 30 <= args.max_seconds <= 600:
        parser.error("--max-seconds must be in [30, 600]")
    if args.case is not None:
        if args.output is None or args.output_dir is not None:
            parser.error("internal --case requires --output and no --output-dir")
        selected = next((case for case in cases(args.matrix) if case.name == args.case), None)
        if selected is None:
            parser.error("case is not in the pinned legacy matrix")
        return _run_case(args.matrix, selected, args.output)
    if args.output_dir is None or args.output is not None:
        parser.error("--output-dir is required for a matrix run")
    return _run_matrix(args.matrix, args.output_dir, args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())

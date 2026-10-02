"""Correlate six archived spatial-memory cases with an Instruments GPU trace.

This records host intervals and PyTorch allocator observations only. GPU
kernel duration, occupancy, resource residency, and physical footprint must
be read from the separately saved Instruments trace. No GPU work runs merely
by importing this module.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.spatial_bvh import MortonTwoLevelBVH  # noqa: E402
from mps_pointops._flat_search_mps import knn_indices  # noqa: E402

ARCHIVE_DIR = ROOT / "bench/results/spatial-memory-v090-m5pro-clean"
ARCHIVED_COMMIT = "62fcc0f295e127be47a475fbc122d4de686b4c58"
ARCHIVED_CASES = (
    ("collapsed-q256-bvh-serial", "bvh-serial", "collapsed", 256, 16.0),
    ("collapsed-q256-bvh-split", "bvh-split", "collapsed", 256, 16.0),
    ("mixed-q4096-bvh-serial", "bvh-serial", "cluster-sparse", 4096, 0.015625),
    ("mixed-q65536-bvh-serial", "bvh-serial", "cluster-sparse", 65536, 0.015625),
    ("mixed-q65536-native-full-scan", "native-full-scan", "cluster-sparse", 65536, 0.015625),
    ("uniform-q65536-bvh-serial", "bvh-serial", "uniform", 65536, 16.0),
)
PINNED_SOURCE_FILES = (
    "bench/measure_spatial_memory.py",
    "bench/bench_spatial_bvh.py",
    "bench/spatial_bvh.py",
    "bench/spatial_bvh.metal",
    "bench/spatial_keys.py",
    "bench/spatial_keys.metal",
    "mps_pointops/_flat_search_mps.py",
    "mps_pointops/kernels/flat_search.metal",
)
EXPECTED_KERNELS = {
    "bvh-serial": ["bvh_knn_f32"],
    "bvh-split": ["bvh_seed_f32", "bvh_search_micro_f32", "bvh_merge_micro_f32"],
    "native-full-scan": ["flat_knn_indices"],
}


@dataclass(frozen=True)
class Case:
    name: str
    fixture: dict[str, Any]
    archive_path: str
    archive_sha256: str
    archived_memory: dict[str, Any] | None
    derived_from_archive: bool = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(*argv: str) -> str:
    result = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed: {' '.join(argv)}: {result.stderr.strip()}")
    return result.stdout.strip()


def _expected_fixture(path: str, distribution: str, queries: int, cell_size: float) -> dict[str, Any]:
    return {
        "path": path,
        "distribution": distribution,
        "points": 1_000_000,
        "queries": queries,
        "k": 16,
        "extent": 1024.0,
        "cell_size": cell_size,
        "seed": 20261002,
    }


def load_cases(*, include_uniform_scan: bool = False, archive_dir: Path = ARCHIVE_DIR) -> list[Case]:
    """Reject archive/source drift before dispatching any GPU work."""
    cases: list[Case] = []
    source_map: dict[str, str] | None = None
    for name, path, distribution, queries, cell_size in ARCHIVED_CASES:
        archive = archive_dir / f"{name}.json"
        record = json.loads(archive.read_text())
        if record.get("source_commit") != ARCHIVED_COMMIT or record.get("source_dirty") is not False:
            raise ValueError(f"{archive}: archived source provenance changed")
        expected = _expected_fixture(path, distribution, queries, cell_size)
        fixture = record.get("fixture", {})
        if any(fixture.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{archive}: fixture differs from the fixed six-case matrix")
        expected_source = record.get("source_sha256", {})
        if set(expected_source) != set(PINNED_SOURCE_FILES):
            raise ValueError(f"{archive}: source hash file set changed")
        if source_map is None:
            source_map = expected_source
        elif expected_source != source_map:
            raise ValueError(f"{archive}: archived cases disagree about source hashes")
        for source, digest in expected_source.items():
            if _sha256(ROOT / source) != digest:
                raise ValueError(f"{source}: current source differs from the archived implementation")
        cases.append(Case(
            name=name,
            fixture=fixture,
            archive_path=str(archive.relative_to(ROOT)),
            archive_sha256=_sha256(archive),
            archived_memory={
                "allocator_high_water_bytes": record["allocator_high_water_bytes"],
                "sampled_high_water_bytes": record["sampled_high_water_bytes"],
            },
        ))
    if include_uniform_scan:
        reference = cases[-1]
        cases.append(Case(
            name="uniform-q65536-native-full-scan",
            fixture={**reference.fixture, "path": "native-full-scan"},
            archive_path=reference.archive_path,
            archive_sha256=reference.archive_sha256,
            archived_memory=None,
            derived_from_archive=True,
        ))
    return cases


def validate_intervals(repeats: list[dict[str, Any]]) -> None:
    """Keep an exported record usable for timeline correlation."""
    if not repeats:
        raise ValueError("no query intervals")
    previous_end = -1
    for row in repeats:
        start = row["host_perf_start_ns"]
        end = row["host_perf_end_ns"]
        if start < previous_end or end <= start:
            raise ValueError("query host intervals overlap or have non-positive duration")
        if row["host_unix_end_ns"] <= row["host_unix_start_ns"]:
            raise ValueError("invalid wall-clock correlation interval")
        if row["host_synchronized_ns"] != end - start:
            raise ValueError("host synchronized duration is not derived from perf-counter endpoints")
        previous_end = end


def _memory() -> dict[str, int]:
    return {
        "tensor_bytes": int(torch.mps.current_allocated_memory()),
        "driver_bytes": int(torch.mps.driver_allocated_memory()),
    }


def _run_case(case: Case, warmups: int, repeats: int) -> dict[str, Any]:
    # The benchmark extra contains SciPy; keep the CPU-only schema checks
    # importable in the base CI environment that installs torch and NumPy.
    from bench.bench_spatial_bvh import _fixture

    args = argparse.Namespace(**case.fixture)
    points_cpu, query_cpu, detail = _fixture(args)
    fixture_sha256 = hashlib.sha256(points_cpu.tobytes() + query_cpu.tobytes()).hexdigest()
    points = torch.from_numpy(points_cpu).to("mps")
    query = torch.from_numpy(query_cpu).to("mps")
    origin = torch.zeros(3, dtype=torch.float32, device="mps")
    ptr_x = torch.tensor([0, len(points)], device="mps")
    ptr_y = torch.tensor([0, len(query)], device="mps")
    if case.fixture["path"].startswith("bvh"):
        index = MortonTwoLevelBVH.build(points, origin, case.fixture["cell_size"])
        def query_once() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return index.knn(query, case.fixture["k"],
                             parallel_microtrees=case.fixture["path"] == "bvh-split")
    else:
        index = None
        def query_once() -> torch.Tensor:
            return knn_indices(points, query, ptr_x, ptr_y, case.fixture["k"])
    torch.mps.synchronize()
    for _ in range(warmups):
        warm = query_once()
        torch.mps.synchronize()
        del warm
    torch.mps.synchronize()
    before = _memory()
    samples: list[dict[str, Any]] = []
    pid = os.getpid()
    for repeat in range(repeats):
        torch.mps.synchronize()
        torch.accelerator.reset_peak_memory_stats()
        print(f"INSTRUMENTS_QUERY_NEXT pid={pid} case={case.name} repeat={repeat}", flush=True)
        start_unix_ns = time.time_ns()
        start_ns = time.perf_counter_ns()
        result = query_once()
        torch.mps.synchronize()
        end_ns = time.perf_counter_ns()
        end_unix_ns = time.time_ns()
        print(f"INSTRUMENTS_QUERY_END pid={pid} case={case.name} repeat={repeat} "
              f"unix_ns={end_unix_ns} perf_ns={end_ns}", flush=True)
        index_result = result[1] if isinstance(result, tuple) else result
        if index_result.shape != (case.fixture["queries"], case.fixture["k"]) or index_result.dtype != torch.int64:
            raise RuntimeError(f"{case.name}: unexpected index output contract")
        samples.append({
            "repeat": repeat,
            "host_unix_start_ns": start_unix_ns,
            "host_unix_end_ns": end_unix_ns,
            "host_perf_start_ns": start_ns,
            "host_perf_end_ns": end_ns,
            "host_synchronized_ns": end_ns - start_ns,
            "memory_after_query": _memory(),
            "allocator_peak_during_query": {
                "tensor_bytes": int(torch.accelerator.max_memory_allocated()),
                "reserved_bytes": int(torch.accelerator.max_memory_reserved()),
            },
        })
        del result
    validate_intervals(samples)
    torch.mps.synchronize()
    after = _memory()
    del index, points, query, origin, ptr_x, ptr_y, points_cpu, query_cpu
    gc.collect()
    torch.mps.empty_cache()
    torch.mps.synchronize()
    return {
        "name": case.name,
        "fixture": {**case.fixture, **detail},
        "fixture_points_queries_sha256": fixture_sha256,
        "archive_reference": {
            "path": case.archive_path,
            "sha256": case.archive_sha256,
            "derived_case": case.derived_from_archive,
            "archived_2026_10_02_memory": case.archived_memory,
        },
        "expected_query_kernel_symbols": EXPECTED_KERNELS[case.fixture["path"]],
        "before_query_repeats": before,
        "after_query_repeats": after,
        "warmups": warmups,
        "query_repeats": samples,
        "host_synchronized_median_ms": statistics.median(
            row["host_synchronized_ns"] for row in samples) / 1e6,
    }


@contextlib.contextmanager
def _signposts(enabled: bool) -> Iterator[None]:
    if enabled:
        # PyTorch documents MPS operation OS signposts. This does not expose
        # GPU occupancy and is not a GPU-kernel timing counter.
        with torch.mps.profiler.profile(mode="interval", wait_until_completed=False):
            yield
    else:
        yield


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--include-uniform-scan", action="store_true")
    parser.add_argument("--mps-signposts", action="store_true",
                        help="enable PyTorch MPS operation OS signposts for Instruments")
    parser.add_argument("--pause-before-s", type=float, default=30.0)
    parser.add_argument("--pause-after-s", type=float, default=10.0)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0" or os.environ.get("PYTORCH_MPS_FAST_MATH") != "0":
        raise RuntimeError("requires fallback=0 and Fast Math=0 before Python starts")
    if not torch.backends.mps.is_available():
        raise RuntimeError("requires a physical MPS device")
    if not 1 <= args.warmups <= 3 or not 5 <= args.repeats <= 7:
        raise ValueError("warmups must be 1..3 and query repeats must be 5..7")
    if not 0 <= args.pause_before_s <= 600 or not 0 <= args.pause_after_s <= 600:
        raise ValueError("Instruments pauses must be in [0, 600] seconds")
    output_dir = args.output_dir.resolve()
    if output_dir == ROOT or ROOT in output_dir.parents:
        raise ValueError("write profiling evidence outside the source tree")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    cases = load_cases(include_uniform_scan=args.include_uniform_scan)
    if _command("git", "status", "--porcelain"):
        raise RuntimeError("a clean source checkout is required for profiling")
    if not all(hasattr(torch.accelerator, name) for name in
               ("reset_peak_memory_stats", "max_memory_allocated", "max_memory_reserved")):
        raise RuntimeError("PyTorch accelerator peak allocator API unavailable")
    output_dir.mkdir(parents=True)
    metadata = {
        "utc_started": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
        "source_commit": _command("git", "rev-parse", "HEAD"),
        "source_dirty": False,
        "source_sha256": {name: _sha256(ROOT / name) for name in
                          (*PINNED_SOURCE_FILES, "bench/profile_spatial_instruments.py")},
        "archive_commit": ARCHIVED_COMMIT,
        "hardware": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                   capture_output=True, text=True, check=False).stdout.strip(),
        "physical_ram_bytes": subprocess.run(["sysctl", "-n", "hw.memsize"],
                                              capture_output=True, text=True, check=False).stdout.strip(),
        "macos": platform.mac_ver()[0],
        "torch": torch.__version__,
        "mps_fast_math": "0", "mps_fallback": "0",
        "mps_recommended_max_memory_bytes": int(torch.mps.recommended_max_memory()),
        "mps_signposts": args.mps_signposts,
        "mps_signpost_wait_until_completed": False,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "include_uniform_scan": args.include_uniform_scan,
        "trace_capture": "external Instruments session; attach separately; trace not embedded",
        "host_time_scope": "perf_counter start/end spans include Python dispatch and MPS synchronize; wall-clock timestamps aid trace correlation; neither is GPU kernel duration",
        "memory_scope": "PyTorch tensor/driver endpoints and per-query accelerator allocator peaks; no whole-device or physical-footprint measurement",
        "gpu_kernel_duration_ns": None,
        "gpu_occupancy": None,
        "cases": [],
    }
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"INSTRUMENTS_ATTACH_READY pid={os.getpid()} pause_s={args.pause_before_s} "
          f"source={metadata['source_commit']}", flush=True)
    time.sleep(args.pause_before_s)
    try:
        with _signposts(args.mps_signposts):
            for case in cases:
                print(f"INSTRUMENTS_CASE_START case={case.name} pid={os.getpid()}", flush=True)
                result = _run_case(case, args.warmups, args.repeats)
                path = output_dir / f"{case.name}.json"
                path.write_text(json.dumps(result, indent=2) + "\n")
                metadata["cases"].append({"name": case.name, "path": path.name,
                                          "sha256": _sha256(path)})
                manifest.write_text(json.dumps(metadata, indent=2) + "\n")
                print(f"INSTRUMENTS_CASE_COMPLETE case={case.name} file={path}", flush=True)
    except Exception as exc:
        metadata["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        metadata["utc_finished_or_failed"] = datetime.now(timezone.utc).isoformat()
        manifest.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"INSTRUMENTS_CAPTURE_COMPLETE pid={os.getpid()} pause_s={args.pause_after_s}",
              flush=True)
        time.sleep(args.pause_after_s)


if __name__ == "__main__":
    main()

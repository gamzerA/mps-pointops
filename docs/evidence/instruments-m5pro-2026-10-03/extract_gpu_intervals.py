"""Reduce exported Metal System Trace tables to the launched target process.

The output omits PIDs, absolute paths, and rows from unrelated processes. It
distinguishes host Metal API encoding from device GPU-active intervals. The
trace does not attach a shader name to each GPU interval, so this script does
not claim per-shader timing or occupancy. Inputs are private xctrace XML
exports and the published query-runs directory; no workload is launched.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _trace_bundle_summary(path: Path) -> dict[str, str | int]:
    if not path.is_dir():
        raise NotADirectoryError(path)
    files = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        files.append({
            "path": file.relative_to(path).as_posix(),
            "bytes": file.stat().st_size,
            "sha256": _hash(file),
        })
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "trace": path.name,
        "canonical_manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "file_count": len(files),
        "total_file_bytes": sum(item["bytes"] for item in files),
    }


def _root_and_refs(path: Path) -> tuple[ET.Element, dict[str, ET.Element]]:
    root = ET.parse(path).getroot()
    refs = {element.attrib["id"]: element for element in root.iter() if "id" in element.attrib}
    return root, refs


def _resolve(element: ET.Element | None, refs: dict[str, ET.Element]) -> ET.Element | None:
    while element is not None and "ref" in element.attrib:
        element = refs.get(element.attrib["ref"])
    return element


def _pid(process: ET.Element | None, refs: dict[str, ET.Element]) -> int | None:
    process = _resolve(process, refs)
    value = _resolve(process.find("pid"), refs) if process is not None else None
    return int(value.text) if value is not None and value.text else None


def _value(element: ET.Element | None, refs: dict[str, ET.Element]) -> str | None:
    element = _resolve(element, refs)
    return element.text if element is not None else None


def _fmt(element: ET.Element | None, refs: dict[str, ET.Element]) -> str | None:
    element = _resolve(element, refs)
    return element.attrib.get("fmt") if element is not None else None


def _origin_ns(toc: Path) -> tuple[int, str, int]:
    root = ET.parse(toc).getroot()
    raw = root.findtext("./run/info/summary/start-date")
    if not raw:
        raise ValueError("trace start-date is missing")
    timestamp = datetime.fromisoformat(raw)
    if timestamp.tzinfo is None:
        raise ValueError("trace start-date needs a timezone")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = timestamp.astimezone(timezone.utc) - epoch
    ns = (delta.days * 86400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1000
    target = root.find("./run/info/target/process")
    if target is None or target.attrib.get("type") != "launched" or not target.attrib.get("pid"):
        raise ValueError("trace must identify one launched target process")
    return ns, raw, int(target.attrib["pid"])


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for start, stop in sorted(intervals):
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def _gpu_rows(path: Path, pid: int) -> list[dict[str, int | str]]:
    root, refs = _root_and_refs(path)
    rows: list[dict[str, int | str]] = []
    for row in root.findall(".//row"):
        if _pid(row.find("process"), refs) != pid:
            continue
        identifiers = row.findall("metal-command-buffer-id")
        if len(identifiers) != 2:
            raise ValueError("GPU interval lacks command-buffer/encoder identifiers")
        start = _value(row.find("start-time"), refs)
        duration = _value(row.find("duration"), refs)
        if start is None or duration is None:
            raise ValueError("GPU interval lacks start or duration")
        rows.append({
            "start_trace_ns": int(start),
            "duration_ns": int(duration),
            "channel": _fmt(row.find("gpu-channel-name"), refs) or "",
            "state": _fmt(row.find("gpu-state"), refs) or "",
            "command_buffer_id": int(_value(identifiers[0], refs)),
            "encoder_id": int(_value(identifiers[1], refs)),
        })
    return rows


def _encoder_rows(path: Path, pid: int) -> list[dict[str, int | str]]:
    root, refs = _root_and_refs(path)
    node = next((node for node in root.findall("node")
                 if node.find("schema") is not None
                 and node.find("schema").attrib.get("name") == "metal-application-encoders-list"), None)
    if node is None:
        raise ValueError("host encoder list is missing")
    rows: list[dict[str, int | str]] = []
    for row in node.findall("row"):
        if _pid(row.find("process"), refs) != pid:
            continue
        identifiers = row.findall("metal-command-buffer-id")
        labels = row.findall("metal-object-label")
        if len(identifiers) != 2 or len(labels) < 3:
            raise ValueError("host encoder list has unexpected schema")
        rows.append({
            "start_trace_ns": int(_value(row.find("start-time"), refs)),
            "duration_ns": int(_value(row.find("duration"), refs)),
            "generic_encoder_label": _fmt(labels[2], refs) or "",
            "command_buffer_id": int(_value(identifiers[0], refs)),
            "encoder_id": int(_value(identifiers[1], refs)),
        })
    return rows


def _compiled_expected_shaders(path: Path, pid: int, expected: set[str]) -> list[str]:
    root, refs = _root_and_refs(path)
    node = next((node for node in root.findall("node")
                 if node.find("schema") is not None
                 and node.find("schema").attrib.get("name") == "metal-shader-profiler-shader-list"), None)
    if node is None:
        raise ValueError("shader list is missing")
    found: set[str] = set()
    for row in node.findall("row"):
        if _pid(row.find("process"), refs) != pid:
            continue
        label = _fmt(row.find("metal-object-label"), refs) or ""
        for name in expected:
            if label.startswith(name + " ("):
                found.add(name)
    return sorted(found)


def _profiler_row_counts(path: Path, toc_path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    toc = ET.parse(toc_path).getroot()
    schemas = {
        "gpu-shader-profiler-sample": "gpu_shader_profiler_samples",
        "gpu-shader-profiler-interval": "gpu_shader_profiler_intervals",
        "metal-shader-profiler-intervals": "metal_shader_profiler_intervals",
    }
    mapping: dict[str, str] = {}
    for index, table in enumerate(toc.findall("./run/data/table"), 1):
        schema = table.attrib.get("schema")
        if schema in schemas:
            key = f"/table[{index}]"
            if schemas[schema] in mapping.values():
                raise ValueError(f"duplicate profiler table: {schema}")
            mapping[key] = schemas[schema]
    if set(mapping.values()) != set(schemas.values()):
        raise ValueError("trace TOC lacks a required profiler table")
    counts = {name: -1 for name in schemas.values()}
    for node in root.findall("node"):
        xpath = node.attrib.get("xpath", "")
        for suffix, name in mapping.items():
            if xpath.endswith(suffix):
                counts[name] = len(node.findall("row"))
    if any(count < 0 for count in counts.values()):
        raise ValueError("shader profiler exports are incomplete")
    return counts


def extract(exports_dir: Path, publication_dir: Path, trace_bundle: Path) -> dict[str, object]:
    manifest_path = publication_dir / "query-runs/manifest.json"
    toc_path = exports_dir / "query-matrix-system-toc.xml"
    gpu_path = exports_dir / "query-matrix-gpu-intervals.xml"
    host_path = exports_dir / "query-matrix-host-encoders.xml"
    shader_path = exports_dir / "query-matrix-shader-and-execution.xml"
    profiler_path = exports_dir / "query-matrix-shader-intervals.xml"
    availability_path = publication_dir / "counter-availability.json"
    manifest = json.loads(manifest_path.read_text())
    trace_summary = _trace_bundle_summary(trace_bundle)
    if trace_summary != manifest.get("query_trace_bundle"):
        raise ValueError("private query trace bundle differs from the published digest")
    origin, trace_start, pid = _origin_ns(toc_path)
    if "pid" in manifest and int(manifest["pid"]) != pid:
        raise ValueError("manifest target PID differs from trace target")
    if manifest["source_dirty"] is not False:
        raise ValueError("query runs must come from a clean source checkout")
    case_records: dict[str, dict] = {}
    for case in manifest["cases"]:
        case_path = publication_dir / "query-runs" / case["path"]
        if _hash(case_path) != case["sha256"]:
            raise ValueError(f"query run changed: {case['path']}")
        record = json.loads(case_path.read_text())
        if record.get("name") != case["name"]:
            raise ValueError(f"query run name mismatch: {case['path']}")
        case_records[case["name"]] = record
    gpu = _gpu_rows(gpu_path, pid)
    encoders = _encoder_rows(host_path, pid)
    encoder_map = {(r["command_buffer_id"], r["encoder_id"]): r for r in encoders}
    expected: set[str] = set()
    for case in manifest["cases"]:
        record = case_records[case["name"]]
        expected.update(record["expected_query_kernel_symbols"])
    compiled = _compiled_expected_shaders(shader_path, pid, expected)
    profiler_counts = _profiler_row_counts(profiler_path, toc_path)
    if not expected.issubset(compiled):
        raise ValueError(f"target process is missing expected compiled shaders: {expected - set(compiled)}")
    output_cases: list[dict[str, object]] = []
    for case in manifest["cases"]:
        record = case_records[case["name"]]
        repeats: list[dict[str, object]] = []
        for repeat in record["query_repeats"]:
            lo, hi = int(repeat["host_unix_start_ns"]), int(repeat["host_unix_end_ns"])
            observed: list[tuple[int, int, dict[str, int | str]]] = []
            for row in gpu:
                if row["state"] != "Active":
                    continue
                start = origin + int(row["start_trace_ns"])
                stop = start + int(row["duration_ns"])
                if start < hi and stop > lo:
                    observed.append((start, stop, row))
            clips = [(max(lo, start), min(hi, stop)) for start, stop, _ in observed]
            # The exported trace origin is printed only to milliseconds. A
            # one-millisecond offset sweep exposes sensitivity at host edges;
            # it is not a confidence interval for instrument accuracy.
            alignment_unions = []
            for shift in (-1_000_000, 0, 1_000_000):
                shifted = []
                for row in gpu:
                    if row["state"] != "Active":
                        continue
                    start = origin + int(row["start_trace_ns"]) + shift
                    stop = start + int(row["duration_ns"])
                    if start < hi and stop > lo:
                        shifted.append((max(lo, start), min(hi, stop)))
                alignment_unions.append(_union_ns(shifted))
            by_encoder: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
            for (start, stop, row), clipped in zip(observed, clips):
                by_encoder[(int(row["command_buffer_id"]), int(row["encoder_id"]))].append(clipped)
            groups: list[dict[str, int | str]] = []
            for key, intervals in by_encoder.items():
                host = encoder_map.get(key)
                first = min(a for a, _ in intervals)
                last = max(b for _, b in intervals)
                groups.append({
                    "command_buffer_id": key[0],
                    "encoder_id": key[1],
                    "generic_encoder_label": host["generic_encoder_label"] if host else "unmatched",
                    "host_encoding_start_trace_ns": host["start_trace_ns"] if host else -1,
                    "host_encoding_duration_ns": host["duration_ns"] if host else -1,
                    "gpu_active_interval_count": len(intervals),
                    "gpu_active_first_unix_ns": first,
                    "gpu_active_last_unix_ns": last,
                    "gpu_active_sum_ns": sum(b - a for a, b in intervals),
                    "gpu_active_union_ns": _union_ns(intervals),
                })
            groups.sort(key=lambda group: group["gpu_active_first_unix_ns"])
            if not clips:
                raise ValueError(f"no target GPU interval for {case['name']} repeat {repeat['repeat']}")
            repeated = {
                "repeat": repeat["repeat"],
                "host_unix_start_ns": lo,
                "host_unix_end_ns": hi,
                "host_synchronized_ns": int(repeat["host_synchronized_ns"]),
                "target_gpu_active_interval_count": len(clips),
                "target_gpu_active_sum_ns": sum(b - a for a, b in clips),
                "target_gpu_active_union_ns": _union_ns(clips),
                "target_gpu_active_first_unix_ns": min(a for a, _ in clips),
                "target_gpu_active_last_unix_ns": max(b for _, b in clips),
                "intervals_crossing_host_start": sum(start < lo for start, _, _ in observed),
                "intervals_crossing_host_end": sum(stop > hi for _, stop, _ in observed),
                "total_clipped_ns": sum((b - a) - (clip_b - clip_a)
                                        for (a, b, _), (clip_a, clip_b) in zip(observed, clips)),
                "gpu_active_union_ns_if_trace_origin_minus_1ms": alignment_unions[0],
                "gpu_active_union_ns_if_trace_origin_plus_1ms": alignment_unions[2],
                "encoder_groups": groups,
            }
            repeats.append(repeated)
        output_cases.append({
            "name": case["name"],
            "query_run_sha256": case["sha256"],
            "expected_query_kernel_symbols": record["expected_query_kernel_symbols"],
            "median_host_synchronized_ns": int(statistics.median(r["host_synchronized_ns"] for r in repeats)),
            "median_target_gpu_active_union_ns": int(statistics.median(r["target_gpu_active_union_ns"] for r in repeats)),
            "query_allocator_peak_tensor_bytes_max": max(
                int(r["allocator_peak_during_query"]["tensor_bytes"])
                for r in record["query_repeats"]
            ),
            "query_allocator_peak_reserved_bytes_max": max(
                int(r["allocator_peak_during_query"]["reserved_bytes"])
                for r in record["query_repeats"]
            ),
            "driver_allocated_checkpoint_bytes_max": max(
                [int(record["before_query_repeats"]["driver_bytes"]),
                 int(record["after_query_repeats"]["driver_bytes"])]
                + [int(r["memory_after_query"]["driver_bytes"])
                   for r in record["query_repeats"]]
            ),
            "repeats": repeats,
        })
    inputs = (manifest_path, toc_path, gpu_path, host_path, shader_path, profiler_path,
              availability_path)
    return {
        "scope": "launched-target-only Metal System Trace extraction; PID omitted",
        "source_commit": manifest["source_commit"],
        "source_dirty": manifest["source_dirty"],
        "private_original_manifest_sha256": manifest.get("private_original_manifest_sha256"),
        "query_trace_bundle": trace_summary,
        "trace_start_date": trace_start,
        "trace_start_date_resolution_ns": 1_000_000,
        "trace_alignment_note": "Trace start-date has millisecond precision; wall-clock alignment may be uncertain by about 1 ms. Host spans and trace GPU intervals use different clocks.",
        "identity_note": "Shader list proves expected shaders compiled in target process, but GPU interval records attach only generic encoder labels. Match to a named shader is an inference from pinned source dispatch, not a direct shader profiler observation.",
        "gpu_measurement": "PID-filtered GPU Active intervals, clipped to each synchronized host query span; overlap union avoids double counting; this is not shader occupancy or whole-device utilization.",
        "compiled_expected_shaders": compiled,
        "shader_profiler_rows": profiler_counts,
        "gpu_occupancy": None,
        "counter_availability_sha256": _hash(availability_path),
        "input_sha256": {path.name: _hash(path) for path in inputs},
        "target_gpu_interval_count": len(gpu),
        "target_host_encoder_interval_count": len(encoders),
        "cases": output_cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exports-dir", required=True, type=Path,
                        help="private xctrace XML exports directory")
    parser.add_argument("--publication-dir", required=True, type=Path,
                        help="directory containing query-runs and counter-availability.json")
    parser.add_argument("--trace-bundle", required=True, type=Path,
                        help="private query-matrix-system.trace directory for digest verification")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = extract(args.exports_dir, args.publication_dir, args.trace_bundle)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"launched target: {result['target_gpu_interval_count']} GPU intervals, {len(result['cases'])} cases")
    for case in result["cases"]:
        print(f"{case['name']}: median host {case['median_host_synchronized_ns']/1e6:.3f} ms, "
              f"target GPU Active union {case['median_target_gpu_active_union_ns']/1e6:.3f} ms")


if __name__ == "__main__":
    main()

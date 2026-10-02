"""Extract target-process memory series from local Instruments traces.

It exports XML to a temporary directory, then retains only numeric rows for
the single target PID. The trace directory digest is the
SHA-256 of canonical JSON containing each relative file path, size and SHA-256.

Run with a full Xcode developer directory, for example:
    DEVELOPER_DIR=/path/to/Xcode.app/Contents/Developer python3 extract_xctrace.py \
      --trace-root /path/to/local-traces --output-dir /path/to/private-output

No workload is launched. Existing traces are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


FIXTURES = (
    "mixed-q65536-bvh",
    "uniform-q65536-bvh",
    "mixed-q4096-bvh",
    "mixed-q65536-scan",
    "collapsed-q256-bvh",
    "collapsed-q256-split",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def trace_manifest(path: Path) -> dict:
    files = []
    for file in sorted(p for p in path.rglob("*") if p.is_file()):
        files.append({
            "path": file.relative_to(path).as_posix(),
            "bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        })
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "trace": path.name,
        "canonical_manifest_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "file_count": len(files),
        "total_file_bytes": sum(item["bytes"] for item in files),
        "files": files,
    }


def export(trace: Path, args: list[str], output: Path) -> ET.Element:
    output.unlink(missing_ok=True)
    subprocess.run(
        ["xcrun", "xctrace", "export", "--input", str(trace), *args,
         "--output", str(output)],
        check=True, stdout=subprocess.DEVNULL,
    )
    return ET.parse(output).getroot()


def selected_table(toc: ET.Element, schema: str) -> None:
    tables = [
        table for table in toc.findall(".//table")
        if table.get("schema") == schema and table.get("target-pid") == "SINGLE"
    ]
    if len(tables) != 1:
        raise ValueError(f"expected one target-only {schema} table; got {len(tables)}")


def rows_with_schema(xml: ET.Element, expected_schema: str):
    schema = xml.find(".//schema")
    if schema is None or schema.get("name") != expected_schema:
        raise ValueError(f"unexpected exported schema for {expected_schema}")
    names = [column.findtext("mnemonic") for column in schema]
    refs = {(item.tag, item.get("id")): item for item in xml.iter() if item.get("id")}

    def value(row, mnemonic):
        cell = row[names.index(mnemonic)]
        while cell.get("ref"):
            cell = refs[(cell.tag, cell.get("ref"))]
        return cell

    return xml.findall(".//row"), value


def target_pid(cell: ET.Element) -> int:
    pid = cell.findtext(".//pid")
    if pid is None:
        raise ValueError("target process PID missing in exported row")
    return int(pid)


def extract_metal(trace: Path, allocator_path: Path, tmp: Path) -> dict:
    toc = export(trace, ["--toc"], tmp / "toc.xml")
    selected_table(toc, "metal-current-allocated-size")
    xml = export(trace, [
        "--xpath",
        '/trace-toc/run/data/table[@schema="metal-current-allocated-size" and @target-pid="SINGLE"]',
    ], tmp / "metal.xml")
    rows, value = rows_with_schema(xml, "metal-current-allocated-size")
    if not rows:
        raise ValueError(f"no Metal allocation rows in {trace}")
    series = []
    for row in rows:
        series.append({
            "pid": target_pid(value(row, "process")),
            "start_raw": int(value(row, "start").text),
            "duration_raw": int(value(row, "duration").text),
            "current_allocated_bytes": int(value(row, "current-allocated-size").text),
        })
    pids = {item["pid"] for item in series}
    if len(pids) != 1:
        raise ValueError(f"more than one target PID in {trace}: {pids}")
    allocator = json.loads(allocator_path.read_text())
    peak = max(item["current_allocated_bytes"] for item in series)
    return {
        "fixture": trace.stem,
        "trace": trace.name,
        "allocator_json": allocator_path.name,
        "allocator_json_sha256": file_sha256(allocator_path),
        "source_commit": allocator["source_commit"],
        "source_dirty": allocator["source_dirty"],
        "source_sha256": allocator["source_sha256"],
        "target_pid": next(iter(pids)),
        "row_count": len(series),
        "observed_current_allocated_max_bytes": peak,
        "allocator_sampled_driver_high_water_bytes": allocator["sampled_high_water_bytes"]["driver_bytes"],
        "observed_max_equals_allocator_sampled_high_water": (
            peak == allocator["sampled_high_water_bytes"]["driver_bytes"]
        ),
        "series": series,
    }


def extract_activity(trace: Path, allocator_path: Path, tmp: Path) -> dict:
    toc = export(trace, ["--toc"], tmp / "activity-toc.xml")
    selected_table(toc, "sysmon-process")
    selected_table(toc, "activity-monitor-process-live")
    xml = export(trace, [
        "--xpath",
        '/trace-toc/run/data/table[@schema="sysmon-process" and @target-pid="SINGLE"]',
    ], tmp / "activity.xml")
    rows, value = rows_with_schema(xml, "sysmon-process")
    if not rows:
        raise ValueError("Activity Monitor trace has no target rows")
    series = []
    for row in rows:
        series.append({
            "pid": int(value(row, "pid").text),
            "time_raw": int(value(row, "time").text),
            "physical_footprint_bytes": int(value(row, "memory-physical-footprint").text),
            "resident_size_bytes": int(value(row, "memory-resident-size").text),
        })
    pids = {item["pid"] for item in series}
    if len(pids) != 1:
        raise ValueError(f"more than one Activity Monitor PID: {pids}")
    allocator = json.loads(allocator_path.read_text())
    return {
        "trace": trace.name,
        "allocator_json": allocator_path.name,
        "allocator_json_sha256": file_sha256(allocator_path),
        "source_commit": allocator["source_commit"],
        "source_dirty": allocator["source_dirty"],
        "source_sha256": allocator["source_sha256"],
        "metric_scope": (
            "Sampled target-process physical footprint and resident size from "
            "Activity Monitor sysmon-process; neither is GPU-only or an exact peak."
        ),
        "target_pid": next(iter(pids)),
        "row_count": len(series),
        "observed_physical_footprint_max_bytes": max(x["physical_footprint_bytes"] for x in series),
        "observed_resident_size_max_bytes": max(x["resident_size_bytes"] for x in series),
        "series": series,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    trace_root = args.trace_root.resolve()
    output_dir = args.output_dir.resolve()
    if not trace_root.is_dir():
        raise NotADirectoryError(trace_root)
    if trace_root == output_dir or trace_root in output_dir.parents:
        raise ValueError("output directory must be outside the raw trace root")
    output = {
        "metric_scope": (
            "Metal currentAllocatedSize is device memory allocated for the app, not "
            "a physical GPU/process memory peak. These are observed trace maxima."
        ),
        "fixtures": [],
    }
    manifests = []
    with tempfile.TemporaryDirectory(prefix="mps-xctrace-export-") as temp:
        tmp = Path(temp)
        for fixture in FIXTURES:
            trace = trace_root / f"{fixture}.trace"
            allocator = trace_root / f"{fixture}-allocator.json"
            if not trace.is_dir() or not allocator.is_file():
                raise FileNotFoundError(f"missing archived trace or allocator JSON for {fixture}")
            output["fixtures"].append(extract_metal(trace, allocator, tmp))
            manifests.append(trace_manifest(trace))
            print(fixture, output["fixtures"][-1]["observed_current_allocated_max_bytes"])
        activity = trace_root / "mixed-q65536-bvh-activity.trace"
        if activity.is_dir():
            activity_allocator = trace_root / "mixed-q65536-bvh-activity-allocator.json"
            output["activity_monitor"] = extract_activity(activity, activity_allocator, tmp)
            manifests.append(trace_manifest(activity))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "target-memory-series.json").write_text(json.dumps(output, indent=2) + "\n")
    (output_dir / "trace-tree-manifest.json").write_text(json.dumps(manifests, indent=2) + "\n")
    print("wrote target-memory-series.json and trace-tree-manifest.json")


if __name__ == "__main__":
    main()

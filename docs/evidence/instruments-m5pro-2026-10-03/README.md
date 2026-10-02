# M5 Pro Instruments memory evidence

This directory supports the [spatial Instruments report](../../spatial-instruments-v090.md).

- [`measurements.json`](measurements.json): integer-byte allocator peaks and checkpoints, six target-process Metal `currentAllocatedSize` series, and a separate nine-sample Activity Monitor process-memory pilot. It omits PIDs, absolute paths, and unrelated processes.
- [`trace-tree-manifest.json`](trace-tree-manifest.json): relative file paths, byte lengths, and SHA-256 hashes for the six Game Memory bundles and one Activity Monitor bundle. Raw bundles are not in the repository because a trace can include unrelated process and device metadata.
- [`extract_xctrace.py`](extract_xctrace.py): target-only `xctrace export` selection and canonical bundle-digest algorithm. It launches no workload. A reviewer with the archived traces can run it with `--trace-root` and `--output-dir` outside that root.
- [`query-runs/manifest.json`](query-runs/manifest.json): a separate seven-case Metal System Trace query matrix, source commit `6052590f9e21e31f8459e0669822fd6f122e55ca`, with five synchronized repetitions per case. The seven original case JSON files retain fixture hashes, host timestamps, and same-run allocator readings. The published manifest omits the target PID, records the SHA-256 of its private original, and commits to the private query trace bundle using a canonical digest without exposing its internal file list.
- [`gpu-query-intervals.json`](gpu-query-intervals.json): PID-filtered GPU `Active` interval unions inside each query window and matched generic host encoder timing. The summary omits PIDs, absolute paths, and unrelated-process rows. Its GPU timings are query aggregates, not directly named-kernel timings.
- [`extract_gpu_intervals.py`](extract_gpu_intervals.py): reproduces the query summary from the private trace bundle and `xctrace` XML exports plus the published query runs. It checks the trace bundle digest and every original case JSON hash, derives the target PID from the trace TOC, and selects profiler tables by schema instead of fixed table number. It launches no workload.
- [`allocator-runs/`](allocator-runs/): seven unchanged source-pinned allocator JSONs; their SHA-256 values match `measurements.json`.
- [`counter-availability.json`](counter-availability.json) and [`counter-probes/`](counter-probes/): settings and original logs from bounded counter probes. Profile 3 was rejected with shader profiling on and off. Plain Metal System Trace completed without that warning. This does not establish that all M5 counter configurations or tool versions are unsupported.

The source commit is `8932fca29456bd38690f072b0979feabec6de2d0`. All seven allocator records reported `source_dirty=false`, the same source SHA-256 map, Apple M5 Pro, macOS 26.5.2, PyTorch 2.14.1, Safe Math, and disabled MPS CPU fallback. The source hashes were checked against that commit. Xcode 26.6 (17F113) exported the target-only tables. For Game Memory, the XPath was `/trace-toc/run/data/table[@schema="metal-current-allocated-size" and @target-pid="SINGLE"]`; for Activity Monitor, it was `/trace-toc/run/data/table[@schema="sysmon-process" and @target-pid="SINGLE"]`.

Every `*.trace` path in the manifest is relative to the bundle root. `canonical_manifest_sha256` hashes the UTF-8 encoding of `json.dumps(files, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`, where `files` is sorted by relative path and each entry contains `path`, `bytes`, and `sha256`. It commits to local trace content without publishing the trace itself.

`MTLDevice.currentAllocatedSize`, PyTorch tensor allocator peak, Activity Monitor process physical footprint, and resident size are different metrics. The maxima in this directory are maxima **observed among exported rows or samples**, not exact transient or physical GPU-memory peaks. Trace and polling overhead make their wall times unsuitable as replacements for unprofiled search benchmarks. Target-process query GPU activity is included for the separate seven-case trace; GPU occupancy and per-kernel GPU time are not included.

## Reproducing the GPU query reduction

With the private `query-matrix-system.trace` and full Xcode, use `xcrun xctrace export --input TRACE --toc --output query-matrix-system-toc.xml`. Export the following schemas to the indicated private XML filenames with `--xpath '/trace-toc/run/data/table[@schema="SCHEMA"]'`. For filenames with multiple schemas, join the individual XPath expressions using `|`; the reducer looks up each table by schema in the TOC.

| Private XML filename | Required schema names |
| --- | --- |
| `query-matrix-gpu-intervals.xml` | `metal-gpu-intervals` |
| `query-matrix-host-encoders.xml` | `metal-application-encoders-list` |
| `query-matrix-shader-and-execution.xml` | `metal-shader-profiler-shader-list` |
| `query-matrix-shader-intervals.xml` | `gpu-shader-profiler-sample`, `gpu-shader-profiler-interval`, `metal-shader-profiler-intervals` |

The original private exports also contained supplementary application, execution-point, and signpost tables; the reducer ignores them. After exporting, run:

```bash
python3 extract_gpu_intervals.py \
  --trace-bundle /path/to/query-matrix-system.trace \
  --exports-dir /path/to/private-exports \
  --publication-dir . \
  gpu-query-intervals.json
```

The published `input_sha256` values pin the original private XML exports. A later re-export with different supplementary tables or Xcode formatting may have different XML hashes while producing the same reduced measurements. The reducer verifies all seven query-run file hashes before reading interval data.

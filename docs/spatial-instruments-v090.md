# Instruments memory study: million-point spatial search

This study adds an Instruments observation to the [PyTorch allocator study](spatial-memory-v090.md). It measures **six one-million-point searches on an Apple M5 Pro** and one separate Activity Monitor pilot. The full integer-byte results and target-only numeric series are in [`measurements.json`](evidence/instruments-m5pro-2026-10-03/measurements.json). The captures were made on 2026-10-03 KST from clean source commit `8932fca29456bd38690f072b0979feabec6de2d0`.

## Environment and metric boundaries

Apple M5 Pro, 48 GiB unified memory, macOS 26.5.2, Xcode 26.6 (17F113), PyTorch 2.14.1, `PYTORCH_ENABLE_MPS_FALLBACK=0`, and `PYTORCH_MPS_FAST_MATH=0`. Each fixture ran in a fresh process. All use `N=1,000,000`, `k=16`, seed `20261002`; the query count, point distribution, and search path differ by row. The six allocator records and the Activity pilot report the same clean commit and identical source-file SHA-256 maps. Those eight source hashes also match the files at that commit.

The metrics answer different questions:

- **Tensor allocator peak:** PyTorch's [maximum allocated tensor memory](https://docs.pytorch.org/docs/2.14/generated/torch.accelerator.memory.max_memory_allocated.html), reset before each synchronized stage. It includes short-lived tensors but excludes other Metal and process allocations.
- **Driver checkpoints:** `torch.mps.driver_allocated_memory()` after each synchronized stage. The run maximum is sampled by a poller requested at 0.5 ms, so it can miss shorter transients. [PyTorch documents this counter](https://docs.pytorch.org/docs/2.14/generated/torch.mps.driver_allocated_memory.html) as memory allocated by the Metal driver for the process, including cached allocator and MPS/MPSGraph allocations.
- **Instruments Metal maximum:** the largest target-process row in the Game Memory export's `metal-current-allocated-size` table. Its schema identifies [`MTLDevice.currentAllocatedSize`](https://developer.apple.com/documentation/metal/mtldevice/currentallocatedsize), the size of Metal resources allocated by the device. It is an **observed maximum over exported rows**, not an exact transient peak or a measurement of physical GPU residency. Apple distinguishes [Metal resource allocations from memory footprint](https://developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app).
- **Activity Monitor pilot:** a different process run measured `sysmon-process.memory-physical-footprint` and `memory-resident-size`. These are sampled **process** metrics, not GPU-only measurements, and are not simultaneous with the six Game Memory captures. They cannot be added to or subtracted from the Metal values.

## Six Game Memory captures

Values below are MiB (`2²⁰` bytes), rounded to three decimals. Each triplet is **input transfer / index build / query**; `—` means the full-scan path has no BVH build. The JSON preserves integer bytes and all 102–119 target-process Metal samples per fixture.

| Fixture and path | Tensor allocator peak by stage | Overall tensor peak | Driver after each stage | Instruments Metal observed maximum | Metal rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed density, Q=4,096, serial BVH | 11.548 / 59.001 / 20.563 | 59.001 | 1,032.688 / 1,032.688 / 1,032.688 | 1,036.141 | 113 |
| Mixed density, Q=65,536, serial BVH | 12.251 / 59.705 / 33.688 | 59.705 | 1,032.688 / 1,032.688 / 1,032.688 | 1,036.141 | 113 |
| Mixed density, Q=65,536, native full scan | 12.251 / — / 20.251 | 20.251 | 1,032.688 / — / 1,032.688 | 1,036.141 | 102 |
| Uniform, Q=65,536, serial BVH | 12.251 / 59.705 / 33.688 | 59.705 | 1,032.688 / 1,032.688 / 1,032.688 | 1,036.141 | 113 |
| Collapsed, Q=256, serial BVH | 11.504 / 58.958 / 19.743 | 58.958 | 1,032.688 / 1,032.688 / 1,032.688 | 1,036.141 | 113 |
| Collapsed, Q=256, split BVH | 11.504 / 58.958 / 26.119 | 58.958 | 1,032.688 / 1,032.688 / 1,032.688 | 1,036.141 | 119 |

The observed Metal maximum is **1,086,472,192 bytes** in every capture. It equals that capture's separately polled driver maximum, so the two readings do not establish two different memory peaks. The allocator records place the maximum during **input transfer**, before BVH construction or search. The repeated value therefore does not show that BVH and full scan have equal incremental memory cost. Their tensor allocator peaks differ, as the table shows. All six traces reported one target PID each; no unrelated-process rows enter the published series.

The mixed and uniform Q=65,536 captures both use `cell_size=0.015625`; the collapsed captures use `cell_size=16`. The earlier uniform performance archive used a different cell size and is **not** a matched-input timing comparison for this memory capture.

The Game Memory export also includes Metal resource creation and destruction events. In the mixed Q=65,536 BVH capture, `Resource Size` is absent from six allocation and seven deallocation events. Summing the available events would not reconstruct a complete live-resource maximum. The export did not expose a VM Tracker footprint table; its `virtual-memory` schema records page-fault intervals, not a resident-memory series. This does not establish whether VM Tracker was active in the recording configuration.

## Source-configured shader resources

The following is a static audit of the **actual profiled paths** at `N=1,000,000`, `k=16`. It is not a GPU occupancy measurement. The BVH host uses `group_size=128` for serial and split search, `64` for its microtree build, and `128` for its macro build; the `1,024`-thread FPS design is a different operator. The flat full-scan kNN host uses `256` threads: eight 32-lane SIMD groups, one query per SIMD group. It calls a device width probe and raises if the width is not 32; the BVH path does not make that SIMD-width assumption. See the [BVH host](../bench/spatial_bvh.py), [BVH shader](../bench/spatial_bvh.metal), [Morton host](../bench/spatial_keys.py), [flat host](../mps_pointops/_flat_search_mps.py), and [flat shader](../mps_pointops/kernels/flat_search.metal).

| Custom kernel | Host threads/group; groups at profiled size | Source-declared threadgroup arrays | Thread-private fixed arrays in source |
| --- | ---: | ---: | --- |
| `spatial_morton_keys_f32` (BVH build) | 256; 3,907 | 0 B | — |
| `bvh_build_micro_f32` | 64; 128 | 4,608 B | — |
| `bvh_build_macro_f32` | 128; 1 | 9,216 B | — |
| `bvh_knn_f32` (serial) | 128; 512 at Q=65,536, 32 at Q=4,096, 2 at Q=256 | 0 B | `float[32]`, `uint[32]`, `uint[32]` stack |
| `bvh_seed_f32` (split Q=256) | 128; 2 | 0 B | `float[32]`, `uint[32]` |
| `bvh_search_micro_f32` (split Q=256) | 128; 256 | 0 B | `float[32]`, `uint[32]`, `uint[16]` stack |
| `bvh_merge_micro_f32` (split Q=256) | 128; 2 | 0 B | `float[32]`, `uint[32]` |
| `flat_knn_indices` (full scan Q=65,536) | 256; 8,192 | 16,384 B | `float[8]`, `uint[8]` insertion scratch |

The BVH has 8,192 padded leaves and 128 microtrees at this point count. Its micro build declares two `float3[128]` arrays and one `uint[128]` array: `2 × 128 × 16 + 128 × 4 = 4,608` B. Macro build declares two `float3[256]` and one `uint[256]`: `9,216` B. A Metal `float3` array element occupies a 16-byte slot under the [Metal Shading Language layout](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf). Flat kNN declares two `[8][256]` four-byte arrays: `2 × 8 × 256 × 4 = 16,384` B even though the run requests `k=16`. These are **source-declared** sizes, not compiler-reported resource usage. Thread-private arrays may stay in registers or spill; their register count and the device's achieved occupancy were not measured. The BVH index build also calls PyTorch's stable `argsort`, whose internal kernels are outside this source audit.

## Separate Activity Monitor pilot

For the mixed-density Q=65,536 serial BVH fixture, a separate Activity Monitor trace yielded nine target-process `sysmon-process` rows. The largest **sampled process physical footprint** was **1,367,606,304 bytes (1,304.251 MiB)**; the largest **sampled resident size** was **399,966,208 bytes (381.438 MiB)**. The `activity-monitor-process-live` table also exposes the same maxima. These are observations from this pilot, not exact peaks or GPU-only allocations. Apple describes [VM footprint and resident size as different memory views](https://developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app).

## Separate GPU activity query matrix

A later **Metal System Trace** captured seven query cases in one launched Python process on the same M5 Pro, macOS 26.5.2, PyTorch 2.14.1, Safe Math, and disabled MPS CPU fallback. This run used clean source commit `6052590f9e21e31f8459e0669822fd6f122e55ca`, two warmups and five synchronized repetitions per case. All cases use `N=1,000,000` and `k=16`. The uniform full-scan row was added as a matched-fixture comparison; it was not in the earlier six-case memory archive. The [original seven query-run JSON files and sanitized manifest](evidence/instruments-m5pro-2026-10-03/query-runs/manifest.json) retain the fixture hashes, per-repeat host timestamps, and allocator readings. Their hashes are checked by the [extractor](evidence/instruments-m5pro-2026-10-03/extract_gpu_intervals.py).

The GPU column below is the **union of target-process GPU `Active` intervals that overlap each synchronized query call**, after clipping intervals to the host call bounds. This prevents overlapping interval rows from being counted twice. It includes any target GPU work during that call, including validation or copy work; it is not a named-kernel duration. Host time includes dispatch and `torch.mps.synchronize()`. All times are milliseconds; memory columns are MiB (`2²⁰` bytes). The tensor column is the maximum query-stage PyTorch allocator peak over five repetitions. The driver column is the maximum of the **before/after and per-repeat driver checkpoints from this same traced run**, not a transient peak.

| Fixture and path | GPU Active median (5) | GPU Active min–max | Host synchronized median | Query tensor peak | Driver checkpoint max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed density, Q=4,096, serial BVH | 57.339 | 57.261–58.947 | 58.531 | 20.563 | 1,032.688 |
| Mixed density, Q=65,536, serial BVH | 231.755 | 226.515–242.666 | 238.487 | 33.688 | 1,032.688 |
| Mixed density, Q=65,536, native full scan | 1,109.297 | 1,080.286–1,163.480 | 1,139.718 | 20.251 | 1,032.688 |
| Uniform, Q=65,536, serial BVH | 184.552 | 184.153–185.920 | 191.572 | 33.688 | 1,032.688 |
| Uniform, Q=65,536, native full scan | 1,145.668 | 1,132.609–1,155.064 | 1,193.467 | 20.251 | 1,032.688 |
| Collapsed, Q=256, serial BVH | 228.096 | 228.062–228.195 | 228.789 | 19.743 | 1,032.703 |
| Collapsed, Q=256, split BVH | 9.131 | 8.908–9.374 | 10.102 | 26.119 | 1,032.688 |

The [PID-filtered numeric export](evidence/instruments-m5pro-2026-10-03/gpu-query-intervals.json) records all 35 host windows, GPU interval unions and sums, matched generic host encoder timings, boundary clipping, the target-row count, input hashes, and profiler-table row counts. It omits PIDs, absolute paths, and data from other applications. The sanitized [query manifest](evidence/instruments-m5pro-2026-10-03/query-runs/manifest.json) also preserves a canonical SHA-256 digest, file count, and total bytes for the private query trace bundle without publishing its internal file list; the extractor checks that digest. The trace start time is exported only to millisecond precision; shifting its origin by ±1 ms changed any reported per-repeat GPU union by at most 0.440 ms. This is an alignment sensitivity check, not a general timing error bound.

The target shader list contains the expected BVH and full-scan function names, and the host and GPU records match by command-buffer and encoder ID. GPU records label those encoders generically, without a shader-name binding. The shader-profiler sample and interval tables contain zero rows. Thus even the long serial BVH Compute encoder cannot be published as a directly measured `bvh_knn_f32` duration, and **GPU occupancy remains unmeasured**. The [counter availability record](evidence/instruments-m5pro-2026-10-03/counter-availability.json) documents that the two tested counter configurations were rejected on this setup; it does not establish that all counters are unsupported.

This trace uses a different commit and profiling configuration from the six Game Memory captures and the earlier unprofiled spatial benchmark. It supports a within-trace comparison of these seven instrumented queries; its timing should not replace the unprofiled benchmark latency or be joined with the separate Activity Monitor memory run.

## Provenance and remaining limits

The published [`trace-tree-manifest.json`](evidence/instruments-m5pro-2026-10-03/trace-tree-manifest.json) lists SHA-256, byte size, and relative path for every file in the six Game Memory bundles and one Activity Monitor bundle. Each bundle digest is SHA-256 of a canonical JSON list of those entries. [`extract_xctrace.py`](evidence/instruments-m5pro-2026-10-03/extract_xctrace.py) documents the exact target-PID table selection and digest algorithm. The later query trace is bound separately by the sanitized query manifest's digest. The raw trace bundles remain private because they can contain unrelated process and device metadata; the published numeric JSON omits PIDs, absolute paths, and unrelated-process data. The local allocator JSON hashes and source-file hashes are retained in the published numeric record.

The source-pinned [allocator report](spatial-memory-v090.md) at `62fcc0f...` is an earlier run; its `1,036.156 MiB` sampled driver value must not be substituted for the six Game Memory captures' `1,036.141 MiB`. Instruments and the poller add overhead, so the traced times are **not** replacements for unprofiled search benchmarks. These captures do not provide an exact whole-device physical GPU-memory peak, GPU occupancy, or per-kernel GPU duration. Those require separately validated instruments/counter captures before publication.

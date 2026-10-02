# v0.9 spatial search: MPS memory measurement

## What the numbers mean

The measurement program is [`bench/measure_spatial_memory.py`](../bench/measure_spatial_memory.py).
It runs **one path per fresh process** and separates CPU-to-MPS input transfer,
BVH index build, and search. Before each stage it synchronizes the MPS device
and resets the [PyTorch accelerator peak allocator statistics](https://docs.pytorch.org/docs/stable/generated/torch.accelerator.memory.max_memory_allocated.html).
`allocator_peak.tensor_bytes` is the maximum live tensor allocation seen by
PyTorch's allocator *during that stage*, including temporary tensors that
have been freed before the stage ends. `allocator_peak.reserved_bytes` is
memory managed by PyTorch's caching allocator. These are peaks for **that
allocator**, not the total GPU or process physical-memory peak.

The program additionally polls [`torch.mps.current_allocated_memory()`](https://docs.pytorch.org/docs/stable/generated/torch.mps.current_allocated_memory)
and [`torch.mps.driver_allocated_memory()`](https://docs.pytorch.org/docs/main/generated/torch.mps.driver_allocated_memory.html)
at a requested 0.5 ms interval. The first excludes cached MPS allocator
blocks. The second includes cached allocator blocks and MPS/MPSGraph
allocations. Its largest *sample* may miss short transients; it is not an
exact peak. Recorded maximum sampling gaps show the actual coverage. No
counter in this run measures the whole system's instantaneous GPU memory
residency or Apple Silicon's unified-memory physical footprint.

The tiny warmup compiles the selected path, then `torch.mps.empty_cache()`
releases unused cached blocks. The script records the remaining baseline,
keeps reference/query tensors alive across stages, and synchronizes before
sampling and after every stage. The stage wall times include sampling
overhead and **must not** replace the performance results in the spatial
benchmark reports.

## M5 Pro observations, 2026-10-02

Environment: Apple M5 Pro, 48 GiB unified memory, macOS 26.5.2,
PyTorch 2.14.1, `PYTORCH_ENABLE_MPS_FALLBACK=0`,
`PYTORCH_MPS_FAST_MATH=0`. Each row uses a fresh Python process.
For mixed density, `N=1,000,000`, 90% of points are in a side-one cube and
the rest in a side-1024 background; half of the queries are sampled from
each region. The collapsed case makes all points and queries coincident.
All rows have `k=16`, seed `20261002`. Values are MiB (2²⁰ bytes).

| Fixture and path | Input transfer tensor peak | Index-build tensor peak | Query tensor peak | Overall tensor peak | Sampled driver maximum |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mixed, Q=4,096, serial BVH | 11.548 | 59.001 | 20.563 | 59.001 | 1,036.156 |
| Mixed, Q=65,536, serial BVH | 12.251 | 59.705 | 33.688 | 59.705 | 1,036.156 |
| Mixed, Q=65,536, native Metal full scan | 12.251 | — | 20.251 | 20.251 | 1,036.156 |
| Uniform, Q=65,536, serial BVH | 12.251 | 59.705 | 33.688 | 59.705 | 1,036.156 |
| Collapsed, Q=256, serial BVH | 11.504 | 58.958 | 19.743 | 58.958 | 1,036.156 |
| Collapsed, Q=256, split BVH | 11.504 | 58.958 | 26.119 | 58.958 | 1,036.156 |

For the mixed Q=65,536 run, persistent BVH-owned index tensors occupy
8.067 MiB, while its returned `(distance, index, stats)` tensors occupy
13.250 MiB. The 59.705 MiB build peak includes sorting temporaries, the
resident points/query tensors, and the index; it is **not** a claim that the
index alone costs 59.705 MiB. The native full-scan path returns only the
index tensor (8.000 MiB). On the collapsed Q=256 fixture, split traversal's
query-stage peak exceeds serial traversal by 6.376 MiB because it allocates
per-microtree scratch. An endpoint-only sample misses part of that scratch:
the split query's sampled live-tensor maximum was 19.743 MiB, while the
allocator peak counter recorded 26.119 MiB.

All six runs reached an allocator **reserved** peak of 1,032.000 MiB.
The driver counter's largest sample was 1,036.156 MiB, mostly during input
transfer. These values include reusable allocations and cannot be added to
the live-tensor number or described as BVH-specific additional memory.
The cause of the roughly 1 GiB reservation needs a Metal resource trace;
the counter alone does not attribute it to the BVH, a transfer, or a
framework cache.

The six [raw JSON records](../bench/results/spatial-memory-v090-m5pro-clean/)
contain integer byte counts and per-stage sample coverage. All six record
the same measurement-script SHA-256 prefix `3595381c861a` and BVH shader
SHA-256 prefix `e466dd288013`; full hashes are in each JSON. The measurement
program's peak counter catches short-lived allocations that its sampler can
miss, while the driver counter remains sampled only. With the same N, Q and
K, the uniform and mixed fixtures had identical observed tensor and driver
peaks; this is an observation for those two generated inputs, not a general
density-independence claim.

These observations used clean Git commit
`62fcc0f295e127be47a475fbc122d4de686b4c58`. All six records have
`source_dirty=false` and identical source SHA-256 maps; the recorded hashes
also match the measurement program and spatial/native source files at that
commit. The numbers are M5 Pro allocator measurements, not a whole-device
GPU-memory peak. No Instruments capture or M1 measurement was made in that
2026-10-02 run. The subsequent [M5 Pro Instruments study](spatial-instruments-v090.md)
adds six target-process Metal allocation traces and a separate sampled
process-footprint pilot; the original table remains unchanged.

## Reproduction

Use a new output name for each run; the script refuses to overwrite evidence.
The project environment must contain PyTorch with the unified accelerator
memory APIs. They were verified here on 2.14.1. Run paths sequentially so
they do not compete for the GPU or unified memory.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/measure_spatial_memory.py \
  --path bvh-serial --distribution cluster-sparse \
  --points 1000000 --queries 65536 --k 16 --cell-size 0.015625 \
  --output bench/results/memory-mixed-q65536-bvh.json

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/measure_spatial_memory.py \
  --path native-full-scan --distribution cluster-sparse \
  --points 1000000 --queries 65536 --k 16 --cell-size 0.015625 \
  --output bench/results/memory-mixed-q65536-native.json
```

Use `--distribution collapsed --queries 256 --cell-size 16` for the serial
versus split comparison; set `--path bvh-split` for the split run. The
`sample_count`, maximum actual sample gap, first/last value, peak counter,
fixture, tool versions, Git state, and source hashes are in each JSON.

## Metal resource and physical-footprint gate

Apple documents [Game Memory in Instruments](https://developer.apple.com/documentation/xcode/analyzing-the-memory-usage-of-your-metal-app)
with **Metal Resource Events** for buffer allocation/deallocation and
**VM Tracker** for process footprint over time. The
[Metal system trace](https://developer.apple.com/metal/tools/) can also show
GPU and memory activity. For an independent total-Metal-resource and physical
footprint report:

1. Install full Xcode with Instruments and confirm the Game Memory template
   exists. The 2026-10-03 follow-up used Xcode 26.6 with a launchable debug
   interpreter. Its source-pinned traces and metric limits are documented in
   the [Instruments report](spatial-instruments-v090.md).
2. Start a fresh run of the same script with `--pause-before-s 30
   --pause-after-s 30`; it prints `INSTRUMENTS_ATTACH_READY` and its PID
   after warmup. Attach Instruments to that Python PID, select **Game
   Memory**, and begin recording before the input-transfer stage starts.
3. Inspect Metal allocations over transfer, build, and query. Validate which
   process-memory columns the selected instrument actually exports. The
   follow-up's Activity Monitor pilot exposed sampled physical footprint and
   resident size; its Game Memory export did not expose a VM Tracker footprint
   table. Preserve raw traces privately, publish target-only numeric rows,
   and record tool versions, JSON and source hashes. Do not treat the
   `virtual-memory` page-fault table as a footprint series.
4. Repeat on a clean release commit and on M1 when that device is available.
   A resource-allocation peak, PyTorch allocator peak, driver allocation,
   and resident physical footprint are different quantities; label each
   separately in the release report.

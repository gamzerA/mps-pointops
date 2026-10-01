# Compact voxel API benchmark: physical M5 Pro

The [benchmark runner](../bench/bench_voxel_api.py) measures the experimental
[`mps_pointops.voxel`](voxel-api-contract.md) API on a physical Apple M5 Pro.
Raw per-iteration timings and per-case memory records are in the [Safe Math
JSON](bench-voxel-api-m5pro-torch214-safe-2026-10-02.json) and [Fast Math
JSON](bench-voxel-api-m5pro-torch214-fast-2026-10-02.json). This is a compact
voxelization and feature-downsampling workload, not PyG `voxel_grid`, graph
`avg_pool`, or a fused Metal operator.

## Reproduction and fixture

- Source commit with the benchmark script: `4cd02b3fca3ba94967c140f229f080a213a49475`.
  Script SHA-256: `51d44007ad35d65965e393925f108922dd7d333fedc3f213af988b109c4f04f7`.
  `mps_pointops/voxel.py` SHA-256:
  `523b56bbcbd85afe35dd131da7f519eca033fe15b0a9bc088b59dc6e76ab00ca`.
- Physical Apple M5 Pro, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1.
  Installed-distribution name/version SHA-256:
  `7663844160b75808375ef103c2dd5e8d87471f3bac47f28deab6c90e2ec0f8ae`.
  The JSON records the executable basename, OS, source hashes, and mode. The
  historical M5 Pro JSON has its local executable path redacted to its basename
  after capture; only this environment metadata changed, not measurements.
- Two separate Safe/Fast processes with `PYTORCH_ENABLE_MPS_FALLBACK=0` and
  `PYTORCH_MPS_FAST_MATH=0` or `1`. Each process completed all 24 combinations:
  `N=20,000/100,000/500,000`, uniform/ragged four-batch sizes, dense/sparse
  cell occupancy, and CPU/MPS. Each case has two warmups and five measured
  iterations in its own worker process. The worker separation makes process
  peak RSS specific to that case. The Safe/Fast input SHA-256 values match for
  every corresponding case.
- The fixed fixture uses finite float32 `(N,3)` positions, float32 `(N,32)`
  features, cell size `0.5`, batch IDs `[0,3,11,29]`, and seed `2701`.
  Points are globally shuffled, so batches are unsorted. Uniform lengths are
  `N/4` each; ragged lengths are `1%`, `10%`, `25%`, and the remainder. Dense
  cells have up to 16 points; sparse cells have one. Dyadic cell coordinates
  and interior offsets avoid float32 boundary ambiguity. The worker checks
  exact expected voxel count, total point count, and finite gradients outside
  the timed region.

Run from the repository root in an environment with PyTorch and optional
`psutil` for resident-memory sampling. Without `psutil`, the runner uses
macOS `ps` for current RSS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_voxel_api.py \
  --output docs/bench-voxel-api-m5pro-torch214-safe-2026-10-02.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/bench_voxel_api.py \
  --output docs/bench-voxel-api-m5pro-torch214-fast-2026-10-02.json
```

## Timing definition

All numbers below are **median wall milliseconds** from the five raw samples.
The stage timers call `torch.mps.synchronize()` before and after each stage:
`voxelize` forward; fixed-map aggregation forward; and aggregation backward.
The aggregation stage reproduces the `index_add_` and count-division code in
`voxel_downsample` with a precomputed compact map. It is an attribution probe,
not a separately exported API.

Three separate end-to-end series time `voxel_downsample` forward, backward
through a graph constructed before the timer, and a complete forward plus
scalar loss plus backward step. The complete step has one synchronization
before and after the whole step, with **no explicit synchronization between
forward and backward**. The API itself still synchronizes when checking
scalar predicates and producing a data-dependent voxel count. The three
series are independent; their medians should not be added or subtracted.
Inputs are already on the measured device before timing.

| N | Batches | Occupancy | Voxels | Safe CPU / MPS full step ms | Fast CPU / MPS full step ms |
| ---: | --- | --- | ---: | ---: | ---: |
| 20,000 | uniform | dense | 1,252 | 3.27 / 5.65 | 3.49 / 6.45 |
| 20,000 | uniform | sparse | 20,000 | 3.99 / 7.42 | 3.95 / 9.98 |
| 20,000 | ragged | dense | 1,251 | 3.83 / 7.36 | 3.14 / 9.60 |
| 20,000 | ragged | sparse | 20,000 | 3.88 / 8.38 | 4.42 / 11.37 |
| 100,000 | uniform | dense | 6,252 | 16.02 / 3.60 | 15.16 / 6.95 |
| 100,000 | uniform | sparse | 100,000 | 18.25 / 18.18 | 19.47 / 16.98 |
| 100,000 | ragged | dense | 6,251 | 14.83 / 5.67 | 15.30 / 4.23 |
| 100,000 | ragged | sparse | 100,000 | 19.85 / 12.94 | 19.06 / 13.90 |
| 500,000 | uniform | dense | 31,252 | 78.42 / 21.73 | 83.79 / 23.09 |
| 500,000 | uniform | sparse | 500,000 | 107.96 / 31.26 | 100.38 / 39.09 |
| 500,000 | ragged | dense | 31,251 | 79.52 / 23.24 | 80.01 / 22.37 |
| 500,000 | ragged | sparse | 500,000 | 98.11 / 33.93 | 95.48 / 35.52 |

Stage timing for the 500,000-point uniform fixtures illustrates where work
occurs. Full stage samples for all rows are in the JSON files.

| Mode | Occupancy | Device | `voxelize` forward ms | Aggregate forward ms | Aggregate backward ms |
| --- | --- | --- | ---: | ---: | ---: |
| Safe | dense | CPU | 65.68 | 6.24 | 1.44 |
| Safe | dense | MPS | 12.53 | 1.10 | 0.78 |
| Safe | sparse | CPU | 78.91 | 8.27 | 8.65 |
| Safe | sparse | MPS | 19.69 | 5.12 | 10.93 |
| Fast | dense | CPU | 67.91 | 7.06 | 1.73 |
| Fast | dense | MPS | 15.20 | 1.45 | 3.60 |
| Fast | sparse | CPU | 80.88 | 8.43 | 8.60 |
| Fast | sparse | MPS | 10.14 | 2.81 | 11.77 |

The measured MPS full-step range was broad: for 500,000 uniform dense points,
Safe samples spanned **9.45–34.66 ms**; for 500,000 uniform sparse points,
**19.73–62.70 ms**. In the 20,000-point cases, MPS was slower than CPU in
these runs. The 500,000-point medians favor MPS in this fixture, but Safe/Fast
ordering and small differences are not stable conclusions from one five-sample
run. Device contention, cache state, and thermal state were not controlled.

## Memory definition

`rss_current_bytes` is the process resident set at the named checkpoint;
`rss_process_peak_bytes` is the process high-water mark from `getrusage`.
The baseline is taken after importing PyTorch but before generating inputs.
The case-local worker makes that peak specific to one matrix row. MPS also
reports `current_allocated_memory()` (live tensors managed by PyTorch) and
`driver_allocated_memory()` (driver allocation including cache) at checkpoints.
`observed_mps_boundary_max_bytes` is only the largest checkpoint reading; it
is **not** a true transient device-memory peak. GPU allocations and process RSS
can overlap on unified memory, so these columns should not be summed.

Safe Math, 500,000 uniform points, MiB (`2^20` bytes):

| Occupancy | Device | Current RSS after timing | Process peak RSS | MPS current allocated after timing | MPS driver allocated after timing |
| --- | --- | ---: | ---: | ---: | ---: |
| dense | CPU | 459.0 | 459.0 | — | — |
| dense | MPS | 487.1 | 487.1 | 139.4 | 1,064.7 |
| sparse | CPU | 744.7 | 744.7 | — | — |
| sparse | MPS | 486.5 | 486.5 | 139.4 | 1,064.7 |

The MPS driver allocation remained cached after tensor cleanup in these
workers. The JSON preserves baseline, after-inputs, after-timing, and
after-cleanup current and peak readings for all cases. A physical M1 was not
measured: no usable M1 SSH target was configured for noninteractive BatchMode
access during this run. The physical M1 8 GiB memory and latency behavior
cannot be inferred from this M5 Pro result.

# Compact voxel API benchmark: physical M1 and M5 Pro

The [benchmark runner](../bench/bench_voxel_api.py) measures the experimental
[`mps_pointops.voxel`](voxel-api-contract.md) API on physical Apple M1 and M5 Pro
machines. Raw per-iteration timings and per-case memory records are in the
M1 [Safe Math](bench-voxel-api-m1-torch212-safe-2026-10-02.json) and [Fast
Math](bench-voxel-api-m1-torch212-fast-2026-10-02.json) JSON files, and the
M5 Pro [Safe Math](bench-voxel-api-m5pro-torch214-safe-2026-10-02.json) and
[Fast Math](bench-voxel-api-m5pro-torch214-fast-2026-10-02.json) JSON files.
This is a compact voxelization and feature-downsampling workload, not PyG
`voxel_grid`, graph `avg_pool`, or a fused Metal operator. The runner calls
`voxel_downsample` with its default `index_add` pooling backend.

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
- Physical Apple M1, macOS 26.5.2, Python 3.10.6, PyTorch 2.12.0. The M1
  source commit is `dae1bd3c95f0d068ae6ade6d4aaacaa34e8f27af`;
  benchmark-script SHA-256 is the same as M5 Pro. Its `voxel.py` SHA-256 is
  `15d8b88ecf4a7387598fdd6ca71feab352833bb120d1c569f510f7b5c9ac896a`,
  and installed-distribution name/version SHA-256 is
  `74686f2286d4269240fd410fca350e685831aa3b590182d2518c69b86d636361`.
  The M1 source includes the later opt-in fused CSR prototype, but this runner
  did not select it. M1 and M5 Pro differ in source commit and PyTorch version,
  so the cross-device figures are descriptive, not a controlled chip-only
  speed comparison. The public M1 JSONs replace the executable path with its
  basename and append provenance fields containing the original capture
  SHA-256; all measurements are unchanged.
- Two separate Safe/Fast processes with `PYTORCH_ENABLE_MPS_FALLBACK=0` and
  `PYTORCH_MPS_FAST_MATH=0` or `1`. Each process completed all 24 combinations:
  `N=20,000/100,000/500,000`, uniform/ragged four-batch sizes, dense/sparse
  cell occupancy, and CPU/MPS. Each case has two warmups and five measured
  iterations in its own worker process. The worker separation makes process
  peak RSS specific to that case. The Safe/Fast input SHA-256 values match for
  every corresponding case on each machine. The corresponding M1 and M5 Pro
  input SHA-256 values also match for all 24 cases.
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

Use the M1 filenames above instead when collecting the same matrix on M1.
The published M1 Safe and Fast captures contain all 24 canonical cases,
with no resumed or merged rows. Their unredacted SHA-256 values are
`e45b7c5b529b4f42972765dbe20307427e94bbe2b68da096c4a006afddbfa9a3`
and `bb8db2e24394ebe9b69e9a26983811bf34988938cb3ad89f4f4304e3aae8f5da`,
respectively. The raw captures retain a local executable path; the public
copies remove that path and preserve every case and measured sample.

The runner writes its output after every case but does not resume an existing
file. Keep an interrupted JSON unchanged, rerun missing cases with narrowed
`--nodes`, `--batch-shapes`, `--occupancies`, and `--devices` values into new
files, then use [`merge_voxel_api_fragments.py`](../bench/merge_voxel_api_fragments.py)
to validate and assemble the canonical 24-case matrix. The merge tool rejects
different source or environment metadata, missing or reordered cases, and
CPU/MPS input-hash mismatches; it preserves the input files and records their
SHA-256 values in the assembled result.

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

### M5 Pro full step

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

### M1 full step

| N | Batches | Occupancy | Voxels | Safe CPU / MPS full step ms | Fast CPU / MPS full step ms |
| ---: | --- | --- | ---: | ---: | ---: |
| 20,000 | uniform | dense | 1,252 | 5.13 / 12.65 | 5.13 / 12.91 |
| 20,000 | uniform | sparse | 20,000 | 7.10 / 15.12 | 7.24 / 15.23 |
| 20,000 | ragged | dense | 1,251 | 5.04 / 13.41 | 5.08 / 12.48 |
| 20,000 | ragged | sparse | 20,000 | 6.92 / 15.35 | 6.88 / 15.26 |
| 100,000 | uniform | dense | 6,252 | 24.41 / 21.30 | 24.48 / 23.46 |
| 100,000 | uniform | sparse | 100,000 | 37.55 / 34.48 | 37.04 / 37.31 |
| 100,000 | ragged | dense | 6,251 | 23.42 / 22.51 | 23.45 / 24.47 |
| 100,000 | ragged | sparse | 100,000 | 36.43 / 37.21 | 36.87 / 36.92 |
| 500,000 | uniform | dense | 31,252 | 134.30 / 69.22 | 133.66 / 65.96 |
| 500,000 | uniform | sparse | 500,000 | 202.61 / 127.19 | 232.22 / 134.96 |
| 500,000 | ragged | dense | 31,251 | 136.89 / 85.02 | 137.17 / 66.21 |
| 500,000 | ragged | sparse | 500,000 | 199.21 / 130.59 | 198.16 / 129.57 |

On M1, CPU led in all four 20,000-point fixtures. The 100,000-point
results straddled parity: MPS led in three of four Safe cases and one of four
Fast cases. MPS led in all four 500,000-point fixtures in both math modes,
with CPU/MPS median ratios from 1.53 to 2.07. This is a conditional crossover
for the tested fixtures, not a general speedup claim. A five-sample median
cannot establish thermal or run-to-run stability.

Stage timing for the 500,000-point uniform fixtures illustrates where work
occurs. Full stage samples for all rows and both machines are in the JSON files.

M5 Pro:

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

M1:

| Mode | Occupancy | Device | `voxelize` forward ms | Aggregate forward ms | Aggregate backward ms |
| --- | --- | --- | ---: | ---: | ---: |
| Safe | dense | CPU | 105.34 | 11.29 | 6.52 |
| Safe | dense | MPS | 31.72 | 16.98 | 3.65 |
| Safe | sparse | CPU | 120.80 | 39.27 | 30.63 |
| Safe | sparse | MPS | 35.11 | 60.97 | 23.63 |
| Fast | dense | CPU | 105.13 | 11.16 | 6.96 |
| Fast | dense | MPS | 31.76 | 16.22 | 4.99 |
| Fast | sparse | CPU | 117.95 | 37.77 | 34.58 |
| Fast | sparse | MPS | 34.15 | 61.06 | 24.00 |

For M1 sparse cells, fixed-map aggregation forward was slower on MPS than
CPU even at 500,000 points. The full-step gain in that fixture came from the
voxelization portion; the independent stage medians must not be summed to
reconstruct full-step time.

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

M5 Pro Safe Math, 500,000 uniform points, MiB (`2^20` bytes):

| Occupancy | Device | Current RSS after timing | Process peak RSS | MPS current allocated after timing | MPS driver allocated after timing |
| --- | --- | ---: | ---: | ---: | ---: |
| dense | CPU | 459.0 | 459.0 | — | — |
| dense | MPS | 487.1 | 487.1 | 139.4 | 1,064.7 |
| sparse | CPU | 744.7 | 744.7 | — | — |
| sparse | MPS | 486.5 | 486.5 | 139.4 | 1,064.7 |

M1 Safe Math, 500,000 uniform points, MiB (`2^20` bytes):

| Occupancy | Device | Current RSS after timing | Process peak RSS | MPS current allocated after timing | MPS driver allocated after timing |
| --- | --- | ---: | ---: | ---: | ---: |
| dense | CPU | 516.8 | 516.8 | — | — |
| dense | MPS | 169.4 | 373.1 | 137.3 | 1,188.6 |
| sparse | CPU | 765.6 | 765.6 | — | — |
| sparse | MPS | 299.0 | 424.2 | 137.3 | 1,196.6 |

On M1, the largest observed MPS checkpoint values in these rows were 185.0
MiB of current PyTorch allocation (sparse) and 1,196.6 MiB of driver allocation
(both occupancies). These are checkpoint maxima, not transient VRAM peaks. The MPS
driver allocation remained cached after tensor cleanup in these workers. The
JSON files preserve baseline, after-inputs, after-timing, and after-cleanup
current and peak readings for all cases. Neither RSS nor the reported driver
allocation directly measures total physical unified-memory pressure.

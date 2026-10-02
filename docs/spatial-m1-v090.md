# Physical M1 spatial search study (v0.9 development source)

This is a reproducible **development-branch measurement**, not a v0.9 or
v1.0 release claim. The physical machine was an Apple M1 MacBook Pro with
8 GiB unified memory, macOS 26.5.2. All 34 JSON records identify clean source
commit `ef07a9337b40065c18a6c950bda4240c789f31f1`, PyTorch 2.14.1,
`PYTORCH_ENABLE_MPS_FALLBACK=0`, and `PYTORCH_MPS_FAST_MATH=0`. SciPy
1.15.3 was used for the separate CPU `cKDTree` measurements. The M1 had
about 40–44% free RAM and approximately 2.3 GiB swap in use at the initial
inventory; that background state may affect latencies.

The [34 unmodified JSON files](../bench/results/spatial-v090-m1/),
[spatial Safe/Fast pytest logs](evidence/spatial-v090-m1/), and
[Chamfer Safe/Fast pytest logs](evidence/chamfer-v090-m1/) are archived with a
[SHA-256 manifest](evidence/spatial-v090-m1/SHA256SUMS.txt). Run
`python bench/validate_spatial_m1_evidence.py` to verify every archived byte,
clean source revision, 21 recorded source-file hashes, and benchmark parity
counters. The report branch starts from `905fce5384c681b16db0ebd0fbcd892635544ead`, whose source tree is
identical to the measured `ef07a933` tree; the JSONs retain the **measured**
commit. Files were generated on the M1 in fresh processes, saved outside its
source checkout, then copied without editing.

## Correctness and test scope

The M1 Safe Math run of `tests/test_spatial_public.py`,
`tests/test_spatial_bvh.py`, and `tests/test_spatial_bvh_radius.py` recorded
**31 passed, 1 skipped** on PyTorch 2.12.0. The skipped test specifically
requires isolated Fast Math. A separate Fast Math run recorded **5 passed,
27 skipped**; the Safe-only BVH contract tests were intentionally skipped in
Fast Math. These are focused spatial tests, not a full package or release CI
matrix. Both runs disabled MPS CPU fallback. The performance and allocator
records used PyTorch 2.14.1 because its accelerator peak-memory counters ran
on this M1; the 2.12.0 `reset_peak_memory_stats()` attempt raised a PyTorch
allocator assertion. That failed counter attempt did not produce a benchmark
JSON and is not silently included in the table.

The separate M1 Chamfer Safe and Fast Math runs each recorded **76 passed,
2 skipped** on PyTorch 2.12.0 with MPS CPU fallback disabled. Both skipped
tests require PyTorch3D, which was absent from that M1 environment. These
logs establish only this project's internal regression behavior on M1; they
do **not** establish upstream PyTorch3D parity or close the full Chamfer
compatibility gate.

Across **13 public kNN-dispatch records**, all BVH and automatic outputs
matched the native full-scan indices, with zero recorded selected-distance
error. Across **12 dense Ball Query-dispatch records**, BVH first-call and
steady-call outputs matched the full scan: zero index mismatches, zero
squared-distance bit mismatches, and zero original-index order violations.
The 1M-point matrix includes uniform, mixed dense-cluster/sparse-background,
and all-coincident distributions. For Ball Query, the uniform and mixed
fixtures contain both empty and full result rows, exercising padding and
first-`K` selection. These checks cover the generated finite float32 inputs,
not every numerical boundary or public API combination. In five additional
research-BVH versus native-scan comparisons, index mismatches and BVH
fallback counts were zero. The CPU `cKDTree` timings below **do not** establish
its index parity on ties; the benchmark checks BVH against the package's
native-scan oracle.

## Synchronized dispatch timings

Each cell is the median of **three** synchronized host-wall steady-query
samples in milliseconds, with inputs already on MPS; shader compilation and
input transfer are excluded. `N=1,000,000`, `K=16`, seed `20261002` for all
rows. kNN queries are sampled reference points. Ball Query uses independent
queries with `r=12` for uniform, `r=0.03` for mixed, and `r=0.5` for
collapsed data. The mixed fixture puts 90% of points in a dense side-one
cube and the rest in a side-1024 background. `scan` and `BVH` are explicit
paths. **Automatic dispatch chose scan on every M1 fixture**, so BVH timing
must not be described as the current automatic public API's speed.

| Operator / distribution | Q | Scan ms | Explicit BVH ms | BVH / scan |
| --- | ---: | ---: | ---: | ---: |
| kNN / uniform | 4,096 | 303.039 | 77.891 | 0.257 |
| kNN / uniform | 8,192 | 625.136 | 141.559 | 0.226 |
| kNN / uniform | 65,536 | 4,762.511 | 784.613 | 0.165 |
| kNN / mixed | 4,096 | 323.641 | 138.538 | 0.428 |
| kNN / mixed | 8,192 | 624.466 | 146.905 | 0.235 |
| kNN / mixed | 65,536 | 4,867.692 | 901.545 | 0.185 |
| kNN / collapsed | 4,096 | 313.693 | 504.947 | 1.610 |
| kNN / collapsed | 8,192 | 598.371 | 1,098.874 | 1.837 |
| kNN / collapsed | 65,536 | 4,753.238 | 6,038.436 | 1.270 |
| Ball Query / uniform | 4,096 | 721.723 | 14.218 | 0.020 |
| Ball Query / uniform | 8,192 | 1,437.850 | 25.224 | 0.018 |
| Ball Query / uniform | 65,536 | 11,461.166 | 159.031 | 0.014 |
| Ball Query / mixed | 4,096 | 419.306 | 24.852 | 0.059 |
| Ball Query / mixed | 8,192 | 830.372 | 30.838 | 0.037 |
| Ball Query / mixed | 65,536 | 6,595.015 | 204.420 | 0.031 |
| Ball Query / collapsed | 4,096 | 363.852 | 1.407 | 0.004 |
| Ball Query / collapsed | 8,192 | 721.401 | 1.485 | 0.002 |
| Ball Query / collapsed | 65,536 | 5,736.899 | 2.367 | <0.001 |

For kNN, the 1M-point **single-sample staging** runs at `Q=256` (scan
16.555 ms, BVH 35.676 ms) and `Q=2,048` (162.119 ms, 72.827 ms) bound a
uniform-fixture crossover between those query counts; they do not locate a
statistically precise threshold. The collapsed kNN data are a counterexample
to a universal BVH speed claim. Ball Query benefits strongly in these
fixtures, but its current automatic M1 policy still selects scan. First-call
construction and wrapper times, transfer, every raw sample, and allocator
snapshots are in the JSONs; steady-query figures must not be quoted as
one-shot or end-to-end latency.

## CPU `cKDTree` baseline

The separate [research-BVH benchmark](../bench/bench_spatial_bvh.py) measured
both CPU tree construction and query, and a private BVH construction and
query, each as medians of three synchronized samples. These use the same
`N=1,000,000`, `K=16`, seed and distribution family, but are **not the same
wrapper/timing program** as the dispatch table. Addition of separate medians
is an illustrative build-plus-one-query comparison, not the median of paired
end-to-end samples. Units are milliseconds. GPU inputs are preloaded; CPU
`cKDTree` starts from CPU arrays. Neither total includes MPS input transfer.

| Distribution | Q | BVH build | BVH query | BVH sum | cKDTree build | cKDTree query | cKDTree sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 4,096 | 11.127 | 76.832 | 87.959 | 365.583 | 17.772 | 383.355 |
| Uniform | 65,536 | 10.088 | 787.239 | 797.327 | 369.452 | 277.522 | 646.974 |
| Mixed | 4,096 | 10.274 | 138.189 | 148.463 | 362.545 | 17.678 | 380.223 |
| Mixed | 65,536 | 10.383 | 907.154 | 917.537 | 358.587 | 284.674 | 643.261 |
| Collapsed | 4,096 | 10.349 | 549.324 | 559.673 | 39.347 | 9,021.408 | 9,060.755 |

The BVH wins this **one-query-after-build** comparison at `Q=4,096` for
these three distributions, but loses to `cKDTree` at `Q=65,536` in uniform
and mixed data. On reuse, `cKDTree`'s query alone is faster in the uniform
and mixed fixtures. The collapsed case reverses this because its CPU query
is extremely slow with all points coincident. These observations do not
justify a blanket CPU-library speed claim; processor load, memory state and
additional hardware repetitions need controlling before such a claim. No
measurement here includes a full model or data-loading pipeline.

## M1 memory counters and open physical-peak gate

Each [memory record](../bench/results/spatial-v090-m1/) used a fresh Python
process and isolated transfer, index-build, and query stages. PyTorch's
allocator peak catches transient **PyTorch tensor** allocations; the MPS
driver counter was polled and can miss short-lived allocations. Values are
MiB. They are not whole-device physical GPU peaks.

| Fixture / path | Allocator live-tensor peak | Largest sampled driver allocation |
| --- | ---: | ---: |
| Uniform, Q=65,536, serial BVH | 59.705 | 1,035.844 |
| Uniform, Q=65,536, native full scan | 20.251 | 1,035.844 |
| Mixed, Q=65,536, serial BVH | 59.705 | 1,035.844 |
| Collapsed, Q=256, split BVH | 58.958 | 1,036.094 |

The similar driver figures include PyTorch's roughly 1 GiB cached allocator
reservation; they cannot be added to live-tensor figures or attributed to BVH
index storage. The M1 had only Xcode Command Line Tools; `xcrun --find
xctrace` could not find Instruments. Consequently no Metal Resource Events,
VM Tracker trace, or independent physical-memory peak was recorded. The
[Instruments capture procedure](spatial-memory-v090.md#metal-resource-and-physical-footprint-gate)
remains an open release gate. The measured M1 timings and allocator counters
close only the stated fixtures; they do not close the v0.9/v1.0 release plan.

## Reproduction and remaining work

Use a clean checkout of `ef07a933` on physical M1, install PyTorch 2.14.1,
NumPy 2.2.6, SciPy 1.15.3, and run the benchmark scripts under Safe Math with
CPU fallback disabled. For example:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_spatial_dispatch.py \
  --distribution uniform --points 1000000 --queries 4096 \
  --k 16 --repeats 3 --output /tmp/knn-uniform-m1.json

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_spatial_radius_dispatch.py \
  --distribution uniform --points 1000000 --queries 4096 \
  --limit 16 --radius 12 --repeats 3 --output /tmp/radius-uniform-m1.json

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_spatial_bvh.py \
  --distribution uniform --points 1000000 --queries 4096 \
  --k 16 --cell-size 16 --repeats 3 --output /tmp/ckdtree-uniform-m1.json
```

Use fresh processes and unique output paths, one fixture at a time. On the
8 GiB M1, stage from 20k/256 through 100k/256 before 1M and cap each process
runtime; only raise Q after preceding runs finish without a watchdog or OOM.
The 1M/Q=65,536 results were obtained this way. Re-run the complete
cross-device matrix, capture Instruments physical peaks, audit CPU tree
neighbor equivalence outside tie/boundary cases, and evaluate a device-aware
automatic dispatch policy before making public acceleration promises.

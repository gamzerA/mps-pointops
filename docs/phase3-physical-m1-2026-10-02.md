# Physical Apple M1 operator and Phase 4 measurements, 2026-10-02

This record extends the [Phase 3 pinned result matrix](phase3-results-2026-10-01.md)
with **physical Apple M1** evidence. It also tests the proposed Phase 4
performance focus: bidirectional Chamfer and feature-space kNN. Graph
`avg_pool` remains a correctness and topology result, not a speed claim.
Only this M1 and the separately documented M5 Pro have been measured on
physical hardware. M2–M4 remain untested.

## Fixed environment and provenance

- MacBookPro17,1, Apple M1, 8 GiB unified memory, macOS 26.5.2, arm64,
  Python 3.10.6. The checked-out `main` source was
  `f4a886a08938a8a3c9785ecaec962300efd298e0`. The new
  `bench_chamfer_cpu_mps.py` and `probe_feature_knn_tie.py` scripts were
  copied into that checkout for their runs and are committed with this
  report; their exact SHA-256 values are embedded in the respective JSON.
  The base commit alone does not reconstruct those two commands.
- The operator, PyG, Chamfer, FPS, Ball Query, and feature-kNN processes used
  PyTorch 2.12.0, PyG 2.8.0, and pyg-lib 0.7.0+pt212. The **native PyTorch
  scatter probes alone** used a separate PyTorch 2.14.1 environment without
  PyG. Do not merge their timings into a single version comparison.
- `PYTORCH_ENABLE_MPS_FALLBACK=0` was set before importing PyTorch. Safe
  (`PYTORCH_MPS_FAST_MATH=0`) and Fast (`=1`) ran in separate processes.
  Timed MPS calls use `torch.mps.synchronize()` at their stated boundaries.
  Samples, commands, seeds, source hashes, and finer input contracts are in
  the linked JSON. The benchmark directory contains [all 24 JSON records](../bench/results/physical-m1-2026-10-02/).
- Before committing, the sole local checkout prefix in the published pytest
  logs and new Chamfer benchmark JSON was replaced with
  `<physical-m1-checkout>`. No results, counts, timings, or source hashes were
  changed. The original unnormalized captures were retained locally. The
  PyG graph `avg_pool` JSON does not embed a commit or script hash: it was
  run from the source commit above, and
  `bench/bench_pyg28_avg_pool.py` SHA-256 is
  `f61453504df6406c8f0ba8ef7638b841cb704b27b74a2b864d6ba8d3344799ac`.

## Correctness and existing kernels

| Check | Safe | Fast | Scope |
| --- | ---: | ---: | --- |
| Full `pytest -ra tests` | [394 passed, 11 skipped](pytest-physical-m1-torch212-safe-2026-10-02.log) | [393 passed, 12 skipped](pytest-physical-m1-torch212-fast-2026-10-02.log) | The extra Fast skip is the Safe-only nonfinite feature-kNN test. Real `torch_cluster` is not installed; the other skips are listed in each log. |
| Dense Ball Query previous-vs-SIMD differential | [48/48](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-simd-contract.json) | [40/40](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-simd-contract.json) | Poisoned complete output buffers, first-K indices, and squared-distance bits. Fast excludes the eight nonfinite-coordinate cases by policy. |
| Pure PyTorch `F.linear` and `nn.Linear` fixed-bias probe | [No bias omission](diagnostics/mps-pointops-m1-safe-linear.json) | [No bias omission](diagnostics/mps-pointops-m1-fast-linear.json) | Both `max abs(Linear − (xWᵀ+b))` values are zero. The earlier hosted M1 Virtual discrepancy is not a property of all M1 hardware. Its root cause is still unknown. |

The paired [Ball Query Safe](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-ball-ablation.json)
and [Fast](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-ball-ablation.json)
samples used the same preallocated buffers and sorted or random sphere-shell
inputs. Each timed call includes host Metal dispatch and synchronization.
At 100,000 sorted points, 1,024 queries, and K=64, the prior dense kernel
versus SIMD medians were **45.276 → 10.349 ms** (Safe) and
**30.025 → 6.842 ms** (Fast); indices and squared-distance bits matched.
At 100,000 random points the SIMD medians were 5.394 ms (Safe) and
3.504 ms (Fast). Allocation, transfers, and shader compilation are excluded;
these are neither GPU shader-only timings nor public API or SciPy comparisons.

The public FPS strategy comparison at B=1 and 1,024 selected points used
standard-normal coordinates. [Safe](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-fps-production.json)
single → multigroup medians were **361.998 → 117.345 ms** at 500,000 points
and **804.007 → 385.046 ms** at 1,000,000 points. [Fast](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-fps-production.json)
medians were **340.220 → 111.503 ms** and **811.173 → 395.759 ms**.
Every paired iteration returned identical indices. The M1 default `auto`
policy still uses the single-group path; these measurements explicitly chose
the two strategies and do not set an M1 crossover or automatic threshold.

PyG 2.8 graph coarsening passed the full Safe/Fast regression suite. The
synthetic direct `avg_pool` workload was also synchronized at 16,384 and
65,536 nodes: [Safe raw samples](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-avg-pool.json)
and [Fast raw samples](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-avg-pool.json).
At 65,536 nodes in Safe mode, CPU/MPS forward medians were 16.294/22.088 ms
and backward medians 8.986/11.720 ms. This is functional coverage with a
measured MPS slowdown for that fixture; no custom fused graph-pooling kernel
was tested.

## Phase 4: Chamfer and feature-space kNN

The new [CPU/MPS Chamfer benchmark](../bench/bench_chamfer_cpu_mps.py) compares
the **same public `mps_pointops.chamfer_distance` API** on resident float32
inputs. It verifies the bidirectional mean loss and both coordinate gradients
within `rtol=atol=1e-5` before timing. All six size/mode cases passed. CPU is
the repository's tiled PyTorch reference, not SciPy, PyTorch3D CUDA, or an
optimized CPU nearest-neighbor library. Forward and backward are timed as
separate phases; adding their medians is not an end-to-end training-loop
measurement. Inputs are already on their respective devices; transfers are
excluded.

| N per cloud | Mode | CPU forward | MPS forward | CPU backward | MPS backward |
| ---: | --- | ---: | ---: | ---: | ---: |
| 256 | Safe | 1.074 | 1.980 | 0.140 | 0.924 |
| 1,024 | Safe | 9.650 | 2.389 | 0.205 | 0.985 |
| 4,096 | Safe | 151.997 | 4.475 | 0.344 | 0.902 |
| 256 | Fast | 1.076 | 1.987 | 0.136 | 0.923 |
| 1,024 | Fast | 9.584 | 2.315 | 0.200 | 0.936 |
| 4,096 | Fast | 154.089 | 4.906 | 0.364 | 0.961 |

Times are median milliseconds. Raw records: [Safe 256/1,024](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-chamfer-cpu-mps.json),
[Fast 256/1,024](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-chamfer-cpu-mps.json),
[Safe 4,096](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-chamfer-cpu-mps-4096.json),
[Fast 4,096](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-chamfer-cpu-mps-4096.json).
The 4,096-point MPS forward is 33.96× (Safe) or 31.41× (Fast) faster than
**this package CPU reference**; its backward remains slower. This is one
random finite input family, not a universal Chamfer or model speedup.

The existing [feature-kNN benchmark](../bench/bench_feature_knn.py) measured
D=64/128, K=20 with the public dense and flat APIs. Dense Q=N=1,024
medians were CPU/MPS 4.226/7.871 ms at D=64 and 3.454/17.127 ms at D=128
in Safe mode. At Q=N=2,048 they were 5.929/34.310 ms and 6.809/71.331 ms.
Fast showed the same direction. The Metal path also lost to MPS
`cdist+topk` at 2,048 in both dimensions. Flat Q=256,N=1,024 Safe medians
were CPU/MPS 98.230/6.486 ms (D=64) and 167.853/9.534 ms (D=128); that
CPU baseline is this package's flat reference, not an optimized CPU library.
Raw [1,024 Safe](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-feature-knn.json)
/ [Fast](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-feature-knn.json)
and [2,048 Safe](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-feature-knn-2048.json)
/ [Fast](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-feature-knn-2048.json)
contain every sample. These differences do not identify SIMD occupancy or
memory packing as the cause; hardware profiling and an implementation
ablation would be required.

At Q=N=2,048,D=128 the CPU `cdist` path and Metal path differed in **four
output slots across two query rows**, in both modes. The independent
[direct float32 accumulation probe](../bench/probe_feature_knn_tie.py)
recomputed and ranked all 2,048 references for all 2,048 queries, following
the [feature-space contract](feature-knn.md). MPS had **zero** slot mismatches
against that oracle; CPU `cdist` had four, two adjacent swaps, each between
candidates separated by two float32 ULPs in oracle squared distance. See
[Safe](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-feature-knn-tie.json)
and [Fast](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-feature-knn-tie.json).
This resolves the observed four slots as a near-rank difference under the
documented arithmetic contracts. It does not assert universal CPU/MPS index
equality on larger or differently scaled features.

## Concentrated destinations are a severe M1 counterexample

The [native graph scatter](../bench/bench_scatter_fanin_mps.py) PyTorch 2.14.1
probe checks exact dyadic CPU/MPS output and source-gradient bits before
measuring. At 1,048,576 contributions, C32 `scatter_add_` forward rose from
30.892 ms uniform to 537.020 ms at one destination in Safe mode; Fast was
30.941 → 538.841 ms. A scalar `amax` probe also slowed. The separate
[three-channel point-gradient probe](../bench/bench_scatter_xyz_fanin_mps.py)
rose from **3.312 to 4,527.662 ms** (Safe), and **3.338 to 4,578.584 ms**
(Fast) for the same contribution count and destination change. Exact-value
output and gradient checks passed in all 12 graph and six point cases per
mode. Raw [graph Safe](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-scatter-fanin.json)
/ [Fast](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-scatter-fanin.json)
and [point Safe](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-scatter-xyz.json)
/ [Fast](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-scatter-xyz.json)
retain the full distributions and twelve samples. On this 8 GiB device,
memory pressure and thermal state were not controlled; especially the point
probe's Fast backward timing differed markedly from Safe and needs reruns.

The actual single-directional Chamfer
[Safe](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-chamfer-contention.json)
and [Fast](../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-chamfer-contention.json)
fixtures separately forced uniform or concentrated nearest indices. At
B=1,N=2,048, Safe backward medians were **0.968 ms uniform versus 3.814 ms
concentrated**; Fast was **0.860 versus 3.669 ms**. All expected indices and
analytic gradients passed. This is a different fixture and reduction mode
from the bidirectional random Chamfer table above. Neither probe identifies
the GPU primitive or proves Metal atomic contention; GPU-side tracing and a
candidate reduction-kernel ablation are required for a causal claim.

## Decision and remaining experiments

- Keep PyG `avg_pool` as topology/gradient functionality, with no speedup
  claim. At this report's v0.6 source revision, voxel downsampling composed
  PyTorch operations and there was **no fused Metal voxel-pooling kernel**
  to benchmark. The later opt-in v0.7 prototype is documented separately.
- The random large-Chamfer forward result motivates model-level tests, but
  the concentrated case makes backward reduction a design and profiling
  target. Compare candidate segmented reduction against native PyTorch on
  both uniform and concentrated inputs before changing the default path.
- Feature-space kNN needs a tiled/vectorized D=64/128 candidate and
  same-device `cdist+topk` and optimized CPU baselines before a speed claim.
  The M1 dense path was slower at both 1,024 and 2,048 points.
- Physical M2–M4 Safe/Fast evidence, real data distributions, transfer-aware
  pipeline measurements, and controlled power/thermal reruns remain open.
  Nothing here establishes blanket PyG compatibility or a universal M1
  speedup.

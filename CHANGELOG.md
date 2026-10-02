# Changelog

## Unreleased

- Add an opt-in, reusable single-cloud `SpatialIndex` with explicit Metal
  full-scan/BVH selection. Existing dense and flat entry points keep their
  signatures and default kernels. Automatic BVH routing is restricted to a
  measured M5 Pro Safe Math kNN window and a sampled hot-cell guard; other
  inputs retain the existing scan path.
- Extend the private two-level Morton BVH to PyTorch3D-style first-K Ball
  Query. The bounded MPS float32 path preserves original-index order, strict
  radius comparison, `(0,-1)` padding, and first-order coordinate gradients
  through the public index facade. Adversarial M5 Pro tests compare full
  indices and selected squared-distance bits to the existing dense kernel.
- Record stage-separated PyTorch MPS allocator peaks and sampled driver
  allocation in a [source-pinned memory study](docs/spatial-memory-v090.md).
  Total GPU physical-memory peak and physical M1 validation remain open;
  these development changes do not constitute a v0.9 or v1.0 release.

## 0.8.0 — 2026-10-02

This release adds a bounded Pointcept PTv1 compatibility path and a required
upstream Chamfer parity gate for the supported squared-L2 contract. It does not
claim complete Pointcept or PyTorch3D compatibility, or completion of Phase 4.

- Add an opt-in `mps_pointops.compat.install(pointcept=True)` shim for the five
  Pointcept v1.2.1 PTv1 Seg26 `pointops` calls. The pinned synthetic M5 Pro
  Seg26 fixture passed Safe/Fast forward and first-order backward comparison
  with CPU after one documented temporary CUDA-constructor substitution.
  The [contract](docs/pointcept-ptv1-subset.md) records exact offsets,
  distance cutoff, padding, gradient scope, source hashes, and the stricter
  gradient gate that rejects an all-zero-gradient counterexample.
- Add a dedicated [PyTorch3D Chamfer MPS parity workflow](docs/chamfer-upstream-ci.md)
  for the supported squared-L2 subset. It pins the official CPU extension,
  requires MPS with CPU fallback disabled, and runs Safe/Fast separately.
  [Hosted run 36899829775](https://github.com/gamzerA/mps-pointops/actions/runs/36899829775)
  passed 160 cases and 1,080 output/gradient checks per mode with zero failed
  elements; the case-by-case JSON is retained in `docs/results/`.
- Extend the [physical M5 Pro and M1 bidirectional Chamfer contention study](docs/chamfer-large-contention-2026-10-02.md)
  to 32,768 and 65,536 points, with matching input hashes, exact nearest-index
  checks, and independent analytic-gradient checks. At 65,536 points, paired
  concentrated/uniform full-call ratios were 1.00×/1.03× on M5 Pro and
  4.19×/4.23× on M1 in Safe/Fast Math. The M1 result prioritizes a dedicated
  reduction ablation; native PyTorch scatter remains the default until a
  same-input full-call comparison includes grouping and gradient work. Device
  and PyTorch versions differ, so this is not an isolated GPU-generation effect.
- Review L1, normal-vector, and `Pointclouds` PyTorch3D Chamfer behavior in a
  [pinned API scope decision](docs/chamfer-api-scope-v0.8.md). Those extensions
  remain outside the 0.8.0 squared-L2 subset pending their own direct gates.

## 0.7.0 — 2026-10-02

This release adds a measured compact voxel path and an **opt-in experimental**
Metal CSR pooling prototype. Phase 3 has bounded validated coverage for the
pinned PyG 2.8 and legacy shim surfaces documented in the compatibility matrix;
custom scatter reductions, universal PyG compatibility, and M2–M4 physical
validation remain open.

### Added

- `voxel_downsample(..., pool_backend="fused_csr")` reduces mean positions and
  mean/sum features from an existing CSR map in one Metal dispatch on MPS.
  Backward gathers through the exact `inverse` map. The default remains
  `index_add`; the fused path is opt in because severe float32 cancellation
  can produce a result outside the stated tolerance of MPS `index_add_`.
  Integer maps match exactly, and the bounded numerical and gradient contract
  is recorded in [the voxel API document](docs/voxel-api-contract.md).

### Changed

- MPS dense, flat, and PyG kNN entry points now raise a clear `ValueError`
  when the effective requested width exceeds the current kernel constant
  `MAX_K=256`; the flat path no longer silently falls back to a slow
  PyTorch scan. CPU reference inputs retain their wider-k behavior.
- Package CI verifies that the new shader is included in built artifacts and
  runs the opt-in pooling tests under separate Safe/Fast Math processes.

### Validation and measurements

- Synchronized [M5 Pro compact voxel measurements](docs/voxel-api-benchmark.md)
  cover 20k, 100k, and 500k points; uniform and ragged batches; and dense and
  sparse cells, with raw Safe/Fast JSON, stage timing, and MPS allocation
  checkpoints. The 500k-point uniform dense full-step median was
  CPU/MPS **78.42/21.73 ms** in Safe Math and **83.79/23.09 ms** in Fast Math.
  Small 20k fixtures favored CPU. The benchmark measures the default
  `index_add` path, not the fused prototype.
- The same 24-case default-path matrix completed on a physical M1 with
  PyTorch 2.12.0 in separate Safe/Fast processes. At 500k uniform dense,
  full-step CPU/MPS medians were **134.30/69.22 ms** in Safe Math and
  **133.66/65.96 ms** in Fast Math. CPU led all 20k cases; the 100k crossover
  depended on occupancy and batch shape. Input hashes matched across modes
  and the corresponding M5 Pro fixtures; see the [source-pinned report](docs/voxel-api-benchmark.md).
- The focused fused CSR suite passed **32 tests with one expected failure** in
  each M5 Pro math mode. The expected failure records the severe-cancellation
  difference from PyTorch's MPS scatter accumulation; raw samples and a
  float64 oracle are linked from the voxel API document. A separate
  [full-call benchmark](docs/voxel-fused-benchmark.md) found mixed wins and
  losses across 20k–500k fixtures, with synchronized allocator checkpoints;
  no overall speedup or lower peak-memory claim is made for the prototype.
- Physical M1 focused fused CSR runs at the earlier source snapshot with
  identical computation passed [12 tests / 1 expected failure in Safe Math](docs/pytest-v070-m1-fused-safe-2026-10-02.log)
  and [12 tests / 1 expected failure in Fast Math](docs/pytest-v070-m1-fused-fast-2026-10-02.log).
- The integrated M5 Pro suite with fallback disabled passed
  [383 tests / 38 skips / 1 expected failure](docs/pytest-v070-m5pro-torch214-safe-2026-10-02.log)
  in Safe Math and
  [382 tests / 39 skips / 1 expected failure](docs/pytest-v070-m5pro-torch214-fast-2026-10-02.log)
  in Fast Math. The v0.7.0 wheel and source archive built, include both
  license files and `voxel_pool.metal`, and passed content inspection.

## 0.6.0 — 2026-10-02

This release adds experimental feature-space and graph/grid operations, plus
bounded model and physical M1 validation. It does not complete Phase 3 or
establish universal PyG compatibility or a speedup on every Apple GPU.

### Added

- Native Metal feature-space kNN for positive dimensions other than three,
  including D=64 and 128, through the dense, flat, and PyG `pyg::knn` MPS
  entry points. The existing 3D kernels remain in use. The
  [contract](docs/feature-knn.md) specifies direct float32 accumulation,
  deterministic ties, non-finite distances, and an explicit `k <= 256` limit
  for the feature path.
- Bounded legacy `torch_cluster` compatibility paths for `nearest`,
  `grid_cluster`, `graclus_cluster`, and `random_walk` on CPU/MPS. The
  [`random_walk` contract](docs/random-walk-contract.md) uses PyTorch tensor
  operations, not a native Metal kernel, and does not register PyG 2.8's
  separate `torch.ops.pyg.random_walk` operator.
- MPS dispatch for PyG 2.8's `pyg::grid_cluster`, enabling the tested
  `voxel_grid` → `avg_pool_x` path. A separate experimental
  [compact voxel API](docs/voxel-api-contract.md) provides floor-based cells,
  point/voxel maps, and mean/sum feature reduction on CPU/MPS. These APIs
  have different raw grid IDs; no fused Metal voxel-pooling kernel is included.

### Validation and measurements

- The pinned original DGCNN classification model passed one synthetic
  64-point forward/backward fixture, including four dynamic-neighbor stages
  and 4,096 matching indices. Dataset accuracy, full training, and original
  CUDA parity remain untested ([evidence](docs/feature-knn.md)).
- A fixed synthetic PointNet++ SSG segmentation model passed eval forward,
  cross-entropy loss, and input/parameter gradient comparison with the
  original CUDA extension at `atol=rtol=1e-4`. Six of 12 raw local-index
  arrays differed after an FPS tie; the report explains the mapped indices
  and one order-dependent Ball Query cutoff. Labeled-data accuracy and
  training convergence remain untested
  ([report](docs/parity/pointnet2-segmentation.md)).
- PyG 2.8 graph `avg_pool` and `voxel_grid` → `avg_pool_x` were checked on
  bounded synthetic graphs and features, including topology and first-order
  gradients. Pooling uses PyG/PyTorch reductions, not a project Metal kernel
  ([Phase 3 matrix](docs/phase3-results-2026-10-01.md)).
- Physical M1 Safe/Fast full-suite runs reported **394 passed / 11 skipped**
  and **393 passed / 12 skipped**, respectively, with MPS fallback disabled.
  The [M1 report](docs/phase3-physical-m1-2026-10-02.md) records dense Ball
  Query and large-cloud FPS parity, synchronized Chamfer/feature-kNN timings,
  and a severe concentrated-destination `scatter_add_` slowdown in its
  stated fixtures. M2–M4 physical validation remains open.
- Synchronized M5 Pro PyG GCN/GraphSAGE/GAT workloads and M1/M5 native
  scatter probes support [retaining native PyTorch scatter](docs/pyg-survey/2026-10-02-m1-m5-scatter-decision.md)
  for the documented graph path while testing a segmented-reduction candidate
  for skewed destinations. These measurements do not isolate Metal atomics
  as the cause or establish a model-level speedup from a custom reduction.

## 0.5.0 — 2026-10-01

This release makes the new PointNet++ and Chamfer APIs available as
**experimental operators**. It does not complete Phase 2: high-dimensional
kNN and PointNet++ segmentation validation remain open.

### Added

- Experimental `three_nn` and `three_interpolate` for PointNet++
  feature propagation on MPS, with a PyTorch CPU reference and native Metal
  forward/interpolation backward.
  The [contract](docs/pointnet2-propagation.md) records the supported shapes,
  tie order, and gradient surface; full segmentation validation remains open.
- Experimental squared-L2 `chamfer_distance`. Metal returns
  nearest indices and distances, while PyTorch `scatter_add_` accumulates the
  bidirectional first-order gradient. Supported lengths mask padding in both
  passes; point/batch reductions and weights scale the gradient as specified
  in the [contract](docs/chamfer-contract.md).

### Fixed

- Chamfer's Metal nearest search now selects a valid first index even when
  finite float32 coordinates overflow every squared distance to infinity;
  the saved index remains safe for backward gathering (#20). CI also checks
  that both new Python modules and Metal shaders are present in the wheel.
- Match PyTorch3D's all-zero-weight Chamfer output shapes, normal-result slot,
  and gradient connectivity for the supported API subset (#22).

### Survey and measurements

- A [PyG 2.8.0 survey](docs/pyg-survey/2026-10-01-m5-pro-pyg28.md) ran GCN,
  GraphSAGE, and GAT forward/backward on a fixed synthetic 12-node graph on
  an M5 Pro with MPS fallback disabled. No missing operator appeared in that
  tested configuration; it does not establish coverage of other graphs or
  package combinations.
- Synchronized [Safe](bench/results/2026-10-01-apple-m5-pro-chamfer-contention-safe.md)
  and [Fast](bench/results/2026-10-01-apple-m5-pro-chamfer-contention-fast.md)
  Chamfer runs compared uniform and concentrated nearest-index selection at
  batch 4 and 256–16,384 points per cloud. Neither backward timings nor a
  direct `scatter_add_` control showed a consistent contention slowdown in
  this range; larger clouds, other GPUs, and bidirectional losses are untested.
- [Direct PyTorch3D Chamfer comparison](docs/chamfer-upstream-parity-0.5.0.md)
  against a pinned, compiled upstream CPU extension passed 160 finite-input
  cases and 1,080 output/gradient checks per port device on CPU, MPS Safe,
  and MPS Fast. The MPS maximum absolute loss difference was `9.5367e-7`;
  this result does not cover normals, `norm=1`, ties, or PyTorch3D CUDA.
- [Direct PointNet++ CUDA comparison](docs/parity/pointnet2-upstream.md)
  on an RTX 2080 passed two deterministic propagation fixtures against MPS
  Safe and Fast. The 243 selected indices matched exactly; Safe Math had zero
  observed difference in the compared arrays, while Fast Math differed only
  in Euclidean distances by at most `2.3842e-7`. The original CUDA kernel
  source was unchanged; two `setup.py` build settings were adapted for the
  available Windows toolchain.
- Final-candidate M5 Pro [Safe](bench/results/2026-10-01-apple-m5-pro-v050-final-safe.md)
  and [Fast](bench/results/2026-10-01-apple-m5-pro-v050-final-fast.md)
  benchmarks measure the public experimental APIs with synchronization,
  four warmups, and 20 samples per case. The matching
  [Safe JSON](bench/results/2026-10-01-apple-m5-pro-v050-final-safe.json) and
  [Fast JSON](bench/results/2026-10-01-apple-m5-pro-v050-final-fast.json)
  record every timing, source commit `7c1406e`, and source SHA-256 values.

### Verified

- The v0.5.0 release candidate passed **260 tests with 12 expected skips** in
  separate M5 Pro [Safe](docs/pytest-v050-safe-torch214-2026-10-01.log) and
  [Fast](docs/pytest-v050-fast-torch214-2026-10-01.log) processes with MPS
  fallback disabled. The 12 local skips are seven existing `k > n` cases and
  five PyG 2.8 checks requiring optional packages; the pinned PyG CI job
  supplies those packages. The v0.5.0 wheel and source distribution built,
  passed `twine check`, and contain the new operator code and licenses.

## 0.4.0 — 2026-10-01

### Documentation

- Re-measured 20,000/100,000-point FPS, kNN, and dense Ball Query on the
  current M5 Pro source, including a separate sorted Ball Query run. Refreshed
  the README chart and table from those JSON files, recorded hashes for every
  timed kernel, and placed the current 201-pass Safe/Fast results first.

### Added

- A production B=1 multi-threadgroup FPS path for large clouds. On the measured
  M5 Pro, `strategy="auto"` selects it from 500,000 points with at least two
  samples; other Apple GPUs and batches use the original path by default.
  Explicit `single` and `multigroup` strategies allow direct comparisons.
- `mps_pointops.pytorch3d.ball_query` provides PyTorch3D's Ball Query argument
  order and defaults, lengths, optional gathered neighbor coordinates, and
  `KNN(dists, idx, knn)` result for three-dimensional float32 inputs.
  `skip_points_outside_cube` is accepted as a result-preserving hint; the
  adapter does not apply the cube prefilter.

### Changed

- Dense Ball Query now scans consecutive 32-point blocks with a SIMD prefix
  rank, preserving the first-K input order while improving sorted-input
  performance. On an M5 Pro, a paired 100k-point Safe Math ablation measured
  21.43 to 2.91 ms on x-sorted input and 7.66 to 1.40 ms on random input.
- Native kernel startup checks Apple Silicon macOS, an available PyTorch MPS
  backend, and 32-wide simdgroups before compiling the dense shader.
- The README includes an MPS smoke example and separates the archived v0.3.0
  benchmark from measurements of the released SIMD kernel.

### Verified

- With the large-cloud FPS path and PyTorch3D-style adapter, M5 Pro Safe and
  Fast Math each passed 201 tests with 12 expected skips. Paired public-API
  FPS measurements at 1,024 samples gave 193.08 to 29.37 ms for 500,000
  points and 421.69 to 49.39 ms for 1,000,000 points, with identical indices
  in every paired iteration.
- Safe and Fast Math each passed 168 tests with 12 expected skips on M5 Pro.
  A separate sentinel-buffer checker passed 48 Safe and 40 Fast differential
  cases: complete output writes, byte-identical indices and squared distances
  versus the previous Metal kernel, and first-K indices equal to an independent
  CPU oracle on the tested inputs.
- An earlier multi-threadgroup FPS size sweep measured 2,048 to 1,000,000
  points at batch 1 and informed the conservative public M5 Pro dispatch
  threshold. Uneven batches and automatic selection on other Apple GPUs
  remain future work.

## 0.3.0 — 2026-10-01

### Added

- MPS dispatch for pyg-lib's `pyg::fps`, `pyg::knn`, and `pyg::radius`
  operators used by PyG 2.8.0. PyG's `knn_graph` and `radius_graph` use these
  operators; the latter now excludes equal global indices before its neighbor
  cap. The integration keeps pyg-lib's CPU and CUDA dispatch intact.
- An MPS batch-to-pointer bridge for PyG 2.8, whose original path reaches a
  PyTorch CSR conversion without an MPS kernel. CPU conversion remains in PyG.
- A pinned PyG 2.8 integration CI job and PyPI Trusted Publishing workflow.
  This is the first PyPI release of `mps-pointops`.

### Compatibility

- The PyG 2.8 MPS path supports three-dimensional coordinates: float32 FPS
  and kNN, float32 or float16 radius. The float16 radius path computes in
  float32, so boundary membership can differ from pyg-lib's half arithmetic.
- The published package version is now 0.3.0, with project links and metadata
  for PyPI.

## 0.2.0 — 2026-10-01

### Added

- Flat, variable-length 3D point-cloud `fps`, `knn`, and `radius` operations
  with sorted batch vectors, separate reference/query sets, global indices,
  and compact `[query, reference]` edge tensors. Metal kernels handle MPS
  inputs; CPU inputs use PyTorch implementations.
- A `torch_cluster` 1.6.3-style compatibility shim, including `knn_graph` and
  `radius_graph`. The five point-cloud entry points were exercised through
  PyG 2.7.0 on MPS. PyG 2.8.0 uses a separate `torch.ops.pyg` path and is
  outside this shim's scope.

### Fixed

- FPS sample counts now follow the original device-specific degree conversion
  and preserve scalar versus length-one tensor ratio semantics.
- The flat float32 radius search uses `fl32(r * r)` after double-precision
  multiplication, matching the torch-cluster threshold. Dense Ball Query
  retains its PyTorch3D threshold contract.

The M5 Pro PyTorch 2.7.0 MPS Safe and Fast Math runs each passed 147 tests
with 7 expected skips; the raw logs are under `docs/`. CI checks the package
and macOS/Linux configurations.

## 0.1.1 — 2026-10-01

### Fixed

- The minimum PyTorch version is now 2.7 (`torch>=2.7`). 0.1.0 declared
  `torch>=2.6`, but the kernels need `torch.mps.compile_shader`, which PyTorch
  2.6 does not have: with 2.6, 71 of 85 tests fail. Checked with PyTorch 2.7.0,
  which passes the full suite. A PyTorch build without `compile_shader` now
  raises a clear error on first use.
- `ball_query` applies the same `radius` and `K` checks on every device. The CPU
  path used to accept negative, infinite and NaN radii (a negative radius acted
  like its absolute value) and tiny positive radii that the MPS path rejects.
  It also checks input shapes now.
- The source distribution includes `docs/`, `bench/`, `examples/` and `tools/`,
  which the README links to.

### Added

- CI on GitHub Actions: tests on Apple Silicon macOS runners (the MPS tests run
  on the runner GPU, in Metal safe and fast math), including the oldest
  supported combination of Python 3.10 and PyTorch 2.7.0, CPU tests on Linux, and
  a check that the wheel carries the Metal kernels and license files.

## 0.1.0 — 2026-10-01

First public release: Metal kernels for farthest point sampling, k nearest
neighbors and ball query on PyTorch MPS, and stand-ins for the CUDA-only
`pointnet2_ops` and `knn_cuda` packages.

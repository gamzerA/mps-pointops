# Changelog

## Unreleased

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
- The README includes a small MPS smoke example and distinguishes the v0.3.0
  benchmark from the current SIMD experiment.

### Verified

- With the large-cloud FPS path and PyTorch3D-style adapter, M5 Pro Safe and
  Fast Math each passed 201 tests with 12 expected skips. Paired public-API
  FPS measurements at 1,024 samples gave 191.68 to 29.13 ms for 500,000
  points and 417.06 to 48.97 ms for 1,000,000 points, with identical indices.
- Safe and Fast Math each passed 168 tests with 12 expected skips on M5 Pro.
  A separate sentinel-buffer checker passed 48 Safe and 40 Fast differential
  cases: complete output writes, byte-identical indices and squared distances
  versus the previous Metal kernel, and first-K indices equal to an independent
  CPU oracle on the tested inputs.
- A benchmark-only multi-threadgroup FPS experiment measured sizes from 2,048
  to 1,000,000 points at batch 1. It remains outside the public kernel until
  other devices and batch sizes are tested.

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

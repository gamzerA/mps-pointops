# Changelog

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

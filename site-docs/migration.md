# Migration and release scope

The latest published package is not automatically the same as the development
source. Pin a tag in experiments and cite its **version DOI**. The
[concept DOI](https://doi.org/10.5281/zenodo.23076057) identifies the archive
series; the [release list](https://github.com/gamzerA/mps-pointops/releases)
identifies the published code snapshot.

## Moving from v0.8.0 to the development tree

- Dense `ball_query` retains squared distances and `-1` index padding.
  `flat.radius` returns compact graph edges and follows the separate
  `torch_cluster` radius-square rule. Choose the interface by its contract,
  not by a shared operation name.
- Existing dense and flat calls keep their scan kernels. `SpatialIndex` is
  opt-in; only its narrow measured M5 Pro kNN auto case can choose BVH.
  Application code should not depend on its research-only threshold as a
  stable performance guarantee.
- `voxel_downsample(pool_backend="fused_csr")` opts into an experimental
  Metal reducer. The default `index_add` path remains the baseline.
- Private `mps_pointops._sparse_*` and `_subm_*` modules are not a public
  `spconv` interface. Their names, layouts, and behavior can change before a
  dedicated release.
- PyTorch 2.7.0, 2.12.0, and 2.14.1 are **tested points**, not proof that every
  intervening or future PyTorch release behaves identically. Re-run the
  relevant Safe/Fast suites when changing runtime versions.

## Before a 1.0 API freeze

Freeze only public calls with stated argument validation, output shape/dtype,
padding, ordering, gradients, error behavior, and test coverage. Keep opt-in
research interfaces clearly marked until their own release criteria close.
The [v0.9–v1.0 gate](https://github.com/gamzerA/mps-pointops/blob/main/docs/milestones-v0.9-v1.0.md)
requires current-commit CI, measured physical hardware, upstream sparse
convolution comparisons and a model pass, publication artifacts, and
source/version/DOI agreement. A small synthetic fixture proves only the
tested case. This page does not declare those gates complete.

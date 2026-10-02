# Migration and release scope

The latest published package is not automatically the same as the development
source. Pin a tag in experiments and cite its **version DOI**. The
[concept DOI](https://doi.org/10.5281/zenodo.23076057) identifies the archive
series; the [release list](https://github.com/gamzerA/mps-pointops/releases)
identifies the published code snapshot.

## Moving from v0.8.0 to v1.0.0

The proposed v0.9.0 and v0.10.0 milestones were planning targets and had no
release tags. v1.0.0 follows v0.8.0 directly. It combines tested additions
without declaring those entire milestones complete. The established dense
and flat call signatures, result shapes, ordering, and padding rules remain
the same in the documented supported domains.

- Dense `ball_query` retains squared distances and `-1` index padding.
  `flat.radius` returns compact graph edges and follows the separate
  `torch_cluster` radius-square rule. Choose the interface by its contract,
  not by a shared operation name.
- Existing dense and flat calls keep their scan kernels. `SpatialIndex` is
  an opt-in experimental public facade; only its narrow measured M5 Pro kNN
  auto case can choose BVH. Application code should not depend on its routing
  threshold as a stable performance guarantee. Explicit BVH calls have the
  bounded input contract in the [spatial API document](https://github.com/gamzerA/mps-pointops/blob/main/docs/spatial-api-v090.md).
- Chamfer adds experimental L1, normal-vector, `Pointclouds`, and
  variable-dimension tensor paths. The pinned upstream comparisons cover
  stated finite float32 cases; they do not imply every PyTorch3D argument,
  error path, or dtype is supported.
- `voxel_downsample(pool_backend="fused_csr")` opts into an experimental
  Metal reducer. The default `index_add` path remains the baseline.
- Private `mps_pointops._sparse_*`, `_subm_*`, and `_spconv_compat` modules are
  not a public `spconv` interface. They include bounded SubM, strided, and
  saved-key inverse Metal arithmetic and a fixed synthetic OpenPCDet adapter
  check; their names, layouts, and behavior can change before a dedicated
  public sparse release. Do not replace a production `spconv` installation
  with these private modules.
- PyTorch 2.7.0, 2.12.0, and 2.14.1 are **tested points**, not proof that every
  intervening or future PyTorch release behaves identically. Re-run the
  relevant Safe/Fast suites when changing runtime versions.

## v1.0 public API policy

The compatibility promise covers the documented behavior of tested public
calls within their stated input ranges: accepted arguments, output shape and
dtype, ordering, padding, first gradients where documented, and explicit
errors. Changes to those contracts follow semantic versioning. It does not
freeze benchmark crossover thresholds, private modules, or experimental
paths beyond their explicitly documented result contract. Public APIs and
tested limits are indexed [here](api.md); the device/version matrix is
[here](support.md). A PyTorch dependency of `>=2.7` is not a claim that every
later version has been tested.

The [original v0.9–v1.0 plan](https://github.com/gamzerA/mps-pointops/blob/main/docs/milestones-v0.9-v1.0.md)
remains a record of proposed full milestones. Full PyTorch3D Chamfer coverage,
general `spconv` 2.x parity, sparse transpose convolution, broader physical
device coverage, and upstream acceptance remain separate future work. A
fixed synthetic model fixture proves only its tested case.

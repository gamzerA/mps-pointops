# Phase 3 pinned MPS compatibility results, 2026-10-01

This record covers named operations and fixtures on an Apple M5 Pro running
macOS 26.5.2. MPS runs used `PYTORCH_ENABLE_MPS_FALLBACK=0` before importing
PyTorch. Safe and Fast Math were run in separate processes only where both
logs are linked; the GCN/GraphSAGE/GAT survey was a single run per model.
A passing row applies
only to its listed versions, dtype, input family, and assertions. It does not
mean every PyG program or every `torch_cluster` input is supported. The
[acceptance plan](phase3-compatibility-matrix.md) describes the wider gates.

| Surface | Pinned packages and tested input | Result and direct evidence | Source / raw log |
| --- | --- | --- | --- |
| PyG 2.8 direct `fps`, `knn`, `radius` and graph wrappers | PyTorch 2.12.0, PyG 2.8.0, pyg-lib 0.7.0+pt212; float32 MPS inputs in the named test fixtures | Correctness assertions pass with the real pyg-lib schemas and CPU fallback disabled. | [`test_pyg28.py`](../tests/test_pyg28.py); full [Safe](pytest-pyg28-grid-safe-2026-10-01.log) / [Fast](pytest-pyg28-grid-fast-2026-10-01.log) logs from grid PR source `6228275c3c221bcde28687527b62825abdb45859` rebased onto `c3c7cc73bab1b2181815ba38a7d16bf6c0332601`. |
| PyG 2.8 `voxel_grid` IDs | Same pinned packages; finite float32 1D–3D positions and batches | Real pyg-lib CPU differential: 192 deterministic raw cases, plus exact representable cell boundaries. MPS Safe/Fast focused integration: 15 tests each, including the existing PyG search assertions. | [Grid contract](pyg28-grid-cluster-contract.md); full [Safe](pytest-pyg28-grid-safe-2026-10-01.log) **327 passed / 9 skipped**, [Fast](pytest-pyg28-grid-fast-2026-10-01.log) **326 passed / 10 skipped**. |
| PyG 2.8 `voxel_grid → avg_pool_x` feature path | Same pinned packages; finite float32 positions and features, 1D–3D; compact and fixed-size rows | CPU/MPS grid IDs exact; pooled features, batch labels, input and bias-free model-weight gradients within documented tolerance; 256-point single-voxel gradient exact. A hosted Torch 2.12 runner showed a separate bias-bearing `nn.Linear` discrepancy after identical pooling. | [Feature pooling contract and hosted probe](pyg28-voxel-avg-pool.md); full [Safe](pytest-pyg28-voxel-avg-safe-2026-10-01.log) **352 passed / 9 skipped**, [Fast](pytest-pyg28-voxel-avg-fast-2026-10-01.log) **351 passed / 10 skipped**, test source `0c3a62d82f94128886df1ce725428f704e67bd49` based on main `94398da77fb305039c9648380182d55747609ad1`. |
| New compact `mps_pointops.voxel` API | Torch 2.12.0/PyG 2.8.0/pyg-lib 0.7.0+pt212 and separate Torch 2.14.1/torch-cluster 1.6.3; finite float32 1D–3D, nonnegative int64 unsorted/gapped batch IDs | Exact CPU/MPS cell rows, batch isolation, inverse and CSR maps and counts; mean positions, mean/sum features, and first-order gradients pass exact dyadic or stated tolerance tests. This floor-based compact API is distinct from both upstream raw-ID signatures. | [API contract](voxel-api-contract.md); focused 20/20 each mode/environment. Full pinned Torch 2.12 [Safe](pytest-voxel-api-torch212-safe-2026-10-01.log) **390/11**, [Fast](pytest-voxel-api-torch212-fast-2026-10-01.log) **389/12**; Torch 2.14 + real cluster [Safe](pytest-voxel-api-torch214-safe-2026-10-01.log) **373/28**, [Fast](pytest-voxel-api-torch214-fast-2026-10-01.log) **372/29** (passed/skipped). Source `caf28915a56ed0ec8ba9ce75d0e9c377a2cb301d`, M5 Pro macOS 26.5.2, fallback disabled. |
| Native graph aggregation in PyG 2.8 | PyTorch 2.14.1, PyG 2.8.0, no optional pyg-lib or torch-scatter; fixed 12-node model survey plus 4,096/32,768-node synthetic uniform/hub scatter workloads | GCN, GraphSAGE and GAT forward/backward passed on CPU and MPS for the survey fixture. Synchronized scatter profiling did not justify a custom Metal replacement. A separate Safe/Fast numerical probe found CPU/MPS parity but a mathematical-versus-native backward difference at zero extrema for `scatter_reduce_(include_self=False)`. | [Survey result](pyg-survey/2026-10-01-m5-pro-pyg28.md), [raw survey JSON](pyg-survey/2026-10-01-m5-pro-pyg28.json), [scatter profile](pyg-survey/2026-10-01-m5-pro-scatter-profile.md), and [backward probe with raw JSON](pyg-survey/2026-10-01-m5-pro-scatter-backward.md). |
| Legacy `torch_cluster.grid_cluster` | Original 1.6.3 CPU extension; finite float32 3D coordinates | 100 deterministic seeds with 1–100 points each yielded 5,050 matching point IDs; MPS Safe/Fast contracts passed. | [Legacy grid contract](grid-cluster-contract.md), [Safe](pytest-grid-upstream-safe-2026-10-01.log) / [Fast](pytest-grid-upstream-fast-2026-10-01.log) raw logs. |
| Legacy `torch_cluster.nearest` | Original 1.6.3 SciPy CPU path; finite well-separated D=1, 3, 64, 128 float32 features and ragged batches | 408 CPU/MPS indices match in Safe and Fast. A documented 1024-lane tie differs from SciPy CPU; CUDA binary was not run. | [Nearest contract](nearest-contract.md), [CPU](parity/nearest-upstream-cpu-2026-10-01.json) / [MPS Safe](parity/nearest-upstream-mps-safe-2026-10-01.json) / [MPS Fast](parity/nearest-upstream-mps-fast-2026-10-01.json) comparisons. |
| Legacy `torch_cluster.graclus_cluster` | Original 1.6.3 CPU C++ extension; documented finite-weight subset | 40/40 exact CPU label arrays across eight cases and five seeds. MPS Safe/Fast focused suites: 20 passed each, with structural checks for stochastic cases. CUDA matching is a different algorithm. | [Graclus contract](graclus-contract.md), [direct CPU JSON](parity/graclus-163-cpu.json), full [Safe](parity/graclus-full-safe.log) **331 passed / 22 skipped** and [Fast](parity/graclus-full-fast.log) **330 passed / 23 skipped** on main `61a4a076a38147e237d1dec36283ec720d811d6e`. |
| Legacy `torch_cluster.random_walk` | Original 1.6.3 CPU extension and this CPU/MPS int64 COO subset; positive finite `p,q` with documented weight bounds | Uniform nodes and edge IDs match the original CPU extension for 20 random seeds and sorted input fixtures under equal CPU RNG state and float32 default. Biased transitions match the stated distribution within tolerance; CUDA biased bit parity is not claimed. MPS Safe/Fast check edge validity, isolated nodes, and the float32 categorical boundary. | [Stochastic contract](random-walk-contract.md); full PyTorch 2.14.1 + real upstream [Safe](pytest-random-walk-torch214-safe-2026-10-01.log) **353 passed / 28 skipped**, [Fast](pytest-random-walk-torch214-fast-2026-10-01.log) **352 passed / 29 skipped**; full pinned PyTorch 2.12.0 + PyG 2.8 [Safe](pytest-random-walk-pyg28-safe-2026-10-01.log) **370 passed / 11 skipped**, [Fast](pytest-random-walk-pyg28-fast-2026-10-01.log) **369 passed / 12 skipped**. Tested code source `8ccf921834e3fdee43824bf6c8bd2058e90c062b` on main `ad7d703377fcfd34bf743967a0c8e532cab344b8`. |

These rows use different source revisions. A whole-suite count is the sum of
tests available in that revision and is not a benchmark or a count of supported
operators. The pinned PyG environment and the separate PyTorch 2.14.1
environment have different optional packages and therefore different skips.
Each linked contract records the operation's finer input bounds and source
provenance.

## Remaining gates

- The legacy random-walk tests cover a bounded CPU/MPS subset. Biased CUDA
  sampling parity, large-graph speed, and all possible random seeds remain
  unverified.
- `avg_pool` graph coarsening, nonfinite and ambiguous cell-boundary cases,
  performance, and real point-labeled datasets have no passing row here.
  The compact API's inverse maps do not change PyG's raw `voxel_grid` output.
- The graph survey does not cover all models, aggregation variants, PyTorch
  versions, dtypes, or M1–M4 machines. No custom Metal scatter speedup has
  been established.

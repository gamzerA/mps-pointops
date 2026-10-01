# Phase 3 compatibility matrix and release gate

This is the acceptance plan for the proposed 0.6.0–0.7.0 graph and grid work.
It records *tested surfaces*, not a promise that every PyG program on macOS
will run. Record each result with the exact package versions, source commit,
chip, macOS version, PyTorch build, input shape and dtype, and whether
`PYTORCH_ENABLE_MPS_FALLBACK=0` was set.

## Why there are two integration paths

PyG 2.8.0 deprecated `torch-cluster` and moved its optional accelerated
operators to `pyg-lib>=0.6.0` ([PyG 2.8.0 changelog](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/CHANGELOG.md),
[installation guide](https://pytorch-geometric.readthedocs.io/en/2.8.0/install/installation.html)).
The `mps_pointops.compat` `torch_cluster` shim is still useful for older PyG
versions and projects that import `torch_cluster` directly, but completing
that legacy call surface does not establish PyG 2.8 compatibility. The
`mps_pointops.pyg` registration is a separate path for real `pyg::*` schemas.
In particular, PyG 2.8 `Node2Vec` calls `torch.ops.pyg.random_walk`; the
implemented legacy `torch_cluster.random_walk` shim does not register it.

| Surface | Pinned reference for parity | Required evidence before marking complete |
| --- | --- | --- |
| PyG 2.8 direct pool/search | PyG 2.8.0 plus a recorded `pyg-lib` version | `fps`, `knn`, `radius`, and graph wrappers: outputs, first-order gradients where applicable, empty/ragged batches, a real schema, and no CPU fallback |
| Native graph aggregation | PyG 2.8.0 and a recorded PyTorch build | GCN, GraphSAGE, and GAT on documented representative graphs; compare forward and backward to CPU, profile synchronized native scatter, and log every unsupported operator |
| Legacy `torch_cluster` shim | `torch-cluster` 1.6.3 | Each exposed function against the original on CPU/CUDA where available, including ordering, randomness, batching, dtype, and empty inputs; no `NotImplementedError` within the published shim surface |
| Grid and voxel functions | A named, pinned reference implementation per function | Coordinate-to-cell boundary rule, batch isolation, output ordering, negative coordinates, duplicate points, dtype, and CPU/MPS parity |

The first PyG survey covered a fixed 12-node synthetic graph without optional
`pyg-lib` or `torch-scatter` packages. It found no missing operator for those
three models in that configuration; it is not a general compatibility result.
See the [survey record](pyg-survey/2026-10-01-m5-pro-pyg28.md).

## 0.6.0: measure before adding reductions

1. Profile `scatter_add_` and `scatter_reduce_` with MPS synchronization,
   separate Safe/Fast processes, uniform and hub degree distributions, and
   fixed graph sizes. Save the script, raw samples, environment and source
   hashes. Compare operator latency and complete model forward/backward.
2. Identify an actual missing operation, incorrect result, or measured
   bottleneck before replacing a native PyTorch path.
3. For a proposed segmented `sum`/`mean`/`max`/`min`/`argmax`/`argmin` kernel,
   specify empty-segment identities, tie-breaking by original row index,
   output dtype and index dtype, nonfinite values, and gradient behavior.
   Validate capability by a runtime feature check on each device; an MSL
   language-version check alone does not prove float atomic availability.
4. Benchmark correctness and total model time against the native path on each
   supported chip. A kernel merge does not imply a speedup.

The [dated M1/M5 scatter decision](pyg-survey/2026-10-02-m1-m5-scatter-decision.md)
closes the **profiling and current adoption decision** for its pinned synthetic
workloads: retain native PyTorch scatter. A custom Metal reduction, real-graph
ablation, and broader device coverage remain separate work.

## 0.7.0: grid and legacy operator coverage

1. Specify and test `grid_cluster` first, then voxelization and downsampling.
   These are distinct APIs: document returned coordinates, point-to-voxel
   mapping, aggregation and inverse mapping separately.
2. Specify `nearest`, `graclus_cluster`, and `random_walk` against the pinned
   upstream signatures. In particular, record `random_walk`'s seed and
   stochastic comparison policy. Replace shim placeholders only after parity
   tests pass for the supported input domain.
3. Run the two integration paths in the table independently. A release may
   claim “zero failures in the pinned compatibility matrix” only when every
   listed case passes. Keep untested models, devices, dtypes and optional PyG
   packages explicitly outside that claim.

The release checklist should link each matrix row to its raw log and package
hash. A checkbox in the README is marked `[x]` only for its stated scope;
unverified rows remain `[~]` or `[ ]`.

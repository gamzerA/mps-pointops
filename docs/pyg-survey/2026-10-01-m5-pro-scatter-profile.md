# Phase 3 native scatter survey on M5 Pro

The [benchmark script](../../bench/bench_pyg_scatter_mps.py) exercises native
PyTorch aggregation on the MPS backend with PyG 2.8.0 two-layer GCN,
GraphSAGE, and GAT models. Its source SHA-256 for this run is
`e9da3f0cc81dd94671f0c6c03b3b67de8f76c2058405760229e5224f184e6a66`;
the repository base is
`3580b6ceb21fd4c5a88cd89874abf6af9fd6d260` (`v0.5.0`). The full
[coordinated solo Safe JSON](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-solo-safe.json)
and [solo Fast JSON](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-solo-fast.json)
retain every timed sample, graph seed, PyG scatter source hash, PyTorch/PyG
versions, output checks, and CPU profiler input shapes. Readable result tables
are in the matching [Safe](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-solo-safe.md)
and [Fast](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-solo-fast.md)
reports. An earlier run while other MPS tasks were active is preserved as
[Safe](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-safe.json) and
[Fast](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-fast.json)
exploratory raw data, not merged into the solo medians.

## What was verified

- M5 Pro, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1, PyG 2.8.0.
  `PYTORCH_ENABLE_MPS_FALLBACK=0` was set before import. Safe and Fast Math
  ran in separate processes, with four warmups and 20 timed calls per case.
  The optional `pyg-lib`, `torch-scatter`, and `torch-cluster` packages were
  absent in this environment. This benchmark covers PyG 2.8 graph-model
  aggregation through PyTorch native operators. PyG 2.8 point-search
  registration through `pyg-lib` and the separate legacy `torch_cluster`
  compatibility shim are outside this survey.
- Synthetic directed graphs have 4,096 or 32,768 nodes, eight edges per node,
  32 float32 feature channels, and either uniform destinations or 80% of
  edges directed to the first 1% of nodes. The observed maximum in-degree
  is 20–22 for uniform and 701–703 for the hub cases.
- Native `scatter_add_` and `scatter_reduce_` (`sum`, `mean`, `amax`, `amin`)
  succeeded on MPS in both modes. Separate scalar add/amax controls approximate
  GAT attention aggregation with one self-loop per node. All seven controls
  matched CPU within `rtol=atol=1e-4`; maximum absolute difference among all
  recorded values was below `1.2e-4`. This tolerance does not claim bitwise
  agreement or deterministic float atomic order.
- Two-layer GCN, GraphSAGE, and GAT forward and backward completed in all
  24 solo model/mode/graph combinations. For both 4,096-node graph patterns,
  full model output and input feature gradients matched CPU within
  `rtol=atol=5e-4`; the largest recorded output absolute error was below
  `1.4e-5`. The 32,768-node model outputs were checked for finite gradients,
  but CPU numerical parity was not run at that size.
- CPU dispatch tracing showed `aten::scatter_add_` in all three models, and
  `aten::scatter_reduce_` in GAT. For the large graph, the observed GAT
  `scatter_reduce_` source shape was `[294909, 1]`, a scalar per edge/head
  after PyG inserted missing self-loops. The scalar control uses 294,912
  edges because it appends a self-loop even where the random input already
  has one; it is an approximation, not an exact replay. The channel-wide
  controls use `[262144, 32]` and stress feature aggregation instead.

## Timing status and decision

The first run overlapped other Phase 2 MPS work. We repeated both modes after
those tasks released the GPU. The second run is the coordinated solo result;
its absolute device exclusivity cannot be proved. Large GAT forward samples
still ranged from about 15 to 60 ms within a single case, so the oscillation
cannot be attributed solely to the known concurrent agent work. Power state,
runtime behavior, and other macOS GPU activity remain unseparated. The host
timer brackets each call with `torch.mps.synchronize()`; this closes the
asynchronous queue for our process but does not control system scheduling.

For context, the 32,768-node uniform graph produced these coordinated solo
medians in milliseconds:

| Mode | C32 `scatter_add_` | Scalar GAT `scatter_reduce_(amax)` | Two-layer GAT forward | Two-layer GAT backward |
| --- | ---: | ---: | ---: | ---: |
| Safe | 3.237 | 0.187 | 46.137 | 25.345 |
| Fast | 5.649 | 0.197 | 34.644 | 27.830 |

The operations have different inputs and dispatch counts, so these medians
**cannot** be divided to assign a fraction of model time to scatter. The
scalar max control is much smaller than the entire GAT call in both modes,
but does not isolate it from other attention work. PyG's mean aggregator uses
`scatter_add_` for sum and count; the slower isolated native
`scatter_reduce_(mean)` is not evidence of a PyG mean-path bottleneck.

This evidence confirms native support and graph-level correctness. It does
**not** yet justify replacing PyTorch's native scatter with a Metal float
atomic or segmented reduction kernel. A replacement needs the same call
shapes, representative real graphs, GPU-side profiling, repeat runs across
several run orders, and an end-to-end GCN/SAGE/GAT ablation. This is a
separate Phase 3 implementation gate.

## Proposed segmented reduction contract (design only)

Before implementing a Metal kernel, use finite float32 `src[E, C]`, int64
`index[E]` with every destination in `[0, N)`, and a stable grouping step
that forms `ptr[N+1]`. Output values have shape `[N, C]` and float32 dtype;
optional arg indices have shape `[N, C]`, int64 dtype, and refer to the
**original** edge row before grouping. `E=0` and `N=0` return allocated empty
or zero outputs without dispatch. Invalid indices or nonfinite inputs are
rejected until a separate numerical contract is defined.

| Operation | Empty segment | Nonempty segment | Backward for each source edge `e` |
| --- | --- | --- | --- |
| sum | `0` | Fixed-order sum | `g[index[e]]` |
| mean | `0` | Fixed-order sum divided by count | `g[index[e]] / count[index[e]]` |
| max, min | `0` | Extreme value | Split `g` equally among all exact tied extrema, matching the observed native `scatter_reduce_` CPU rule |
| argmax, argmin | `-1` internally | Original row of the lowest-index exact tied extremum | Index has no gradient; value follows the max/min rule |

The tie and empty-index rules are **proposed internal rules**, not a claim of
`torch_scatter`/PyG API parity. An adapter must map their sentinel and tie
behavior after direct upstream tests. Fixed-order reduction can be a
reproducible safe baseline, while float32 CPU/MPS bit parity is not assumed.
Do not select an atomic path from an MSL version string alone: first query
the actual device capability and verify shader compilation and execution on
that device, then compare correctness and end-to-end speed with the safe
path. No such kernel or runtime capability probe is included in this PR.

## Reproduce

Install PyTorch with MPS support and `torch-geometric==2.8.0`, then run in
separate processes:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python3 bench/bench_pyg_scatter_mps.py --output bench/results/local-scatter-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python3 bench/bench_pyg_scatter_mps.py --output bench/results/local-scatter-fast.json
```

The script times synchronized host wall time, including Python dispatch and
kernel launch; it does not report isolated GPU kernel duration. Its CPU
profiler captures operator names, counts, and shapes, not MPS execution time.
The [PyG 2.8 scatter source](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/utils/_scatter.py)
shows its sum/mean/min/max choices; [PyTorch 2.14 `scatter_reduce_`
documentation](https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.scatter_reduce_.html)
defines the `include_self=False` behavior used by the controls.

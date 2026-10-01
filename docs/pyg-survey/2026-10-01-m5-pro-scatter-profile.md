# Phase 3 native scatter survey on M5 Pro

The [benchmark script](../../bench/bench_pyg_scatter_mps.py) exercises native
PyTorch aggregation on the MPS backend with PyG 2.8.0 two-layer GCN,
GraphSAGE, and GAT models. Its source SHA-256 for this run is
`e9da3f0cc81dd94671f0c6c03b3b67de8f76c2058405760229e5224f184e6a66`;
the repository base is
`3580b6ceb21fd4c5a88cd89874abf6af9fd6d260` (`v0.5.0`). The full
[Safe JSON](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-safe.json)
and [Fast JSON](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-fast.json)
retain every timed sample, graph seed, PyG scatter source hash, PyTorch/PyG
versions, output checks, and CPU profiler input shapes. Readable result tables
are in the matching [Safe](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-safe.md)
and [Fast](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-fast.md)
reports.

## What was verified

- M5 Pro, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1, PyG 2.8.0.
  `PYTORCH_ENABLE_MPS_FALLBACK=0` was set before import. Safe and Fast Math
  ran in separate processes, with four warmups and 20 timed calls per case.
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
  24 model/mode/graph combinations. For both 4,096-node graph patterns,
  full model output and input feature gradients matched CPU within
  `rtol=atol=5e-4`; the largest recorded output absolute error was below
  `1.2e-5`. The 32,768-node model outputs were checked for finite gradients,
  but CPU numerical parity was not run at that size.
- CPU dispatch tracing showed `aten::scatter_add_` in all three models, and
  `aten::scatter_reduce_` in GAT. For the large graph, the observed GAT
  `scatter_reduce_` source shape was `[294909, 1]`, a scalar per edge/head
  after PyG inserted missing self-loops. The scalar control uses 294,912
  edges because it appends a self-loop even where the random input already
  has one; it is an approximation, not an exact replay. The channel-wide
  controls use `[262144, 32]` and stress feature aggregation instead.

## Timing status and decision

The current timing numbers are **exploratory**. Other Phase 2 MPS jobs were
running on the same M5 Pro during capture. Some large GAT forward samples
alternate between approximately 15 and 50 ms within one case, so the device
was not isolated enough for a quantitative speedup claim. The host timer
brackets each call with `torch.mps.synchronize()`, but synchronization cannot
remove competition from other GPU processes.

As context, on the 32,768-node uniform graph the Safe/Fast medians were
3.404/3.113 ms for a single 32-channel `scatter_add_`, 0.199/0.198 ms for a
single scalar `scatter_reduce_(amax)`, and 25.702/34.342 ms for the whole
two-layer GAT forward. These operations have different inputs and dispatch
counts, so these medians **cannot** be divided to assign a fraction of model
time to scatter. PyG's mean aggregator uses `scatter_add_` for sum and count;
the slower isolated native `scatter_reduce_(mean)` is not evidence of a PyG
mean-path bottleneck.

This evidence confirms native support and meaningful graph-level correctness.
It does **not** yet justify replacing PyTorch's native scatter with a Metal
float-atomic or segmented reduction kernel. The next performance gate is a
solo-GPU rerun, ideally across several run orders, followed by an end-to-end
ablation of a candidate reduction on the same GCN/SAGE/GAT call sites and
feature shapes. If a kernel is warranted, segmented reduction should be the
deterministic correctness baseline; a device-specific float-atomic path needs
runtime capability detection and measured improvement. That implementation
is a separate Phase 3 task.

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

# Native PyTorch scatter in PyG 2.8 graph workloads on MPS

Captured 2026-10-01T11:47:53.519708+00:00; code base `3580b6ceb21fd4c5a88cd89874abf6af9fd6d260`; benchmark SHA-256 `e9da3f0cc81dd94671f0c6c03b3b67de8f76c2058405760229e5224f184e6a66`.

Apple M5 Pro; macOS 26.5.2; Python 3.12.13; PyTorch 2.14.1; PyG 2.8.0. `PYTORCH_ENABLE_MPS_FALLBACK=0` was fixed before importing PyTorch.
Reproduce: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 python3 bench/bench_pyg_scatter_mps.py --output bench/results/2026-10-01-apple-m5-pro-pyg-scatter-fast.json`

4 warmups, 20 timed calls per case. All values below are synchronized host-wall medians in milliseconds.

## Native operation controls

| Nodes | Edges | Channels | Destination | Max in-degree | `scatter_add_` | `scatter_reduce_` sum | mean | amax | amin | GAT scalar add | GAT scalar amax |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32768 | 32 | uniform | 20 | 0.410 | 0.708 | 1.980 | 0.702 | 1.269 | 0.160 | 0.138 |
| 4096 | 32768 | 32 | hub | 703 | 0.410 | 0.753 | 0.974 | 0.963 | 0.736 | 0.150 | 0.179 |
| 32768 | 262144 | 32 | uniform | 22 | 3.113 | 8.486 | 14.862 | 9.172 | 9.733 | 0.158 | 0.198 |
| 32768 | 262144 | 32 | hub | 701 | 3.308 | 9.080 | 13.289 | 9.213 | 8.778 | 0.201 | 0.324 |

## Two-layer PyG models

| Nodes | Edges | Destination | Model | Forward | Backward | Observed CPU-dispatch operators |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 4096 | 32768 | uniform | GCN | 4.870 | 1.996 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | uniform | GraphSAGE | 1.287 | 1.479 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | uniform | GAT | 6.235 | 2.957 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 4096 | 32768 | hub | GCN | 2.225 | 1.602 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | hub | GraphSAGE | 2.121 | 1.542 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | hub | GAT | 4.528 | 2.376 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 32768 | 262144 | uniform | GCN | 17.728 | 24.273 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | uniform | GraphSAGE | 13.840 | 13.243 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | uniform | GAT | 34.342 | 27.116 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 32768 | 262144 | hub | GCN | 19.157 | 27.325 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | hub | GraphSAGE | 13.407 | 13.001 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | hub | GAT | 35.525 | 27.964 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |

## Scope and interpretation

- Graphs are seeded synthetic directed edges. Uniform destinations are sampled across all nodes; hub destinations send 80% of edges to the first 1% of nodes. These are stress shapes, not a public dataset.
- Channel-wide isolated operations use a precomputed contiguous float32 edge-feature source, int64 destination indices expanded across channels, and a preallocated output. The GAT scalar controls append one self-loop per node, matching its attention reduction shape. Output reset is outside timing. All seven MPS outputs are compared with CPU using rtol=atol=1e-4; reduction order is not bit-stable.
- Models use PyG 2.8 GCN, GraphSAGE and GAT, two layers, dropout 0, float32. The smaller graph cases also compare model output and input gradient with CPU (rtol=atol=5e-4). Full model timings include normalization, gather, aggregation, dense layers, autograd and dispatch.
- CPU profiler operator names/counts establish that a native scatter path was invoked, but its self CPU time is *not* GPU time. Input shapes are in the raw JSON. The isolated scatter median cannot be divided by the model median to assign a model bottleneck percentage: message shapes and dispatch counts differ.
- `scatter_reduce_` uses `include_self=False` for sum, mean, amax and amin; this matches PyG's MPS min/max reduction path. PyG mean aggregation itself uses `scatter_add_` for sums and counts. The raw JSON retains every timing sample and correctness result.
- A custom Metal segmented reduction or float-atomic path should be built only after an end-to-end bottleneck is established and compared on more hardware and graph distributions.

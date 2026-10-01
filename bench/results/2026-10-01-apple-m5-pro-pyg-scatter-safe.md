# Native PyTorch scatter in PyG 2.8 graph workloads on MPS

Captured 2026-10-01T11:47:29.323570+00:00; code base `3580b6ceb21fd4c5a88cd89874abf6af9fd6d260`; benchmark SHA-256 `e9da3f0cc81dd94671f0c6c03b3b67de8f76c2058405760229e5224f184e6a66`.

Apple M5 Pro; macOS 26.5.2; Python 3.12.13; PyTorch 2.14.1; PyG 2.8.0. `PYTORCH_ENABLE_MPS_FALLBACK=0` was fixed before importing PyTorch.
Reproduce: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python3 bench/bench_pyg_scatter_mps.py --output bench/results/2026-10-01-apple-m5-pro-pyg-scatter-safe.json`

4 warmups, 20 timed calls per case. All values below are synchronized host-wall medians in milliseconds.

## Native operation controls

| Nodes | Edges | Channels | Destination | Max in-degree | `scatter_add_` | `scatter_reduce_` sum | mean | amax | amin | GAT scalar add | GAT scalar amax |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32768 | 32 | uniform | 20 | 0.418 | 1.212 | 0.972 | 0.772 | 0.757 | 0.144 | 0.189 |
| 4096 | 32768 | 32 | hub | 703 | 0.395 | 0.916 | 0.961 | 1.282 | 0.718 | 0.165 | 0.169 |
| 32768 | 262144 | 32 | uniform | 22 | 3.404 | 8.853 | 13.567 | 9.573 | 9.636 | 0.161 | 0.199 |
| 32768 | 262144 | 32 | hub | 701 | 3.357 | 8.791 | 14.081 | 9.268 | 8.461 | 0.191 | 0.382 |

## Two-layer PyG models

| Nodes | Edges | Destination | Model | Forward | Backward | Observed CPU-dispatch operators |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 4096 | 32768 | uniform | GCN | 4.996 | 1.956 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | uniform | GraphSAGE | 1.339 | 2.018 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | uniform | GAT | 3.503 | 2.538 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 4096 | 32768 | hub | GCN | 2.519 | 2.478 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | hub | GraphSAGE | 1.199 | 1.628 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | hub | GAT | 5.022 | 2.766 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 32768 | 262144 | uniform | GCN | 18.953 | 24.954 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | uniform | GraphSAGE | 11.508 | 15.683 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | uniform | GAT | 25.702 | 28.809 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 32768 | 262144 | hub | GCN | 18.910 | 24.071 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | hub | GraphSAGE | 13.788 | 13.207 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | hub | GAT | 38.704 | 29.776 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |

## Scope and interpretation

- Graphs are seeded synthetic directed edges. Uniform destinations are sampled across all nodes; hub destinations send 80% of edges to the first 1% of nodes. These are stress shapes, not a public dataset.
- Channel-wide isolated operations use a precomputed contiguous float32 edge-feature source, int64 destination indices expanded across channels, and a preallocated output. The GAT scalar controls append one self-loop per node, matching its attention reduction shape. Output reset is outside timing. All seven MPS outputs are compared with CPU using rtol=atol=1e-4; reduction order is not bit-stable.
- Models use PyG 2.8 GCN, GraphSAGE and GAT, two layers, dropout 0, float32. The smaller graph cases also compare model output and input gradient with CPU (rtol=atol=5e-4). Full model timings include normalization, gather, aggregation, dense layers, autograd and dispatch.
- CPU profiler operator names/counts establish that a native scatter path was invoked, but its self CPU time is *not* GPU time. Input shapes are in the raw JSON. The isolated scatter median cannot be divided by the model median to assign a model bottleneck percentage: message shapes and dispatch counts differ.
- `scatter_reduce_` uses `include_self=False` for sum, mean, amax and amin; this matches PyG's MPS min/max reduction path. PyG mean aggregation itself uses `scatter_add_` for sums and counts. The raw JSON retains every timing sample and correctness result.
- A custom Metal segmented reduction or float-atomic path should be built only after an end-to-end bottleneck is established and compared on more hardware and graph distributions.

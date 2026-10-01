# Native PyTorch scatter in PyG 2.8 graph workloads on MPS

Captured 2026-10-01T11:51:37.924578+00:00; code base `404677cdac5f88f2a7ec92ff25937ab66a914f60`; benchmark SHA-256 `e9da3f0cc81dd94671f0c6c03b3b67de8f76c2058405760229e5224f184e6a66`.

Apple M5 Pro; macOS 26.5.2; Python 3.12.13; PyTorch 2.14.1; PyG 2.8.0. `PYTORCH_ENABLE_MPS_FALLBACK=0` was fixed before importing PyTorch.
Reproduce: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python3 bench/bench_pyg_scatter_mps.py --output bench/results/2026-10-01-apple-m5-pro-pyg-scatter-solo-safe.json`

4 warmups, 20 timed calls per case. All values below are synchronized host-wall medians in milliseconds.

## Native operation controls

| Nodes | Edges | Channels | Destination | Max in-degree | `scatter_add_` | `scatter_reduce_` sum | mean | amax | amin | GAT scalar add | GAT scalar amax |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32768 | 32 | uniform | 20 | 0.403 | 0.700 | 1.442 | 0.723 | 1.524 | 0.176 | 0.147 |
| 4096 | 32768 | 32 | hub | 703 | 0.397 | 1.045 | 0.966 | 1.151 | 0.746 | 0.164 | 0.198 |
| 32768 | 262144 | 32 | uniform | 22 | 3.237 | 8.572 | 13.154 | 8.450 | 9.825 | 0.177 | 0.187 |
| 32768 | 262144 | 32 | hub | 701 | 3.134 | 8.948 | 12.038 | 9.391 | 8.245 | 0.198 | 0.385 |

## Two-layer PyG models

| Nodes | Edges | Destination | Model | Forward | Backward | Observed CPU-dispatch operators |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 4096 | 32768 | uniform | GCN | 1.875 | 2.432 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | uniform | GraphSAGE | 1.115 | 1.644 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | uniform | GAT | 4.884 | 2.478 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 4096 | 32768 | hub | GCN | 3.805 | 1.417 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | hub | GraphSAGE | 2.338 | 1.546 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | hub | GAT | 5.314 | 3.079 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 32768 | 262144 | uniform | GCN | 16.564 | 22.013 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | uniform | GraphSAGE | 12.606 | 13.705 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | uniform | GAT | 46.137 | 25.345 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 32768 | 262144 | hub | GCN | 20.547 | 22.041 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | hub | GraphSAGE | 13.856 | 13.180 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | hub | GAT | 34.317 | 23.477 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |

## Scope and interpretation

- Graphs are seeded synthetic directed edges. Uniform destinations are sampled across all nodes; hub destinations send 80% of edges to the first 1% of nodes. These are stress shapes, not a public dataset.
- Channel-wide isolated operations use a precomputed contiguous float32 edge-feature source, int64 destination indices expanded across channels, and a preallocated output. The GAT scalar controls append one self-loop per node, matching its attention reduction shape. Output reset is outside timing. All seven MPS outputs are compared with CPU using rtol=atol=1e-4; reduction order is not bit-stable.
- Models use PyG 2.8 GCN, GraphSAGE and GAT, two layers, dropout 0, float32. The smaller graph cases also compare model output and input gradient with CPU (rtol=atol=5e-4). Full model timings include normalization, gather, aggregation, dense layers, autograd and dispatch.
- CPU profiler operator names/counts establish that a native scatter path was invoked, but its self CPU time is *not* GPU time. Input shapes are in the raw JSON. The isolated scatter median cannot be divided by the model median to assign a model bottleneck percentage: message shapes and dispatch counts differ.
- `scatter_reduce_` uses `include_self=False` for sum, mean, amax and amin; this matches PyG's MPS min/max reduction path. PyG mean aggregation itself uses `scatter_add_` for sums and counts. The raw JSON retains every timing sample and correctness result.
- A custom Metal segmented reduction or float-atomic path should be built only after an end-to-end bottleneck is established and compared on more hardware and graph distributions.

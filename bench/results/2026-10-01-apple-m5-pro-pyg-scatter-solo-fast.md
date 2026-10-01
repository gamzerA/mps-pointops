# Native PyTorch scatter in PyG 2.8 graph workloads on MPS

Captured 2026-10-01T11:51:58.669421+00:00; code base `404677cdac5f88f2a7ec92ff25937ab66a914f60`; benchmark SHA-256 `e9da3f0cc81dd94671f0c6c03b3b67de8f76c2058405760229e5224f184e6a66`.

Apple M5 Pro; macOS 26.5.2; Python 3.12.13; PyTorch 2.14.1; PyG 2.8.0. `PYTORCH_ENABLE_MPS_FALLBACK=0` was fixed before importing PyTorch.
Reproduce: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 python3 bench/bench_pyg_scatter_mps.py --output bench/results/2026-10-01-apple-m5-pro-pyg-scatter-solo-fast.json`

4 warmups, 20 timed calls per case. All values below are synchronized host-wall medians in milliseconds.

## Native operation controls

| Nodes | Edges | Channels | Destination | Max in-degree | `scatter_add_` | `scatter_reduce_` sum | mean | amax | amin | GAT scalar add | GAT scalar amax |
| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32768 | 32 | uniform | 20 | 0.454 | 0.658 | 1.573 | 0.728 | 0.697 | 0.181 | 0.168 |
| 4096 | 32768 | 32 | hub | 703 | 0.505 | 0.741 | 1.316 | 1.060 | 0.749 | 0.186 | 0.159 |
| 32768 | 262144 | 32 | uniform | 22 | 5.649 | 9.693 | 15.263 | 10.037 | 8.891 | 0.302 | 0.197 |
| 32768 | 262144 | 32 | hub | 701 | 3.296 | 8.157 | 13.955 | 9.325 | 7.822 | 0.198 | 0.305 |

## Two-layer PyG models

| Nodes | Edges | Destination | Model | Forward | Backward | Observed CPU-dispatch operators |
| ---: | ---: | --- | --- | ---: | ---: | --- |
| 4096 | 32768 | uniform | GCN | 2.372 | 1.731 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | uniform | GraphSAGE | 1.231 | 1.646 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | uniform | GAT | 4.957 | 3.474 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 4096 | 32768 | hub | GCN | 5.166 | 2.108 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | hub | GraphSAGE | 1.798 | 2.059 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 4096 | 32768 | hub | GAT | 5.876 | 3.214 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 32768 | 262144 | uniform | GCN | 18.769 | 21.253 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | uniform | GraphSAGE | 12.668 | 14.408 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | uniform | GAT | 34.644 | 27.830 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |
| 32768 | 262144 | hub | GCN | 17.259 | 22.871 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | hub | GraphSAGE | 10.249 | 17.960 | aten::gather×2, aten::index_add_×2, aten::scatter_add_×4 |
| 32768 | 262144 | hub | GAT | 35.021 | 28.506 | aten::gather×4, aten::index_add_×8, aten::scatter_add_×4, aten::scatter_reduce_×2 |

## Scope and interpretation

- Graphs are seeded synthetic directed edges. Uniform destinations are sampled across all nodes; hub destinations send 80% of edges to the first 1% of nodes. These are stress shapes, not a public dataset.
- Channel-wide isolated operations use a precomputed contiguous float32 edge-feature source, int64 destination indices expanded across channels, and a preallocated output. The GAT scalar controls append one self-loop per node, matching its attention reduction shape. Output reset is outside timing. All seven MPS outputs are compared with CPU using rtol=atol=1e-4; reduction order is not bit-stable.
- Models use PyG 2.8 GCN, GraphSAGE and GAT, two layers, dropout 0, float32. The smaller graph cases also compare model output and input gradient with CPU (rtol=atol=5e-4). Full model timings include normalization, gather, aggregation, dense layers, autograd and dispatch.
- CPU profiler operator names/counts establish that a native scatter path was invoked, but its self CPU time is *not* GPU time. Input shapes are in the raw JSON. The isolated scatter median cannot be divided by the model median to assign a model bottleneck percentage: message shapes and dispatch counts differ.
- `scatter_reduce_` uses `include_self=False` for sum, mean, amax and amin; this matches PyG's MPS min/max reduction path. PyG mean aggregation itself uses `scatter_add_` for sums and counts. The raw JSON retains every timing sample and correctness result.
- A custom Metal segmented reduction or float-atomic path should be built only after an end-to-end bottleneck is established and compared on more hardware and graph distributions.

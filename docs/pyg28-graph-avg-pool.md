# PyG 2.8 graph `avg_pool` on MPS

This is a **tested integration path**, not a new `avg_pool` kernel. In PyG
2.8.0, graph `avg_pool(cluster, data)` uses PyTorch tensor operations for
feature and position means, edge rewiring, duplicate-edge sums, and batch
metadata. Those operations ran on MPS in the bounded fixture below with
`PYTORCH_ENABLE_MPS_FALLBACK=0`. If `cluster` comes from PyG `voxel_grid`,
call `mps_pointops.pyg.register_mps()` first to install the existing
`pyg::grid_cluster` MPS dispatch. Direct `avg_pool` of a supplied cluster
needs no mps-pointops registration.

## Pinned upstream contract

Reference stack: PyTorch **2.12.0**, PyG **2.8.0**, and pyg-lib
**0.7.0+pt212**. The contract below was read from the tagged PyG
[`avg_pool`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/avg_pool.py),
[`pool_edge`, `pool_batch`, `pool_pos`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/pool.py),
and [`coalesce`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/utils/_coalesce.py)
implementations. No upstream code was copied into this project.

Let `c_i` be the cluster ID of node `i`, `x_i` its feature vector, and `p_i`
its position. The sorted unique IDs become consecutive coarse nodes
`0,...,U-1`. For coarse node `u`, let `C_u = {i : c_i maps to u}` and
`n_u = |C_u|`. PyG computes

```text
x'_u = (sum_{i in C_u} x_i) / n_u
p'_u = (sum_{i in C_u} p_i) / n_u
b'_u = b_{perm_u}, where perm_u is the last input node in C_u
```

The batch value is meaningful when every member of a cluster has the same
batch ID. Batch IDs may have gaps. The caller must keep cluster IDs distinct
across graphs and pass edges within their graph; `avg_pool` does not enforce
either condition. A `voxel_grid` cluster computed with PyG's batch dimension
provides graph-separated IDs for valid batched point sets.

Each input edge `(i,j)` is mapped to coarse edge `(u(c_i),u(c_j))`.
Self-loops **after mapping** are removed. Remaining duplicates are coalesced
and sorted by `(source,target)`. With edge attributes `a_e`, PyG uses

```text
a'_(u,v) = sum_{e: mapped(e)=(u,v), u != v} a_e.
```

The tested output is a `DataBatch` containing `x`, `pos`, `edge_index`,
`edge_attr`, and `batch`; attributes not passed to PyG's `avg_pool` output
constructor are outside this contract. For loss gradients `g^x_u`, `g^p_u`,
and `g^a_(u,v)` from those outputs, the chain rule gives

```text
dL/dx_i = g^x_(u(c_i)) / n_(u(c_i))
dL/dp_i = g^p_(u(c_i)) / n_(u(c_i))
dL/da_e = 0 when mapped(e) is a self-loop;
         = g^a_(u(c_i),u(c_j)) otherwise.
```

These formulas refer to fixed integer cluster and edge indices. No gradient
through cluster assignment or `voxel_grid` cell boundaries is claimed.

## Tested scope

The [regression](../tests/test_pyg28_avg_pool_graph.py) uses finite float32
node features, positions, and two-column edge attributes; int64 cluster,
batch, and edge indices; ten nodes in unequal four/six-node graphs with a
missing batch ID; nonconsecutive cluster IDs; parallel edges, self-loops,
and an empty graph. It checks exact CPU/MPS integer topology and batch
labels; CPU/MPS float32 pooled values and first-order gradients within
`rtol=1e-5, atol=1e-6`; and zero edge-attribute gradients for removed
self-loops. A separate path checks `voxel_grid -> avg_pool` with the real
pyg-lib schema and MPS registration, including loss and `x`, `pos`, and
`edge_attr` backward parity. This scope is finite and boundary-safe.

Float32 reduction order can vary on CPU and MPS; this fixture does not
promise bitwise equality of sums. It does not establish performance gains,
arbitrary PyG model compatibility, mixed-graph clusters, nonfinite inputs,
other dtypes, gradients through clustering, or behavior on M1–M4 hardware.
The hosted Torch 2.12 MPS [`nn.Linear` bias discrepancy](pyg28-voxel-avg-pool.md)
is independent of this graph pooling test, which does not use `nn.Linear`.

Run Safe and Fast Math in separate processes:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  pytest -ra tests/test_pyg28_avg_pool_graph.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  pytest -ra tests/test_pyg28_avg_pool_graph.py
```

On a physical Apple **M5 Pro**, macOS **26.5.2**, the pinned PyTorch 2.12.0
suite passed **394 tests with 11 skips** in Safe Math and **393 with 12
skips** in Fast Math. The new focused graph test passed **4/4** in each
process. The [Safe](pytest-pyg28-graph-avg-pool-safe-2026-10-01.log) and
[Fast](pytest-pyg28-graph-avg-pool-fast-2026-10-01.log) raw pytest logs record
the full suite. All commands used CPU fallback disabled. CI's
`macos-latest` runner may report a virtual Apple chip; it is functional
CI coverage, not a physical M1–M4 benchmark.

## Synchronized coarsening measurement

The [benchmark script](../bench/bench_pyg28_avg_pool.py) measures PyG's
direct `avg_pool(cluster, data)` forward and backward, including edge
rewiring/coalescing. It does **not** include `voxel_grid`, device transfer,
data generation, or Python model layers. Three graph batches have unequal
sizes and no cross-batch edges. Each coarse node groups four input nodes;
each node contributes eight edges, including duplicates after clustering.
Features have 32 columns, positions 3, and edge attributes 4. Inputs are
resident on each device before timing. MPS is synchronized before the
forward, between forward and backward, and after backward. The loss is the
sum of means of squares of pooled `x`, `pos`, and `edge_attr`. Each result
uses three warmups and the median of ten timed repetitions. The reported
**backward** interval begins before constructing that loss and ends after
autograd backward and the MPS synchronization; it is not isolated kernel-only
gradient time.

| Math | Nodes / edges | CPU forward / backward | MPS forward / backward |
| --- | ---: | ---: | ---: |
| Safe | 16,384 / 131,072 | 2.619 / 1.353 ms | 3.933 / 1.527 ms |
| Safe | 65,536 / 524,288 | 8.953 / 3.713 ms | 12.055 / 2.778 ms |
| Fast | 16,384 / 131,072 | 2.387 / 1.172 ms | 5.776 / 1.100 ms |
| Fast | 65,536 / 524,288 | 8.707 / 3.671 ms | 11.105 / 4.018 ms |

The [Safe raw samples](bench-pyg28-graph-avg-pool-safe-2026-10-01.json) and
[Fast raw samples](bench-pyg28-graph-avg-pool-fast-2026-10-01.json) include
all repetitions, version/device metadata, and the synthetic shape. MPS
forward samples varied substantially: for 65,536 nodes, **5.145–23.187 ms**
in Safe and **5.470–21.318 ms** in Fast. These data do not establish an MPS
speedup or a stable Safe/Fast difference. There is no physical M1–M4 result.

The exact benchmark commands were:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_pyg28_avg_pool.py --nodes 16384 65536 \
  --warmups 3 --repeats 10 --output docs/bench-pyg28-graph-avg-pool-safe-2026-10-01.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/bench_pyg28_avg_pool.py --nodes 16384 65536 \
  --warmups 3 --repeats 10 --output docs/bench-pyg28-graph-avg-pool-fast-2026-10-01.json
```

The benchmark script was committed unchanged after the measurement at
`a0b904d2264cdb3cf5fad09f301fffe0171183ac`; both JSON files annotate
that Git commit ID and the script's SHA-256
`f61453504df6406c8f0ba8ef7638b841cb704b27b74a2b864d6ba8d3344799ac`
without modifying any timing samples. The composed-path
gradient test and refreshed full-suite logs are in
`491ca7d71d91ce9c45b190935e4dd8a3dab962a2`. Their raw log paths are
linked above. The cross-operation scope is recorded in the dated
[Phase 3 results matrix](phase3-results-2026-10-01.md) and associated PR.
The broader [Phase 3 release gate](https://github.com/gamzerA/mps-pointops/issues/41)
remains open; this measured fixture is one bounded part of that gate.

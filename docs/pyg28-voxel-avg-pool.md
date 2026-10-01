# PyG 2.8 voxel feature mean pooling on MPS

This integration covers the **feature** path `voxel_grid(pos, ...)` followed by
`avg_pool_x(cluster, x, batch, ...)` for finite float32 1D–3D spatial
coordinates and float32 features. It uses the existing `pyg::grid_cluster`
MPS registration and PyG's own mean reduction. It adds no new Metal shader,
PyG operator schema, or replacement for PyG's CPU implementation. The legacy
`torch_cluster.grid_cluster` shim is separate.

## Pinned upstream contract

The reference versions are PyG **2.8.0**, `pyg-lib` **0.7.0**, and PyTorch
**2.12.0**. The tagged [voxel-grid wrapper](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/voxel_grid.py)
produces int64 mixed-radix IDs using the existing `pyg::grid_cluster` schema;
the [grid-ID contract](pyg28-grid-cluster-contract.md) specifies its arithmetic.
The tagged [`avg_pool_x` implementation](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/avg_pool.py),
[`consecutive_cluster`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/consecutive.py),
and [mean `scatter`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/utils/_scatter.py)
define the following behavior. The upstream source is cited, not copied.

Let `c_i` be the raw grid ID of point `i`, `x_i` its feature row, and `b_i`
its batch ID. For `size=None`, let `u_0 < ... < u_(U-1)` be the sorted unique
values of `c`. PyG assigns consecutive output row `t` to raw ID `u_t` and
computes

```text
n_t       = #{i : c_i = u_t}
pooled[t] = (sum_{i : c_i = u_t} x_i) / n_t
batch'[t] = b_i for any i with c_i = u_t
```

The grid's batch coordinate makes all members of one valid voxel share the
same batch ID, so the member selected for `batch'[t]` does not change its
value. Empty raw IDs are omitted. The returned pair is `(pooled, batch')`
with shapes `(U, C)` and `(U,)` for a feature matrix `x: (N,C)`.

For integer `size=s`, PyG does **not** compact IDs. It sets `B` from the
explicit `batch_size` or `max(batch)+1`, allocates `B*s` output rows, and uses
each raw `c_i` directly as the output row. Valid IDs must lie in
`{0,...,B*s-1}`. The formula is

```text
n_j       = #{i : c_i = j}
pooled[j] = (sum_{i : c_i = j} x_i) / max(n_j, 1),  j = 0,...,B*s-1
```

Empty rows are zero. The tagged implementation returns `(pooled, None)` even
though its docstring describes the fixed-size result as a bare tensor. With
explicit grid bounds and batches numbered from zero, `s` should be the
product of the spatial grid extents if each batch is meant to occupy exactly
one contiguous `s`-row block. Passing a merely large enough `s` does not
reindex raw IDs.

The mean comes from two PyTorch `scatter_add_` calls, one for counts and one
for feature sums, then a division. For a loss `L`, the feature gradient is

```text
dL/dx_i = (dL/dpooled[row(c_i)]) / n_(row(c_i)).
```

No gradient is defined for integer grid IDs, and this path does not provide
coordinate gradients through voxel assignment. Float32 addition order can
vary between CPU and MPS, so forward and backward parity uses `rtol=1e-5`,
`atol=1e-6`; the separate 256-point single-voxel test checks the exact
`1/256` feature gradient.

## MPS behavior and reproduction

In the pinned environment on an M5 Pro with macOS 26.5.2 and
`PYTORCH_ENABLE_MPS_FALLBACK=0`, `voxel_grid` without `register_mps()` first
fails at `pyg::grid_cluster` with `NotImplementedError`. After
`mps_pointops.pyg.register_mps()`, both the compact and fixed-size
`voxel_grid -> avg_pool_x -> Linear -> loss.backward()` paths ran on MPS in
separate Safe and Fast Math processes. PyG's mean scatter and its backward
used existing PyTorch MPS operations; this observation does not establish a
speedup. The pinned CPU outputs and gradients are the reference.

Run the focused regression in separate processes because Metal Math mode is
cached within a process:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest tests/test_pyg28_voxel_avg_pool.py -q
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python -m pytest tests/test_pyg28_voxel_avg_pool.py -q
```

The focused tests cover 1D, 2D, and 3D positions; missing batch number 1;
`size=None` and fixed-size output shapes and batch metadata; a model loss;
input-feature and linear-weight gradients; and 256 points in one voxel. They
also check the tagged PyG/pyg-lib CPU output. The package remains usable
without optional PyG and pyg-lib; those differential tests then skip.

The scope excludes nonfinite points, ambiguous float32 cell boundaries,
float16/float64 features, higher spatial dimensions, negative or out-of-range
fixed-size cluster IDs, and empty inputs. The separate graph-coarsening
[`avg_pool`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/avg_pool.py)
path includes edge rewiring, position pooling, and graph batching; it is not
validated here. Real point-cloud datasets, larger hardware coverage, and
performance measurements remain separate work.

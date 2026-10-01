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

This is the chain-rule formula; the implementation accumulates counts in the
feature dtype, so very large float32 counts and sums can round. Our exact
gradient fixture uses 256 points, for which the count is represented exactly.
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
`voxel_grid -> avg_pool_x -> bias-free Linear -> loss.backward()` paths ran on
MPS in separate Safe and Fast Math processes. PyG's mean scatter and its
backward used existing PyTorch MPS operations; this observation does not
establish a speedup. The pinned CPU outputs and gradients are the reference.

A bias-bearing `nn.Linear` was tested initially on the M5 Pro and passed,
but the [hosted macOS CI diagnostic run](https://github.com/gamzerA/mps-pointops/actions/runs/36865707274/job/110380541934)
on the `macos-26-arm64` runner image with Python 3.12.10 and Torch 2.12.0
diverged: grid IDs and pooled features matched the CPU, and the transferred
Linear weight and bias matched bit for bit. Every projected value omitted its
corresponding bias; the largest output difference was `0.75`. For the
four-row compact case the CPU loss was `12.3370438`, while MPS returned
`9.2584095`, matching the CPU computation with bias set to zero. This is a
large projection difference, not a reduction-order or tolerance effect.

A [second hosted job](https://github.com/gamzerA/mps-pointops/actions/runs/36866677241/job/110383773339)
at PR head `d6c608b4bb010843b83b01235fb85b9899f758e0`
split the projection into `xWᵀ`, `b`, explicit `xWᵀ+b`, `torch.addmm`, and
`nn.Linear`, first for a fixed matrix without PyG and then for PyG's pooled
features. Both inputs showed the same behavior; the grid IDs were equal and
the pooled tensors had maximum absolute difference `0.0`. On the independent
input, CPU and MPS `xWᵀ` norms were `5.9161038` and `5.9161034`, the bias
norm was about `0.9354143`, and explicit addition and `addmm` both printed
norm `7.4795647`. MPS `nn.Linear` instead printed the bias-free norm
`5.9161034`; `max|Linear-(xWᵀ+b)|=0.75` and `max|Linear-xWᵀ|=0.0`.
For the pooled input, explicit addition and `addmm` printed norm
`12.1673546`, while MPS `nn.Linear` printed the bias-free norm `10.5404415`
with the same maximum differences. The `addmm` tensor values matched explicit
addition at the log's printed precision; this probe did not calculate their
elementwise maximum difference. The [raw probe excerpt](ci-pyg28-linear-bias-probe-2026-10-01.log)
preserves all printed tensors and norms. This isolates the symptom from PyG
pooling, but does not identify the internal cause of the MPS Linear behavior.
The portable integration fixture therefore uses a bias-free projection.
Bias-bearing model parity across MPS environments remains unverified.

### Standalone framework probe

The earlier hosted probe ran inside a PyG integration test process. The
[standalone diagnostic](../tools/diagnose_mps_linear.py) starts a fresh process
that imports only PyTorch. It fixes the same four input rows, `3 × 4` weight,
and three bias values; records `xWᵀ`, explicit `xWᵀ+b`, `torch.addmm`,
`torch.nn.functional.linear`, and `nn.Linear`; calls
`torch.mps.synchronize()` before every MPS-to-CPU readback; and prints the
elementwise and maximum differences. It makes no numerical assertion, so a
framework discrepancy cannot fail the package's product tests. The
[diagnostic workflow](../.github/workflows/diagnose-mps-linear.yml) pins Torch
2.12.0 on hosted macOS and runs Safe and Fast Math in separate Python
processes, without installing this package or PyG.

On the local M5 Pro, macOS 26.5.2, Python 3.12.13, and Torch 2.12.0, both
[Safe Math](diagnostics/mps-linear-m5-pro-torch212-safe-2026-10-01.json) and
[Fast Math](diagnostics/mps-linear-m5-pro-torch212-fast-2026-10-01.json) fresh
processes gave `max|nn.Linear-(xWᵀ+b)| = 0`,
`max|functional.linear-(xWᵀ+b)| = 0`, and exact CPU/MPS equality for every
reported tensor. The same fixed input in the previous hosted integration
process had `max|nn.Linear-(xWᵀ+b)| = 0.75`; its
`functional.linear` result was not recorded. These observations establish
environment or process variability, without identifying a cause or assigning
responsibility to PyTorch internals. A fresh hosted process is required to
separate those possibilities further.

The following independent snippet reproduces the projection comparison.
It passed on the M5 Pro with pinned Torch 2.12.0; the hosted job's full probe
used this same model and weights alongside a pooled-input case:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python - <<'PY'
from copy import deepcopy
import torch

x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 7
cpu = torch.nn.Linear(4, 3)
with torch.no_grad():
    cpu.weight.copy_(torch.arange(12).reshape(3, 4) / 19)
    cpu.bias.copy_(torch.tensor([-0.25, 0.5, 0.75]))
mps = deepcopy(cpu).to('mps')
print('bias on CPU:', cpu.bias.detach())
print('bias copied to MPS:', mps.bias.detach().cpu())
print('CPU output:', cpu(x).detach())
print('MPS output:', mps(x.to('mps')).detach().cpu())
PY
```

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
input-feature and bias-free linear-weight gradients; and 256 points in one voxel. They
also check the tagged PyG/pyg-lib CPU output. The package remains usable
without optional PyG and pyg-lib; those differential tests then skip.

On the source revision `0c3a62d82f94128886df1ce725428f704e67bd49`, based
on `main` commit `94398da77fb305039c9648380182d55747609ad1`, the pinned
M5 Pro environment passed the full suite with **352 passed, 9 skipped** in
Safe Math and **351 passed, 10 skipped** in Fast Math. The focused tests
passed **8/8** in each mode. See the raw
[Safe](pytest-pyg28-voxel-avg-safe-2026-10-01.log) and
[Fast](pytest-pyg28-voxel-avg-fast-2026-10-01.log) pytest logs. In the
separate Torch 2.14.1 environment without optional PyG/pyg-lib, the full
suite passed **331 with 30 skips** in Safe Math and **330 with 31 skips** in
Fast Math; see its [Safe](pytest-pyg28-voxel-avg-optional-safe-2026-10-01.log)
and [Fast](pytest-pyg28-voxel-avg-optional-fast-2026-10-01.log) logs. The
optional integration tests skipped there. All four logs contain no personal
absolute paths and record correctness, not performance.

The scope excludes nonfinite points, ambiguous float32 cell boundaries,
float16/float64 features, higher spatial dimensions, negative or out-of-range
fixed-size cluster IDs, and empty inputs. The separate graph-coarsening
[`avg_pool`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/avg_pool.py)
path includes edge rewiring, position pooling, and graph batching; it is not
validated here. Real point-cloud datasets, larger hardware coverage, and
performance measurements remain separate work.

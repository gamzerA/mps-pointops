# Experimental PyG 2.8 voxel-grid dispatch

This slice adds an MPS implementation to the **existing**
`pyg::grid_cluster` operator from `pyg-lib`. It does not define that schema or
replace its CPU/CUDA implementations. The legacy
`torch_cluster.grid_cluster` shim has a separate, 3D-only contract in
[grid-cluster-contract.md](grid-cluster-contract.md).

## Pinned upstream contract

The reference versions are PyG **2.8.0** and `pyg-lib` **0.7.0**. Their
[PyG wrapper](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/voxel_grid.py),
[operator schema](https://github.com/pyg-team/pyg-lib/blob/0.7.0/pyg_lib/csrc/ops/cluster.cpp),
[CPU implementation](https://github.com/pyg-team/pyg-lib/blob/0.7.0/pyg_lib/csrc/ops/cpu/cluster_kernel.cpp), and
[upstream tests](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/test/nn/pool/test_voxel_grid.py)
establish the following behavior. No upstream source is copied here.

The raw schema is
`pyg::grid_cluster(Tensor pos, Tensor size, Tensor? start=None, Tensor? end=None) -> Tensor`.
The PyG call is `voxel_grid(pos, size, batch=None, start=None, end=None)`.
For spatial dimension `d`, PyG appends the per-point batch number as dimension
`d+1` of `pos`, appends `1` to `size`, appends `0` to an explicit `start`, and
appends `batch.max()` to an explicit `end`. When `batch` is absent, it uses an
all-zero batch vector. Scalar or list `size`, `start`, and `end` values are
expanded to spatial dimension `d` by the PyG wrapper before the raw call.
Consequently, the raw operator receives `D=d+1` columns.

For each dimension `j` in `0,...,D-1`, let `s_j` and `e_j` be the supplied
bounds or the componentwise minimum and maximum of `pos`. For positive cell
size `h_j`, the finite float32 calculation is

```text
cell_j(i) = int64(trunc_0(fl32(fl32(pos[i,j] - s_j) / h_j)))
extent_j  = int64(trunc_0(fl32(fl32(e_j - s_j) / h_j))) + 1
id(i)     = sum_j cell_j(i) * product_{t<j} extent_t
```

`fl32` denotes rounding to float32. The subtraction and division are
separate float32 tensor operations;
`trunc_0` means truncation toward zero, including negative quotients.
`id` is an `int64` mixed-radix identifier on the input device, **not** a
compact consecutive label. Because PyG puts the batch number in the most
significant dimension, two points in the same spatial cell from different
batches receive different IDs. No gradient is defined for the integer output.

## Current MPS scope

Call `mps_pointops.pyg.register_mps()` after installing a `pyg-lib` wheel
matching the installed PyTorch. It registers the MPS implementation for
`pyg::grid_cluster` alongside the existing `fps`, `knn`, and `radius`
implementations. The `torch_geometric.nn.voxel_grid` wrapper then works on MPS
for finite `float32` spatial coordinates of dimension 1, 2, or 3. Equivalently,
the raw operator accepts a nonempty `(N, D)` float32 tensor with
`D in {2, 3, 4}`, and same-device float32 `(D,)` vectors for `size` and
optional bounds. Sizes must be positive; bounds must be ordered and the mixed
radix arithmetic must fit in `int64`. The current path uses PyTorch MPS tensor
operations, with no custom Metal shader or speed claim. The MPS result stays
on MPS. CPU calls continue to use `pyg-lib`'s original CPU kernel.

Empty point sets, nonfinite values, nonpositive sizes, inverted bounds,
half/double precision, higher dimensions, and arithmetic overflow are outside
this experimental MPS contract. Values within a few float32 ulps of a cell
boundary may differ between CPU, MPS Safe Math, MPS Fast Math, and CUDA because
division rounding is device dependent; the tests include representable cell
boundaries and compare finite inputs away from ambiguous boundaries. This is
not a complete replacement for every `pyg-lib` grid input.

## Reproduction and follow-up

The pinned reference is PyTorch 2.12.0, PyG 2.8.0, and
`pyg-lib==0.7.0+pt212` from the matching macOS wheel index. With those
packages installed, run each mode in a separate process:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest tests/test_pyg28_grid.py tests/test_pyg28.py -q
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python -m pytest tests/test_pyg28_grid.py tests/test_pyg28.py -q
```

The tests compare 192 deterministic raw CPU cases directly with
`pyg-lib` 0.7.0, check negative coordinates and exact cell boundaries, and
compare both the raw operator and PyG's batched wrapper between CPU and MPS.
They also verify the real dispatcher schema and MPS output device/dtype.

On an M5 Pro with macOS 26.5.2 and MPS fallback disabled, the isolated pinned
reference environment passed the complete test suite with **308 passed,
9 skipped** in Safe Math and **307 passed, 10 skipped** in Fast Math. The two
legacy `torch_cluster` differential tests skipped because that separate
extension was absent; seven existing `k > n` tests skipped in both modes, and
the feature-space kNN nonfinite test is Safe-only. The focused PyG operator
tests passed **15/15** in each mode. See the unmodified pytest output with
warning details disabled: [Safe](pytest-pyg28-grid-safe-2026-10-01.log) and
[Fast](pytest-pyg28-grid-fast-2026-10-01.log).

In the separate Torch 2.14.1 environment without optional PyG or pyg-lib,
the full suite passed **295 with 22 skips** in Safe Math and **294 with
23 skips** in Fast Math. The new optional differential tests skipped, while
the independent CPU contract checks still passed. See
[Safe](pytest-pyg28-grid-optional-safe-2026-10-01.log) and
[Fast](pytest-pyg28-grid-optional-fast-2026-10-01.log). These logs contain
no personal absolute paths. They measure correctness, not performance.

**Voxel downsampling is a separate follow-up.** In PyG 2.8,
[`avg_pool_x`](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/avg_pool.py)
first makes cluster IDs consecutive when `size=None`, then averages features
by cluster and maps each cluster to a batch. When `size` is given, its output
has `batch_size * size` rows and retains empty cluster slots. That reduction,
its feature gradients, `torch.unique`/scatter behavior, graph coarsening in
`avg_pool`, and a voxel-downsampling performance claim are not part of this
grid-ID registration. They need their own pinned parity and MPS profiling.

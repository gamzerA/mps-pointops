# Experimental `grid_cluster` slice

`mps_pointops.grid.grid_cluster(pos, size, start=None, end=None)` is available
through the optional `torch_cluster` shim after `mps_pointops.compat.install()`.
It runs with PyTorch tensor operators on CPU or MPS. This first slice has no
custom Metal grid shader and makes no speed claim.

The supported input is a nonempty, finite `(N, 3)` float32 position tensor;
`size`, `start`, and `end` are `(3,)` float32 tensors on the same device. `size`
must be positive. Omitted bounds are the componentwise minimum and maximum of
`pos`. The result is an `(N,)` int64 tensor of **linear voxel identifiers** on
the input device. These IDs are not renumbered to consecutive cluster labels.
`torch_cluster.grid_cluster` 1.6.3 has no `batch` argument; batching is outside
this slice. It also requires a Tensor with one size per coordinate dimension:
the real 1.6.3 CPU extension rejects both a Python scalar and a one-element
size Tensor for three-dimensional `pos`. No gradient is defined for integer
voxel IDs.

For each dimension `d` in `{0, 1, 2}`, let

```text
g[d] = trunc_to_zero((pos[d] - start[d]) / size[d])
n[d] = trunc_to_zero((end[d] - start[d]) / size[d]) + 1
id   = g[0] + n[0] * g[1] + n[0] * n[1] * g[2]
```

The float32 subtraction and division happen before the int64 conversion.
In particular, `trunc_to_zero(-0.5) = 0`; replacing it with `floor` would not
match upstream for points below an explicitly supplied `start`. The output may
be negative if a point is outside the stated bounds. Inputs whose mixed-radix
products overflow int64, nonfinite values, and nonpositive sizes have no
parity guarantee. CPU and MPS floating-point division may disagree very close
to cell boundaries; the included exact-boundary and random differential tests
define the currently measured coverage, not a universal bitwise promise.

`torch_cluster` 1.6.3's [Python signature](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/torch_cluster/grid.py),
[CPU implementation](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cpu/grid_cpu.cpp),
and [CUDA implementation](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cuda/grid_cuda.cu)
establish this contract. The local implementation independently expresses
those arithmetic rules with PyTorch operators; no upstream code is copied.

Run the contract tests with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest tests/test_grid_cluster.py -q
```

If a real `torch_cluster==1.6.3` CPU extension is importable, the final test
also compares the outputs directly. If it is unavailable, that test skips
while the fixed-value CPU/MPS tests still run. The optional comparison must
not import the local `torch_cluster` shim in place of the real extension.

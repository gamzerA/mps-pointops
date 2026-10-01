# Compact voxelization and downsampling API

This is a **new** `mps_pointops.voxel` API, currently experimental. It is not
the `torch_geometric.nn.voxel_grid` or `torch_cluster.grid_cluster` call
surface. Those return raw mixed-radix IDs, and their cell calculation uses
truncation toward zero. This API uses mathematical floor, compact rows, and
explicit maps. It does not register a new `pyg::*` operator or implement PyG
graph coarsening (`avg_pool`). The prior [PyG 2.8 grid](pyg28-grid-cluster-contract.md)
and [feature-pooling](pyg28-voxel-avg-pool.md) paths remain separate.

## Inputs and cell rule

Call `voxelize(pos, size, batch=None, *, start=None, end=None)` or
`voxel_downsample(pos, size, batch=None, features=None, *, start=None,
end=None, feature_reduce="mean")` from `mps_pointops.voxel`.

- `pos` is finite float32 `(N,D)` on CPU or MPS, `D ∈ {1,2,3}`. `N=0` is
  supported. `batch` is optional nonnegative int64 `(N,)` on the same device;
  its rows need not be sorted, contiguous, or equally sized, and batch IDs
  may have gaps. Missing batch means batch zero.
- `size` is a finite positive float32 scalar or `D`-vector. Python numbers
  and sequences are converted to float32 on the input device. A tensor must
  already be float32 on that device. `start` and `end` follow the same vector
  rules. `start=None` means the zero origin. Without `end`, `start` is only an
  origin, so points below it are allowed and have negative cells.
- When `end` is supplied, it must exceed `start` in every dimension and every
  point must lie in the half-open region `[start,end)`. A point exactly on
  `start` belongs to cell zero; a point exactly on `end` raises `ValueError`.
  This is a validation boundary, not a clipping operation. Empty inputs are
  allowed with valid vectors and ordered bounds.

For dimension `d`, the implemented float32 cell expression is

```text
c[i,d] = int64(floor(fl32(fl32(pos[i,d] - start[d]) / size[d])))
```

Here `fl32` denotes a float32 tensor-operation result. A quotient that is
nonfinite or has magnitude at least `2^62` is rejected before int64 conversion.
The bound is conservative; it avoids unsafe conversion. Integer cell rows
are not packed into a mixed-radix scalar, so there is no key collision from a
product of grid extents. Float32 quotient values within a few ULPs of an
integer may land on different sides of a boundary on CPU, MPS Safe, and MPS
Fast Math. The parity tests use exact dyadic boundaries and half-integer
interiors. Subnormal sizes and coordinates can show additional device
differences from flush-to-zero behavior and are not a parity guarantee.

## Output order and inverse maps

Let `K_i = (batch_i, c[i,0], …, c[i,D-1])`. The unique keys are sorted in
ascending lexicographic order, first by numeric batch ID, then by spatial
cell coordinates. This order is independent of input row order and retains
empty gaps in batch IDs only as absent rows. For `V` unique keys:

| Field | Shape / dtype | Meaning |
| --- | --- | --- |
| `voxel_coords` | `(V,D)` int64 | Spatial cell for output row `v` |
| `batch` | `(V,)` int64 | Original batch ID for row `v` |
| `inverse` | `(N,)` int64 | `inverse[i]` is point `i`'s output row |
| `counts` | `(V,)` int64 | Number of points in row `v` |
| `point_order` | `(N,)` int64 | Original point indices, grouped by voxel and stable within each group |
| `ptr` | `(V+1,)` int64 | The points for row `v` are `point_order[ptr[v]:ptr[v+1]]` |

Thus `inverse` maps points to voxels, while `point_order` and `ptr` provide
the voxel-to-points CSR map. All fields stay on the input device. For `N=0`,
`V=0`, all vectors are empty except `ptr=[0]`, and the coordinate shape is
`(0,D)`.

## Aggregation and gradients

`voxel_downsample` returns a `VoxelDownsample` with the `Voxelization` above,
mean position `(V,D)` and optional float32 features `(V,C)` (`C ≥ 1`).
Features must be finite and on the same device. `feature_reduce="mean"` is
the default; `"sum"` is also supported. Positions are always averaged.
For row `v`, with `I_v={i: inverse[i]=v}` and `n_v=counts[v]`:

```text
position[v] = (Σ_{i∈I_v} pos[i]) / n_v
feature_mean[v] = (Σ_{i∈I_v} features[i]) / n_v
feature_sum[v] = Σ_{i∈I_v} features[i]
```

The integer assignment and maps have no gradient. Autograd differentiates
the aggregation at fixed assignment. For an upstream position gradient `g_v`,
each member receives `g_v/n_v`. Features receive their upstream row gradient
divided by `n_v` for mean, or unscaled for sum. These are first-order
contracts; second-order support has not been separately tested. Empty input
returns empty views that preserve `requires_grad` for positions/features.
MPS scatter-add accumulation order can change float32 low bits, so the
integer maps are checked exactly and aggregate values within a documented
floating-point tolerance. This implementation makes no speed claim.

## Implementation and verification scope

The code uses PyTorch `torch.unique(dim=0, sorted=True)`, stable `argsort`,
and `index_add_` on CPU/MPS. Our code does not explicitly copy input tensors
to CPU; PyTorch's internal implementation of these operators is outside this
claim. Finite,
bound, and range checks read GPU scalar predicates and therefore synchronize
MPS; `unique` also creates a data-dependent output shape. This initial
correctness path is not a latency-optimized Metal kernel. The hosted
Torch 2.12 `nn.Linear` bias discrepancy noted in the separate
[PyG feature-pooling record](pyg28-voxel-avg-pool.md) is not used to judge
this API's inverse map or aggregation correctness.

Run Safe and Fast in separate processes, always disabling CPU fallback:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest tests/test_voxel.py -q -ra
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python -m pytest tests/test_voxel.py -q -ra
```

On an Apple M5 Pro, macOS 26.5.2 (25F84), Python 3.12.13, source commit
`62d2ca0358d6e6a226e0384c2beb2a834a4f7dbd`, all runs used
`PYTORCH_ENABLE_MPS_FALLBACK=0` and separate processes:

| Environment | Safe Math | Fast Math |
| --- | --- | --- |
| Torch 2.12.0, PyG 2.8.0, pyg-lib 0.7.0+pt212 | [390 passed, 11 skipped](pytest-voxel-api-torch212-safe-2026-10-01.log) | [389 passed, 12 skipped](pytest-voxel-api-torch212-fast-2026-10-01.log) |
| Torch 2.14.1, real torch-cluster 1.6.3; no optional PyG/pyg-lib | [373 passed, 28 skipped](pytest-voxel-api-torch214-safe-2026-10-01.log) | [372 passed, 29 skipped](pytest-voxel-api-torch214-fast-2026-10-01.log) |

The new API's focused tests passed **20/20** in each mode and each environment.
The full logs record the current suite including optional-dependency skips;
they are not performance measurements. Exact integer maps cover
unsorted/gapped batches, negative and representable boundary cells, empty
input, and feature/position backward. `uv build --offline --no-build-isolation`
produced a wheel and source distribution; the wheel contains
`mps_pointops/voxel.py`. Only M5 Pro was measured here; M1–M4 and larger
inputs remain untested.

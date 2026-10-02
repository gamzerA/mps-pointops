# Numerical contracts

The mathematical selection rule, the stored float32 result, and parity with
a CUDA/CPU library are separate claims. Ties and radius boundaries are part
of the API contract; binary equivalence across all hardware is not implied.

## Selection and output

For query `q_i`, reference `x_j`, and coordinate dimension `D`, the real
squared distance is

```text
s(i,j) = sum[d=0..D-1] (q[i,d] - x[j,d])^2.
```

Dense `knn` selects the smallest `k` distances, breaking exact ties by the
smaller reference index, and returns **Euclidean** distances `sqrt(s)`.
Dense Ball Query retains the first `K` matching reference indices in their
original input order and returns **squared** distances. A valid zero distance
is possible; use `index >= 0`, not `distance != 0`, to recognize padding.

The dense Ball Query threshold is

```text
r32 = fl32(r)
R2_dense = fl32(r32 * r32)
accept(i,j) iff s32(i,j) < R2_dense.
```

`fl32` is a float32 rounding operation. The flat `torch_cluster`-style radius
path instead uses `R2_flat = fl32(r * r)` after the Python double-precision
product. These can differ by one float32 ULP, so cross-API boundary comparisons
must use the intended threshold. The dense kernel also takes a normalized
path for very small radii to reduce subnormal flush errors. Its exact
threshold, FTZ limits, FMA source order, and edge-case tests are in the
[full Ball Query numerical contract](https://github.com/gamzerA/mps-pointops/blob/main/docs/ball-query-math.md).

FPS starts at the requested index and repeatedly chooses the point whose
distance to its nearest selected center is greatest. Exact ties choose the
smallest input index. `flat.fps` computes the sample count from `ratio` using
the `torch_cluster` dtype and rounding rule; for deterministic parity set
`random_start=False`. See the [README equations](https://github.com/gamzerA/mps-pointops/blob/main/README.md#the-operators-in-equations).

## Differentiation

Neighbor indices and radius membership are discrete and are not differentiated.
With a selected Ball Query pair, upstream distance gradient `G`, and
`delta=q-x`, the first-order contributions are `2G*delta` to the query and
`-2G*delta` to the reference point. Multiple references to one point are
accumulated; floating-point summation order may vary. The
[Chamfer contract](https://github.com/gamzerA/mps-pointops/blob/main/docs/chamfer-contract.md)
defines both nearest-neighbor directions, padding masks, weights, point and
batch reduction divisors, L1 subgradients at coincident coordinates, and
normal-loss scope. A tested gradient formula does not make the nearest-index
decision itself differentiable.

## Metal math modes

| Mode | How to start the process | Interpretation |
| --- | --- | --- |
| Safe | `PYTORCH_MPS_FAST_MATH=0` | Primary numerical contract and regression baseline. |
| Fast | `PYTORCH_MPS_FAST_MATH=1` | Separate observed test mode. Nonfinite handling and boundary bits are not promoted to Safe-mode guarantees. |

Set `PYTORCH_ENABLE_MPS_FALLBACK=0` during parity runs so a missing Metal path
cannot be hidden by CPU fallback. Call `torch.mps.synchronize()` around timed
regions; an API return alone does not mean the GPU finished. Compare Safe and
Fast in **different Python processes**, since shader compilation settings are
cached. Subnormal flush and compiler contraction can affect the lowest bits
even when the source makes FMA order explicit. The
[Metal probe and methods](https://github.com/gamzerA/mps-pointops/blob/main/docs/ball-query-math.md#컴파일-설정과-검증-기준)
record the observed behavior and limits.

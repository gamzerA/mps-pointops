# PointNet++ feature propagation contract

`mps_pointops.pointnet2.three_nn` and
`mps_pointops.pointnet2.three_interpolate` independently implement the call
shapes documented by
[`pointnet2_ops.pointnet2_utils`](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/pointnet2_ops_lib/pointnet2_ops/pointnet2_utils.py).
The upstream CUDA implementation is a compatibility reference, not source for
the Metal kernels. No upstream code is copied into this project. This feature
was introduced experimentally in v0.5.0. A fixed synthetic PointNet++ SSG
segmentation forward/backward run was compared with the original CUDA
extension for v0.6.0 ([report](parity/pointnet2-segmentation.md)). Labeled-data
accuracy and training convergence remain untested.

## Inputs and outputs

| Operation | Inputs | Output |
| --- | --- | --- |
| `three_nn(unknown, known)` | Coordinates `unknown: (B,N,3)`, `known: (B,M,3)`, float32, same device, `M >= 3` for nonempty queries | Euclidean distances `(B,N,3)` float32 and indices `(B,N,3)` int32 |
| `three_interpolate(features, indices, weights)` | Source features `(B,C,M)` float32, indices and **caller-provided** weights `(B,N,3)`; indices int32 or int64 | Interpolated features `(B,C,N)` float32 |

For a given query, the three indices are distinct and ordered by squared
distance, then by smaller input index on an exact tie. Distances are square
roots of those squared distances. `three_nn` returns no coordinate gradient,
as the PointNet++ operation does not differentiate neighbor selection or
distance. Queries, known coordinates, features, and weights should be finite;
non-finite values are outside this contract. An empty batch or query dimension
returns an empty result without launching a Metal kernel. There must be at
least one source feature when interpolation has nonempty indices.

The original PointNet++ feature tensor is channel-first `(B,C,M)`. A model
holding `(B,M,C)` data must transpose before calling `three_interpolate`.
This API does not calculate inverse-distance weights: callers can use
`three_nn`'s distances to construct them, including their chosen zero-distance
rule, and then pass the resulting weights explicitly.

## Equations

For query `q[b,n]` and known point `x[b,m]`, the squared Euclidean distance is

$$
s_{bnm}=\sum_{d=0}^{2}(q_{bnd}-x_{bmd})^2.
$$

`three_nn` chooses the three smallest pairs `(s[b,n,m], m)`, with selected
index `I[b,n,t]` at rank `t`, and returns `sqrt(s[b,n,I[b,n,t]])`.
The Metal and CPU reference paths accumulate the three squares in x, y, z
order without fused multiply-add contraction. They can still disagree at
near ties owing to device arithmetic and square-root rounding.

For input features `F` and supplied weights `W`, interpolation is

$$
Y_{bcn}=\sum_{t=0}^{2}W_{bnt}F_{bc,I_{bnt}}.
$$

Holding `I` and `W` fixed, the chain rule yields the source-feature gradient

$$
\frac{\partial L}{\partial F_{bcm}}
=\sum_{n=0}^{N-1}\sum_{t=0}^{2}
  [I_{bnt}=m]W_{bnt}\frac{\partial L}{\partial Y_{bcn}}.
$$

Repeated indices contribute repeatedly. The upstream autograd wrapper returns
an explicit zero tensor for the weight gradient. This is an API choice, not the
mathematical derivative: when treating the weights as independent variables,
$\partial Y_{bcn}/\partial W_{bnt}=F_{bc,I_{bnt}}$. This compatibility API
reproduces upstream's zero tensor when weights require gradients. Integer
indices are not differentiable. This is a
first-order backward operation; second-order gradients are not supported on
the native Metal path.

## Implementation and numerical policy

The forward Metal search keeps the best three candidates in registers per
query. Interpolation launches one thread per output element. Backward uses
integer compare-and-swap to add float32 contributions to source features;
this avoids assuming device support for native float atomics. Addition order
can vary between executions when multiple queries refer to the same point,
so the least significant gradient bits are not deterministic. The CPU
reference uses PyTorch gather and reduction operations.

The public wrapper checks every supplied index before native memory access.
That check synchronizes a scalar MPS result to the host; a future trusted-index
path can remove the synchronization after provenance is established. This
version prioritizes memory safety. The current search is linear in `M` per
query; the interpolation backward performs three atomic additions per output
element. No speedup against CUDA or CPU is claimed without a benchmark.

## Reproduction

On an Apple Silicon Mac with PyTorch 2.7 or later, run Safe and Fast Math in
separate processes because the setting is cached by the Metal runtime:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python -m pytest -q tests/test_pointnet2.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 python -m pytest -q tests/test_pointnet2.py
```

The tests cover exact ties, non-contiguous inputs, empty dimensions, invalid
indices, repeated indices, forward values, and the feature gradient against
an independent PyTorch reference. The fixed synthetic segmentation comparison
is documented in the [model-level report](parity/pointnet2-segmentation.md);
labeled-data accuracy and training convergence remain future validation.

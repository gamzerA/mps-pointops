# Chamfer distance experimental contract

Version 0.5.0 includes an experimental squared-L2 bidirectional Chamfer
implementation. MPS uses
`kernels/chamfer_nn.metal` to find each nearest point; CPU and CUDA use a tiled
PyTorch search. Both searches cost $O(BPQ)$ time. The CPU/CUDA path bounds
temporary pair storage. The Metal path uses one 32-lane SIMD group per query
and scans each reference cloud without constructing a pair matrix. The Metal
kernel writes a squared distance and int64 index for every query row: padded
rows receive `(0, -1)`. Its distance is the MPS forward result; the saved
index selects the pair for the backward formula below. MPS execution is tested
with `PYTORCH_ENABLE_MPS_FALLBACK=0`.

For cloud `x` with valid points \(x_i\) and cloud `y` with valid points
\(y_j\), let \(a(i)\) be the lowest-index point in `y` at minimum rounded
squared distance from \(x_i\), and \(b(j)\) the analogous point in `x`.
Without reductions or weights the two output tensors are

$$
d^x_i=\|x_i-y_{a(i)}\|_2^2,\qquad
d^y_j=\|y_j-x_{b(j)}\|_2^2.
$$

`point_reduction="sum"` sums valid point distances in each cloud;
`"mean"` divides each directional sum by its own valid length. The
bidirectional loss adds those two directional results. `"max"` takes the
larger of the two directional maximum distances. `point_reduction=None`
returns both padded per-point tensors and requires `batch_reduction=None`.
`single_directional=True` returns only the `x` to `y` term. Optional
nonnegative finite batch weights multiply each directional point distance;
`batch_reduction="mean"` divides the batch sum by the sum of weights (or by
the batch size when no weights are supplied). With all-zero weights the result
and numerical gradients are zero. This case has the upstream special return
shape described below.

For the un-reduced squared-L2 sum, the selected-index gradient includes both
the point's own search and every reverse search that selected it:

$$
\frac{\partial L}{\partial x_i}=2(x_i-y_{a(i)})
+2\sum_{j:b(j)=i}(x_i-y_j).
$$

The analogous $y_j$ gradient swaps $x$ and $y$. On MPS, a custom autograd
function saves the Metal-selected indices. Its backward forms
$2g_i(x_i-y_{a(i)})$ for upstream scalar $g_i$ and uses PyTorch's native
`scatter_add_` to accumulate the negative contribution into `y`. The reverse
direction supplies the other term. No project-specific Metal floating-point
atomic kernel is needed for this first-order path. The indices have no
gradient, and MPS second-order gradients are unsupported.
With `mean`, `sum`, `max`, weights, or batch reduction, autograd applies the
respective reduction weights or selected maximum. Nearest indices themselves
have no gradient. Gradients are piecewise defined away from nearest-neighbor
ties; ties use the lower reference index.

For example, with mean point reduction, query $x_i$ receives upstream scale
$w_b/(L^x_b\sum_c w_c)$ under weighted batch mean, where $L^x_b$ is the
valid query length. The reverse direction uses $L^y_b$ in its own mean.
Without batch weights, the batch denominator is $\max(B,1)$; batch sum has
no batch denominator; point sum has no point denominator. With all-zero
weights, the loss and first-order gradient values are zero. These scales are
applied before the native scatter, through the upstream `grad_distances`.

Inputs must have dense shapes `(B, P, 3)` and `(B, Q, 3)`, matching devices
and dtypes, with at least one valid point per cloud in each nonempty batch.
MPS accepts float32; CPU and CUDA accept float32 or float64. Optional lengths
must be int64 vectors on the same device, within the padded dimension. Valid
coordinates must be finite. Padded distances and gradients are zero, even if
padded coordinates are nonfinite. Validation reads a few device scalars and
therefore synchronizes MPS with the host.

The ordinary return shape follows PyTorch3D's loss tuple `(loss, None)`;
normals and `norm=1` are not implemented and raise explicitly. `abs_cosine`
is accepted but has no effect when normals are absent. At **all-zero batch
weights**, pinned upstream PyTorch3D 0.7.9 (commit
`88e182f989c80836f4bd744e0d9cb1852762ce01`) takes an early exit before
nearest-neighbor search. Its broadcast produces `(B, B)` zeros before batch
reduction even when `point_reduction=None`; it also returns a zero tensor for
the normal-loss slot although normals were absent. The only exception is
bidirectional `point_reduction="max"`, whose normal-loss slot stays `None`.
The port mirrors these observed API signatures for finite valid points and
all-zero weights. In single-directional mode, upstream disconnects `y` from
the graph (`y.grad is None`), while `x` and `weights` have present zero
gradients; the port does likewise. The port masks padded coordinates in this
zero path, so nonfinite padding cannot contaminate the anchor. Such padding
is outside the direct upstream parity matrix. This is a subset of the
[PyTorch3D Chamfer API](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py).
The Metal search accumulates squared coordinate differences in x, y, z order
with FMA contraction disabled in source. Safe and Fast Math modes can still
differ on underflow or near-tie inputs; exact cross-backend equality at those
boundaries is not promised. Repeated point ties use the lowest reference
index, including when finite coordinates make every float32 squared distance
overflow to infinity. Many-to-one nearest selections can contend during
scatter accumulation. The first measurement below tests this effect on one
M5 Pro; a later physical M1 measurement is linked below. Floating-point
scatter accumulation can
also change its last bits with execution order; bitwise repeatability of MPS
gradients is not promised. The implementation and equations here were
independently written for this project; no PyTorch3D source was copied.

## First contention measurement

The [Safe](../bench/results/2026-10-01-apple-m5-pro-chamfer-contention-safe.md)
and [Fast](../bench/results/2026-10-01-apple-m5-pro-chamfer-contention-fast.md)
M5 Pro measurements use `single_directional=True`, four clouds of 256 to
16,384 points, 20 timed repeats per case, and synchronized forward/backward
timings. Uniform selection sends each query to a different reference;
concentrated selection sends all queries
to reference zero. The paired median concentrated/uniform backward ratios
were 0.79–1.08 in Safe Math and 0.94–1.08 in Fast Math. A direct
`scatter_add_` control also showed no consistent concentrated slowdown in
this range. These results support retaining PyTorch's native accumulation for
the current implementation. They do not establish its behavior on larger
clouds, other Apple GPUs, or different gradient distributions.

The later [physical M1 report](phase3-physical-m1-2026-10-02.md) found a
different distribution response on an 8 GiB M1. For B=1,N=2,048,
concentrated selection increased single-directional Chamfer backward from
0.968 to 3.814 ms (Safe) or 0.860 to 3.669 ms (Fast); indices and analytic
gradients still passed. On separate random finite bidirectional inputs,
the MPS forward was faster than the **package's CPU reference** at N=1,024
and 4,096, but its backward was slower. These are different fixtures, so
they do not conflict. The M1 report links the raw Safe/Fast samples and a
native scatter stress test. None of these timings identifies Metal atomics
as the cause, compares an optimized CPU nearest-neighbor library, or proves
an end-to-end model speedup.

## Larger bidirectional M5 Pro probe

The [2026-10-02 large-cloud record](chamfer-large-contention-2026-10-02.md)
extends the controlled M5 Pro study to **bidirectional** mean loss at
`B=1,N=32,768` and `65,536`. It compares in-order one-to-one, randomly
permuted one-to-one, and both-directions all-to-one nearest maps. Both index
maps, the loss, and analytic first-order gradients passed before timing. Safe
and Fast used separate processes, CPU fallback was disabled, and raw
synchronized forward, backward, full loss plus backward, and paired native
scatter samples are retained.

At 65,536 points, concentrated native two-scatter median was 0.489 ms Safe
or 0.499 ms Fast, versus 0.297 and 0.292 ms for in-order one-to-one. The
paired full-loss concentrated/uniform ratios were 1.00× and 1.03×. Backward
alone varied much more, including a Safe paired ratio of 2.87×, so the full
loss and raw sample spread are needed for the adoption decision. The current
M5 Pro default remains native PyTorch scatter for these synthetic cases;
this does not establish a result for physical M1 or a general GPU mechanism.

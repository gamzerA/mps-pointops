# Chamfer distance experimental contract

This branch adds a squared-L2 bidirectional Chamfer implementation. MPS uses
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
and gradients are zero.

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
weights, the loss and first-order gradients are zero. These scales are
applied before the native scatter, through the upstream `grad_distances`.

Inputs must have dense shapes `(B, P, 3)` and `(B, Q, 3)`, matching devices
and dtypes, with at least one valid point per cloud in each nonempty batch.
MPS accepts float32; CPU and CUDA accept float32 or float64. Optional lengths
must be int64 vectors on the same device, within the padded dimension. Valid
coordinates must be finite. Padded distances and gradients are zero, even if
padded coordinates are nonfinite. Validation reads a few device scalars and
therefore synchronizes MPS with the host.

The return shape follows PyTorch3D's loss tuple `(loss, None)`; normals and
`norm=1` are not implemented and raise explicitly. `abs_cosine` is accepted
but has no effect when normals are absent. This is a subset of the
[PyTorch3D Chamfer API](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/loss/chamfer.py).
The Metal search accumulates squared coordinate differences in x, y, z order
with FMA contraction disabled in source. Safe and Fast Math modes can still
differ on underflow or near-tie inputs; exact cross-backend equality at those
boundaries is not promised. Repeated point ties use the lowest reference
index. Many-to-one nearest selections can contend during scatter accumulation;
whether this dominates runtime depends on the input and remains to be
benchmarked. Floating-point scatter accumulation can also change its last
bits with execution order; bitwise repeatability of MPS gradients is not
promised. The implementation and equations here were independently written
for this project; no PyTorch3D source was copied.

## First contention measurement

The [Safe](../bench/results/2026-10-01-apple-m5-pro-chamfer-contention-safe.md)
and [Fast](../bench/results/2026-10-01-apple-m5-pro-chamfer-contention-fast.md)
M5 Pro measurements use four clouds of 256 to 16,384 points, 20 timed repeats
per case, and synchronized forward/backward timings. Uniform selection sends
each query to a different reference; concentrated selection sends all queries
to reference zero. The paired median concentrated/uniform backward ratios
were 0.97–1.11 in Safe Math and 0.98–1.04 in Fast Math. A direct
`scatter_add_` control also showed no consistent concentrated slowdown in
this range. These results support retaining PyTorch's native accumulation for
the current implementation. They do not establish its behavior on larger
clouds, other Apple GPUs, or different gradient distributions.

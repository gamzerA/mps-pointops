# Experimental L1 Chamfer parity for v0.9 development

This is development evidence, not a v0.9 release or a full PyTorch3D Chamfer
compatibility claim. The oracle is PyTorch3D 0.7.9 at commit
[`88e182f989c80836f4bd744e0d9cb1852762ce01`](https://github.com/facebookresearch/pytorch3d/commit/88e182f989c80836f4bd744e0d9cb1852762ce01).
See its [CPU kNN backward](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/csrc/knn/knn_cpu.cpp)
for the selected L1 subgradient.

For query point `x_i` and reference points `y_j`, the new search selects

\[
j^*(i)=\operatorname*{argmin}_{j\in\{0,\ldots,L_y-1\}}
       \left(\sum_{d=0}^{2}|x_{i,d}-y_{j,d}|,\ j\right),
\qquad
\ell_i=\sum_{d=0}^{2}|x_{i,d}-y_{j^*(i),d}|.
\]

The pair notation makes the smallest reference index win an equal-distance
tie. For upstream parity, its chosen derivative at an exactly equal coordinate
is **-1**, rather than the usual `torch.sign(0)=0`:

\[
s_{i,d}=\begin{cases}+1,&x_{i,d}>y_{j^*(i),d},\\-1,&x_{i,d}\le y_{j^*(i),d},\end{cases}
\quad
\frac{\partial L}{\partial x_{i,d}}=g_i s_{i,d},\qquad
\frac{\partial L}{\partial y_{j,d}}=-\sum_{i:j^*(i)=j}g_i s_{i,d}.
\]

`g_i` is the gradient arriving after the existing length mask, point/batch
reduction, and optional batch weight. The reverse Chamfer direction adds its
own selected-pair contributions. Padded rows receive zero gradient. The
derivative is a selected subgradient at the cusp, not a claim that the L1
function is differentiable there. Higher-order gradients remain unsupported.

## Direct comparison and repeatability

The [differential script](../tests/verify_upstream_chamfer.py) now covers
both `norm=1` and `norm=2`, 480 cases and 3,240 checked output/gradient
objects per target. The raw records are [CPU](evidence/chamfer-l1/chamfer-l1-parity-cpu.json),
[M5 Pro Safe](evidence/chamfer-l1/chamfer-l1-parity-mps-safe.json), and
[M5 Pro Fast](evidence/chamfer-l1/chamfer-l1-parity-mps-fast.json): each has
zero failing objects and zero failing elements. The MPS cases ran with
`PYTORCH_ENABLE_MPS_FALLBACK=0` in separate Safe/Fast processes. The CPU
oracle environment uses a different PyTorch version from the M5 Pro runner;
the exact versions are in each raw record.

The regression suite additionally places equal-distance candidates at
reference indices 0, 31, 32, and 64, runs 17 queries in each of two batches,
and checks the selected index over 32 repeated Metal dispatches. It tests an
exactly coincident pair in isolation: its zero L1 loss has query gradient
`(-1,-1,-1)` and reference gradient `(+1,+1,+1)`. Safe and Fast Math each
passed the selected Chamfer/spatial regression set (86 tests) on M5 Pro.

Normal loss and PyTorch3D `Pointclouds` inputs remain unsupported. The
[v0.8 scope decision](chamfer-api-scope-v0.8.md) lists their acceptance gates.

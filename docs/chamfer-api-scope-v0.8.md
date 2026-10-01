# Chamfer API scope decision for v0.8

## Decision

Keep the v0.8 Chamfer API at its documented dense, squared-L2 subset. Do not
advertise `norm=1`, normal loss, or PyTorch3D `Pointclouds` input as supported
in v0.8. The function already exposes the upstream argument names, but it
explicitly rejects `norm=1` and non-`None` normals; `Pointclouds` inputs do not
pass its tensor validation. This document fixes the contract and acceptance
gates for a later, separately reviewed expansion. It does not change exports
or versioning.

The reference is official PyTorch3D **0.7.9 at commit
[`88e182f989c80836f4bd744e0d9cb1852762ce01`](https://github.com/facebookresearch/pytorch3d/commit/88e182f989c80836f4bd744e0d9cb1852762ce01)**,
the same pin as the [existing direct parity run](chamfer-upstream-parity-0.5.0.md).
Relevant source is its
[`loss/chamfer.py`](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/loss/chamfer.py),
[`ops/knn.py`](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/ops/knn.py),
[`csrc/knn/knn_cpu.cpp`](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/csrc/knn/knn_cpu.cpp),
[`csrc/knn/knn.cu`](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/csrc/knn/knn.cu),
and [`structures/pointclouds.py`](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/structures/pointclouds.py).
The local pinned checkout's SHA-256 for `loss/chamfer.py` is
`de49c3d5f1176a747efc7084b8afe72e8639ce3c4490c8bb128864dfb493c917`.
Current package behavior is defined in
[`mps_pointops/chamfer.py`](../mps_pointops/chamfer.py) and
[`chamfer-contract.md`](chamfer-contract.md).

## Gap and precise future contract

| Surface | Pinned upstream behavior | Current package | Expansion gate |
| --- | --- | --- | --- |
| `norm=1` | kNN minimizes the **sum of absolute coordinate differences** and returns that L1 sum, not the square root of the L2 result. `norm=2` returns squared L2. | Rejects every norm except 2. Both CPU search and Metal search rank by squared L2. | Implement L1 ranking and output on each supported device, including backward for selected pairs; compare against the pinned CPU extension. |
| `x_normals`, `y_normals`, `abs_cosine` | When **both** normal arrays exist, gather the reference normal at each nearest point and return a separate normal loss: `1 - abs(cosine)` by default, or `1 - cosine` when `abs_cosine=False`. The cosine uses `eps=1e-6`. With only one normal array, ordinary calls return `None` for the normal loss. | Any supplied normal array raises `NotImplementedError`; ordinary return is `(loss, None)`. | Match normal result shape, reduction, masking, weighting, and first-order gradients, including the one-sided and zero-weight branches. |
| `Pointclouds` objects | For each object, upstream obtains `points_padded()`, `num_points_per_cloud()`, and `normals_padded()`. Supplied `*_lengths` and `*_normals` arguments for that object are **replaced** by its contents. Tensor inputs use the supplied arrays. | Dense tensor shapes `(B,P,3)` and `(B,Q,3)` only. | Decide an optional PyTorch3D dependency boundary and test object/tensor mixes, list-built heterogeneous clouds, explicit-argument precedence, and gradients through padded views. |

Source details for these rows: upstream `loss/chamfer.py` lines 38–74,
89–150, and 238–292; `csrc/knn/knn_cpu.cpp` lines 35–49 and 101–123;
`structures/pointclouds.py` lines 546–577. These are line numbers at the
pinned commit, rather than moving `main` links.

The L1 backward has a non-obvious upstream convention. Its CPU and CUDA kNN
backward use `+1` when a query coordinate is strictly greater than its selected
reference coordinate and `-1` otherwise, **including equality**. The selected
reference receives the negative contribution. This is a chosen subgradient at
an L1 cusp, so finite-difference gradcheck belongs only on fixtures away from
coordinate equality and nearest-neighbor ties. A parity implementation must
either match this upstream equality convention or explicitly narrow its parity
claim. The current MPS backward is fixed to `2 * (query - reference)` and
therefore cannot serve L1 unchanged. See upstream `csrc/knn/knn_cpu.cpp`
lines 112–123 and `csrc/knn/knn.cu` lines 503–515; local
`mps_pointops/chamfer.py` lines 133–166.

For normals, nearest-neighbor selection is made from positions using the
chosen norm; normal similarity does not choose a different neighbor. Normal
loss is masked at padded query rows, multiplied by batch weights, then follows
the same point `sum`/`mean` and batch `sum`/`mean` reductions as distance.
Bidirectional reduced loss adds both directions; unreduced loss is a tuple of
directional arrays, or one array with `single_directional=True`. A normal-only
backward should reach valid normal tensors through cosine and the gathered
reference normals on ordinary positive-weight inputs, but not point coordinates
through the nondifferentiable kNN indices. Combined distance and normal losses still give position gradients
from the distance term. Check this with isolated upstream normal-loss
backward; do not infer it from a combined-loss test. See upstream
`loss/chamfer.py` lines 111–150 and 261–292 and `ops/knn.py` lines 91–94.

`point_reduction="max"` has no normal-loss definition. Upstream rejects
explicit tensor normals before unpacking objects (`loss/chamfer.py` lines
243–247); object-carried normals can pass that precheck and then reach an
assertion in the one-direction helper (line 135). A future port should give a
clear `ValueError` for this object combination and record that deliberate
error-path deviation, rather than expose an assertion as public behavior.
Empty clouds are also outside the current package contract; expanding to
objects must not silently imply support for them.

The existing all-zero-weight behavior needs its own cases when normal inputs
are added. Pinned upstream exits before kNN, returns graph-connected zero
terms shaped `(B,B)` before batch reduction, and uses positions rather than
normal arrays for the zero normal term. In that branch a normal input need not
have a connected gradient. The current package already mirrors the unusual
shapes and normal-output slot for its supported subset; the expansion must
compare output type, shape, and gradient **presence**, not just zero values.
See upstream `loss/chamfer.py` lines 100–107 and the
[existing parity report](chamfer-upstream-parity-0.5.0.md).

## Acceptance gates for an expansion release

1. **Preserve the v0.8 subset.** Existing dense `norm=2`, no-normal CPU/MPS
   tests and the pinned 160-case direct parity matrix remain passing. Retain
   explicit rejection of L1 and normals; do not advertise object inputs or
   claim full PyTorch3D parity in the README.
2. **L1 distance.** Differential tests against the pinned upstream CPU
   extension cover L1 versus squared-L2 neighbor disagreement, exact ties,
   equal coordinates (the upstream `-1` subgradient), unequal lengths, padding,
   one-way/two-way output, all point and batch reductions, and absent,
   partial-zero, and all-zero batch weights. Test CPU and MPS Safe/Fast in
   separate processes with `PYTORCH_ENABLE_MPS_FALLBACK=0`. Compare first-order
   x/y/weight gradients and gradient presence. Gradcheck of the port on CPU
   uses untied, unequal-coordinate float64 fixtures only; this is separate
   from differential tests against upstream's float32 extension.
3. **Normals.** Compare both `abs_cosine` modes, parallel/opposite/orthogonal
   and zero normals, one-sided normals, repeated gathered references, unequal
   lengths, valid/padded rows, weights, every permitted reduction, and both
   directions. Test distance and normal outputs separately and together;
   compare normal gradients and assert whether position/normal gradients are
   present. Make `max` plus normals an explicit documented error. Zero-vector
   behavior must follow `F.cosine_similarity(..., eps=1e-6)`, including its
   first derivative on each tested backend.
4. **Objects.** With the pinned PyTorch3D version installed, compare tensor
   versus `Pointclouds` representation for homogeneous and heterogeneous
   nonempty clouds, with and without normals, object/tensor mixes, and
   explicit lengths/normals supplied alongside an object. Verify output and
   gradients to original list or padded tensor leaves. Importing this package
   without PyTorch3D must still work; if object support is optional, errors
   without the dependency must be explicit.
5. **Numerical and evidence gate.** On finite, well-separated float32
   fixtures, use the existing elementwise predicate
   `abs(port - upstream) <= 2e-5 + 2e-4 * abs(upstream)` for outputs and
   first-order gradients; assert exact structures, shapes, masks, and gradient
   presence. Keep exact-index tests for constructed ties and document any
   Safe/Fast last-bit differences. Save case counts, failures, raw results,
   source hashes, device, math mode, and reproduction commands. Broader
   float64 CPU tolerances, if needed, require a stated numeric reason.

These are separate acceptance gates because each feature changes a different
part of the implementation. L1 changes Metal search and backward; normals add
gathered values, reductions, and gradients; objects add input conversion and
an optional external type. The current v0.8 work should finish its supported
subset and performance acceptance without folding these untested branches
into the release. A later release can add each branch once its own gate passes.

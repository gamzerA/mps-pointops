# Tensor-coordinate Chamfer prototype

This branch extends dense tensor inputs to `(B, P, D)` and `(B, Q, D)` for
positive coordinate dimension `D`. The pinned PyTorch3D reference accepts
this shape, but the previous `mps-pointops` implementation required `D=3`.
The existing 3D Metal kernels and their operation order remain unchanged.
This is a prototype for review, not a release or a claim of full PyTorch3D
API compatibility.

## Scope and algorithm

The new Metal search handles `D != 3` in float32. CPU/CUDA tensor inputs use
the existing tiled PyTorch search generalized to loop over all coordinates.
For query $q_i$ and reference $r_j$, the two search metrics are

$$
d_1(q_i,r_j)=\sum_{c=0}^{D-1}|q_{ic}-r_{jc}|,
\qquad
d_2(q_i,r_j)=\sum_{c=0}^{D-1}(q_{ic}-r_{jc})^2.
$$

The metric is accumulated in increasing coordinate order. The lowest valid
reference index wins an equal rounded distance, including when all finite
coordinate pairs overflow to `+inf` in squared L2. Padded query rows write
distance `0` and index `-1`. The 1D, 2D, and 4D oracle matrix covers L1 and
squared L2; no square root is applied. The selected-pair first derivatives
follow the existing Chamfer contract: $\partial d_2/\partial q_c=2(q_c-r_c)$
and $\partial d_1/\partial q_c=+1$ when $q_c>r_c$, otherwise $-1$, including
coincident coordinates. The reference contribution is negated and scattered
to each selected point.

The existing feature-space kNN output cannot be used directly here: its
published distance is Euclidean rather than squared L2, it has no L1 or
per-cloud lengths path, and it discards `+inf` candidate distances. A
dedicated generalized Chamfer kernel preserves the pinned reference's
selection and gradient rules. Its scan remains $O(BPQD)$ time. The 3D path
keeps its measured kernel and `O(BPQ)` time bound.

On CPU/CUDA, the `(batch tile, query tile, reference tile, D)` difference
tensor is budgeted at about 16 MiB. Tile sizes account for `D`, dtype size,
and batch count; large batches are split as well. This bounds that temporary,
not all allocator memory. For an extreme `D` where one pair exceeds 16 MiB,
one pair remains the minimum tile.

Normals remain supported only for `D=3`; supplying normals at other
dimensions raises `NotImplementedError`. The MPS path accepts float32 only;
CPU/CUDA also accept float64. Both paths require finite valid coordinates,
and the implementation supports first-order gradients only. The usual
floating-point caveat applies at near ties: coordinate accumulation and
Safe/Fast Math may change the winner when distances differ by only rounding
error. No bitwise cross-device parity is promised for those inputs.

## Reproduce the direct upstream comparison

The opt-in verifier requires a source checkout of PyTorch3D at commit
`88e182f989c80836f4bd744e0d9cb1852762ce01` with its CPU extension built.
See `.github/workflows/chamfer-upstream-parity.yml` for the pinned PyTorch
version and the C++20 build-flag change required by that version. The
algorithmic upstream source is not modified. Run each MPS math mode in a
separate process; compilation options are cached inside a process.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  PYTHONPATH=/path/to/pytorch3d \
  python tests/verify_upstream_chamfer_nd.py \
    --upstream-root /path/to/pytorch3d \
    --upstream-build-note 'CPU extension; setup.py C++20 flags only; torch 2.14.1' \
    --devices mps --out /tmp/chamfer-nd-safe.json

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  PYTHONPATH=/path/to/pytorch3d \
  python tests/verify_upstream_chamfer_nd.py \
    --upstream-root /path/to/pytorch3d \
    --upstream-build-note 'CPU extension; setup.py C++20 flags only; torch 2.14.1' \
    --devices mps --out /tmp/chamfer-nd-fast.json
```

The verifier compares 384 configurations per target: three dimensions, two
metrics, random and tie/coincident fixtures, four point reductions, both
directions, and four weight settings. Unequal valid lengths and finite padded
rows are included. Each case checks loss structure, normal-loss slot,
gradient presence, values, first coordinate and weight gradients, and valid
nearest indices in both directions when applicable. Values use the established `atol=2e-5, rtol=2e-4` predicate;
indices and structures must match exactly. The JSON records the pinned source,
source hashes, runtime, math mode, per-case checks, and failures. The existing
3D direct-upstream matrix remains a separate required regression gate.

## Local result at source commit `2da9c7c`

On an Apple M5 Pro (48 GB), macOS with PyTorch 2.14.1, all three direct
comparisons against the pinned PyTorch3D CPU extension passed:

| Port target | Cases | Checks | Failed | Largest absolute value difference | Raw record |
| --- | ---: | ---: | ---: | ---: | --- |
| CPU | 384 | 2,976 | 0 | `1.90735e-6` | [CPU JSON](evidence/chamfer-variable-dim/cpu.json) |
| MPS Safe Math, fallback disabled | 384 | 2,976 | 0 | `3.81470e-6` | [Safe JSON](evidence/chamfer-variable-dim/mps-safe.json) |
| MPS Fast Math, fallback disabled | 384 | 2,976 | 0 | `3.81470e-6` | [Fast JSON](evidence/chamfer-variable-dim/mps-fast.json) |

Each record reports `port.commit=2da9c7c28c6dfd07f9238e49b0f36afa0074508e`
and `port.dirty=false`. The existing D=3 direct-upstream matrix also passed
480 cases and 3,240 checks in each Safe/Fast mode at this working tree before
the evidence commit. The focused unit suite passed 110 tests with two
environment skips in each MPS mode; CPU-oracle spot checks at `D=64,128` for
both norms matched outputs and first gradients within the same tolerance.

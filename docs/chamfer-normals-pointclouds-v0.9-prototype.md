# Experimental Chamfer normals and Pointclouds inputs

This is v0.9 development evidence, not a release claim. The direct oracle is
PyTorch3D 0.7.9 at commit
[`88e182f989c80836f4bd744e0d9cb1852762ce01`](https://github.com/facebookresearch/pytorch3d/commit/88e182f989c80836f4bd744e0d9cb1852762ce01),
using its CPU extension. Its [Chamfer source](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/loss/chamfer.py)
selects neighbors from positions, then gathers their normals.

For query normal n_i, selected reference normal m_j, and cosine similarity
c_i = cosine_similarity(n_i, m_j; eps=1e-6), the per-point normal loss is
1 - abs(c_i) when `abs_cosine=True`, and 1 - c_i otherwise. The selected
index comes from the positional L1 or squared-L2 search, including its
lowest-index tie rule. Normal gradients flow through cosine similarity and
the gather into selected normal vectors. A normal-only loss does not
differentiate coordinates through the discrete nearest-neighbor index.
Query padding is zeroed before point and batch reductions. Weights apply
before point reduction; distance and normals use the same mean denominators.
`point_reduction="max"` with any normals raises `ValueError`.

For an optional PyTorch3D `Pointclouds` object, the adapter reads
`points_padded()`, `num_points_per_cloud()`, and `normals_padded()`; these
replace explicitly supplied lengths and normals for that object. Tensor
inputs do not require PyTorch3D to be installed. Object/tensor mixed calls
are included in the direct parity matrix.

## Empty and all-zero-weight behavior

A cloud with zero valid points contributes zero positional distance and
gradient. If a nonempty query faces a zero-length reference cloud whose
normal tensor has allocated rows, PyTorch3D `knn_gather` supplies a zero
normal, so the valid-query normal loss is one. If that reference normal
tensor has zero allocated rows, upstream raises an index error; the
prototype does the same. `Pointclouds([])` on both sides returns scalar
zero under the default reduction.

The all-zero-weight branch preserves upstream's `(B, B)` intermediate
shape and graph-connected zero output. Its exceptional normal output is
anchored to positions; explicit normal tensors have no connected gradient.
With only one normal argument, the ordinary normal output is `None`.

## Direct parity evidence

The [extended verifier](../tests/verify_upstream_chamfer_extended.py)
covers L1 and squared-L2, both cosine modes, parallel/opposite/orthogonal
and zero normals, one-sided normals, all-zero and partially zero weights,
heterogeneous and empty clouds, one/two directional outputs, and tensor,
object/tensor, and object/object inputs. It compares output shapes,
gradient presence, values, and first derivatives of both total and
normal-only losses. The elementwise numeric rule is
`abs(port - upstream) <= 2e-5 + 2e-4 * abs(upstream)`.

| Target | Cases | Checked values or structures | Failures | Largest absolute difference |
| --- | ---: | ---: | ---: | ---: |
| CPU | 252 | 3,720 | 0 | 4.77e-7 |
| M5 Pro Safe Math | 252 | 3,720 | 0 | 1.91e-6 |
| M5 Pro Fast Math | 252 | 3,720 | 0 | 1.91e-6 |

Raw records: [CPU](evidence/chamfer-v090-extended/chamfer-extended-cpu.json),
[M5 Pro Safe](evidence/chamfer-v090-extended/chamfer-extended-mps-safe.json),
[M5 Pro Fast](evidence/chamfer-v090-extended/chamfer-extended-mps-fast.json).
Each includes the upstream commit, port commit and dirty-worktree flag,
source hashes, environment, and per-case checks. These first records were
made in an uncommitted development worktree; the source hashes identify
the measured code until a clean post-commit run replaces them. The
physical M1 was unavailable for this gate.

Reproduce from a checkout with the pinned upstream CPU extension built:

```bash
PYTHONPATH=.:/path/to/pytorch3d \
  python tests/verify_upstream_chamfer_extended.py \
  --upstream-root /path/to/pytorch3d --devices cpu --out /tmp/chamfer-cpu.json

PYTHONPATH=.:/path/to/pytorch3d PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python tests/verify_upstream_chamfer_extended.py \
  --upstream-root /path/to/pytorch3d --devices mps --out /tmp/chamfer-safe.json

PYTHONPATH=.:/path/to/pytorch3d PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python tests/verify_upstream_chamfer_extended.py \
  --upstream-root /path/to/pytorch3d --devices mps --out /tmp/chamfer-fast.json
```

On M5 Pro, the Chamfer regression suite passed 78 tests in each separate
Safe/Fast process with MPS fallback disabled. The existing L1 tie test
still repeats 32 Metal dispatches, and the coincident-point test checks
PyTorch3D's `-1` L1 subgradient.

Limits: MPS search remains brute force; normal-loss requests add a second
nearest-index dispatch per direction; the matrix uses finite float32
inputs; higher derivatives remain unsupported. Optional dependency and
packaging behavior must be checked before a release claim.

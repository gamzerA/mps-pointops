# Feature-space kNN contract and DGCNN validation

This development branch adds native Metal kNN for feature vectors with a
positive dimension other than three. The existing three-dimensional dense and
flat kernels are unchanged. FPS, radius search, Ball Query, and PointNet++
geometry inputs still require three coordinates.

## Public contract

| Surface | Inputs | Output | MPS limit |
|:--|:--|:--|:--|
| `mps_pointops.knn(query, ref, k)` | `(B,M,D)`, `(B,N,D)`, same `D >= 1` | Euclidean float32 distances and int64 local indices `(B,M,k)` | float32, `k <= min(N,256)` |
| `mps_pointops.flat.knn(x, y, k, batch_x, batch_y)` | `(N,D)`, `(M,D)`, sorted batch IDs | int64 `[query, reference]` global edges, no padding | float32, effective `k <= 256` for `D != 3` |
| `mps_pointops.pyg.register_mps()` `pyg::knn` | same flat features plus batch pointers | the same global edges | the same MPS feature-kNN limit |

Each query selects the `k` references with the smallest **float32 squared L2
distance**. Exact equal squared distances are ordered by smaller reference
index, regardless of the scrambled 32-point visit order. The dense output
returns Euclidean distance by taking the square root *after* selection. The
flat output removes `-1` padding for clouds containing fewer than `k`
references; an empty reference batch produces no edges for its queries.
The search is forward-only: no gradient flows through distances or indices.

For `D != 3`, the Metal source computes a direct, dimension-ordered sum:

```text
delta_d = fl32(x_jd - q_id)
s_0 = fl32(delta_0 * delta_0)
s_d = fl32(s_(d-1) + fl32(delta_d * delta_d)), d = 1 ... D-1
```

The source disables implicit FMA contraction in that helper. This avoids
using the cancellation-prone identity `||x||² + ||q||² - 2 x·q` as the Metal
baseline. Metal Safe/Fast Math, transcendental square root, subnormal
handling, compiler changes, and CPU `cdist` may still produce different bits.
Near ties can therefore select different neighbors across implementations;
tests require exact indices for their finite fixtures and use a tolerance on
Euclidean distances. No universal bitwise CPU/CUDA parity is claimed.

The feature kernel discards candidates whose accumulated squared distance is
non-finite. If fewer than `k` finite distances remain, dense unused slots are
`(inf, -1)` and flat unused slots are removed. A non-finite query similarly
has no valid feature neighbors in Safe Math. Fast Math's handling of NaN/Inf
is outside the validated contract. Even finite coordinates can overflow the
float32 square/sum and become invalid; callers needing valid neighbors must
keep their feature scale in a finite-distance range. The existing `D=3`
kernels have their separately documented behavior.

The MPS `D != 3` path raises a `ValueError` when effective `k > 256`; it does
not silently use PyTorch search. The existing 3D flat/PyG `k > 256` fallback
is a Phase 1 policy issue tracked separately. CPU tensors continue to use the
PyTorch reference implementation. `cosine=True` is unsupported.

## Direct and model-level evidence

The [new test suite](../tests/test_feature_knn.py) compares `D=1,2,4,64,128`
Metal results with an independent scalar float32 oracle. It includes exact
duplicates/ties, noncontiguous dense inputs, ragged and empty flat batches,
native-path enforcement, PyG wrapper/graph calls, and EdgeConv forward and
gradient checks. A four-stage self-contained dynamic-graph classifier checks
neighbor indices, logits, and gradients through a complete graph-rebuilding
path. On M5 Pro/macOS 26.5.2/PyTorch 2.14.1 with fallback disabled, the full
project suite passed [282 tests in Safe Math](pytest-feature-knn-safe-torch214-2026-10-01.log)
and [282 in Fast Math](pytest-feature-knn-fast-torch214-2026-10-01.log), with
12 skips in each (seven `k > n` parametrizations and five optional PyG tests).
Safe and Fast were separate processes.

For upstream model validation, [the verifier](../tools/dgcnn_upstream_parity.py)
loads the author's [DGCNN PyTorch model](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py)
from a *separate, clean* checkout at commit
`f765b469a67730658ba554e97dc11723a7bab628`. The upstream `model.py`
SHA-256 is `9be404728fa66eb5f9fc9a47d8553a9a75423f6e7d7b235496be354cfeb9b6e5`;
its repository declares MIT. No upstream source is bundled into this project.
For this experiment, the verifier changes only the hard-coded
`device = torch.device('cuda')` line to `device = x.device` **in memory**. The
CPU baseline retains the author's GEMM/topk neighbor selection. The MPS run
replaces only that module's kNN call with `mps_pointops.knn`; the original
DGCNN layers and copied parameters are otherwise used in both runs.

The fixed synthetic classifier fixture is batch 2, 64 points, `k=8`,
embedding dimension 128, dropout 0, and evaluation mode with gradients from
the mean squared logits. Four graph stages search in dimensions
`3, 64, 64, 128`. The [Safe](parity/dgcnn-upstream-safe.json) and
[Fast](parity/dgcnn-upstream-fast.json) raw results each record **0/4,096
neighbor-index mismatches**, maximum absolute logit error `3.3528e-8`,
maximum input-gradient error `2.7285e-11` (Safe) or `2.8649e-11` (Fast),
and maximum parameter-gradient error `3.2597e-9`. This validates one complete
classification-model forward/backward path on a synthetic fixture. It does
not measure ModelNet classification accuracy, full training, other point
counts, or original CUDA execution. The author's original CPU kNN uses a
different distance expression, so near-tie parity beyond this fixture is
unproven.

Reproduce with a clean pinned checkout and the local MPS environment:

```bash
git clone https://github.com/WangYueFt/dgcnn.git /private/tmp/dgcnn-upstream
git -C /private/tmp/dgcnn-upstream checkout f765b469a67730658ba554e97dc11723a7bab628
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python tools/dgcnn_upstream_parity.py --upstream /private/tmp/dgcnn-upstream \
  --output docs/parity/dgcnn-upstream-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python tools/dgcnn_upstream_parity.py --upstream /private/tmp/dgcnn-upstream \
  --output docs/parity/dgcnn-upstream-fast.json
```

## Performance and packaging

The [synchronized benchmark script](../bench/bench_feature_knn.py) records
public API timing on the M5 Pro in separate Safe/Fast processes, with CPU
PyTorch and MPS `cdist+topk` baselines. The direct accumulation path keeps
only `k <= 256` candidates per query and does not allocate a full `(M,N)`
distance matrix. The PyTorch `cdist+topk` control materializes a logical
`M*N*4` byte float32 matrix per batch; this figure is a shape-derived lower
bound, not a measured allocator peak. At `D=64,128`, the direct kernel is a
correctness baseline, not a documented speedup over the optimized matrix
path. Flat API timings also include batch-pointer construction and validation.

An isolated sdist and wheel build was checked against the source byte hash:
`mps_pointops/kernels/feature_knn.metal` SHA-256
`501f3b95c132900bd0ed919306de2154e665af5293e0cad3125307543a50ae27`
appeared byte-for-byte in both archives. This is a development-branch build,
not a published PyPI artifact.

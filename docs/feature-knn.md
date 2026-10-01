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
project suite passed [283 tests in Safe Math](pytest-feature-knn-safe-torch214-2026-10-01.log)
with 13 skips, and [282 in Fast Math](pytest-feature-knn-fast-torch214-2026-10-01.log)
with 14 skips. Seven skips are `k > n` parametrizations, six require optional
PyG packages, and the Fast process skips the Safe-only NaN/overflow case.
Safe and Fast were separate processes.

For upstream model validation, [the verifier](../tools/dgcnn_upstream_parity.py)
loads the author's [DGCNN PyTorch model](https://github.com/WangYueFt/dgcnn/blob/f765b469a67730658ba554e97dc11723a7bab628/pytorch/model.py)
from a *separate, clean* checkout at commit
`f765b469a67730658ba554e97dc11723a7bab628`. The upstream `model.py`
SHA-256 is `9be404728fa66eb5f9fc9a47d8553a9a75423f6e7d7b235496be354cfeb9b6e5`;
its repository declares MIT, and the `LICENSE` SHA-256 is
`288c5357e9620f022174625a153eb2423d5c14a8d9838bb4a9ef3deadf10549d`.
No upstream source is bundled into this project.
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
public API timing on the M5 Pro in separate Safe/Fast processes, with four
warmups and 20 timed calls per case. Each MPS timing includes dispatch,
allocation, and `torch.mps.synchronize()`. Raw samples, source commit
`8819a233800a3f8e99a5bd237a07623bed33d9ca`, six source SHA-256 values,
and CPU/MPS index mismatch counts are in the
[Safe 1,024-point](../bench/results/2026-10-01-apple-m5-pro-feature-knn-safe.json),
[Fast 1,024-point](../bench/results/2026-10-01-apple-m5-pro-feature-knn-fast.json),
[Safe DGCNN 64-point](../bench/results/2026-10-01-apple-m5-pro-feature-knn-dgcnn64-safe.json),
and [Fast DGCNN 64-point](../bench/results/2026-10-01-apple-m5-pro-feature-knn-dgcnn64-fast.json)
JSON files. Each case had zero CPU/MPS index mismatches on its seeded fixture.

| Public dense kNN shape | Safe Metal | Safe MPS `cdist+topk` | Fast Metal | Fast MPS `cdist+topk` |
|:--|--:|--:|--:|--:|
| Q=N=64, D=64, k=8 | 0.180 ms | 0.611 ms | 0.144 ms | 0.634 ms |
| Q=N=64, D=128, k=8 | 0.173 ms | 0.646 ms | 0.180 ms | 0.430 ms |
| Q=N=1,024, D=64, k=20 | 0.704 ms | 0.775 ms | 0.629 ms | 0.858 ms |
| Q=N=1,024, D=128, k=20 | 1.283 ms | 0.749 ms | 1.324 ms | 0.765 ms |

For the flat public API at Q=256, N=1,024, k=20, Safe/Fast Metal medians
were `2.325/1.898 ms` at D=64 and `2.731/2.504 ms` at D=128. Flat API
timings include batch-pointer construction and validation; the small
Q=N=64 flat cases still took `2.217–2.725 ms` on MPS. These timings are
not directly comparable with the dense kernel alone. CPU PyTorch reference
medians and all 20 samples are in the raw JSON.

The direct accumulation path keeps only `k <= 256` candidates per query and
does not allocate a full `(M,N)` distance matrix. At Q=N=1,024, a float32
`cdist` result is logically `1,024*1,024*4 = 4,194,304` bytes per batch;
the dense native outputs are `1,024*20*(4+8) = 245,760` bytes. These are
shape-derived quantities, **not measured allocator peaks**. The new kernel
is faster than MPS `cdist+topk` in the shown D=64 dense cases, but slower at
D=128 with 1,024 points. The direct path is therefore a numerical baseline;
a tiled/vectorized D=128 path is a performance follow-up. No general GPU or
CPU speedup is claimed from these fixtures.

An isolated sdist and wheel build was checked against the source byte hash:
`mps_pointops/kernels/feature_knn.metal` SHA-256
`501f3b95c132900bd0ed919306de2154e665af5293e0cad3125307543a50ae27`
appeared byte-for-byte in both archives. This is a development-branch build,
not a published PyPI artifact.

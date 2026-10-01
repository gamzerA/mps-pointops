# Pointcept v1.2.1 PTv1 subset on CPU and MPS

The opt-in [`mps_pointops.pointcept` shim](../mps_pointops/pointcept.py) covers
the five `pointops` calls used by Pointcept v1.2.1
`point_transformer_seg.py`: `farthest_point_sampling`, `knn_query`,
`grouping`, `knn_query_and_group`, and `interpolation`. It is registered as
`pointops` only after `mps_pointops.compat.install(pointcept=True)` and does
not change the package's public exports or version. This is a PTv1 Seg26
subset, not general Pointcept, Point Transformer V2, or CUDA pointops parity.

## Call contract

`xyz` and `new_xyz` are flat float32 `(N,3)` and `(M,3)` tensors on the same
CPU or MPS device. `feat` is `(N,C)` float32. `offset` and `new_offset` are
same-length int32/int64 vectors of **cumulative batch end indices**, with no
leading zero: `[4,9]` means batches `[0:4]` and `[4:9]`. They must be
nondecreasing and end at the respective point counts. Empty batches are
supported.

| Function | Result | Differentiation |
| --- | --- | --- |
| `farthest_point_sampling(xyz, offset, new_offset)` | Global int32 `(M,)` indices, first point of each nonempty batch first. Each output batch count must not exceed its input count. | No gradient through selection. |
| `knn_query(nsample, xyz, offset, new_xyz=None, new_offset=None)` | Global int32 `(M,K)` indices and float32 **Euclidean** distances. Neighbors sort by distance, then smaller reference index. Missing slots have index `-1`, distance `100000` (the square root of upstream's `1e10` initializer). | Selection and distances have no coordinate gradient. |
| `grouping(idx, feat, xyz, new_xyz=None, with_xyz=False)` | `(M,K,C)` features; with `with_xyz`, `(M,K,3+C)` with relative `xyz[idx]-new_xyz`. `-1` slots are all zero. | Feature and valid relative-position gradients propagate. |
| `knn_query_and_group(...)` | Grouped tensor and the query indices; supplied indices are reused. | As above. |
| `interpolation(xyz, new_xyz, feat, offset, new_offset, k=3)` | `(M,C)` inverse-Euclidean-distance interpolation using `1/(distance+1e-8)`. `-1` slots have zero weight; an empty reference batch returns zero. | Feature gradients propagate with fixed weights; no coordinate gradient. |

MPS kNN supports at most 256 neighbors. CPU kNN uses a chunked stable sort,
and MPS uses the package's Metal kNN kernel. Equal-distance ordering matches
between CPU and MPS in the focused grid fixture; the upstream CUDA heap's
tie order is not guaranteed to match. Like the pinned CUDA heap, a candidate
is accepted only when its squared distance is **strictly less than `1e10`**;
the exact 100,000-unit boundary and farther points retain `-1`/`100000`
padding on both CPU and MPS. The upstream plain `interpolation`
function can index the last feature row through `-1` padding and assign it a
small nonzero weight. This shim deliberately treats missing neighbors as
absent and renormalizes valid weights, so CUDA interpolation values need not
match when a batch has fewer than `k` references.

## Pinned Seg26 integration probe

The [harness](../bench/probe_pointcept_ptv1.py) reads official Pointcept tag
`v1.2.1`, commit `21fb5c51d8c622550b8e22c8ead984511e11d54a`. Its
temporary package symlinks the official builder, registry, misc, and PTv1
utility files unchanged. It loads the official Seg26 model source with **one
explicit substitution**, recorded in each JSON:

```python
torch.cuda.IntTensor(n_o)
# becomes
torch.tensor(n_o, dtype=torch.int32, device=p.device)
```

No `torch.cuda` function is monkeypatched, no source file in the Pointcept
checkout is modified, and unrelated Pointcept model families are not
imported. The official Seg26 source SHA-256 is
`1697a1d74af918f08c4c15415a3e31345921ec0254e58063e7d23d3700a37a75`;
the temporary one-substitution source SHA-256 is
`3b79a8e79f9991ccd039fea4157b56a7fd18977c64e0e801a2e6dbe1e5666111`.
The harness rejects a dirty pinned checkout, including untracked files, and
records SHA-256 values for all four unchanged official files that it loads.
The shim/probe source commit is `9199d1743536bb5bc30e4f2a1db30200f3082de7`;
their SHA-256 values are
`7118a712592f289551d08e0714605eda7fe6dfb3b5cf13f0534582e8b8d51aa1`
and `31a19be52ebf58cd8e41473ad8a780d346bc247d88039471244d6bbef76f61a3`.

The fixed probe has two 256-point batches, six input channels, 13 classes,
seed `2701`, and a mean-square-logit loss. The model is in **eval mode with
autograd enabled** so BatchNorm uses its initial running statistics while
forward and backward are both exercised. CPU and MPS receive the same initial
weights and inputs. The acceptance rule for logits is
`abs(cpu-mps) <= 0.002 + 0.005 * abs(cpu)`. Each gradient has a separate
pointwise rule, `abs(cpu-mps) <= 1e-7 + 0.005 * abs(cpu)`, and must have
nonzero CPU and MPS L2 norms with relative L2 error at most `0.01`.
The CPU coordinate, feature, and first-weight gradients have maximum absolute
values around `2.01e-5`, `1.83e-5`, and `6.36e-4` in the random fixture;
zero MPS gradients fail the new gate. A focused regression explicitly checks
this failure case. The seeded probe also replaces each actual CPU gradient
with zeros and asserts that the old logit-scale tolerance would pass all three
while the new gradient rule rejects all three. Every raw JSON records both
outcomes. `PYTORCH_ENABLE_MPS_FALLBACK=0` is mandatory; Safe and Fast Math run
in separate processes.

On a physical Apple M5 Pro, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1,
einops 0.8.1, the [Safe](pointcept-ptv1-seg26-m5pro-safe-2026-10-02.json)
and [Fast](pointcept-ptv1-seg26-m5pro-fast-2026-10-02.json) fixed 3D cloud
probes both passed. Maximum absolute CPU/MPS differences were:

| Tensor | Safe | Fast |
| --- | ---: | ---: |
| Logits `(512,13)` | `1.04e-7` | `8.94e-8` |
| Coordinate gradient `(512,3)` | `4.55e-12` | `3.64e-12` |
| Feature gradient `(512,6)` | `3.64e-12` | `2.96e-12` |
| First encoder weight gradient `(32,6)` | `1.16e-10` | `1.16e-10` |

A second, tie-heavy line-grid fixture also passed in
[Safe](pointcept-ptv1-seg26-m5pro-line-safe-2026-10-02.json) and
[Fast](pointcept-ptv1-seg26-m5pro-line-fast-2026-10-02.json) Math, with
maximum logit difference `1.19e-7` in both. Across all four probes, the
largest gradient relative L2 error was `3.70e-7`, well below `0.01`.
This fixture exposed an earlier CPU top-k tie
ordering mismatch; the final shim uses stable CPU sorting to make its stated
smaller-index rule explicit. The focused CPU/MPS shim and probe tests in
[`tests/test_pointcept.py`](../tests/test_pointcept.py) and
[`tests/test_pointcept_probe.py`](../tests/test_pointcept_probe.py) passed
**19/19** in each Safe and Fast process.
These tests used the `pytest` CLI against a regular installation of this
checkout in a temporary package target, matching CI's installed-package
import path; the probe test loads the benchmark script by file path.

Reproduce with the official tag checkout and `einops==0.8.1` available on
`PYTHONPATH` or installed in the test environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/probe_pointcept_ptv1.py \
  --pointcept-root /private/tmp/pointcept-v121 \
  --output docs/pointcept-ptv1-seg26-m5pro-safe-2026-10-02.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/probe_pointcept_ptv1.py \
  --pointcept-root /private/tmp/pointcept-v121 \
  --output docs/pointcept-ptv1-seg26-m5pro-fast-2026-10-02.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/probe_pointcept_ptv1.py \
  --pointcept-root /private/tmp/pointcept-v121 --fixture line \
  --output docs/pointcept-ptv1-seg26-m5pro-line-safe-2026-10-02.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/probe_pointcept_ptv1.py \
  --pointcept-root /private/tmp/pointcept-v121 --fixture line \
  --output docs/pointcept-ptv1-seg26-m5pro-line-fast-2026-10-02.json
```

These are one-run functional probes, not performance benchmarks. Training-mode
BatchNorm, other PTv1 variants, real CUDA pointops comparison, M1 hardware,
and larger clouds remain untested. A separate hosted M1 Virtual Torch 2.12
[`nn.Linear` bias probe](pyg28-voxel-avg-pool.md#standalone-framework-probe)
previously found a framework-level bias omission; this physical M5 Pro Torch
2.14.1 model parity does not establish behavior on that runtime.

<h1 align="center"><img src="docs/assets/pointops-mark.svg" width="42" height="42" alt=""> mps-pointops</h1>

<p align="center">
  <a href="https://github.com/gamzerA/mps-pointops/actions/workflows/ci.yml"><img src="https://github.com/gamzerA/mps-pointops/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.10 or later"></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/PyTorch-2.7%2B-EE4C2C?logo=pytorch&amp;logoColor=white" alt="PyTorch 2.7 or later"></a>
  <a href="#license"><img src="https://img.shields.io/badge/License-Apache--2.0%20AND%20MIT-4B5563" alt="Apache-2.0 AND MIT license"></a>
</p>

**Point-cloud operators for PyTorch on Apple Silicon.** Native Metal kernels
run farthest point sampling, k nearest neighbors, and Ball Query on MPS.
Compatibility stand-ins cover supported `pointnet2_ops`, `knn_cuda`, and
`torch_cluster` call sites; CPU tensors use PyTorch reference implementations.

<p align="center">
  <img src="docs/assets/pointops-hero.svg" width="1200" alt="Three operator diagrams: FPS chooses spread-out centers; kNN ranks neighbors by distance; Ball Query retains the first K points inside a radius.">
</p>

[Quick start](#quick-start) · [Results](#benchmark) ·
[Equations](#the-operators-in-equations) · [Compatibility](#compatibility) ·
[Numerical contract](docs/ball-query-math.md)

## Quick start

Requires an Apple Silicon Mac, Python 3.10 or later, and PyTorch 2.7 or later
with MPS. The Metal kernels compile on first use.

```bash
python -m pip install "git+https://github.com/gamzerA/mps-pointops.git@v0.1.1"
```

```python
import torch
from mps_pointops import ball_query, furthest_point_sample, knn

xyz = torch.randn(1, 100_000, 3, device="mps")
centers_idx = furthest_point_sample(xyz, 1024)
centers = xyz.gather(1, centers_idx[..., None].expand(-1, -1, 3))
distance, neighbor_idx = knn(centers, xyz, 128)
distance2, radius_idx = ball_query(centers, xyz, 0.1, 64)
```

| Operator | Selection rule | Native result |
|:--|:--|:--|
| FPS | Farthest from the already selected centers | `int64` center indices |
| kNN | Nearest `k`, sorted by squared distance then index | Euclidean distances, `int64` indices |
| Ball Query | First `K` inside a strict radius, in input order | Squared distances, `int64` indices; `-1` padding |

For a query $q_i$ and reference point $x_j$, the shared distance is
$s_{ij}=\sum_{d=0}^{2}(q_{id}-x_{jd})^2$; each operator selects indices by a
different rule. [The equations](#the-operators-in-equations) give the full
selection and gradient formulas. On MPS, Ball Query supports coordinate
gradients for its squared distances; FPS and kNN do not implement backward.

**Measured on one M5 Pro, with 100k randomly ordered reference points:** FPS
**5.2×**, kNN **2.9×**, and Ball Query **1.6×** faster than the fastest tested
CPU library for each operation. Ball Query is slower on spatially sorted input
(32.5 ms versus SciPy's 23.2 ms at 100k points). The
[benchmark](#benchmark) states the setup and links the raw results.

For existing CUDA-oriented imports, call `mps_pointops.compat.install()`
*before* importing `pointnet2_ops`, `knn_cuda`, or `torch_cluster`:

```python
import mps_pointops.compat
mps_pointops.compat.install()

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from torch_cluster import fps, knn as flat_knn, radius
```

The native API is dense and batched. The flat API in `mps_pointops.flat`
supports sorted batch vectors and global indices. See
[Compatibility](#compatibility) for the stand-ins' behavior and
[License](#license) for component notices.

## Why

Point-MAE based 3D anomaly detection, such as the
[MulSen-AD](https://github.com/ZZZBBBZZZ/MulSen-AD) baseline, groups points with
`pointnet2_ops.furthest_point_sample` and `knn_cuda.KNN`. Both are CUDA only, so
the code does not run on a Mac at all. The usual workaround is to rewrite them
in plain PyTorch and run on MPS. That works, but it is slow, and a 48 GB M5 Pro
ends up slower than its own CPU.

The goal of this project is drop-in Metal kernels for these ops that beat the
best CPU implementations on the same machine.

## Real data: a MulSen-AD 3D detector gives the same results as on CUDA

The Point-MAE 3D-only anomaly detector from the MulSen-AD baseline (MulSen-AD's
released feature extractor, coreset memory bank and object score) was fit and
scored on the Mac GPU with `compat.install()` providing `pointnet2_ops` and
`knn_cuda`, and compared with the same runs made earlier on CUDA with the real
extensions (Windows, RTX 2080, PyTorch 2.9.1 + CUDA 13).

- 45 runs: 15 MulSen-AD categories x 3 seeds, fit on normal samples of a
  frozen research split and scored on its validation samples. The same sample
  IDs and labels were used on both machines.
- Object AUROC, object AP, 3D-label AUROC and 3D-label AP: identical to the
  CUDA runs in all 45 runs.
- Per-sample anomaly scores: largest relative difference 9.85e-5, and the same
  ranking of samples in every run.

This is a validation-set comparison from a separate research project, so the
split, scores and runner scripts are not part of this repository. Setup:
Apple M5 Pro, macOS 26.5.2, PyTorch 2.14.1, this package at commit 80bbca5.

## Real data: MulSen-AD grouping

[examples/mulsen_grouping.py](examples/mulsen_grouping.py) loads MulSen-AD
point clouds the way MulSen-AD's dataset code does (open3d, duplicate vertices
removed, centered) and runs MulSen-AD's own `models.models.Group(num_group=1024,
group_size=128)`, unmodified, on MPS with `compat.install()`.

30 clouds, 2 from each of the 15 classes, 21,168 to 117,259 points, Apple M5
Pro:

| | min | median | max |
|---|---:|---:|---:|
| MulSen `Group` on MPS with mps-pointops (FPS + gather + kNN + indexing) | 10.5 ms | 36.0 ms | 49.1 ms |
| Best CPU libraries (fpsample FPS + scipy cKDTree kNN, nothing else) | 40.1 ms | 163.4 ms | 221.9 ms |
| Plain PyTorch on MPS (FPS loop + `cdist`/`topk`) | 131.1 ms | 330.6 ms | 423.7 ms |

- mps-pointops is 3.0x to 5.0x faster than the CPU libraries (median 4.5x)
  and 7.6x to 13x faster than plain PyTorch on MPS (median 9.2x).
- FPS centers match the pointnet2_ops-contract reference and neighbors match
  an exact float32 oracle: 0 mismatches in all 30 clouds.
- None of these clouds has points within the near-origin cutoff that
  pointnet2_ops skips (see [Compatibility](#compatibility)), so that rule did
  not come into play here.

Per-cloud numbers: [examples/results/mulsen_grouping.json](examples/results/mulsen_grouping.json).
The full MulSen-AD pipeline also needs pretrained DINO ViT-B/8 and Point-MAE
weights and has not been run yet.

## Benchmark

![M5 Pro measured speedups at 100,000 randomly ordered reference points: FPS 5.2 times, kNN 2.9 times, Ball Query 1.6 times faster than the fastest tested CPU libraries](docs/assets/m5-pro-speedup.svg)

The figure compares each Metal kernel with the fastest tested CPU library for
that operation on the **same M5 Pro**. It uses batch 1, 100,000 reference
points, 1,024 samples or queries, random input order, and the median of five
runs. SciPy times include KD-tree construction; device transfer is excluded.
**Sorted-input exception:** Ball Query takes 32.5 ms on Metal versus 23.2 ms
with SciPy. The chart is generated directly from the committed
[benchmark JSON](bench/results/2026-10-01-apple-m5-pro.json) and
[Ball Query JSON](bench/results/2026-10-01-apple-m5-pro-ball-query-port.json)
using [this script](tools/render_readme_assets.py).

The synthetic points lie near a unit sphere and use MulSen-AD scale. Full
result tables: [FPS/kNN random order](bench/results/2026-10-01-apple-m5-pro.md),
[FPS/kNN sorted order](bench/results/2026-10-01-apple-m5-pro-sorted.md),
[Ball Query random order](bench/results/2026-10-01-apple-m5-pro-ball-query-port.md),
and [Ball Query sorted order](bench/results/2026-10-01-apple-m5-pro-ball-query-port-sorted.md).

Apple M5 Pro, 48 GB, macOS 26.5.2, torch 2.14.1, random point order:

| op | points | **mps-pointops (Metal)** | torch on MPS | torch on CPU | best CPU library |
|---|---:|---:|---:|---:|---:|
| FPS (1024 samples) | 20,000 | **6.1 ms** | 112.4 ms | 95.2 ms | 35.0 ms (fpsample) |
| | 100,000 | **33.3 ms** | 247.5 ms | 578.2 ms | 174.0 ms (fpsample) |
| kNN (1024 queries, k=128) | 20,000 | **3.8 ms** | 15.1 ms | 10.9 ms | 5.2 ms (scipy cKDTree) |
| | 100,000 | **6.1 ms** | 120.9 ms | 40.2 ms | 17.6 ms (scipy cKDTree) |
| Ball query (1024 queries, K=64, r=0.1) | 20,000 | **6.5 ms** | 57.2 ms | 35.9 ms | 5.6 ms (scipy cKDTree) |
| | 100,000 | **13.2 ms** | 286.4 ms | 169.8 ms | 21.3 ms (scipy cKDTree) |

What this shows:

- **The FPS and kNN Metal kernels beat the tested CPU libraries**: FPS is 5.2x to 5.8x
  faster than fpsample and kNN is 1.4x to 2.9x faster than scipy's KD-tree
  (including its build). Against plain PyTorch on MPS, FPS is 7x to 18x
  faster and kNN 4x to 20x faster.
- **Plain PyTorch FPS on MPS grows much slower than the work.** 5x the points
  took it from 112 ms to 248 ms. Our hypothesis is a fixed cost per step (1024
  sequential steps of several small kernels each: dispatch, scheduling,
  synchronization). This has not been profiled. The Metal kernel runs all 1024
  steps in one dispatch.
- **Plain PyTorch kNN is not exact.** `cdist` uses a matrix multiply here, so
  distances are off by up to 4.9e-4 and 40 to 313 neighbors land in the wrong
  position; at 100k points 1 to 4 of the 131,072 true neighbors are missing.
- **Point order matters for kNN, and the kernel handles it.** Real scans store
  points in spatial order. With points sorted by x, the Metal kNN takes 2.6 to
  5.4 ms, about the same as random order. Without the scrambled scan order
  described below it took 32 to 36 ms at 100k points.
- **Ball Query is faster at 100k with random point order:** 13.2 ms versus
  scipy's 21.3 ms including tree construction. At 20k, scipy is faster
  (5.6 ms versus 6.5 ms). With points sorted by x, the 100k Metal time is
  32.5 ms versus scipy's 23.2 ms. Each query scans points in input order and
  stops after K hits, so storage order affects runtime.

Timings move by a few ms, sometimes more, between runs. Inputs are already
resident on each implementation's device; transfer time is outside the timer.
The scipy times include building the KD-tree. fpsample's QuickFPS
(`bucket_fps_kdline_sampling`) is absent because in fpsample 1.0.2 it ignores
`start_idx` and returns a different, sorted sample set.

### Correctness checks

Each benchmark row records counts, not percentages, so a single mismatch stays
visible:

- FPS: indices that differ from the PyTorch CPU reference.
- kNN: against an exact float32 oracle (squared distances rounded like the
  kernel, sorted by distance then index): true neighbors missing, neighbors in
  the wrong position, and the largest distance error.
- Ball query: indices and squared distances that differ from the
  PyTorch3D-contract reference on CPU.

On the listed synthetic inputs, the Metal kernels had 0 index mismatches.
Ball Query's maximum squared-distance error against the separate-operation
CPU reference was 1.9e-9; the FPS and kNN checks reported no distance error.
These are observations on one M5 Pro, not a guarantee for every input, GPU or
compiler. Before the flat API was added, the v0.1.1 baseline with PyTorch 2.7.0
and MPS available reported **105 passed, 7 skipped** in each of the Safe and
Fast Math test processes
([Safe log](docs/pytest-safe-torch27-2026-10-01.log),
[Fast log](docs/pytest-fast-torch27-2026-10-01.log)).
For the flat API implementation at commit `fe4a683f33ca0ad9599f1c884a356c46df5e4213`,
PyTorch 2.7.0 on the same M5 Pro reported **127 passed, 7 skipped** in each
separate process ([Safe log](docs/pytest-flat-safe-torch27-2026-10-01.log),
[Fast log](docs/pytest-flat-fast-torch27-2026-10-01.log)).

### Run it

```bash
uv venv --python 3.12 && uv pip install torch numpy scipy fpsample pytest
.venv/bin/python bench/bench_pointops.py                 # random point order
.venv/bin/python bench/bench_pointops.py --order sorted  # spatially sorted
.venv/bin/python -m pytest tests

# MulSen-AD grouping on real data (needs open3d and timm too)
.venv/bin/python examples/mulsen_grouping.py \
    --mulsen-code path/to/MulSen-AD --data path/to/MulSen_AD --per-class 2
```

Results from other Apple Silicon chips are welcome as pull requests.

## Compatibility

`mps_pointops.compat.install()` registers `pointnet2_ops`,
`pointnet2_ops.pointnet2_utils`, `knn_cuda` and `torch_cluster` in
`sys.modules`, unless a real package with that name is already importable.
Call it before importing code that uses those names. `force=True` replaces an
already loaded or importable package; the default preserves it.

| stand-in | behavior |
|---|---|
| `pointnet2_utils.furthest_point_sample(xyz, npoint)` | Metal kernel. int32 output, starts at index 0, and never picks points with x² + y² + z² <= 1e-3, as pointnet2_ops does |
| `pointnet2_utils.gather_operation`, `grouping_operation` | `torch.gather`, differentiable |
| `pointnet2_utils.ball_query(radius, nsample, xyz, new_xyz)` | Metal on MPS. int32, empty slots repeat the first neighbor, no neighbor gives all zeros |
| `knn_cuda.KNN(k, transpose_mode)` | Metal kernel. Same layouts as knn_cuda, Euclidean distances, no gradients |

### Flat `torch_cluster` subset

`mps_pointops.flat` exposes `fps`, `knn`, and `radius` for flat `(N, 3)` point
coordinates. Batch vectors must be sorted, such as `[0, 0, 1, 1, 1]`; missing
batch IDs represent empty clouds. A separate offset array is built for the
reference and query sets, which may have different sizes. All returned indices
are global indices into the corresponding flat input.

```python
import torch
from mps_pointops import flat

x = torch.tensor([[0., 0, 0], [2., 0, 0], [10., 0, 0]], device="mps")
batch_x = torch.tensor([0, 0, 1], device="mps")
y = torch.tensor([[1., 0, 0], [11., 0, 0]], device="mps")
batch_y = torch.tensor([0, 1], device="mps")

centers = flat.fps(x, batch_x, ratio=0.5, random_start=False)
knn_edges = flat.knn(x, y, 2, batch_x, batch_y)
radius_edges = flat.radius(x, y, 1.1, batch_x, batch_y, max_num_neighbors=32)
```

`fps` returns `int64` sampled point indices, with `ceil(ratio * N_b)` samples
per non-empty batch (`ratio=None` means `0.5`; `random_start=True` by default).
It also accepts an explicit `ptr` offset array. `knn` and `radius` return
`int64` `edge_index` tensors of shape `[2, E]`: row 0 is a query index into
`y` and row 1 is a reference index into `x`. kNN is ordered by squared distance,
then reference index; if a batch has fewer than `k` references, it emits only
the available edges. Radius search uses strict `distance² < r²` and takes up
to `max_num_neighbors` matches in reference input order. Padded internal slots
are removed before returning the edge tensor. CPU inputs use PyTorch; MPS
inputs use Metal for supported sizes, with a PyTorch fallback for kNN `k > 256`.
On MPS, FPS and kNN require float32; radius accepts float32 or float16 and
the same positive-radius lower bound as the native Ball Query contract.
Flat FPS uses one threadgroup per cloud and scans only that cloud's
offset range. Very uneven batch sizes can still leave a long-running group;
splitting one FPS sequence across groups would need synchronization after each
selected point and remains a performance task.

This follows the `fps`, `knn`, and `radius` call signatures of
[`torch_cluster` 1.6.3](https://github.com/rusty1s/pytorch_cluster/tree/1.6.3/torch_cluster)
for three-dimensional coordinates. Cosine kNN and `ignore_same_index=True` are
unsupported and raise an error. The shim also provides `knn_graph` and
`radius_graph` using these searches, including `loop` and `flow`. It exposes
explicitly unsupported `grid_cluster`, `graclus_cluster`, `random_walk`, and
`nearest` placeholders so that importing PyG 2.7.0 succeeds; calling those
placeholders raises `NotImplementedError`.

[PyG 2.7.0](https://github.com/pyg-team/pytorch_geometric/blob/2.7.0/torch_geometric/nn/pool/__init__.py)
calls these `torch_cluster` functions directly. Its `fps`, `knn`, `radius`,
`knn_graph`, and `radius_graph` entry points were exercised with MPS tensors
on an M5 Pro. In contrast,
[PyG 2.8.0](https://github.com/pyg-team/pytorch_geometric/blob/2.8.0/torch_geometric/nn/pool/__init__.py)
calls separate `torch.ops.pyg` operators. The shim does not replace the latter
path or provide the rest of `torch_cluster`.

Near ties can resolve differently from the CUDA packages. FPS and kNN round
their squared distances without FMA; Ball Query uses explicit FMA. The CUDA
kernels use their own arithmetic and reduction order.

The native API (`mps_pointops.furthest_point_sample`, `mps_pointops.knn`,
`mps_pointops.ball_query`) returns int64 indices; native FPS does not skip
points near the origin. It uses dense, batched `(B, N, 3)` tensors, while the
flat API above uses `(N, 3)` tensors and optional batch vectors.

## The operators in equations

For batch `b`, let `q[b, i]` be query `i`, where `0 ≤ i < Q`, and let
`x[b, j]` be reference point `j`, where `0 ≤ j < P`. Both have three
coordinates, indexed by `d = 0, 1, 2`. The mathematical squared distance is

$$
s_{bij} = \sum_{d=0}^{2}\bigl(q_{bid}-x_{bjd}\bigr)^2.
$$

### Farthest point sampling

Starting at `c₀ = start_idx`, keep each point's distance to its **closest
already selected center**, then choose the farthest (smaller index on a tie):

$$
m_j^{(t)} = \min_{0\le u\le t}
  \sum_{d=0}^{2}(x_{bjd}-x_{b,c_u,d})^2,
\qquad
c_{t+1} = \min\{j:m_j^{(t)}=\max_{\ell}m_{\ell}^{(t)}\}.
$$

The outer minimum makes the smaller input index win a tie. This is the
native FPS rule for finite coordinates;
degenerate clouds can select an index more than once. The PointNet2 stand-in
also skips points near the origin except for its initial center.

### k nearest neighbors

For each query, sort candidate indices by squared distance and then input
index. The returned distance is **Euclidean**, while sorting uses its square:

$$
\pi_{bi}=\mathrm{argsort}_{j}\bigl(s_{bij},j\bigr),
\qquad
I_{bik}=\pi_{bi}[k],
\qquad
D_{bik}=\sqrt{s_{bi,I_{bik}}}.
$$

The MPS kernel uses this tie rule; the CPU fallback uses
`torch.cdist(...).topk(...)` and can resolve near ties differently.

### Ball Query: first K within a radius

The radius is rounded to `float32` **before** it is squared, matching the
PyTorch3D threshold construction:

$$
R_2=\mathrm{fl}_{32}\left(
  \mathrm{fl}_{32}(r)\cdot\mathrm{fl}_{32}(r)
\right),\qquad
J_{bi}=\bigl[j\in\{0,\ldots,P-1\}:s_{bij}<R_2\bigr]_{\text{input order}}.
$$

$$
(I_{bik},S_{bik})=
\begin{cases}
\bigl(J_{bi}[k],s_{bi,J_{bi}[k]}\bigr),
  & k<\min(K,|J_{bi}|),\\
(-1,0), & \text{otherwise}.
\end{cases}
$$

These equations give the selection contract. The Metal kernel accumulates
the distance with explicit FMA operations and uses a normalized comparison
at very small radii; near a floating-point boundary, its result can differ
from evaluating the real-valued `s` above. See the
[numerical contract](docs/ball-query-math.md) for the exact policy.

The boundary is strict (`<`), and **first K in input order** is different
from the k nearest points. A zero squared distance can be a real match:
check `I >= 0` to detect padding. On MPS, `S` is `float32`, `I` is
`int64`, and the inputs can be `float32` or `float16`.

For a fixed selected index, let `G[b,i,k] = ∂L/∂S[b,i,k]`. The squared-distance
gradient first gives
`∂s/∂q[b,i,d] = 2(q[b,i,d] - x[b,j,d])` and
`∂s/∂x[b,j,d] = -2(q[b,i,d] - x[b,j,d])`. The chain rule then propagates
to both coordinate tensors:

$$
\frac{\partial L}{\partial q_{bid}}
=2\sum_{k:I_{bik}\ge0}G_{bik}
  \bigl(q_{bid}-x_{b,I_{bik},d}\bigr),
\qquad
\frac{\partial L}{\partial x_{bjd}}
=2\sum_{i,k:I_{bik}=j}G_{bik}
  \bigl(x_{bjd}-q_{bid}\bigr).
$$

Indices and the Python scalar radius have no gradient. The equations describe
the selected squared-distance function, not differentiation through the
discrete neighbor choice or float32 rounding. At tiny radii the Metal kernel
uses a normalized comparison, and FMA/flush behavior can change boundary
bits; [the detailed contract](docs/ball-query-math.md) gives its precise
policy, derivation, tests, and limitations.
The formulas state established geometric operations; the
[provenance note](docs/ball-query-provenance.md) separates paper concepts,
external implementation contracts, and this project's code.

## How the kernels work

All three kernels are in [mps_pointops/kernels/](mps_pointops/kernels/) and are
compiled at runtime with `torch.mps.compile_shader`. FPS and kNN turn off FMA
contraction and sum squared distances as ((dx² + dy²) + dz²). Ball Query uses
an explicit FMA sequence and a documented policy for very small radii.

**FPS** ([fps.metal](mps_pointops/kernels/fps.metal))

- One threadgroup of 1024 threads per point cloud runs all `npoint` steps, so
  sampling is a single dispatch instead of thousands of small ones.
- Each thread owns every 1024th point and keeps its running minimum squared
  distance to the sampled set.
- Each step finds the farthest point with a `simd_max` / `simd_min` reduction
  inside simdgroups, then across simdgroups through threadgroup memory. Ties go
  to the smaller index, like `torch.argmax`.
- A batch of B clouds uses B threadgroups. With batch 1 only one GPU core is
  busy, so there is room left for a multi-threadgroup version.

**kNN** ([knn.metal](mps_pointops/kernels/knn.metal))

- One simdgroup of 32 threads per query. The lanes compute distances to 32
  reference points at a time.
- Each query keeps its current k nearest points in threadgroup memory, sorted
  by (squared distance, index). A point is inserted only if it beats the
  current k-th entry. After the list fills up that is rare, so most of the work
  is computing distances.
- Chunks of 32 points are visited in a scrambled order: a stride near
  0.618 x the chunk count, coprime to it, so every chunk is visited once. In
  storage order, a spatially sorted scan walks toward each query and nearly
  every point becomes a new nearest one, which made the kernel 6x to 8x slower.
  The result does not depend on the order, since the list is sorted by
  (distance, index).

FPS and kNN assume 32-wide simdgroups, as on Apple GPUs so far. The first
call checks the width on the GPU and raises an error if it is different.

**Ball Query** ([ball_query.metal](mps_pointops/kernels/ball_query.metal))

- One lane scans reference points for each query in input order, keeping the
  first K matches and stopping once K are found.
- The native output is squared distance plus `int64` index with `-1` padding.
  The PointNet2 stand-in converts indices to `int32` and repeats the first
  neighbor for padding. The Metal kernel supports float32/float16 coordinates
  and coordinate gradients.
- [Math and floating-point contract](docs/ball-query-math.md) records the
  radius-square rounding, subnormal behavior, boundary policy and backward
  equations. PyTorch3D's full call signature and bitwise parity remain future
  compatibility work.

## Contracts

The pure PyTorch versions in [mps_pointops/reference.py](mps_pointops/reference.py)
provide CPU fallbacks and benchmark baselines. MPS boundary arithmetic for
Ball Query is specified separately in the numerical contract.

- `furthest_point_sample(xyz, npoint, start_idx=0, skip_near_origin=False)`:
  starts at `start_idx`, ties go to the smaller index, and once every point is
  taken the remaining slots repeat index 0. Float32 only on MPS.
- `knn(query, ref, k)`: Euclidean distances and indices, sorted by squared
  distance and then by index. `k <= N`, and `k <= 256` on MPS. Float32 only on
  MPS. (The reference itself uses `cdist` and `topk`, so it is only the
  baseline; the tests use an exact oracle.)
- `ball_query(query, ref, radius, K)`: PyTorch3D-style first-K contract. It
  returns the first `K` points in input order satisfying strict radius
  membership, with index `-1` and distance `0` padding. The threshold is
  `fl32(fl32(radius) * fl32(radius))`. MPS uses an explicit FMA accumulation
  and a small-radius normalization policy, so boundary decisions and final
  distance bits can differ from the separate-operation CPU reference. See
  [the numerical contract](docs/ball-query-math.md).

## Roadmap

1. ~~Metal kernel for FPS~~
2. ~~Metal kernel for kNN~~
3. ~~Drop-in stand-ins for `pointnet2_ops` and `knn_cuda`, checked on real
   MulSen-AD data~~
4. ~~Metal kernel for ball query~~ — improve performance on spatially sorted
   inputs and finish the PyTorch3D API compatibility surface
5. ~~MulSen-AD Point-MAE 3D detector on MPS, matching the CUDA runs~~ — the
   full TripleAD pipeline (RGB + IR + 3D) is next
6. `torch_cluster`-style flat/ragged `radius`, `knn` and `fps` — the 3D
   subset and PyG 2.7.0 direct-import path are implemented; operator
   registration for PyG 2.8.0 and wider coordinate dimensions remain

## License

Apache-2.0 for the repository. The Ball Query kernel, Python implementation,
contract tests, numerical documentation and probe were ported from an earlier
MIT-licensed local prototype; its full notice is retained in
[LICENSES/MIT-ball-query.txt](LICENSES/MIT-ball-query.txt). No PyTorch3D or
PointNet++ source was copied into those files. See the
[provenance note](docs/ball-query-provenance.md).

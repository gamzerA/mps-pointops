# mps-pointops

Point cloud ops for PyTorch on Apple Silicon (MPS): farthest point sampling,
k nearest neighbors and ball query.

**Status:** farthest point sampling and kNN have Metal kernels. Ball query is
benchmarked but still pure PyTorch.

```python
import torch
from mps_pointops import furthest_point_sample, knn

xyz = torch.randn(1, 100_000, 3, device="mps")
idx = furthest_point_sample(xyz, 1024)           # (1, 1024), like pointnet2_ops
centers = xyz[:, idx[0]]
dist, nbr = knn(centers, xyz, 128)                # (1, 1024, 128), like knn_cuda
```

## Why

Point-MAE based 3D anomaly detection, such as the
[MulSen-AD](https://github.com/ZZZBBBZZZ/MulSen-AD) baseline, groups points with
`pointnet2_ops.furthest_point_sample` and `knn_cuda.KNN`. Both are CUDA only, so
the code does not run on a Mac at all. The usual workaround is to rewrite them
in plain PyTorch and run on MPS. That works, but it is slow, and a 48 GB M5 Pro
ends up slower than its own CPU.

The goal of this project is drop-in Metal kernels for these ops that beat the
best CPU implementations on the same machine.

## Benchmark

MulSen-AD scale: 20k to 100k points per object, batch 1, 1024 FPS centers,
k = 128 neighbors (Point-MAE `num_group` and `group_size`). Median of 5 runs.
Full tables: [random point order](bench/results/2026-10-01-apple-m5-pro.md),
[spatially sorted point order](bench/results/2026-10-01-apple-m5-pro-sorted.md).

Apple M5 Pro, 48 GB, macOS 26.5.2, torch 2.14.1, random point order:

| op | points | **mps-pointops (Metal)** | torch on MPS | torch on CPU | best CPU library |
|---|---:|---:|---:|---:|---:|
| FPS (1024 samples) | 20,000 | **7.7 ms** | 133.7 ms | 117.8 ms | 34.8 ms (fpsample) |
| | 100,000 | **32.6 ms** | 192.7 ms | 619.3 ms | 173.6 ms (fpsample) |
| kNN (1024 queries, k=128) | 20,000 | **3.2 ms** | 18.6 ms | 10.6 ms | 5.0 ms (scipy cKDTree) |
| | 100,000 | **5.7 ms** | 116.1 ms | 39.6 ms | 17.8 ms (scipy cKDTree) |
| Ball query (1024 queries, K=64, r=0.1) | 20,000 | not yet | 29.8 ms | 10.1 ms | 5.0 ms (scipy cKDTree) |
| | 100,000 | not yet | 224.0 ms | 51.0 ms | 21.2 ms (scipy cKDTree) |

What this shows:

- **The Metal kernels beat the best CPU libraries**: FPS is 4.5x to 5.3x
  faster than fpsample, and kNN is 1.6x to 3.1x faster than scipy's KD-tree
  (including its build). Against the same algorithms in plain PyTorch on MPS,
  FPS is 6x to 17x faster and kNN 6x to 20x faster. Results are identical to
  the references.
- **Plain PyTorch FPS on MPS is overhead bound.** Going from 20k to 100k points
  (5x the work) only takes it from 134 ms to 193 ms. The loop runs 1024
  sequential steps of a few small kernels each, and the flat scaling says most
  of the time is dispatch and scheduling, not math. The Metal kernel runs all
  1024 steps in one dispatch.
- **Plain PyTorch kNN and ball query are slower on MPS than on CPU.** `cdist`
  builds the full 1024 x N distance matrix before `topk` picks from it. Ball
  query is 3x to 4.4x slower than the same code on CPU and is next.
- **Point order matters for kNN, and the kernel handles it.** Real scans store
  points in spatial order. With points sorted by x, the Metal kNN takes 2.4 to
  5.4 ms, the same as random order. Without the scrambled scan order described
  below it took 32 to 36 ms at 100k points.

Every implementation returns the same result as the PyTorch CPU reference
(FPS indices, ball query indices) or 100% recall against exact float64 search
(kNN).

Caveats: synthetic points near a unit sphere surface, batch 1, one machine.
Timings move by a few ms between runs. The scipy times include building the
KD-tree. scipy's ball query uses float64 and `<= r`, so a point exactly on the
boundary can differ from the PyTorch3D contract. fpsample's QuickFPS
(`bucket_fps_kdline_sampling`) is not in the table because in fpsample 1.0.2 it
ignores `start_idx` and returns a different, sorted sample set.

### Run it

```bash
uv venv --python 3.12 && uv pip install torch numpy scipy fpsample pytest
.venv/bin/python bench/bench_pointops.py                 # random point order
.venv/bin/python bench/bench_pointops.py --order sorted  # spatially sorted
.venv/bin/python -m pytest tests
```

Options: `--sizes`, `--ops fps knn ball_query`, `--order`, `--npoint`, `--k`,
`--ball-k`, `--radius`, `--repeat`. Results go to `bench/results/` as JSON and
Markdown. Results from other Apple Silicon chips are welcome as pull requests.

## How the kernels work

Both are in [mps_pointops/kernels/](mps_pointops/kernels/) and are compiled at
runtime with `torch.mps.compile_shader`. FMA contraction is turned off in both,
so squared distances round exactly like the PyTorch reference, which is why
the results match bit for bit.

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

## Contracts

The pure PyTorch versions in [mps_pointops/reference.py](mps_pointops/reference.py)
define the behavior the kernels must match. Tensors that are not on MPS fall
back to them.

- `furthest_point_sample(xyz, npoint, start_idx=0)`: starts at `start_idx`
  (0 by default, like `pointnet2_ops`), ties go to the smaller index, and once
  every point is taken the remaining slots repeat index 0. Float32 only on MPS.
- `knn(query, ref, k)`: Euclidean distances and indices, like
  `knn_cuda.KNN(k, transpose_mode=True)`, sorted by squared distance and then
  by index. `k <= N`, and `k <= 256` on MPS. Float32 only on MPS.
- `ball_query(query, ref, radius, K)`: PyTorch3D contract. The first `K`
  points in input order with squared distance `< radius**2`, padded with
  index -1 and distance 0.

## Roadmap

1. ~~Metal kernel for FPS~~
2. ~~Metal kernel for kNN~~
3. Metal kernel for ball query
4. Drop-in shims for `pointnet2_ops` and `knn_cuda`, with MulSen-AD as the
   end-to-end example

## License

Apache-2.0

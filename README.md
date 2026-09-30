# mps-pointops

Point cloud ops for PyTorch on Apple Silicon (MPS): farthest point sampling,
k nearest neighbors and ball query.

**Status:** farthest point sampling has a Metal kernel. kNN and ball query are
benchmarked but still pure PyTorch.

```python
import torch
from mps_pointops import furthest_point_sample

xyz = torch.randn(1, 100_000, 3, device="mps")
idx = furthest_point_sample(xyz, 1024)  # (1, 1024) int64, same as pointnet2_ops
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
Full table: [bench/results/2026-10-01-apple-m5-pro.md](bench/results/2026-10-01-apple-m5-pro.md).

Apple M5 Pro, 48 GB, macOS 26.5.2, torch 2.14.1:

| op | points | **mps-pointops (Metal)** | torch on MPS | torch on CPU | best CPU library |
|---|---:|---:|---:|---:|---:|
| FPS (1024 samples) | 20,000 | **7.7 ms** | 132.3 ms | 118.9 ms | 35.0 ms (fpsample) |
| | 100,000 | **31.8 ms** | 197.1 ms | 586.4 ms | 173.2 ms (fpsample) |
| kNN (1024 queries, k=128) | 20,000 | not yet | 13.0 ms | 11.0 ms | 5.1 ms (scipy cKDTree) |
| | 100,000 | not yet | 122.9 ms | 39.0 ms | 17.2 ms (scipy cKDTree) |
| Ball query (1024 queries, K=64, r=0.1) | 20,000 | not yet | 29.7 ms | 10.2 ms | 6.6 ms (scipy cKDTree) |
| | 100,000 | not yet | 216.4 ms | 52.5 ms | 23.6 ms (scipy cKDTree) |

What this shows:

- **The Metal FPS kernel is 4.5x to 6x faster than the best CPU library**
  (fpsample) and 6x to 17x faster than the same algorithm in plain PyTorch on
  MPS, with identical indices.
- **Plain PyTorch FPS on MPS is overhead bound.** Going from 20k to 100k points
  (5x the work) only takes it from 132 ms to 197 ms. The loop runs 1024
  sequential steps of a few small kernels each, and the flat scaling says most
  of the time is dispatch and scheduling, not math. The Metal kernel runs all
  1024 steps in one dispatch.
- **kNN and ball query in plain PyTorch are slower on MPS than on CPU**, by
  1.2x to 4x, and 2.5x to 9x slower than scipy's KD-tree. `cdist` builds the
  full 1024 x N distance matrix before `topk` picks from it. These are next.

Every implementation returns the same result as the PyTorch CPU reference
(FPS indices, ball query indices) or 100% recall against exact float64 search
(kNN).

Caveats: synthetic points near a unit sphere surface, batch 1, one machine.
Timings move by a few ms between runs. The scipy times include building the
KD-tree. fpsample's QuickFPS (`bucket_fps_kdline_sampling`) is not in the table
because in fpsample 1.0.2 it ignores `start_idx` and returns a different,
sorted sample set.

### Run it

```bash
uv venv --python 3.12 && uv pip install torch numpy scipy fpsample pytest
.venv/bin/python bench/bench_pointops.py
.venv/bin/python -m pytest tests
```

Options: `--sizes`, `--ops fps knn ball_query`, `--npoint`, `--k`,
`--ball-k`, `--radius`, `--repeat`. Results go to `bench/results/` as JSON and
Markdown. Results from other Apple Silicon chips are welcome as pull requests.

## How the FPS kernel works

[mps_pointops/kernels/fps.metal](mps_pointops/kernels/fps.metal), compiled at
runtime with `torch.mps.compile_shader`:

- One threadgroup of 1024 threads per point cloud runs all `npoint` steps, so
  sampling is a single dispatch instead of thousands of small ones.
- Each thread owns every 1024th point and keeps its running minimum squared
  distance to the sampled set.
- Each step finds the farthest point with a `simd_max` / `simd_min` reduction
  inside simdgroups, then across simdgroups through threadgroup memory. Ties go
  to the smaller index, like `torch.argmax`.
- FMA contraction is turned off so distances round exactly like the PyTorch
  reference, which is why the indices match bit for bit.

A batch of B clouds uses B threadgroups. With batch 1 only one GPU core is
busy, so there is room left for a multi-threadgroup version.

## Contracts

The pure PyTorch versions in [mps_pointops/reference.py](mps_pointops/reference.py)
define the behavior the kernels must match. Tensors that are not on MPS fall
back to them.

- `furthest_point_sample(xyz, npoint, start_idx=0)`: starts at `start_idx`
  (0 by default, like `pointnet2_ops`), ties go to the smaller index, and once
  every point is taken the remaining slots repeat index 0. Float32 only on MPS.
- `knn(query, ref, k)`: Euclidean distances and indices, ascending, like
  `knn_cuda.KNN(k, transpose_mode=True)`.
- `ball_query(query, ref, radius, K)`: PyTorch3D contract. The first `K`
  points in input order with squared distance `< radius**2`, padded with
  index -1 and distance 0.

## Roadmap

1. ~~Metal kernel for FPS~~
2. Metal kernel for kNN
3. Metal kernel for ball query
4. Drop-in shims for `pointnet2_ops` and `knn_cuda`, with MulSen-AD as the
   end-to-end example

## License

Apache-2.0

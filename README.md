# mps-pointops

Point cloud ops for PyTorch on Apple Silicon (MPS): farthest point sampling,
k nearest neighbors and ball query.

**Status:** benchmark only. The Metal kernels are not written yet. This repo
currently measures how slow the existing options are, so the kernels have a
concrete target.

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

| op | points | torch on MPS | torch on CPU | best CPU library |
|---|---:|---:|---:|---:|
| FPS (1024 samples) | 20,000 | 144.9 ms | 125.6 ms | 35.0 ms (fpsample) |
| | 100,000 | 190.6 ms | 635.0 ms | 175.8 ms (fpsample) |
| kNN (1024 queries, k=128) | 20,000 | 18.0 ms | 10.9 ms | 5.2 ms (scipy cKDTree) |
| | 100,000 | 118.3 ms | 42.9 ms | 19.2 ms (scipy cKDTree) |
| Ball query (1024 queries, K=64, r=0.1) | 20,000 | 42.2 ms | 10.1 ms | 5.1 ms (scipy cKDTree) |
| | 100,000 | 214.6 ms | 52.4 ms | 22.8 ms (scipy cKDTree) |

What this shows:

- **FPS on MPS is overhead bound.** Going from 20k to 100k points (5x the work)
  only takes it from 145 ms to 191 ms. The loop runs 1024 sequential steps of a
  few small kernels each, 0.13 to 0.19 ms per step, and the flat scaling says
  most of that is dispatch and scheduling overhead, not math. One fused Metal
  kernel should remove it.
- **kNN and ball query in plain PyTorch are slower on MPS than on CPU**, by
  1.7x to 4x, and 3.5x to 9x slower than scipy's KD-tree. `cdist` builds the
  full 1024 x N distance matrix before `topk` picks from it.
- **The bar to beat is the CPU library column**, not the MPS column.

All implementations return the same result as the PyTorch CPU reference
(FPS indices, ball query indices) or 100% recall against exact float64 search
(kNN).

Caveats: synthetic points near a unit sphere surface, batch 1, one machine.
The scipy times include building the KD-tree. fpsample's QuickFPS
(`bucket_fps_kdline_sampling`) is not in the table because in fpsample 1.0.2 it
ignores `start_idx` and returns a different, sorted sample set.

### Run it

```bash
uv venv --python 3.12 && uv pip install torch numpy scipy fpsample
.venv/bin/python bench/bench_pointops.py
```

Options: `--sizes`, `--ops fps knn ball_query`, `--npoint`, `--k`,
`--ball-k`, `--radius`, `--repeat`. Results go to `bench/results/` as JSON and
Markdown. Results from other Apple Silicon chips are welcome as pull requests.

## Contracts

The pure PyTorch versions in [mps_pointops/reference.py](mps_pointops/reference.py)
define the behavior the kernels must match:

- `furthest_point_sample(xyz, npoint)`: starts at index 0, like `pointnet2_ops`.
- `knn(query, ref, k)`: Euclidean distances and indices, ascending, like
  `knn_cuda.KNN(k, transpose_mode=True)`.
- `ball_query(query, ref, radius, K)`: PyTorch3D contract. The first `K`
  points in input order with squared distance `< radius**2`, padded with
  index -1 and distance 0.

## Roadmap

1. Metal kernel for FPS
2. Metal kernel for kNN
3. Metal kernel for ball query
4. Drop-in shims for `pointnet2_ops` and `knn_cuda`, with MulSen-AD as the
   end-to-end example

## License

Apache-2.0

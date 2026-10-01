# Public flat API timings on one Apple M5 Pro, 2026-10-02

This measures `mps_pointops.flat.fps`, `flat.knn`, and `flat.radius` through
their public Python calls. The [Safe raw JSON](../bench/results/2026-10-02-apple-m5-pro-flat-public-safe.json)
and [Fast raw JSON](../bench/results/2026-10-02-apple-m5-pro-flat-public-fast.json)
contain all 20 samples per cell, exact-output checks, the command inputs,
source file SHA-256 values, and the source commit
`584f4580b2ff899d2e73e4a2dcdaf0bf490912be` (clean tree during timing).

## Fixture and timing boundary

- Physical Apple M5 Pro, 48 GiB, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1.
  `PYTORCH_ENABLE_MPS_FALLBACK=0`; Safe and Fast Math ran in separate processes.
- Two uneven batches numbered 0 and 2, with an empty batch slot 1. Reference
  points are finite dyadic float32 3D coordinates in random input order. `N`
  is 1,024 or 4,096; query count is `N/32`. FPS uses `ratio=1/64` and
  `random_start=False`; kNN uses `k=16`; radius uses `r=2.0` and at most 16
  neighbors. Exact CPU/MPS output equality passed for every timed operation.
- Inputs are on the target device before timing. Each public call includes
  validation, allocation, kernel dispatch, and edge compaction. It excludes
  CPU↔MPS transfer, initial shader compilation, and output readback.
  `torch.mps.synchronize()` bounds every timed MPS call. Five warmups precede
  20 samples; the table shows median synchronized host wall milliseconds.
- CPU is this package's PyTorch implementation, not an optimized CPU library
  or the real `torch-cluster` extension. These synthetic inputs do not measure
  transfer-aware pipelines, random cloud sizes, other `k` values, or other GPUs.

| Mode | N / queries | Operation | CPU ms | MPS ms |
| --- | ---: | --- | ---: | ---: |
| Safe | 1,024 / 32 | FPS | 0.212 | 2.607 |
| Safe | 1,024 / 32 | kNN | 1.152 | 3.901 |
| Safe | 1,024 / 32 | Radius | 0.456 | 2.819 |
| Safe | 4,096 / 128 | FPS | 1.447 | 2.312 |
| Safe | 4,096 / 128 | kNN | 19.665 | 4.377 |
| Safe | 4,096 / 128 | Radius | 3.102 | 4.297 |
| Fast | 1,024 / 32 | FPS | 0.220 | 1.216 |
| Fast | 1,024 / 32 | kNN | 1.116 | 4.048 |
| Fast | 1,024 / 32 | Radius | 0.444 | 2.911 |
| Fast | 4,096 / 128 | FPS | 1.453 | 1.865 |
| Fast | 4,096 / 128 | kNN | 19.851 | 3.896 |
| Fast | 4,096 / 128 | Radius | 3.083 | 3.485 |

Only the 4,096-point kNN case beat this CPU reference in both modes. MPS
samples varied substantially within each run (for example, Safe 4,096-point
kNN ranged from 2.215 to 6.463 ms); use the linked samples rather than a
single median when evaluating a change. No general flat API speedup or
cross-device threshold follows from this fixture.

Reproduce from the recorded source commit on the same pinned environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_flat_public.py --warmups 5 --repeats 20 --output flat-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/bench_flat_public.py --warmups 5 --repeats 20 --output flat-fast.json
```

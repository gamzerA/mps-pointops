# Benchmark: Apple M5 Pro, 48 GB

- macOS 26.5.2, Python 3.12.13, torch 2.14.1, scipy 1.18.1, fpsample 1.0.2
- batch=1, FPS npoint=1024, kNN queries=1024 k=128, ball query queries=1024 K=64 radius=0.1, point order=sorted
- median of 5 runs after 2 warmup, PYTORCH_ENABLE_MPS_FALLBACK=unset
- speedup: torch (MPS) time / this time (>1 means faster than pure PyTorch on MPS)

## Farthest point sampling

| points | implementation | median (ms) | speedup | check |
|---:|---|---:|---:|---|
| 20,000 | torch (MPS) | 113.5 | 1.00x | idx mismatches 0/1024 |
| 20,000 | mps-pointops Metal (MPS) | 6.5 | 17.34x | idx mismatches 0/1024 |
| 20,000 | torch (CPU) | 95.4 | 1.19x | idx mismatches 0/1024 |
| 20,000 | fpsample vanilla (CPU) | 34.7 | 3.27x | idx mismatches 0/1024 |
| 50,000 | torch (MPS) | 167.6 | 1.00x | idx mismatches 0/1024 |
| 50,000 | mps-pointops Metal (MPS) | 16.2 | 10.35x | idx mismatches 0/1024 |
| 50,000 | torch (CPU) | 417.6 | 0.40x | idx mismatches 0/1024 |
| 50,000 | fpsample vanilla (CPU) | 86.5 | 1.94x | idx mismatches 0/1024 |
| 100,000 | torch (MPS) | 245.7 | 1.00x | idx mismatches 0/1024 |
| 100,000 | mps-pointops Metal (MPS) | 34.4 | 7.14x | idx mismatches 0/1024 |
| 100,000 | torch (CPU) | 574.0 | 0.43x | idx mismatches 0/1024 |
| 100,000 | fpsample vanilla (CPU) | 172.8 | 1.42x | idx mismatches 0/1024 |

## k nearest neighbors

| points | implementation | median (ms) | speedup | check |
|---:|---|---:|---:|---|
| 20,000 | torch cdist+topk (MPS) | 13.7 | 1.00x | missing 0/131072, position mismatches 70, max dist err 4.9e-04 |
| 20,000 | mps-pointops Metal (MPS) | 2.6 | 5.30x | missing 0/131072, position mismatches 0, max dist err 0.0e+00 |
| 20,000 | torch cdist+topk (CPU) | 16.6 | 0.83x | missing 0/131072, position mismatches 64, max dist err 4.9e-04 |
| 20,000 | scipy cKDTree build+query (CPU) | 5.1 | 2.69x | missing 0/131072, position mismatches 0, max dist err 2.1e-08 |
| 50,000 | torch cdist+topk (MPS) | 52.6 | 1.00x | missing 0/131072, position mismatches 150, max dist err 4.9e-04 |
| 50,000 | mps-pointops Metal (MPS) | 3.9 | 13.40x | missing 0/131072, position mismatches 0, max dist err 0.0e+00 |
| 50,000 | torch cdist+topk (CPU) | 35.6 | 1.48x | missing 0/131072, position mismatches 142, max dist err 4.9e-04 |
| 50,000 | scipy cKDTree build+query (CPU) | 9.4 | 5.62x | missing 0/131072, position mismatches 0, max dist err 1.2e-08 |
| 100,000 | torch cdist+topk (MPS) | 116.4 | 1.00x | missing 3/131072, position mismatches 303, max dist err 4.9e-04 |
| 100,000 | mps-pointops Metal (MPS) | 5.4 | 21.52x | missing 0/131072, position mismatches 0, max dist err 0.0e+00 |
| 100,000 | torch cdist+topk (CPU) | 60.5 | 1.92x | missing 4/131072, position mismatches 313, max dist err 4.9e-04 |
| 100,000 | scipy cKDTree build+query (CPU) | 17.9 | 6.51x | missing 0/131072, position mismatches 2, max dist err 1.1e-08 |

## Ball query

| points | implementation | median (ms) | speedup | check |
|---:|---|---:|---:|---|
| 20,000 | torch mask+topk (MPS) | 47.8 | 1.00x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 20,000 | torch mask+topk (CPU) | 31.2 | 1.53x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 20,000 | scipy cKDTree build+query (CPU) | 5.3 | 9.10x | idx mismatches 0/65536 |
| 50,000 | torch mask+topk (MPS) | 146.7 | 1.00x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 50,000 | torch mask+topk (CPU) | 79.7 | 1.84x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 50,000 | scipy cKDTree build+query (CPU) | 12.4 | 11.80x | idx mismatches 0/65536 |
| 100,000 | torch mask+topk (MPS) | 292.5 | 1.00x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 100,000 | torch mask+topk (CPU) | 163.6 | 1.79x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 100,000 | scipy cKDTree build+query (CPU) | 22.1 | 13.26x | idx mismatches 0/65536 |


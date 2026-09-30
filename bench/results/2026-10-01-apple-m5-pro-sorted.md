# Benchmark: Apple M5 Pro, 48 GB

- macOS 26.5.2, Python 3.12.13, torch 2.14.1, scipy 1.18.1, fpsample 1.0.2
- batch=1, FPS npoint=1024, kNN queries=1024 k=128, ball query queries=1024 K=64 radius=0.1, point order=sorted
- median of 5 runs after 2 warmup, PYTORCH_ENABLE_MPS_FALLBACK=unset
- speedup: torch (MPS) time / this time (>1 means faster than pure PyTorch on MPS)

## Farthest point sampling

| points | implementation | median (ms) | speedup | check |
|---:|---|---:|---:|---|
| 20,000 | torch (MPS) | 135.5 | 1.00x | idx match 100.0% |
| 20,000 | mps-pointops Metal (MPS) | 7.7 | 17.69x | idx match 100.0% |
| 20,000 | torch (CPU) | 119.0 | 1.14x | idx match 100.0% |
| 20,000 | fpsample vanilla (CPU) | 34.6 | 3.92x | idx match 100.0% |
| 50,000 | torch (MPS) | 133.0 | 1.00x | idx match 100.0% |
| 50,000 | mps-pointops Metal (MPS) | 16.5 | 8.06x | idx match 100.0% |
| 50,000 | torch (CPU) | 209.1 | 0.64x | idx match 100.0% |
| 50,000 | fpsample vanilla (CPU) | 86.5 | 1.54x | idx match 100.0% |
| 100,000 | torch (MPS) | 192.5 | 1.00x | idx match 100.0% |
| 100,000 | mps-pointops Metal (MPS) | 34.8 | 5.53x | idx match 100.0% |
| 100,000 | torch (CPU) | 631.8 | 0.30x | idx match 100.0% |
| 100,000 | fpsample vanilla (CPU) | 175.3 | 1.10x | idx match 100.0% |

## k nearest neighbors

| points | implementation | median (ms) | speedup | check |
|---:|---|---:|---:|---|
| 20,000 | torch cdist+topk (MPS) | 13.4 | 1.00x | recall 100.00% |
| 20,000 | mps-pointops Metal (MPS) | 2.4 | 5.49x | recall 100.00% |
| 20,000 | torch cdist+topk (CPU) | 16.2 | 0.83x | recall 100.00% |
| 20,000 | scipy cKDTree build+query (CPU) | 5.1 | 2.63x | recall 100.00% |
| 50,000 | torch cdist+topk (MPS) | 51.8 | 1.00x | recall 100.00% |
| 50,000 | mps-pointops Metal (MPS) | 4.3 | 12.03x | recall 100.00% |
| 50,000 | torch cdist+topk (CPU) | 33.3 | 1.56x | recall 100.00% |
| 50,000 | scipy cKDTree build+query (CPU) | 9.5 | 5.47x | recall 100.00% |
| 100,000 | torch cdist+topk (MPS) | 117.3 | 1.00x | recall 100.00% |
| 100,000 | mps-pointops Metal (MPS) | 5.4 | 21.82x | recall 100.00% |
| 100,000 | torch cdist+topk (CPU) | 64.4 | 1.82x | recall 100.00% |
| 100,000 | scipy cKDTree build+query (CPU) | 18.9 | 6.20x | recall 100.00% |

## Ball query

| points | implementation | median (ms) | speedup | check |
|---:|---|---:|---:|---|
| 20,000 | torch cdist+mask+topk (MPS) | 34.4 | 1.00x | idx match 100.00% |
| 20,000 | torch cdist+mask+topk (CPU) | 10.2 | 3.39x | idx match 100.00% |
| 20,000 | scipy cKDTree build+query (CPU) | 5.1 | 6.75x | idx match 100.00% |
| 50,000 | torch cdist+mask+topk (MPS) | 96.8 | 1.00x | idx match 100.00% |
| 50,000 | torch cdist+mask+topk (CPU) | 24.9 | 3.88x | idx match 100.00% |
| 50,000 | scipy cKDTree build+query (CPU) | 11.2 | 8.62x | idx match 99.96% |
| 100,000 | torch cdist+mask+topk (MPS) | 230.6 | 1.00x | idx match 100.00% |
| 100,000 | torch cdist+mask+topk (CPU) | 50.1 | 4.61x | idx match 100.00% |
| 100,000 | scipy cKDTree build+query (CPU) | 22.3 | 10.34x | idx match 100.00% |


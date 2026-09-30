# Benchmark: Apple M5 Pro, 48 GB

- macOS 26.5.2, Python 3.12.13, torch 2.14.1, scipy 1.18.1, fpsample 1.0.2
- batch=1, FPS npoint=1024, kNN queries=1024 k=128, ball query queries=1024 K=64 radius=0.1, point order=sorted
- median of 5 runs after 2 warmup, PYTORCH_ENABLE_MPS_FALLBACK=0
- speedup: torch (MPS) time / this time (>1 means faster than pure PyTorch on MPS)

## Ball query

| points | implementation | median (ms) | speedup | check |
|---:|---|---:|---:|---|
| 20,000 | torch mask+topk (MPS) | 67.3 | 1.00x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 20,000 | mps-pointops Metal (MPS) | 6.0 | 11.21x | idx mismatches 0/65536, max dist^2 err 1.9e-09 |
| 20,000 | torch mask+topk (CPU) | 34.6 | 1.94x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 20,000 | scipy cKDTree build+query (CPU) | 4.9 | 13.81x | idx mismatches 0/65536 |
| 100,000 | torch mask+topk (MPS) | 277.6 | 1.00x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 100,000 | mps-pointops Metal (MPS) | 32.5 | 8.54x | idx mismatches 0/65536, max dist^2 err 1.9e-09 |
| 100,000 | torch mask+topk (CPU) | 165.6 | 1.68x | idx mismatches 0/65536, max dist^2 err 0.0e+00 |
| 100,000 | scipy cKDTree build+query (CPU) | 23.2 | 11.96x | idx mismatches 0/65536 |

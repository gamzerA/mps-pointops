# Experimental multi-threadgroup FPS on M5 Pro

This is an ablation, not a public FPS implementation. The experimental Metal
source is [`../fps_multigroup.metal`](../fps_multigroup.metal), invoked by
[`../bench_fps_multigroup.py`](../bench_fps_multigroup.py). Raw timings and
environment metadata are in
[`2026-10-01-apple-m5-pro-fps-multigroup.json`](2026-10-01-apple-m5-pro-fps-multigroup.json).

## Method

- Apple M5 Pro, 48 GiB, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1; `PYTORCH_MPS_FAST_MATH` unset.
- One float32 synthetic noisy unit-sphere cloud per case, seed 42, batch size 1. Random and x-sorted storage order use the same points. Start index 0.
- Existing package FPS: one 1024-thread threadgroup, one dispatch for the whole sampling sequence.
- Experimental FPS: 256 threads per group, contiguous 4096-point chunks. Each sampling step launches one parallel distance-update/partial-max kernel and one global partial-max kernel. It uses `2 × (npoint − 1)` dispatches, with one initial output fill.
- Three warm-up calls and five timed calls for each path. The single/multi execution order alternates each repetition to reduce order bias. Timings are end-to-end wall-clock milliseconds, including Python dispatch overhead; `torch.mps.synchronize()` brackets each call. Medians below are computed from the five raw times.
- Both paths returned identical indices in all 40 size/order/sample cases (25,600 compared output slots), and in separate duplicate-point and all-skipped tie/fallback checks. This validates these finite inputs, not the full production contract or all hardware.

## Results

| Point order | Points | Samples | Existing FPS (ms) | Multi-group (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 2,048 | 256 | 0.39 | 1.92 | 0.20× |
| Random | 2,048 | 1,024 | 1.02 | 7.66 | 0.13× |
| Random | 4,096 | 256 | 0.50 | 2.32 | 0.22× |
| Random | 4,096 | 1,024 | 1.44 | 9.16 | 0.16× |
| Random | 8,192 | 256 | 0.76 | 2.32 | 0.33× |
| Random | 8,192 | 1,024 | 2.24 | 9.00 | 0.25× |
| Random | 16,384 | 256 | 1.30 | 2.33 | 0.56× |
| Random | 16,384 | 1,024 | 4.71 | 9.16 | 0.51× |
| Random | 32,768 | 256 | 2.34 | 2.32 | 1.01× |
| Random | 32,768 | 1,024 | 8.78 | 9.37 | 0.94× |
| Random | 65,536 | 256 | 4.31 | 2.32 | 1.86× |
| Random | 65,536 | 1,024 | 16.85 | 9.28 | 1.81× |
| Random | 100,000 | 256 | 6.63 | 2.74 | 2.42× |
| Random | 100,000 | 1,024 | 25.29 | 10.82 | 2.34× |
| Random | 250,000 | 256 | 22.69 | 3.56 | 6.37× |
| Random | 250,000 | 1,024 | 90.00 | 13.49 | 6.67× |
| Random | 500,000 | 256 | 49.15 | 7.33 | 6.70× |
| Random | 500,000 | 1,024 | 194.15 | 29.49 | 6.58× |
| Random | 1,000,000 | 256 | 106.74 | 12.39 | 8.61× |
| Random | 1,000,000 | 1,024 | 433.28 | 49.81 | 8.70× |
| X-sorted | 32,768 | 256 | 2.32 | 2.33 | 1.00× |
| X-sorted | 32,768 | 1,024 | 8.87 | 9.80 | 0.90× |
| X-sorted | 65,536 | 256 | 4.36 | 2.55 | 1.71× |
| X-sorted | 65,536 | 1,024 | 16.74 | 9.21 | 1.82× |
| X-sorted | 500,000 | 256 | 49.20 | 7.88 | 6.25× |
| X-sorted | 500,000 | 1,024 | 193.93 | 29.92 | 6.48× |
| X-sorted | 1,000,000 | 256 | 106.59 | 12.20 | 8.73× |
| X-sorted | 1,000,000 | 1,024 | 431.48 | 49.35 | 8.74× |

The table selects the x-sorted crossover and large-cloud rows for compactness;
the linked JSON contains all 40 cases and each of the five raw times.

## Interpretation and next steps

At 32,768 points the paths are near parity; at 65,536 points the multi-group
path is faster for both sample counts and both storage orders. This gives a
measured crossover interval on this M5 Pro, not a universal threshold. The
large-cloud speedup appears despite paying for two dispatches per selected
point. These measurements do not show GPU-core occupancy directly. A
production path would also need variable batch sizes, full numeric and edge
case tests, tuning on other Apple GPUs and macOS versions, and benchmark runs
against any new single-group optimizations before choosing a dispatcher.

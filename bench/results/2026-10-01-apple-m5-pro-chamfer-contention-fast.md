# MPS Chamfer scatter contention benchmark

- UTC: 2026-10-01T09:48:51.578319+00:00
- Device: Apple M5 Pro; macOS 26.5.2; PyTorch 2.14.1
- Math mode: `1`; CPU fallback: `0`
- Reproduction: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 python bench/bench_chamfer_contention.py --batch 4 --sizes 256 1024 2048 4096 8192 16384 --warmup 4 --repeat 20 --output bench/results/2026-10-01-apple-m5-pro-chamfer-contention-fast.json`
- 4 warmups and 20 measured runs per case; synchronized wall time, Python launch included.

## Results

All timings are medians in milliseconds. F/B is backward divided by forward. Concentrated / uniform is the median of paired per-iteration ratios.

| B | N | Pattern | Forward | Backward | F/B | Direct scatter |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 256 | uniform | 1.059 | 0.451 | 0.43× | 0.197 |
| 4 | 256 | concentrated | 1.090 | 0.463 | 0.42× | 0.208 |
| | | concentrated / uniform | | 1.04× | | 1.02× |
| 4 | 1024 | uniform | 1.322 | 0.462 | 0.35× | 0.215 |
| 4 | 1024 | concentrated | 1.289 | 0.466 | 0.36× | 0.209 |
| | | concentrated / uniform | | 1.03× | | 0.98× |
| 4 | 2048 | uniform | 1.948 | 0.558 | 0.29× | 0.241 |
| 4 | 2048 | concentrated | 1.909 | 0.576 | 0.30× | 0.248 |
| | | concentrated / uniform | | 1.03× | | 0.99× |
| 4 | 4096 | uniform | 1.820 | 0.386 | 0.21× | 0.188 |
| 4 | 4096 | concentrated | 1.806 | 0.390 | 0.22× | 0.182 |
| | | concentrated / uniform | | 1.01× | | 0.97× |
| 4 | 8192 | uniform | 3.265 | 0.436 | 0.13× | 0.189 |
| 4 | 8192 | concentrated | 3.264 | 0.425 | 0.13× | 0.189 |
| | | concentrated / uniform | | 0.98× | | 0.99× |
| 4 | 16384 | uniform | 10.631 | 0.697 | 0.07× | 0.241 |
| 4 | 16384 | concentrated | 10.644 | 0.709 | 0.07× | 0.246 |
| | | concentrated / uniform | | 1.02× | | 1.01× |

## Method and limits

- Same batch, query/reference shapes, float32 dtype and valid point counts in both patterns. Uniform selects each reference once per batch; concentrated selects reference 0 for every query. Expected indices and analytic mean-reduction gradients are checked before timing.
- The timed Chamfer path is `chamfer_distance(x, y, single_directional=True)` with default point and batch means. Its forward includes public input validation and Metal nearest search; its backward includes autograd, elementwise gradient work, allocation and PyTorch `scatter_add_`. The direct control times only `scatter_add_` into a preallocated destination, reset outside the timer.
- Forward and backward are synchronized separately before/after timing. Case order alternates. Compilation, CPU tensor creation and correctness checks are outside timed runs.
- Timing ratios alone cannot establish atomic contention as the cause, nor prove that a custom Metal reduction kernel will be faster. End-to-end autograd overhead, allocator behavior and thermal state also affect these wall times. This run has no padding, weights or reverse Chamfer term.

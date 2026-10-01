# MPS Chamfer scatter contention benchmark

- UTC: 2026-10-01T09:54:32.924478+00:00
- Device: Apple M5 Pro; macOS 26.5.2; PyTorch 2.14.1
- Math mode: `0`; CPU fallback: `0`
- Base Chamfer code commit (last edit to module or Metal kernel): `21073b67e1b3d6c83537c2b64f9599bdc25c19e8`
- Reproduction: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python bench/bench_chamfer_contention.py --batch 4 --sizes 256 1024 2048 4096 8192 16384 --warmup 4 --repeat 20 --output bench/results/2026-10-01-apple-m5-pro-chamfer-contention-safe.json`
- 4 warmups and 20 measured runs per case; synchronized wall time, Python launch included.

## Results

All timings are medians in milliseconds. B/F is backward divided by forward. Concentrated / uniform is the median of paired per-iteration ratios.

| B | N | Pattern | Forward | Backward | B/F | Direct scatter |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 256 | uniform | 1.073 | 0.483 | 0.45× | 0.201 |
| 4 | 256 | concentrated | 1.111 | 0.537 | 0.48× | 0.196 |
| | | concentrated / uniform | | 1.08× | | 0.97× |
| 4 | 1024 | uniform | 1.609 | 0.640 | 0.40× | 0.212 |
| 4 | 1024 | concentrated | 1.314 | 0.499 | 0.38× | 0.234 |
| | | concentrated / uniform | | 0.79× | | 1.02× |
| 4 | 2048 | uniform | 2.047 | 0.644 | 0.31× | 0.268 |
| 4 | 2048 | concentrated | 2.052 | 0.661 | 0.32× | 0.257 |
| | | concentrated / uniform | | 1.08× | | 0.95× |
| 4 | 4096 | uniform | 1.597 | 0.408 | 0.26× | 0.182 |
| 4 | 4096 | concentrated | 1.709 | 0.442 | 0.26× | 0.177 |
| | | concentrated / uniform | | 1.00× | | 1.03× |
| 4 | 8192 | uniform | 3.535 | 0.561 | 0.16× | 0.199 |
| 4 | 8192 | concentrated | 3.473 | 0.581 | 0.17× | 0.192 |
| | | concentrated / uniform | | 0.99× | | 1.01× |
| 4 | 16384 | uniform | 11.008 | 0.793 | 0.07× | 0.252 |
| 4 | 16384 | concentrated | 10.999 | 0.818 | 0.07× | 0.266 |
| | | concentrated / uniform | | 1.00× | | 1.02× |

## Method and limits

- Same batch, query/reference shapes, float32 dtype and valid point counts in both patterns. Uniform selects each reference once per batch; concentrated selects reference 0 for every query. Expected indices and analytic mean-reduction gradients are checked before timing.
- The timed Chamfer path is `chamfer_distance(x, y, single_directional=True)` with default point and batch means. Its forward includes public input validation and Metal nearest search; its backward includes autograd, elementwise gradient work, allocation and PyTorch `scatter_add_`. The direct control times only `scatter_add_` into a preallocated destination, reset outside the timer.
- Forward and backward are synchronized separately before/after timing. Case order alternates. Compilation, CPU tensor creation and correctness checks are outside timed runs.
- Timing ratios alone cannot establish atomic contention as the cause, nor prove that a custom Metal reduction kernel will be faster. End-to-end autograd overhead, allocator behavior and thermal state also affect these wall times. This run has no padding, weights or reverse Chamfer term.

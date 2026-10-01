# MPS Chamfer scatter contention benchmark

- UTC: 2026-10-01T09:54:48.575766+00:00
- Device: Apple M5 Pro; macOS 26.5.2; PyTorch 2.14.1
- Math mode: `1`; CPU fallback: `0`
- Base Chamfer code commit (last edit to module or Metal kernel): `21073b67e1b3d6c83537c2b64f9599bdc25c19e8`
- Reproduction: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 python bench/bench_chamfer_contention.py --batch 4 --sizes 256 1024 2048 4096 8192 16384 --warmup 4 --repeat 20 --output bench/results/2026-10-01-apple-m5-pro-chamfer-contention-fast.json`
- 4 warmups and 20 measured runs per case; synchronized wall time, Python launch included.

## Results

All timings are medians in milliseconds. B/F is backward divided by forward. Concentrated / uniform is the median of paired per-iteration ratios.

| B | N | Pattern | Forward | Backward | B/F | Direct scatter |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 256 | uniform | 1.087 | 0.462 | 0.42× | 0.199 |
| 4 | 256 | concentrated | 1.048 | 0.454 | 0.43× | 0.218 |
| | | concentrated / uniform | | 1.00× | | 1.09× |
| 4 | 1024 | uniform | 1.569 | 0.570 | 0.36× | 0.208 |
| 4 | 1024 | concentrated | 1.337 | 0.527 | 0.39× | 0.204 |
| | | concentrated / uniform | | 0.95× | | 1.03× |
| 4 | 2048 | uniform | 2.034 | 0.629 | 0.31× | 0.252 |
| 4 | 2048 | concentrated | 1.986 | 0.618 | 0.31× | 0.227 |
| | | concentrated / uniform | | 1.02× | | 0.94× |
| 4 | 4096 | uniform | 1.619 | 0.374 | 0.23× | 0.173 |
| 4 | 4096 | concentrated | 1.579 | 0.375 | 0.24× | 0.172 |
| | | concentrated / uniform | | 0.94× | | 1.01× |
| 4 | 8192 | uniform | 3.316 | 0.478 | 0.14× | 0.194 |
| 4 | 8192 | concentrated | 3.300 | 0.477 | 0.14× | 0.198 |
| | | concentrated / uniform | | 1.08× | | 1.04× |
| 4 | 16384 | uniform | 10.837 | 0.691 | 0.06× | 0.241 |
| 4 | 16384 | concentrated | 10.928 | 0.712 | 0.07× | 0.246 |
| | | concentrated / uniform | | 1.04× | | 1.03× |

## Method and limits

- Same batch, query/reference shapes, float32 dtype and valid point counts in both patterns. Uniform selects each reference once per batch; concentrated selects reference 0 for every query. Expected indices and analytic mean-reduction gradients are checked before timing.
- The timed Chamfer path is `chamfer_distance(x, y, single_directional=True)` with default point and batch means. Its forward includes public input validation and Metal nearest search; its backward includes autograd, elementwise gradient work, allocation and PyTorch `scatter_add_`. The direct control times only `scatter_add_` into a preallocated destination, reset outside the timer.
- Forward and backward are synchronized separately before/after timing. Case order alternates. Compilation, CPU tensor creation and correctness checks are outside timed runs.
- Timing ratios alone cannot establish atomic contention as the cause, nor prove that a custom Metal reduction kernel will be faster. End-to-end autograd overhead, allocator behavior and thermal state also affect these wall times. This run has no padding, weights or reverse Chamfer term.

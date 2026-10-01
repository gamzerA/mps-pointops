# MPS Chamfer scatter contention benchmark

- UTC: 2026-10-01T09:48:41.578890+00:00
- Device: Apple M5 Pro; macOS 26.5.2; PyTorch 2.14.1
- Math mode: `0`; CPU fallback: `0`
- Reproduction: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python bench/bench_chamfer_contention.py --batch 4 --sizes 256 1024 2048 4096 8192 16384 --warmup 4 --repeat 20 --output bench/results/2026-10-01-apple-m5-pro-chamfer-contention-safe.json`
- 4 warmups and 20 measured runs per case; synchronized wall time, Python launch included.

## Results

All timings are medians in milliseconds. F/B is backward divided by forward. Concentrated / uniform is the median of paired per-iteration ratios.

| B | N | Pattern | Forward | Backward | F/B | Direct scatter |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 256 | uniform | 1.112 | 0.498 | 0.45× | 0.207 |
| 4 | 256 | concentrated | 1.131 | 0.553 | 0.49× | 0.216 |
| | | concentrated / uniform | | 1.11× | | 1.01× |
| 4 | 1024 | uniform | 1.282 | 0.484 | 0.38× | 0.217 |
| 4 | 1024 | concentrated | 1.269 | 0.470 | 0.37× | 0.217 |
| | | concentrated / uniform | | 0.99× | | 0.99× |
| 4 | 2048 | uniform | 1.933 | 0.574 | 0.30× | 0.251 |
| 4 | 2048 | concentrated | 1.950 | 0.564 | 0.29× | 0.244 |
| | | concentrated / uniform | | 0.97× | | 1.00× |
| 4 | 4096 | uniform | 1.720 | 0.373 | 0.22× | 0.176 |
| 4 | 4096 | concentrated | 1.754 | 0.384 | 0.22× | 0.181 |
| | | concentrated / uniform | | 1.03× | | 1.00× |
| 4 | 8192 | uniform | 3.281 | 0.457 | 0.14× | 0.192 |
| 4 | 8192 | concentrated | 3.290 | 0.447 | 0.14× | 0.193 |
| | | concentrated / uniform | | 0.98× | | 1.01× |
| 4 | 16384 | uniform | 10.963 | 0.764 | 0.07× | 0.254 |
| 4 | 16384 | concentrated | 11.083 | 0.767 | 0.07× | 0.252 |
| | | concentrated / uniform | | 1.01× | | 0.95× |

## Method and limits

- Same batch, query/reference shapes, float32 dtype and valid point counts in both patterns. Uniform selects each reference once per batch; concentrated selects reference 0 for every query. Expected indices and analytic mean-reduction gradients are checked before timing.
- The timed Chamfer path is `chamfer_distance(x, y, single_directional=True)` with default point and batch means. Its forward includes public input validation and Metal nearest search; its backward includes autograd, elementwise gradient work, allocation and PyTorch `scatter_add_`. The direct control times only `scatter_add_` into a preallocated destination, reset outside the timer.
- Forward and backward are synchronized separately before/after timing. Case order alternates. Compilation, CPU tensor creation and correctness checks are outside timed runs.
- Timing ratios alone cannot establish atomic contention as the cause, nor prove that a custom Metal reduction kernel will be faster. End-to-end autograd overhead, allocator behavior and thermal state also affect these wall times. This run has no padding, weights or reverse Chamfer term.

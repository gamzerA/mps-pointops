# Large bidirectional Chamfer fan-in on one M5 Pro, 2026-10-02

This is a bounded extension of the earlier [single-directional contention
probe](../bench/bench_chamfer_contention.py). It measures the public
`chamfer_distance(x, y)` bidirectional squared-L2 loss and two native PyTorch
`scatter_add_` calls at **32,768 and 65,536 points per cloud**. The raw
[Safe](../bench/results/2026-10-02-apple-m5-pro-chamfer-contention-large-safe.json)
and [Fast](../bench/results/2026-10-02-apple-m5-pro-chamfer-contention-large-fast.json)
JSON contain every sample, input tensor SHA-256, source file SHA-256, exact
index and analytic gradient checks, memory guard settings, and MPS driver
allocation. Both runs used clean source commit
`e1303c8b34cc6d306bb4c1d4cae4d431c6b01bc9`.

## Method

- Physical Apple M5 Pro, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1.
  `PYTORCH_ENABLE_MPS_FALLBACK=0`; Safe and Fast Math ran in separate
  processes. Inputs were resident float32 MPS tensors before timing.
- `B=1`; `N=32,768` or `65,536` for both `x` and `y`. `uniform` has one
  matching destination per point in input order. `random` permutes that
  one-to-one mapping. `concentrated` directs every point to destination zero
  in **both** nearest-search directions. All patterns have the same shapes,
  dtypes, and `2BN²` directional distance evaluations. The coordinates are
  synthetic and chosen to make expected nearest indices unambiguous.
- The benchmark checks both nearest-index tensors exactly against known maps,
  then checks the bidirectional mean loss and both coordinate gradients
  against an independent double-precision fixed-neighbor formula. Maximum
  absolute gradient errors in the concentrated 65,536-point case were
  `1.24e-4` (Safe) and `1.92e-4` (Fast); all checks passed `rtol=5e-4,
  atol=2e-6`. The one-to-one cases had zero reported gradient error.
- Forward, backward, and a fresh full loss plus backward are timed separately
  with `torch.mps.synchronize()` at each boundary. Thus full E2E is measured
  directly, **not** formed by adding phase medians. The native control times
  two `scatter_add_` calls into preallocated buffers with unit sources;
  resetting those buffers is outside its timer. That control does not include
  Chamfer's gradient arithmetic or allocator work. Pattern order rotates by
  iteration; three warmups precede twelve raw samples per case.
- A preflight guard rejects a case if its conservative three-pattern buffer
  estimate exceeds 256 MiB or one bidirectional search exceeds ten billion
  distance pairs. A second guard checks process MPS driver allocation against
  4 GiB after each size. At 65,536 points the estimate was 96 MiB, the pair
  count was 8.59 billion, and observed driver allocation was 32.69 MiB.

All values below are median synchronized host wall milliseconds.

| Math | N | Pattern | Forward | Backward | Full E2E | Two scatters |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Safe | 32,768 | Uniform | 27.374 | 2.093 | 28.785 | 0.261 |
| Safe | 32,768 | Random one-to-one | 27.522 | 1.179 | 30.426 | 0.244 |
| Safe | 32,768 | Concentrated | 26.603 | 1.374 | 27.876 | 0.342 |
| Safe | 65,536 | Uniform | 96.386 | 1.023 | 95.777 | 0.297 |
| Safe | 65,536 | Random one-to-one | 95.910 | 1.294 | 97.085 | 0.347 |
| Safe | 65,536 | Concentrated | 94.324 | 3.172 | 98.042 | 0.489 |
| Fast | 32,768 | Uniform | 28.382 | 0.907 | 27.516 | 0.235 |
| Fast | 32,768 | Random one-to-one | 27.358 | 2.162 | 26.733 | 0.244 |
| Fast | 32,768 | Concentrated | 25.145 | 1.188 | 27.913 | 0.300 |
| Fast | 65,536 | Uniform | 93.163 | 1.725 | 95.383 | 0.292 |
| Fast | 65,536 | Random one-to-one | 95.156 | 1.899 | 96.573 | 0.304 |
| Fast | 65,536 | Concentrated | 99.146 | 1.777 | 99.427 | 0.499 |

At 65,536 points the **paired per-iteration** concentrated/uniform native
scatter ratio was 1.57× Safe and 1.73× Fast. The corresponding complete-loss
ratios were 1.00× and 1.03×. Safe backward's paired ratio was 2.87×, while
Fast backward's was 1.01×; the 32,768-point backward ratios reversed that
pattern. Backward samples were variable: at 65,536, Safe concentrated
backward ranged from 1.149 to 4.969 ms. The raw records should be used for
any finer comparison.

## Decision and limits

**Keep native PyTorch scatter in the public M5 Pro Chamfer path for these
fixtures.** The measured scatter-pair control is under 0.5 ms at the median,
while full bidirectional loss plus backward is roughly 28–99 ms. The tested
whole-loss timing does not show a stable gain that would justify adopting a
dedicated reduction kernel. A segmented candidate could still be worthwhile
for a different fan-in distribution or device, but it needs a direct
same-input ablation including grouping, buffer creation, gradients, and full
loss time before replacing the default.

These host wall timings do not identify a Metal atomic primitive or prove
atomic contention. The control uses unit contributions, while Chamfer's
backward uses scaled coordinate differences. Power/thermal state and GPU-side
traces were not controlled. The earlier [physical M1 report](phase3-physical-m1-2026-10-02.md)
found a much larger penalty for a different, smaller, single-directional
fixture; **no new M1 run was possible for this study**. This M5 decision must
not be extended to M1 or to real point-cloud training workloads.

Reproduce from the recorded source commit, with the same environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_chamfer_contention_large.py --output large-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/bench_chamfer_contention_large.py --output large-fast.json
```

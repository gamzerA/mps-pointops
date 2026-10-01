# Large bidirectional Chamfer fan-in on M5 Pro and M1, 2026-10-02

This is a bounded extension of the earlier [single-directional contention
probe](../bench/bench_chamfer_contention.py). It measures the public
`chamfer_distance(x, y)` bidirectional squared-L2 loss and two native PyTorch
`scatter_add_` calls at **32,768 and 65,536 points per cloud**. The raw M5 Pro
[Safe](../bench/results/2026-10-02-apple-m5-pro-chamfer-contention-large-safe.json)
and [Fast](../bench/results/2026-10-02-apple-m5-pro-chamfer-contention-large-fast.json)
and physical M1
[Safe](../bench/results/2026-10-02-apple-m1-chamfer-contention-large-safe.json)
and [Fast](../bench/results/2026-10-02-apple-m1-chamfer-contention-large-fast.json)
JSON contain every sample, input tensor SHA-256, source file SHA-256, exact
index and analytic gradient checks, memory guard settings, and MPS driver
allocation. The corresponding M1
[Safe](bench-chamfer-contention-large-apple-m1-safe-2026-10-02.log) and
[Fast](bench-chamfer-contention-large-apple-m1-fast-2026-10-02.log) raw
process logs were preserved; both processes exited with status 0. M5 ran from
clean commit
`e1303c8b34cc6d306bb4c1d4cae4d431c6b01bc9`; M1 ran from clean commit
`221705f545f2ce20895d4f779e5dd995ced96b2f`. The three measured source
files have identical SHA-256 on both commits.

## Method

- Physical Apple M5 Pro, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1;
  physical Apple M1 with 8 GiB unified memory, macOS 26.5.2, Python 3.10.6,
  PyTorch 2.12.0. Each device used `PYTORCH_ENABLE_MPS_FALLBACK=0`, and Safe
  and Fast Math ran in separate processes. Inputs were resident float32 MPS
  tensors before timing. PyTorch version and hardware both differ, so the
  cross-device wall-time difference cannot be assigned solely to the GPU.
- `B=1`; `N=32,768` or `65,536` for both `x` and `y`. `uniform` has one
  matching destination per point in input order. `random` permutes that
  one-to-one mapping. `concentrated` directs every point to destination zero
  in **both** nearest-search directions. All patterns have the same shapes,
  dtypes, and `2BN²` directional distance evaluations. The coordinates are
  synthetic and chosen to make expected nearest indices unambiguous. Every
  recorded input tensor SHA-256, including the random permutation and expected
  index maps, matches between M1 and M5 Pro for the same case.
- The benchmark checks both nearest-index tensors exactly against known maps,
  then checks the bidirectional mean loss and both coordinate gradients
  against an independent double-precision fixed-neighbor formula. Maximum
  absolute gradient errors in the concentrated 65,536-point case were
  `1.24e-4` (M5 Safe), `1.92e-4` (M5 Fast), `2.67e-5` (M1 Safe), and
  `2.10e-5` (M1 Fast); all checks passed `rtol=5e-4, atol=2e-6`.
  The one-to-one cases had zero reported gradient error.
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
  4 GiB on M5 Pro or 2 GiB on the 8 GiB M1 after each size. At 65,536 points
  the estimate was 96 MiB, the pair count was 8.59 billion, and observed
  driver allocation was 32.69 MiB on M5 Pro and 52.75 MiB on M1. Both M1
  modes completed both sizes without a guard trip or out-of-memory error.

All values below are median synchronized host wall milliseconds. Each cell
comes from twelve samples; the process performed three warmups first.

### M5 Pro

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

### M1

| Math | N | Pattern | Forward | Backward | Full E2E | Two scatters |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Safe | 32,768 | Uniform | 87.205 | 5.138 | 88.053 | 1.553 |
| Safe | 32,768 | Random one-to-one | 86.820 | 5.103 | 88.124 | 1.550 |
| Safe | 32,768 | Concentrated | 86.647 | 453.639 | 542.385 | 821.845 |
| Safe | 65,536 | Uniform | 326.001 | 5.639 | 325.878 | 1.688 |
| Safe | 65,536 | Random one-to-one | 324.143 | 5.532 | 326.020 | 1.790 |
| Safe | 65,536 | Concentrated | 323.970 | 1058.610 | 1364.108 | 1916.255 |
| Fast | 32,768 | Uniform | 87.216 | 5.048 | 88.028 | 1.541 |
| Fast | 32,768 | Random one-to-one | 86.943 | 5.084 | 88.005 | 1.552 |
| Fast | 32,768 | Concentrated | 86.746 | 451.846 | 536.546 | 811.224 |
| Fast | 65,536 | Uniform | 325.990 | 5.690 | 325.929 | 1.729 |
| Fast | 65,536 | Random one-to-one | 324.331 | 5.662 | 326.060 | 1.826 |
| Fast | 65,536 | Concentrated | 324.110 | 990.001 | 1380.841 | 1910.644 |

The table medians are shown for scale. The following ratios instead pair the
concentrated and uniform samples by iteration **before** taking their median;
they need not equal a ratio formed from two table medians.

| Device | Math | N | Backward ratio | Full E2E ratio | Two-scatter ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| M5 Pro | Safe | 32,768 | 0.68× | 0.97× | 1.39× |
| M5 Pro | Safe | 65,536 | 2.87× | 1.00× | 1.57× |
| M5 Pro | Fast | 32,768 | 1.23× | 0.95× | 1.18× |
| M5 Pro | Fast | 65,536 | 1.01× | 1.03× | 1.73× |
| M1 | Safe | 32,768 | 88.29× | 6.17× | 532.41× |
| M1 | Safe | 65,536 | 189.98× | 4.19× | 1107.78× |
| M1 | Fast | 32,768 | 87.95× | 6.10× | 529.52× |
| M1 | Fast | 65,536 | 172.27× | 4.23× | 1097.85× |

At 65,536 points, M5 Pro Safe concentrated backward samples ranged from
1.149 to 4.969 ms. The raw sample arrays should be used for any finer
comparison.

## Decision and limits

**Keep native PyTorch scatter as the current default; prioritize a dedicated
reduction candidate for concentrated M1 workloads.** On M5 Pro the measured
two-scatter control is below 0.5 ms at the median, and concentration has no
consistent full-loss penalty in these fixtures. On M1, by contrast, the same
synthetic concentration pattern makes the measured *whole bidirectional loss
and backward* 4.19–6.17× slower than uniform in Safe Math and 4.23–6.10×
slower in Fast Math. The M1 two-scatter control shows a much larger penalty,
but it is not the full backward path. A segmented or gather-style candidate
is justified for M1; it should replace the default only after a same-input
ablation proves faster full loss plus backward **including** grouping, buffer
creation, and gradient computation while meeting index and gradient parity.
No candidate was benchmarked here, so this study does not assert its speedup.

These synchronized host-wall measurements do not identify a Metal atomic
primitive or prove hardware-level atomic contention. The scatter control
uses unit contributions, whereas Chamfer backward uses scaled coordinate
differences. Neither power and thermal state nor GPU traces were controlled.
The M1/M5 comparison also mixes PyTorch 2.12.0 and 2.14.1, although the three
measured source files were byte-identical. The earlier
[physical M1 report](phase3-physical-m1-2026-10-02.md) used a smaller,
single-directional fixture; the new results are an independent bidirectional
extension. Neither device result establishes behavior on real point-cloud
training workloads or other Apple Silicon generations.

## Artifact checksums and reproduction

The local copies of the M1 artifacts matched the SHA-256 hashes computed on
the remote M1 before transfer:

| Artifact | SHA-256 |
| --- | --- |
| [Safe JSON](../bench/results/2026-10-02-apple-m1-chamfer-contention-large-safe.json) | `86e92c1fd0f36ef29ead5d64ea4d5c0472e64ccfbf98311b92331ffcdd34e085` |
| [Fast JSON](../bench/results/2026-10-02-apple-m1-chamfer-contention-large-fast.json) | `0e6bb63294f613e76654d0e6e90c71d736ed9c44cb32d9cb5a1aadbe1bf4b9f9` |
| [Safe log](bench-chamfer-contention-large-apple-m1-safe-2026-10-02.log) | `f0d72c3798b41e8d7a002029d9774dc33fe91bcf137982a5319c0144246e1e47` |
| [Fast log](bench-chamfer-contention-large-apple-m1-fast-2026-10-02.log) | `ef8f12f85327d5908ffe368c42eaea9d651ec5d172353b1fd3612f3d08326328` |

Reproduce from either recorded clean commit with the corresponding
environment. On M5 Pro, the command used the script defaults including the
4 GiB driver limit:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_chamfer_contention_large.py --output large-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/bench_chamfer_contention_large.py --output large-fast.json
```

On 8 GiB M1, the same sizes, seed, warmups, and repetitions used the same
benchmark with only the post-size MPS driver limit reduced from 4 GiB to
2 GiB. The Safe and Fast commands ran in separate processes, with no other
MPS benchmark active:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_chamfer_contention_large.py \
  --sizes 32768 65536 --max-driver-mib 2048 --output large-m1-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/bench_chamfer_contention_large.py \
  --sizes 32768 65536 --max-driver-mib 2048 --output large-m1-fast.json
```

# Integrated private SubM path: M5 Pro check

Source commit: `55e8813f5ae7d98bbcba358a5949abe7176cedd6`.
Hardware: Apple M5 Pro. Runtime: macOS 26.5.2, PyTorch 2.14.1. The input is
deterministic, randomly selected unique coordinates in a `64×32×16` grid,
with four input and four output channels and a centered `3×3×3` kernel.
These numbers cover a private prototype, not a public `spconv` adapter.

Each mode ran in a separate process with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
The paired [Safe](pytest-safe.log) and [Fast](pytest-fast.log) raw logs each
show **60 passed** for the CPU coordinate oracle, Metal SubM wrapper, and MPS
rulebook suites. They include exact integer rulebook parity and direct
CPU-rulebook versus GPU-rulebook output/first-gradient comparisons at 1,025
and 10,000 active rows. Full forward output is bit-identical between the two
paths; gradients use `rtol=1e-4, atol=1e-5` because the reductions differ.
The latter is a tolerance result, not a bitwise gradient claim.

The synchronized [Safe](safe.json) and [Fast](fast.json) full-call benchmark
used five warmups and 20 measured repetitions per case. Forward includes CPU
coordinate validation, rulebook construction or transfer, and the Metal
convolution. Backward includes the upstream scalar loss and first gradients.
The table shows medians in milliseconds:

| Math mode | Rows | Rulebook | Forward | Backward |
| --- | ---: | --- | ---: | ---: |
| Safe | 1,025 | CPU | 9.952 | 2.254 |
| Safe | 1,025 | MPS | 0.948 | 0.390 |
| Safe | 10,000 | CPU | 128.186 | 4.737 |
| Safe | 10,000 | MPS | 6.288 | 2.553 |
| Fast | 1,025 | CPU | 9.932 | 2.061 |
| Fast | 1,025 | MPS | 0.940 | 0.381 |
| Fast | 10,000 | CPU | 153.433 | 5.698 |
| Fast | 10,000 | MPS | 7.002 | 4.679 |

Reproduction from the source commit, with the same Python environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest -q tests/test_sparse_rulebook.py tests/test_subm_metal.py tests/test_subm_rulebook_mps.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_subm_conv_backends.py --warmup 5 --repeats 20
```

Repeat both commands in a fresh process with `PYTORCH_MPS_FAST_MATH=1` for
Fast mode. The benchmark records all raw times, software versions, and math
mode in JSON. These are single-host measurements; older chips, larger inputs,
memory peaks, and direct upstream `spconv` parity remain open. `SHA256SUMS`
checks the four raw artifacts.

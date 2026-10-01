# Three-channel point-gradient scatter fan-in, M5 Pro

This extends the [large graph scatter probe](2026-10-01-m5-pro-scatter-fanin-large.md)
with the three float32 channels used when point-coordinate gradients are
accumulated into reference points. It measures native PyTorch `scatter_add_`
on synthetic contribution arrays, **not** a Chamfer Distance kernel or an
end-to-end model. In a first-order Chamfer backward implementation, this
probe's `scatter_add_` **forward** timing corresponds to the accumulation
call. The separately reported autograd backward of `scatter_add_` is an
operator-level derivative measurement, not the Chamfer backward duration.

## Source, environment, and input contract

- The [dedicated script](../../bench/bench_scatter_xyz_fanin_mps.py) imports
  helpers from the unchanged [#42 script](../../bench/bench_scatter_fanin_mps.py).
  Benchmark source commit: `de6a6e3b0ae907bad42228488553006e490d2104`;
  dedicated script SHA-256:
  `39bc82c433cb18ff2a9e4f4d331b81d6a492a2b6ccdc19db72ffc2df9b9a7f64`;
  imported helper SHA-256:
  `ff8f61a707b24b41e67699bb68e415d6639f965dc3778e75ff4c6985bc915220`.
- Physical Apple M5 Pro, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1,
  five PyTorch CPU threads; no PyG package in this environment.
  `PYTORCH_ENABLE_MPS_FALLBACK=0` was set before import. Safe/Fast Math ran in
  separate processes. Each raw JSON records the exact command and environment.
- Logical contributions have shape `[B,N,3]`; the native operation receives
  contiguous `[B·N,3]` values and global destination indices. Each batch has
  its own reference-point range. Main measurements use `B=1`, so
  `single_hot` means **all** contributions go to one destination. Additional
  `B=2` runs use one destination **per batch** and verify index offsets.
- The two main sizes are 262,144 and 1,048,576 contributions, with eight
  nominal contributions per reference point. Uniform, 80%-to-first-1%-hub,
  and single-destination cases share the same source values at each size.
  Inputs are small dyadic float32 values so sums stay exactly representable
  in this fixture. A nonuniform, exactly representable upstream gradient
  tests the source-gradient index mapping. Every one of the 30 cases across
  six processes passed bitwise CPU/MPS output and source-gradient checks,
  including the two-batch cases. This is not a general floating-point
  determinism guarantee.
- MPS forward/backward each have three warmups and twelve synchronized
  `perf_counter_ns` host-wall samples. CPU calls have two warmups and eight
  samples. Output reset is outside forward timing; allocation and forward
  graph construction are outside backward timing. Input generation, device
  transfers, and total model time are excluded.

## Raw data

| Configuration | Safe | Fast |
| --- | --- | --- |
| `B=1`, uniform → hub → single | [JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-safe.json) · [table](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-safe.md) | [JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-fast.json) · [table](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-fast.md) |
| `B=1`, single → hub → uniform | [JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-reverse-safe.json) · [table](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-reverse-safe.md) | [JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-reverse-fast.json) · [table](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-reverse-fast.md) |
| `B=2`, 262,144 contributions | [JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-b2-safe.json) · [table](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-b2-safe.md) | [JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-b2-fast.json) · [table](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-b2-fast.md) |

Every JSON includes the per-call samples, median/min/max/p10/p90, observed
destination counts and top-1% share, source and helper hashes, validation
booleans, and exact CPU/MPS maximum differences. The machine-readable
records distinguish the B=1 global single destination from B=2's one
destination per cloud.

## Observed fan-in sensitivity

Median synchronized host-wall milliseconds for **1,048,576** contributions:

| Mode and pattern order | MPS uniform forward | MPS single forward | Single/uniform | MPS uniform backward | MPS single backward | CPU uniform forward | CPU single forward |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Safe, original | 0.836 | 2.480 | 2.97× | 1.123 | 0.998 | 1.963 | 1.732 |
| Fast, original | 0.802 | 2.204 | 2.75× | 1.528 | 2.044 | 2.065 | 1.933 |
| Safe, reversed | 0.922 | 2.349 | 2.55× | 0.910 | 1.130 | 2.006 | 1.697 |
| Fast, reversed | 0.805 | 2.197 | 2.73× | 1.017 | 1.220 | 1.972 | 1.896 |

At 262,144 contributions, the MPS single/uniform forward ratio was
2.15–2.51× across the same four runs. At 1,048,576 it was 2.55–2.97×.
The hub forward ratio ranged 0.93–1.77× at 262,144 and 1.01–1.51× at
1,048,576; backward ratios were less stable. The CPU columns are a native
PyTorch call baseline under the same input fixture, not a full pipeline
comparison. With one destination, CPU forward was faster than MPS forward
in these four 1,048,576-contribution runs; with uniform destinations, the
measured MPS call was faster. The exact samples and variability are in the
linked JSON rather than a single claimed speedup.

This demonstrates a repeatable **call-level** single-destination penalty on
this M5 Pro fixture. It does not show which GPU primitive PyTorch used,
whether Metal atomics caused the difference, or how a real point cloud's
nearest-neighbor distribution behaves. Timings include host dispatch and
synchronization, and were not collected on physical M1–M4 devices. A
custom reduction kernel still needs direct kernel evidence and an end-to-end
model ablation. This record contributes to the scoped
[Phase 3 release gate](https://github.com/gamzerA/mps-pointops/issues/41).

## Reproduce

Run from the repository root with a PyTorch build exposing MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python3 bench/bench_scatter_xyz_fanin_mps.py \
  --output bench/results/local-scatter-xyz-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python3 bench/bench_scatter_xyz_fanin_mps.py \
  --output bench/results/local-scatter-xyz-fast.json

# Repeat with destination order reversed in two more fresh processes.
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python3 bench/bench_scatter_xyz_fanin_mps.py \
  --patterns single_hot hub uniform \
  --output bench/results/local-scatter-xyz-reverse-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python3 bench/bench_scatter_xyz_fanin_mps.py \
  --patterns single_hot hub uniform \
  --output bench/results/local-scatter-xyz-reverse-fast.json

# Check nontrivial batch offsets separately.
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python3 bench/bench_scatter_xyz_fanin_mps.py --edges 262144 --batches 2 \
  --output bench/results/local-scatter-xyz-b2-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python3 bench/bench_scatter_xyz_fanin_mps.py --edges 262144 --batches 2 \
  --output bench/results/local-scatter-xyz-b2-fast.json
```

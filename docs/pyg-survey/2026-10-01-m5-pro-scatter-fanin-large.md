# Large native MPS scatter fan-in probe

This extends the [PyG 2.8 native scatter survey](2026-10-01-m5-pro-scatter-profile.md)
from 262,144 edges to 1,048,576 and 2,097,152 edges. It isolates native
PyTorch operations and measures both their forward and source-gradient
backward calls. The [script](../../bench/bench_scatter_fanin_mps.py) is a
synthetic graph aggregation stress test, **not** an end-to-end PyG or
point-cloud model benchmark.

## Fixed environment and inputs

- Physical Apple M5 Pro, macOS 26.5.2, Python 3.12.13, PyTorch 2.14.1.
  `PYTORCH_ENABLE_MPS_FALLBACK=0` and Safe/Fast Math were fixed before torch
  import; Safe and Fast ran in separate processes. Repository base commit:
  `0a0ce426fcffe291ee88b805db81bf618c6558fe`. Script SHA-256:
  `ff8f61a707b24b41e67699bb68e415d6639f965dc3778e75ff4c6985bc915220`.
- Each edge count uses eight edges per output node. The same precomputed
  source values are reused across destination patterns at a given scale:
  uniform destinations, 80% to the first 1% of nodes, or all edges to one
  node. The paired JSON records max in-degree and actual top-1% edge share.
- `add_c32` is native `scatter_add_` with 32 float32 message channels;
  `amax_scalar` is native `scatter_reduce_(reduce="amax",
  include_self=False)` with one float32 attention-like value per edge.
  The sum source is a small exact binary lattice. Max sources are positive,
  unique float32 values, avoiding tied maxima and the separately documented
  [zero-extremum backward case](2026-10-01-m5-pro-scatter-backward.md).
- Output buffer reset is outside the forward timer. The output's allocation
  and scatter forward are outside the backward timer. Each measured call
  is bracketed by `torch.mps.synchronize()`. There are three warmups and
  twelve timed calls per case. All 18 cases in each mode matched CPU **bitwise**
  for both output and source gradient on this exact-value fixture. This is
  not a general float32 determinism guarantee.

The complete per-call samples and validation are in [Safe JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-safe.json)
and [Fast JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-fast.json),
with full [Safe](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-safe.md)
and [Fast](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-fast.md)
tables. The same fixture was repeated with destination patterns in reverse
order: [reverse Safe JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-reverse-safe.json)
and [reverse Fast JSON](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-reverse-fast.json),
with [reverse Safe](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-reverse-safe.md)
and [reverse Fast](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-reverse-fast.md)
tables. Each JSON contains the exact command, environment, base commit,
script hash, seed, graph shape, destination statistics, and every timing
sample. All 18 cases in each of the four processes passed the CPU/MPS
exact-value check.

## Largest case: 2,097,152 edges

Medians below are synchronized host-wall milliseconds. Columns retain
identical edge count and source values within each operation and mode.

| Mode | Native operation | Direction | Uniform | 1%-hub | Single destination |
| --- | --- | --- | ---: | ---: | ---: |
| Safe | `scatter_add_`, C32 | Forward | 21.519 | 17.292 | 22.596 |
| Safe | `scatter_add_`, C32 | Backward | 15.904 | 18.007 | 19.260 |
| Safe | `scatter_reduce_(amax)`, C1 | Forward | 0.534 | 0.505 | 1.921 |
| Safe | `scatter_reduce_(amax)`, C1 | Backward | 1.797 | 2.989 | 5.043 |
| Fast | `scatter_add_`, C32 | Forward | 17.075 | 19.018 | 19.935 |
| Fast | `scatter_add_`, C32 | Backward | 20.489 | 16.064 | 20.632 |
| Fast | `scatter_reduce_(amax)`, C1 | Forward | 0.531 | 0.677 | 2.133 |
| Fast | `scatter_reduce_(amax)`, C1 | Backward | 1.104 | 1.051 | 3.339 |

The all-to-one scalar `amax` control is slower than uniform at this scale:
3.6–4.0 times in forward and 2.8–3.0 times in backward in these first two
runs. The reversed-pattern reruns provide a check on ordering:

| Mode/order | `amax` C1 forward uniform | Forward single | Forward ratio | Backward uniform | Backward single | Backward ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Safe, original | 0.534 | 1.921 | 3.60 | 1.797 | 5.043 | 2.81 |
| Safe, reverse | 0.666 | 1.837 | 2.76 | 0.971 | 4.103 | 4.23 |
| Fast, original | 0.531 | 2.133 | 4.02 | 1.104 | 3.339 | 3.02 |
| Fast, reverse | 0.512 | 2.263 | 4.42 | 1.897 | 2.347 | 1.24 |

Thus the **forward** scalar `amax` all-to-one penalty appears in all four
runs. The backward ratio varies, and the channel-wide `scatter_add_`
control does **not** show a consistent fan-in slowdown across sizes, modes,
or orders. Small-case samples vary substantially; between-run power state
and macOS GPU activity were not controlled. The order check is useful but
does not turn the result into a causal GPU-kernel profile.

## Interpretation and next gate

The measured calls are PyTorch native operators. This probe does not inspect
their GPU kernel duration or establish that PyTorch used Metal float atomics.
It therefore cannot attribute the observed `amax` sensitivity to atomic
contention. Backward host-wall time also includes autograd dispatch and the
gradient work inside PyTorch. Prior model-level observations cannot be
divided by these isolated medians to infer a model bottleneck fraction.

These are physical **M5 Pro** measurements. The project's Apple M1 (Virtual)
CI provides a separate functional check; it is not physical M1 performance
evidence. No physical M1–M4 timings or SIMD scheduling comparisons were
collected here. A custom Metal segmented reduction requires a controlled
candidate and end-to-end PyG model ablation, GPU-side profiling, varied real
graph distributions, and physical cross-generation measurements before a
speed or universal compatibility claim.

This result contributes to the scoped [Phase 3 release gate](https://github.com/gamzerA/mps-pointops/issues/41),
which explicitly separates observed fan-in timing from an unproven Metal
atomic explanation.

Reproduce from a checkout with PyTorch MPS support, using two fresh processes:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python3 bench/bench_scatter_fanin_mps.py \
  --output bench/results/local-scatter-fanin-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python3 bench/bench_scatter_fanin_mps.py \
  --output bench/results/local-scatter-fanin-fast.json

# Pattern-order sensitivity, again as two fresh processes:
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python3 bench/bench_scatter_fanin_mps.py \
  --patterns single_hot hub uniform \
  --output bench/results/local-scatter-fanin-reverse-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python3 bench/bench_scatter_fanin_mps.py \
  --patterns single_hot hub uniform \
  --output bench/results/local-scatter-fanin-reverse-fast.json
```

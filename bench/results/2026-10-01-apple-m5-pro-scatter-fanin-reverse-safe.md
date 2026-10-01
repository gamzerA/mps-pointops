# Large native MPS scatter fan-in probe

Captured `2026-10-01T14:34:33.817236+00:00` on Apple M5 Pro, macOS 26.5.2, PyTorch 2.14.1, Fast Math `0`.
Base commit `0a0ce426fcffe291ee88b805db81bf618c6558fe`; script SHA-256 `ff8f61a707b24b41e67699bb68e415d6639f965dc3778e75ff4c6985bc915220`.
Run: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python3 bench/bench_scatter_fanin_mps.py --patterns single_hot hub uniform --output bench/results/2026-10-01-apple-m5-pro-scatter-fanin-reverse-safe.json`

3 warmups and 12 timed calls per case. Times are synchronized host-wall milliseconds; full samples and validation are in the paired JSON.

| Edges | Nodes | Channels | Destinations | Max degree | Top 1% share | Native op | Forward median | Backward median |
| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 262144 | 32768 | 32 | single_hot | 262144 | 1.000 | add_c32 | 1.705 | 1.407 |
| 262144 | 32768 | 1 | single_hot | 262144 | 1.000 | amax_scalar | 0.438 | 0.440 |
| 262144 | 32768 | 32 | hub | 714 | 0.803 | add_c32 | 1.994 | 1.667 |
| 262144 | 32768 | 1 | hub | 714 | 0.803 | amax_scalar | 0.285 | 0.320 |
| 262144 | 32768 | 32 | uniform | 20 | 0.020 | add_c32 | 2.903 | 1.319 |
| 262144 | 32768 | 1 | uniform | 20 | 0.020 | amax_scalar | 0.210 | 0.459 |
| 1048576 | 131072 | 32 | single_hot | 1048576 | 1.000 | add_c32 | 7.361 | 7.850 |
| 1048576 | 131072 | 1 | single_hot | 1048576 | 1.000 | amax_scalar | 1.051 | 1.315 |
| 1048576 | 131072 | 32 | hub | 726 | 0.802 | add_c32 | 7.470 | 9.986 |
| 1048576 | 131072 | 1 | hub | 726 | 0.802 | amax_scalar | 0.508 | 0.561 |
| 1048576 | 131072 | 32 | uniform | 24 | 0.021 | add_c32 | 6.365 | 11.491 |
| 1048576 | 131072 | 1 | uniform | 24 | 0.021 | amax_scalar | 0.332 | 0.556 |
| 2097152 | 262144 | 32 | single_hot | 2097152 | 1.000 | add_c32 | 17.199 | 20.047 |
| 2097152 | 262144 | 1 | single_hot | 2097152 | 1.000 | amax_scalar | 1.837 | 4.103 |
| 2097152 | 262144 | 32 | hub | 729 | 0.802 | add_c32 | 21.913 | 21.742 |
| 2097152 | 262144 | 1 | hub | 729 | 0.802 | amax_scalar | 0.715 | 1.010 |
| 2097152 | 262144 | 32 | uniform | 23 | 0.021 | add_c32 | 13.633 | 20.016 |
| 2097152 | 262144 | 1 | uniform | 23 | 0.021 | amax_scalar | 0.666 | 0.971 |

`add_c32` times native `scatter_add_` on channel-wide graph messages; `amax_scalar` times native `scatter_reduce_(amax, include_self=False)` on one attention-like value per edge. Uniform, 80%-to-1%-hub, and all-to-one destination mappings use the same edge count and source values within each operation and scale. Source/output transfer, output reset, and forward construction for backward are outside timed regions. The fixture uses exact float32 lattice sums or unique positive max values, and every case checks bitwise CPU/MPS output and source gradient.

These observations describe sensitivity to destination fan-in in this PyTorch/MPS version. Host dispatch, synchronization, allocation around backward, power state, and other processes can affect timings. This script does not inspect GPU kernel duration, Metal atomics, or the backend primitive used by PyTorch. It cannot by itself justify a custom Metal reduction kernel or a universal PyG speed claim.

# Large native MPS scatter fan-in probe

Captured `2026-10-01T14:32:34.021113+00:00` on Apple M5 Pro, macOS 26.5.2, PyTorch 2.14.1, Fast Math `0`.
Base commit `0a0ce426fcffe291ee88b805db81bf618c6558fe`; script SHA-256 `ff8f61a707b24b41e67699bb68e415d6639f965dc3778e75ff4c6985bc915220`.
Run: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python3 bench/bench_scatter_fanin_mps.py --output bench/results/2026-10-01-apple-m5-pro-scatter-fanin-safe.json`

3 warmups and 12 timed calls per case. Times are synchronized host-wall milliseconds; full samples and validation are in the paired JSON.

| Edges | Nodes | Channels | Destinations | Max degree | Top 1% share | Native op | Forward median | Backward median |
| ---: | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 262144 | 32768 | 32 | uniform | 20 | 0.020 | add_c32 | 1.865 | 1.338 |
| 262144 | 32768 | 1 | uniform | 20 | 0.020 | amax_scalar | 0.233 | 0.274 |
| 262144 | 32768 | 32 | hub | 714 | 0.803 | add_c32 | 1.296 | 2.130 |
| 262144 | 32768 | 1 | hub | 714 | 0.803 | amax_scalar | 0.398 | 0.307 |
| 262144 | 32768 | 32 | single_hot | 262144 | 1.000 | add_c32 | 2.891 | 1.737 |
| 262144 | 32768 | 1 | single_hot | 262144 | 1.000 | amax_scalar | 0.446 | 0.566 |
| 1048576 | 131072 | 32 | uniform | 24 | 0.021 | add_c32 | 7.291 | 8.856 |
| 1048576 | 131072 | 1 | uniform | 24 | 0.021 | amax_scalar | 0.388 | 0.534 |
| 1048576 | 131072 | 32 | hub | 726 | 0.802 | add_c32 | 5.822 | 9.893 |
| 1048576 | 131072 | 1 | hub | 726 | 0.802 | amax_scalar | 0.421 | 0.913 |
| 1048576 | 131072 | 32 | single_hot | 1048576 | 1.000 | add_c32 | 9.869 | 7.535 |
| 1048576 | 131072 | 1 | single_hot | 1048576 | 1.000 | amax_scalar | 1.044 | 1.482 |
| 2097152 | 262144 | 32 | uniform | 23 | 0.021 | add_c32 | 21.519 | 15.904 |
| 2097152 | 262144 | 1 | uniform | 23 | 0.021 | amax_scalar | 0.534 | 1.797 |
| 2097152 | 262144 | 32 | hub | 729 | 0.802 | add_c32 | 17.292 | 18.007 |
| 2097152 | 262144 | 1 | hub | 729 | 0.802 | amax_scalar | 0.505 | 2.989 |
| 2097152 | 262144 | 32 | single_hot | 2097152 | 1.000 | add_c32 | 22.596 | 19.260 |
| 2097152 | 262144 | 1 | single_hot | 2097152 | 1.000 | amax_scalar | 1.921 | 5.043 |

`add_c32` times native `scatter_add_` on channel-wide graph messages; `amax_scalar` times native `scatter_reduce_(amax, include_self=False)` on one attention-like value per edge. Uniform, 80%-to-1%-hub, and all-to-one destination mappings use the same edge count and source values within each operation and scale. Source/output transfer, output reset, and forward construction for backward are outside timed regions. The fixture uses exact float32 lattice sums or unique positive max values, and every case checks bitwise CPU/MPS output and source gradient.

These observations describe sensitivity to destination fan-in in this PyTorch/MPS version. Host dispatch, synchronization, allocation around backward, power state, and other processes can affect timings. This script does not inspect GPU kernel duration, Metal atomics, or the backend primitive used by PyTorch. It cannot by itself justify a custom Metal reduction kernel or a universal PyG speed claim.

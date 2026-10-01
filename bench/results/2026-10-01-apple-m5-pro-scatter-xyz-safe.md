# Three-channel point-gradient scatter fan-in probe

Captured `2026-10-01T14:45:44.935060+00:00` on Apple M5 Pro, macOS 26.5.2, PyTorch 2.14.1, Fast Math `0`.
Base commit `de6a6e3b0ae907bad42228488553006e490d2104`; script SHA-256 `39bc82c433cb18ff2a9e4f4d331b81d6a492a2b6ccdc19db72ffc2df9b9a7f64`; imported helper SHA-256 `ff8f61a707b24b41e67699bb68e415d6639f965dc3778e75ff4c6985bc915220`.
Run: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python3 bench/bench_scatter_xyz_fanin_mps.py --output bench/results/2026-10-01-apple-m5-pro-scatter-xyz-safe.json`

| Contributions | B | Pattern | Max fan-in | Top 1% share | MPS fwd ms | MPS bwd ms | CPU fwd ms | CPU bwd ms |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 262144 | 1 | uniform | 22 | 0.020 | 0.266 | 0.357 | 0.603 | 0.141 |
| 262144 | 1 | hub | 715 | 0.801 | 0.471 | 0.378 | 0.957 | 0.125 |
| 262144 | 1 | single_hot | 262144 | 1.000 | 0.669 | 0.360 | 0.456 | 0.119 |
| 1048576 | 1 | uniform | 23 | 0.021 | 0.836 | 1.123 | 1.963 | 0.549 |
| 1048576 | 1 | hub | 724 | 0.802 | 0.846 | 0.855 | 2.816 | 0.575 |
| 1048576 | 1 | single_hot | 1048576 | 1.000 | 2.480 | 0.998 | 1.732 | 0.393 |

The logical [B,N,3] contribution tensor is flattened to [B*N,3]. Each source addresses one reference point within its own batch; single_hot means one destination per batch. Sources are exact float32 binary-lattice values, so this fixture checks bitwise CPU/MPS output and a nonuniform-index source gradient. This does not imply general floating-point determinism.

All MPS calls are timed as synchronized host-wall intervals. Output reset is outside forward timing; forward construction is outside backward timing. CPU times use the same call boundaries without MPS synchronization. Input generation and device transfers are excluded. The method cannot identify the PyTorch GPU primitive or attribute a latency difference to Metal atomics.

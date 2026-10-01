# Opt-in fused CSR pooling: M5 Pro full-call comparison

The [benchmark runner](../bench/bench_voxel_pool_backend.py) compares the
default `index_add` and experimental `fused_csr` backends through the public
`voxel_downsample` call. It measures **full forward, scalar loss, and first
backward**, including identical voxel/CSR map construction. The raw [Safe](bench-voxel-fused-m5pro-torch214-safe-2026-10-02.json)
and [Fast](bench-voxel-fused-m5pro-torch214-fast-2026-10-02.json) JSON files
contain every sample and synchronized memory checkpoint.

## Reproduction and provenance

- Physical Apple M5 Pro, macOS 26.5.2, PyTorch 2.14.1, 48 GiB unified memory.
  The benchmark source commit was `dae1bd3c95f0d068ae6ade6d4aaacaa34e8f27af`;
  the benchmark script SHA-256 is
  `174828768cbbbaf7cd811b9d73e97a1f99fc2ed21d93f9a835e12b38f35ac558`.
  Each JSON records the `mps_pointops/voxel.py` SHA-256, software version,
  exact input digest, and all per-case samples. Both backends in a row have
  the same input digest.
- Separate processes used `PYTORCH_ENABLE_MPS_FALLBACK=0` and
  `PYTORCH_MPS_FAST_MATH=0` or `1`. Each backend/fixture ran in a fresh worker
  process with two warmups and five timed repetitions. Timing has one
  `torch.mps.synchronize()` before and after each full step.
- Finite float32 `(N,3)` positions and `(N,32)` features use the fixed seed
  and dyadic cell fixture from the [compact API benchmark](voxel-api-benchmark.md).
  It covers 20k/100k/500k points, uniform/ragged four-batch layouts, and
  dense/sparse cell occupancy. The benchmark checks voxel count and that
  position/feature gradients exist; numerical parity is separately covered
  by the [contract and tests](voxel-api-contract.md).

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_voxel_pool_backend.py \
  --output docs/bench-voxel-fused-m5pro-torch214-safe-2026-10-02.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python bench/bench_voxel_pool_backend.py \
  --output docs/bench-voxel-fused-m5pro-torch214-fast-2026-10-02.json
```

## Full-step timing

Numbers are medians in milliseconds; each cell shows **default / fused**.
The full 24 backend cases per mode, including min/max and all raw samples,
are in the JSON files.

| Points | Batches | Cells | Safe default / fused | Fast default / fused |
| ---: | --- | --- | ---: | ---: |
| 20k | uniform | dense | 8.75 / 8.97 | 7.37 / 7.54 |
| 20k | uniform | sparse | 6.28 / 2.97 | 2.93 / 3.32 |
| 20k | ragged | dense | 4.43 / 8.37 | 5.41 / 4.59 |
| 20k | ragged | sparse | 7.39 / 9.23 | 8.06 / 7.86 |
| 100k | uniform | dense | 6.81 / 5.65 | 3.53 / 4.28 |
| 100k | uniform | sparse | 15.27 / 10.61 | 13.02 / 17.37 |
| 100k | ragged | dense | 3.91 / 11.79 | 12.09 / 11.16 |
| 100k | ragged | sparse | 15.79 / 12.31 | 16.97 / 14.40 |
| 500k | uniform | dense | 21.36 / 22.59 | 26.48 / 23.84 |
| 500k | uniform | sparse | 38.53 / 41.52 | 40.51 / 40.29 |
| 500k | ragged | dense | 21.37 / 26.17 | 26.13 / 23.54 |
| 500k | ragged | sparse | 40.30 / 46.31 | 41.39 / 41.95 |

Fused was faster in **4/12 Safe** and **7/12 Fast** median fixtures. The
results vary by input distribution and math mode; they do not establish a
general full-call speedup. Map construction and its synchronization remain
common to both paths, limiting what one pooling dispatch can save. The
small sample count and broad timing spread also preclude a stable ranking
from differences of a few milliseconds.

## Memory readings

PyTorch 2.14.1 `torch.mps` exposes *current* and *driver allocated* memory,
but no transient peak-memory API. The runner records these values after input
creation and after each synchronized step, plus process high-water RSS. The
largest recorded MPS value is a **checkpoint maximum**, not a true VRAM or
unified-memory peak. Cached driver allocations can remain after an operation;
CPU RSS and MPS allocations overlap in unified memory and cannot be added.

For the 500k uniform fixtures below, the two backends had the same largest
observed PyTorch current allocation and driver allocation (MiB, `2^20`
bytes) in their separate workers:

| Cells | MPS current, either backend | MPS driver, either backend | Process peak RSS, default / fused (Safe) |
| --- | ---: | ---: | ---: |
| dense | 152.8 | 1,064.7 | 486.5 / 486.6 |
| sparse | 238.1 | 1,064.7 | 486.3 / 486.3 |

These checkpoints did not resolve a memory saving from the fused dispatch.
A GPU trace or sufficiently fine external allocation sampler is needed before
claiming a lower *peak* device footprint.

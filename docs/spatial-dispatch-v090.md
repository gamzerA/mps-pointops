# Public SpatialIndex adaptive-dispatch audit (v0.9 research)

This report measures the **public** `mps_pointops.spatial.SpatialIndex.knn` wrapper with `backend="auto"`, `"scan"`, and `"bvh"`. It is a local M5 Pro feasibility result, not a general Apple Silicon speed guarantee. The BVH is still a guarded experimental path; `backend="auto"` keeps the existing scan for devices and inputs outside its measured policy.

## Exact run and provenance

- Hardware: Apple M5 Pro, 48 GB unified memory; macOS 26.5.2; PyTorch 2.14.1.
- `PYTORCH_ENABLE_MPS_FALLBACK=0`, `PYTORCH_MPS_FAST_MATH=0` set **before** Python starts. One fixture per fresh Python process.
- `N=1,000,000`, `K=16`, deterministic seed `20261002`, cube extent `1024`. Uniform references fill the cube. Cluster–sparse references put 90% of points in a side-1 cube at the center and 10% uniformly in the full cube. Collapsed references all share the center coordinate. Unless indicated, queries are sampled from references; cluster–sparse queries are half cluster and half background. Additional independent-query fixtures use `Q=65,536`.
- The Morton `cell_size` supplied to every public wrapper is `16` for uniform/collapsed and `1/64` for cluster–sparse; `origin=(0,0,0)`. The same point and query tensors are shared by all paths.
- Executed source commit: `62fcc0f295e127be47a475fbc122d4de686b4c58` in a **clean detached worktree**. All 15 JSON records have `source_dirty=false`, the same commit, and consistent SHA-256 for every overlapping benchmark, wrapper, private index, shader, and scan source. These hashes also match the files in that checkout.
- Key SHA-256 values: `mps_pointops/spatial.py` = `ad0a71bed4d93b134c718124f6d08f085bea3d43915ddc169471185d835f0b8d`; `mps_pointops/kernels/spatial_bvh.metal` = `e466dd2880136a504106c8fba7e303117a9f2cdd4924dbf5bb3f636f9875c345`. Each raw record carries the complete source map.
- Raw data: [`bench/results/spatial-dispatch-v090-m5pro-clean/`](../bench/results/spatial-dispatch-v090-m5pro-clean/). Each record contains five synchronized wall-time samples except collapsed `Q=2,048` and `Q=65,536` (three samples), the actual selected backend, index mismatch counts, max Euclidean-distance error, one first-call sample, allocator peaks, and MPS driver-memory snapshots.

Run one example from the repository root:

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_spatial_dispatch.py \
  --distribution cluster-sparse --points 1000000 --queries 8192 \
  --k 16 --repeats 5 \
  --output bench/results/spatial-dispatch-v090-m5pro-clean/repeat-cluster-q8192.json
```

The script refuses to overwrite an existing JSON; choose a new output path for a repeat.

## Synchronized timing

All times below are medians in milliseconds. `scan`, `BVH`, and `auto` are **steady query** times with inputs already on MPS, shader compilation excluded, and `torch.mps.synchronize()` before and after each sample. The public wrapper, its validation, and (for BVH) the final square root are included. `first auto` is one separate cold public call that additionally includes policy sampling and possible index construction. Input transfer and explicit BVH construction are recorded separately in each JSON.

| Distribution | Q | Auto chose | Scan | Explicit BVH | Auto | First auto |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Uniform | 2,048 | scan | 33.3 | 30.1 | 33.8 | 44.0 |
| Uniform | 4,096 | scan | 62.2 | 30.0 | 60.3 | 71.2 |
| Uniform | 8,192 | BVH | 115.8 | 32.0 | 32.2 | 56.0 |
| Uniform | 65,536 | BVH | 879.4 | 173.3 | 173.4 | 183.1 |
| 90% cluster + 10% sparse | 2,048 | scan | 32.3 | 53.4 | 33.0 | 43.3 |
| 90% cluster + 10% sparse | 4,096 | scan | 61.0 | 57.1 | 62.4 | 70.6 |
| 90% cluster + 10% sparse | 8,192 | BVH | 115.6 | 53.6 | 53.7 | 75.7 |
| 90% cluster + 10% sparse | 65,536 | BVH | 882.6 | 215.8 | 196.8 | 229.9 |
| All coincident | 2,048 | scan | 32.7 | 232.5 | 33.4 | 41.3 |
| All coincident | 4,096 | scan | 60.9 | 233.6 | 60.8 | 67.7 |
| All coincident | 8,192 | scan | 114.0 | 236.4 | 116.0 | 133.1 |
| All coincident | 65,536 | scan | 863.6 | 1,293.2 | 865.1 | 885.8 |

The additional independent-query `Q=65,536` records show uniform `scan/BVH/auto = 883.1/164.1/166.3 ms` and cluster–sparse `886.8/196.2/220.6 ms`. In all 15 records, **auto-first, auto-steady, and explicit-BVH indices matched the public scan exactly** (zero differing `[Q,K]` cells); maximum absolute distance difference was also zero for these finite fixtures. This does not prove parity for arbitrary inputs or Fast Math.

An extra all-coincident `Q=256` record yielded `scan/BVH/auto = 12.8/9.2/13.2 ms`. The public BVH wrapper uses its parallel-microtree variant at `Q<=256`; it is a different execution path from the serial BVH numbers in the table. The automatic policy still uses scan at this size.

The current M5 Pro policy is intentionally conservative: for `N=1M`, `K=16`, Safe Math and `Q>=8192`, it samples Morton cells and uses the BVH only when the hottest sampled cell occupies less than 5%. At `Q=8,192`, uniform and cluster–sparse had a hottest-cell fraction of `0.00048828125` and selected BVH. The all-coincident fixture had fraction `1.0` and stayed on scan. At `Q=4,096`, explicit BVH beat scan in both measured non-collapsed fixtures, but the cluster–sparse margin was only 3.9 ms (about 6%); these two shapes alone do not justify lowering the default threshold for all workloads.

One first-call sample should not be treated as a stable latency estimate. Separate explicit BVH build samples were roughly 2.5–3.1 ms for the sampled-reference matrix; the cold auto call is longer because it also validates and samples the density policy. The fixed path measurement order and simultaneous residency of auto and explicit BVH objects may affect caches and allocator figures. No CPU KD-tree comparison was included here; the earlier private-kernel benchmarks contain that separate comparison.

## Memory interpretation and remaining gates

Each JSON records PyTorch's tensor/reserved allocator peak for every timed stage plus point-in-time `torch.mps.driver_allocated_memory()` samples. For the `Q=65,536` sampled-reference fixtures, the auto-query stage tensor peak was about **98.4 MB** for uniform and cluster–sparse and **84.3 MB** for collapsed; the explicit BVH's index tensors occupy **8,458,752 bytes**. These figures reflect this benchmark process, where multiple index instances and outputs can coexist. The driver reading near 1.08 GB includes PyTorch's cached allocations. **None of these values is a total system GPU peak or a transient Metal driver peak.** The separate one-path allocator/Instruments procedure in [`spatial-memory-v090.md`](spatial-memory-v090.md) is the appropriate source for a memory claim.

Release gates remain M1 verification when access is restored, broader shape and repeated-process measurements, radius-search routing experiments beyond the measured fixtures, and numerical audit on additional OS/Metal versions. The result supports this guarded M5 Pro dispatch policy only; it does not warrant a general crossover threshold or v0.9/v1.0 release claim.

# Public Ball Query scan/BVH audit (dirty v0.9 development evidence)

This experiment compares `SpatialIndex.ball_query(query, radius, K)` with explicit `backend="scan"` and `backend="bvh"` on the physical M5 Pro. The public `backend="auto"` currently chooses scan for radius queries; the BVH figures are **opt-in research-path measurements**, not a published automatic speed claim.

## Input and numerical contract

- `N=1,000,000`, `K=16`, deterministic seed `20261002`, query sizes `4,096`, `8,192`, and `65,536`. All tensors are MPS float32. Coordinates occupy a cube of side `1024`; Morton origin is zero, with cell sizes `16` (uniform/collapsed) or `1/64` (cluster–sparse).
- Uniform: independent uniform reference and query points, radius `12`. This radius gives mostly partially filled rows, with both no-hit and first-16-full rows in the measured fixtures.
- Cluster–sparse: 90% of references lie in a side-1 central cluster and 10% in the full cube. Half the independent queries sample the cluster and half the sparse background; radius `0.03` gives full and empty rows in this synthetic contrast.
- Collapsed stress case: all references coincide at the cube center. Half the queries coincide and half are displaced by `+1` on the x axis; radius `0.5` gives exactly full versus empty rows. This is an adversarial contract fixture, **not** a typical point-cloud distribution.
- The expected selection is the first `K` neighbors in **original reference-index order**, using strict `distance² < fl32(fl32(radius)·fl32(radius))`, with `(-1, 0)` index/distance padding. The existing native Metal scan is the numerical oracle; a float64 SciPy radius search can differ at float32 boundaries and has a separate ordering contract.

Every record checks the first and repeated BVH output and the `auto` output against scan for exact int64 indices and **bitwise float32 squared distances**. It also counts valid-slot patterns and original-index-order violations. Across the nine fixtures: all mismatch and order-violation counts were zero; every fixture contained at least one empty row and one full first-16 row. This is observed parity for these inputs, not a proof for all radii or compilers.

## Synchronized M5 Pro measurements

Machine: Apple M5 Pro, 48 GB unified memory, macOS 26.5.2, PyTorch 2.14.1. `PYTORCH_ENABLE_MPS_FALLBACK=0` and `PYTORCH_MPS_FAST_MATH=0` were set before each fresh Python process. Shaders were warmed outside timing; each query sample synchronizes MPS before and after host-wall timing. Values are medians in milliseconds from five samples for `Q≤8,192`, three samples for `Q=65,536`. Inputs are already on MPS for query timings. The one-shot first BVH call, including index build, is stored separately in each JSON and should not be treated as a stable latency estimate.

| Distribution | Q | Radius | Empty rows | Full rows | Scan query | BVH query | Auto chose |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Uniform | 4,096 | 12 | 5 | 3 | 172.2 | 6.49 | scan |
| Uniform | 8,192 | 12 | 13 | 9 | 327.9 | 7.25 | scan |
| Uniform | 65,536 | 12 | 114 | 98 | 2,553.0 | 36.39 | scan |
| 90% cluster + 10% sparse | 4,096 | 0.03 | 2,048 | 2,048 | 100.0 | 11.38 | scan |
| 90% cluster + 10% sparse | 8,192 | 0.03 | 4,096 | 4,096 | 191.1 | 12.25 | scan |
| 90% cluster + 10% sparse | 65,536 | 0.03 | 32,768 | 32,768 | 1,476.4 | 51.55 | scan |
| All coincident | 4,096 | 0.5 | 2,048 | 2,048 | 85.5 | 0.49 | scan |
| All coincident | 8,192 | 0.5 | 4,096 | 4,096 | 168.7 | 0.42 | scan |
| All coincident | 65,536 | 0.5 | 32,768 | 32,768 | 1,276.9 | 0.69 | scan |

The collapsed case is especially favorable to a first-K BVH traversal: the coincident rows can certify the first 16 original indices, while the displaced rows prune immediately. It must not be used to predict kNN speed or general Ball Query throughput. Larger radii that fill broad regions, more irregular point ordering, boundary-heavy inputs, and cross-device runs are required before routing the public `auto` radius path to BVH.

## Provenance, memory and reproduction

The checkout base was `b4e0090b58560a21cd4a4415405eb8ff30a26f86`, but the worktree was **dirty** throughout measurement. That base commit alone does not reconstruct the executable source. Every JSON in [`bench/results/spatial-radius-dispatch-v090-m5pro/`](../bench/results/spatial-radius-dispatch-v090-m5pro/) stores the exact SHA-256 of the benchmark, wrapper, private BVH, scan adapter, and Metal shaders; the source hashes are identical across the nine runs. A clean-source rerun is required before release or paper citation.

Run one fixture from the repository root, choosing a new output name because the script refuses overwrites:

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_spatial_radius_dispatch.py \
  --distribution cluster-sparse --points 1000000 --queries 8192 \
  --limit 16 --radius 0.03 --repeats 5 \
  --output bench/results/spatial-radius-dispatch-v090-m5pro/repeat-cluster-q8192.json
```

The JSON also records input-transfer time, the first BVH call, raw query times, PyTorch allocator peaks, and point-in-time MPS driver-memory readings. At `Q=65,536`, the resident BVH index tensors occupy `8,458,752` bytes. **Allocator peaks are not total GPU physical-memory peaks**, and the fixed measurement order keeps multiple path outputs resident. Use the separate one-path allocator and Instruments procedure in [`spatial-memory-v090.md`](spatial-memory-v090.md) for a memory conclusion.

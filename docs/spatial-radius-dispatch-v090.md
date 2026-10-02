# Public Ball Query scan/BVH audit (clean v0.9 research evidence)

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
| Uniform | 4,096 | 12 | 5 | 3 | 171.1 | 6.72 | scan |
| Uniform | 8,192 | 12 | 13 | 9 | 327.5 | 7.24 | scan |
| Uniform | 65,536 | 12 | 114 | 98 | 2,557.5 | 39.09 | scan |
| 90% cluster + 10% sparse | 4,096 | 0.03 | 2,048 | 2,048 | 102.3 | 11.36 | scan |
| 90% cluster + 10% sparse | 8,192 | 0.03 | 4,096 | 4,096 | 206.8 | 12.45 | scan |
| 90% cluster + 10% sparse | 65,536 | 0.03 | 32,768 | 32,768 | 1,468.4 | 44.05 | scan |
| All coincident | 4,096 | 0.5 | 2,048 | 2,048 | 92.6 | 0.40 | scan |
| All coincident | 8,192 | 0.5 | 4,096 | 4,096 | 167.4 | 0.44 | scan |
| All coincident | 65,536 | 0.5 | 32,768 | 32,768 | 1,356.4 | 0.62 | scan |

The collapsed case is especially favorable to a first-K BVH traversal: the coincident rows can certify the first 16 original indices, while the displaced rows prune immediately. It must not be used to predict kNN speed or general Ball Query throughput. Larger radii that fill broad regions, more irregular point ordering, boundary-heavy inputs, and cross-device runs are required before routing the public `auto` radius path to BVH.

## Large-radius dispatch counterexample

A follow-up clean-commit M5 Pro sweep held the **same** independent uniform
`N=1,000,000`, `Q=4,096`, `K=16`, seed, point order, and Safe-Math settings
fixed while changing only the radius. The benchmark's
`--allow-homogeneous-rows` option permits all-full rows; it does not relax
index, distance-bit, or original-index-order parity checks. Each median below
uses three synchronized query samples with inputs resident on MPS and an
already-built BVH for the steady-query column. The separate first BVH call
and all raw samples are in the [four JSON records](../bench/results/spatial-radius-crossover-v090-m5pro/).

| Radius | Full rows / 4,096 | Scan query ms | BVH query ms | Faster path |
| ---: | ---: | ---: | ---: | --- |
| 12 | 3 | 170.934 | 6.415 | BVH |
| 64 | 4,096 | 3.616 | 13.059 | scan |
| 512 | 4,096 | 0.210 | 24.944 | scan |
| 1,024 | 4,096 | 0.186 | 15.475 | scan |

Every record has zero index mismatches, zero float32 distance-bit mismatches,
and zero order violations against the native Metal scan. At radius 512, the
BVH is about **119 times slower** than the scan in this fixture. The native
scan stops as soon as each query has its first `K` matches; a large radius
can make that happen after very few input points. An automatic rule based
only on chip, `N`, `Q`, and `K` would therefore create a severe regression.
The current `auto` radius policy correctly stays on scan. Explicit
`backend="bvh"` remains available for research and for callers that know
their workload. A future automatic rule needs a measured density/radius
predictor that includes its own decision overhead and counterexamples with
different reference orders. This sweep does not establish a universal
crossover radius.

The sweep used source commit `2738661cd89c213c19bbc994b5f2291008018dc3`
with `source_dirty=false`, PyTorch 2.14.1, macOS 26.5.2, and
`PYTORCH_ENABLE_MPS_FALLBACK=0`. Each JSON embeds the source-file SHA-256
map; the adjacent `SHA256SUMS.txt` records the raw JSON hashes. Reproduce a
case in a fresh process, choosing a new output path:

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_spatial_radius_dispatch.py \
  --distribution uniform --points 1000000 --queries 4096 --limit 16 \
  --radius 512 --repeats 3 --allow-homogeneous-rows \
  --output /tmp/spatial-radius-r512-repeat.json
```

## Provenance, memory and reproduction

The executed source was commit `62fcc0f295e127be47a475fbc122d4de686b4c58` in a **clean detached worktree**. Every JSON in [`bench/results/spatial-radius-dispatch-v090-m5pro-clean/`](../bench/results/spatial-radius-dispatch-v090-m5pro-clean/) has `source_dirty=false`, this exact commit, and SHA-256 of the benchmark, wrapper, private BVH, scan adapter, and Metal shaders. Overlapping hashes are identical across the nine radius records and match the files in the executed checkout. The measurement remains limited to the M5 Pro and these synthetic fixtures.

Key SHA-256 values: `mps_pointops/spatial.py` = `ad0a71bed4d93b134c718124f6d08f085bea3d43915ddc169471185d835f0b8d`; `mps_pointops/kernels/spatial_bvh.metal` = `e466dd2880136a504106c8fba7e303117a9f2cdd4924dbf5bb3f636f9875c345`; `mps_pointops/kernels/ball_query.metal` = `2856fa046cbbafb71ec0cd688f071c7dc5d2280e9f04d00471241e24d6f85ba3`. Each raw record carries the complete source map.

Run one fixture from the repository root, choosing a new output name because the script refuses overwrites:

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python bench/bench_spatial_radius_dispatch.py \
  --distribution cluster-sparse --points 1000000 --queries 8192 \
  --limit 16 --radius 0.03 --repeats 5 \
  --output bench/results/spatial-radius-dispatch-v090-m5pro-clean/repeat-cluster-q8192.json
```

The JSON also records input-transfer time, the first BVH call, raw query times, PyTorch allocator peaks, and point-in-time MPS driver-memory readings. At `Q=65,536`, the resident BVH index tensors occupy `8,458,752` bytes. **Allocator peaks are not total GPU physical-memory peaks**, and the fixed measurement order keeps multiple path outputs resident. Use the separate one-path allocator and Instruments procedure in [`spatial-memory-v090.md`](spatial-memory-v090.md) for a memory conclusion.

# Sorted-Morton radius prototype at one million points

This is a **v0.9 development experiment**, not a public operator or release
benchmark. The measured source commit is
`1f1613a17c9b340464de92b6ae82180e8421d993` in the development branch.
Every raw JSON also records SHA-256 hashes for
the benchmark and the two experimental Metal stages. The source worktree was
clean during each measurement.

The prototype creates a 21-bit-per-axis Morton key in Metal, stably sorts
keys with PyTorch MPS, then looks up candidate cells by binary search in a
second Metal query kernel. It writes first-K reference indices in original
input order. The current implementation is single-cloud float32 and exposes
an explicit per-query fallback status when its cell traversal bound is
exceeded; it is **not** connected to `mps_pointops.flat.radius`.

## M5 Pro experiment

Apple M5 Pro, 48 GB, macOS 26.5.2, PyTorch 2.14.1, SciPy 1.18.1; MPS CPU
fallback disabled. Each entry is the median of five synchronized host-wall
samples in milliseconds, after shader compilation. Both CPU and MPS start
with coordinates resident on their own device. CPU cKDTree construction and
query are shown separately; MPS index construction includes invalid-key
checking, stable sorting, and index gathering. Host-to-device input transfer
is recorded separately in each JSON and excluded from these columns.
The uniform fixture occupies the cube `[0, 1024)^3`; the one-cell fixture
occupies a cube of side length `min(cell_size/16, extent/1024) = 1` centered
near `(512, 512, 512)`. Its radius is only `0.01`, while the uniform radius
is `16`, so the rows represent deliberately different workloads.

| Distribution and workload | Math | MPS build | MPS query | cKDTree build | cKDTree query | Index mismatch |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Uniform, N=1M, Q=1,024, r=16, K=16, cell=16 | Safe | 4.080 | 2.589 | 252.358 | 3.817 | 0 |
| Uniform, N=1M, Q=1,024, r=16, K=16, cell=16 | Fast | 3.783 | 1.475 | 252.831 | 3.689 | 0 |
| One-cell cluster, N=1M, Q=16, r=0.01, K=16, cell=16 | Safe | 1.651 | 376.175 | 242.007 | 0.019 | 0 |
| One-cell cluster, N=1M, Q=16, r=0.01, K=16, cell=16 | Fast | 2.797 | 233.255 | 243.720 | 0.019 | 0 |

Raw samples and input-transfer/memory records:
[uniform Safe](../bench/results/2026-10-02-m5-pro-morton-uniform-1m-safe.json),
[uniform Fast](../bench/results/2026-10-02-m5-pro-morton-uniform-1m-fast.json),
[cluster Safe](../bench/results/2026-10-02-m5-pro-morton-clustered-1m-safe.json),
[cluster Fast](../bench/results/2026-10-02-m5-pro-morton-clustered-1m-fast.json).
The benchmark source is [here](../bench/bench_spatial_morton.py). It compared
all returned first-K slots with cKDTree and separately checked eight query
rows against the existing native brute-force flat radius kernel. No mismatch
occurred in these random, finite, non-boundary fixtures. SciPy uses float64
geometry, so exact boundary and tie parity still require the float32 oracle.

The uniform result is promising for this **one** distribution and query
count. The cluster result is the counterexample: one thread scans a million
points in the same cell for each query. Safe Math is slower than cKDTree even
including CPU tree construction; Fast Math is only similar in one-shot time.
Do not extrapolate across the two rows because they use different Q and r.
The reported MPS allocation sample is about 44.7 MB of tensor storage, while
driver allocation is about 1.08 GB after PyTorch/shader initialization; the
latter is not an incremental index-memory estimate or a true peak. The
physical M1 is temporarily disconnected and has no result in this study.

## Decision

Keep the sorted-grid path as a bounded-radius prototype. The next structure
to test for the full v0.9 radius **and exact kNN** gate is a Morton-sorted
linear BVH over small point bricks with AABBs computed from actual points.
This lets a query prune an entire dense brick by a conservative distance
bound instead of scanning every point in a coarse cell. Its topology build,
AABB validity, sorted-first-K radius contract, kNN tie rule, memory use, and
M1/M5 crossover are not implemented or demonstrated yet. The package remains
at v0.8.0 until those gates and the Chamfer scope are complete.

## Experimental kNN brick slice

`bench/spatial_bricks.py` and `bench/spatial_bricks.metal` now add a bounded
research kNN path on the same Morton ordering. Each brick contains at most
128 points. A Metal build pass computes the actual per-brick minimum and
maximum coordinate; a query thread seeds from the nearest box, scans every
other box, and prunes a brick only when its lower bound is **strictly greater**
than the current Kth squared distance. This is intended to retain equally
distant points with smaller original indices; a floating-point bound proof
for arbitrary magnitudes and FTZ is still required. Distances use the flat
Metal kNN order:
`dx*dx + dy*dy + dz*dz`, with contraction disabled. The path currently accepts
single-cloud float32 XYZ and `0 <= K <= 32`; it returns squared distances and
indices with `inf`/`-1` padding. The index borrows the reference tensor; it
must remain unchanged between build and query.

This is not a hierarchical BVH: scanning every brick box costs `O(Q*N/128)`
even when point visits are pruned. Bricks can have overlapping wide AABBs,
especially with all points in one Morton cell, so this is not a general
performance result. The first gate is differential correctness against the
existing flat Metal kNN, including repeated exact ties across brick
boundaries. Full 1M kNN benchmarks, a conservative floating-point bound proof
for arbitrary magnitudes, clustered distribution handling, and M1 checks
remain open before public routing.

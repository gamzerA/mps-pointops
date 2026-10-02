# Sorted-Morton radius prototype at one million points

This is a **v0.9 development experiment**, not a public operator or release
benchmark. The measured source commit is
`1f1613a17c9b340464de92b6ae82180e8421d993` in the development branch.
Every raw JSON also records SHA-256 hashes for
the benchmark and the two experimental Metal stages. The source worktree was
clean during each measurement.
This page preserves the grid-stage research snapshot. Later work added an
[opt-in public `SpatialIndex`](spatial-api-v090.md) with two-level BVH kNN and
first-K Ball Query; its separate [radius audit](spatial-radius-dispatch-v090.md)
and [public dispatch study](spatial-dispatch-v090.md) supersede the status
statements below about those features.

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
for the full v0.9 radius **and exact kNN** gate is a Morton-sorted two-level
BVH over small point bricks with AABBs computed from actual points. This lets
a query prune an entire dense brick by a conservative distance bound instead
of scanning every point in a coarse cell. A private exact-kNN prototype now
implements the hierarchy and measures its M5 Pro crossover below. The
sorted-first-K radius traversal and limited public routing were still open at
the time of this grid study; they now have separate experimental BVH paths.
M1 validation and broader numerical and memory gates remain open. The package
remains at v0.8.0 until those gates and the Chamfer scope are complete.

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
boundaries. A separate private [two-level Metal BVH prototype](spatial-hierarchy-v0.9-design.md)
now measures hierarchical traversal. The separate `SpatialIndex` provides
limited opt-in/public routing; M1 validation and broader routing remain open.

### Bounded AABB-pruning contract and proof assumptions

The brick experiment accepts single-cloud float32 coordinates that are zero
or normal numbers with `|coordinate| <= 2^60`. It rejects NaN, infinity, and
nonzero subnormal **coordinate bits** through integer reinterpretation on MPS;
this check is independent of floating-point FTZ. It also requires all
reference Morton cells to fit 21 bits per axis, fewer than `2^32` points and
queries, and `0 <= K <= 32`. A point and a query in this coordinate domain
have `|x_i - q_i| <= 2^61`, so the sum of three squared differences is at most
`3*2^122`, below float32 overflow. The index holds the reference tensor; its
PyTorch version counter rejects tracked in-place changes after construction.
Raw external buffer mutation remains outside this prototype's contract.

For a Safe Math process (`PYTORCH_MPS_FAST_MATH=0` set **before** Python
starts), the pruning argument is conditional on the Metal compiler preserving
the explicitly staged subtraction, square, and left-to-right addition, with
monotone, sign-symmetric float32 rounding and consistent FTZ. The source
disables FP contraction. For every reference point in a brick, its actual
coordinate satisfies `lo_i <= x_i <= hi_i`: the bounds are min/max of the
original point values, while Morton sorting is only a permutation. Define
`p_i = fl32(x_i - q_i)` and
`l_i = max(fl32(lo_i - q_i), fl32(q_i - hi_i), 0)`.
Monotonicity gives `0 <= l_i <= |p_i|` for each axis. Applying the same
nonnegative float32 squares and additions in the same order yields
`S(l) <= S(p)`. Thus a brick is discarded only when its computed lower bound
is **strictly greater** than the current Kth squared distance; equality is
visited to preserve the lowest-original-index tie rule. Morton cell boundary
rounding cannot remove a point from its brick because the bounds are built
from every sorted point, not from the quantized cell extent.

The source-level proof does not certify arbitrary future Metal compiler
transformations. In Fast Math (`PYTORCH_MPS_FAST_MATH=1`) or an unspecified
mode, this prototype **disables AABB pruning and scans all bricks**. This
eliminates pruning misses, while the selected-pair distance and native full
scan must still be compared under the same mode for numerical parity. The
debug audit counts every would-be winning point inside each pruned brick on
small adversarial fixtures; zero observed audit violations support, but do
not replace, the stated conditional proof. Tests cover Morton boundary
neighbors (`nextafter(1, ±infinity)`), cross-brick exact ties, `4096^2+1+1`
association, tiny normal coordinates whose squares can flush, actual
subnormal-coordinate rejection, and the `2^60` finite-domain edge.

### kNN benchmark scope

`bench/bench_spatial_bricks.py` compares the experimental brick path with the
package's native Metal full-scan kNN and SciPy cKDTree. It synchronizes MPS
before and after each timed call and records raw samples, source commit and
SHA-256 hashes, input transfer separately, and a separate memory-only
allocator sampling pass. The reported sampled high-water is **not** a true
transient GPU peak; Instruments is still required for that measurement. CPU
RSS is a process-lifetime peak. SciPy uses float64 geometry and is only a
diagnostic index comparison; the same-mode native Metal indices and an
independent selected-pair Metal squared-distance kernel are the parity checks.
The default workload guard caps measured `N*Q` at 2,048,000,000 pairs.
`--allow-large-q` explicitly permits up to 65,536,000,000 pairs per run with
at most five synchronized repetitions, so Q=8192 or 65,536 at N=1M is never
silently launched.

### Clean-commit M5 Pro kNN scaling

The 15 raw JSON files in
[`bench/results/spatial-v090-m5pro-clean/`](../bench/results/spatial-v090-m5pro-clean/)
were rerun from clean source commit `a80b385e41f7ae80bfc28445ee3b970731ca3a1f`
(`source_dirty=false`). Each records source-file SHA-256 hashes and five raw
synchronized samples. These measurements are research evidence, not a public
operator performance guarantee.
All rows use 1,000,000 float32 references, `K=16`, seed `20261002`, and M5 Pro
Safe Math with MPS CPU fallback disabled. Uniform references lie in
`[0,1024)^3`, with Morton cell size 16. The mixed input places 90% of
references in a side-1 cube near `(512,512,512)` and 10% uniformly in the
larger cube; half the queries sample each population. Its **fine** Morton
cell size is `1/64`; a separate coarse cell-16 experiment is discussed below.
The table's queries are sampled from references in both cases, so each row
has a zero-distance self-neighbor and does **not** represent independent
model query coordinates. Coordinates are resident on CPU/MPS before timing;
transfer is recorded separately. Build and query
are timed separately. All displayed entries are medians of five raw samples
in milliseconds, excluding shader compilation.

| Input | Q | Brick build | Brick query | Metal full scan query | cKDTree build | cKDTree query |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 16 | 2.453 | 20.121 | 15.351 | 237.481 | 0.024 |
| Uniform | 256 | 2.439 | 54.736 | 14.313 | 206.637 | 0.455 |
| Uniform | 2,048 | 2.436 | 68.894 | 40.960 | 202.124 | 5.232 |
| Uniform | 4,096 | 2.508 | 59.084 | 74.446 | 207.470 | 11.833 |
| Uniform | 8,192 | 2.492 | 70.620 | 140.675 | 202.648 | 23.076 |
| Uniform | 65,536 | 2.476 | 204.133 | 1,074.867 | 229.824 | 210.605 |
| Mixed, fine cell | 16 | 2.453 | 60.145 | 15.246 | 205.043 | 0.025 |
| Mixed, fine cell | 256 | 2.481 | 147.610 | 15.951 | 258.223 | 1.127 |
| Mixed, fine cell | 2,048 | 2.514 | 133.843 | 40.537 | 238.067 | 7.418 |
| Mixed, fine cell | 4,096 | 2.443 | 129.423 | 78.695 | 222.359 | 11.720 |
| Mixed, fine cell | 8,192 | 2.466 | 134.629 | 146.036 | 212.552 | 24.693 |
| Mixed, fine cell | 65,536 | 2.504 | 362.907 | 1,094.657 | 207.362 | 192.778 |

For these sampled workloads, the brick **query** median first beats the
native Metal full scan at Q=4,096 for uniform data and Q=8,192 for the fine
mixed data. These are observed sampled-size crossovers, not a universal
threshold. In this run all five mixed Q=8,192 brick-query samples are below
all five native full-scan samples, but other devices and distributions remain
unmeasured. cKDTree query-only is faster on every mixed row; at uniform
Q=65,536 the GPU brick and cKDTree query raw ranges overlap. The **sum of separately
measured build and query medians** is lower for the GPU path than cKDTree on
these preloaded resident fixtures; it is not a directly timed total-call
sample. CPU-to-MPS transfer is excluded and can change the one-shot result.
The original coarse mixed cell size 16 forces most clustered points
into one Morton cell: its brick query medians are about 216, 212, and 209 ms
at Q=16, 256, and 2,048 respectively. Finer cells improve that counterexample
but do not make the brick path consistently faster than native Metal.

Each measured row returned **zero differing index slots** versus the same-mode
native Metal full scan, and zero bit differences in selected squared distances
versus an independent Metal probe. This is empirical evidence for these
fixtures, not a proof of every input. The mixed Q=8,192 and Q=65,536 rows
have 14 and 120 differing slots versus SciPy respectively; the recorded
classifier identifies **all** of these as exact equal-distance ties in both
float32 and float64. SciPy's tie order is not the native Metal contract. In randomized small
Safe-mode audits over three seeds, three cell sizes and three K values, no
candidate in an actually pruned brick would have beaten the current worst
pair. Fast-mode tests verify that no boxes are pruned.

At Q=65,536 the sampled PyTorch MPS allocator high-water is about 58.8 MB of
current tensor allocation and 1.08 GB of driver allocation. This includes
process initialization and other live tensors, and is neither incremental
index memory nor the true transient GPU peak. The physical M1 remains
disconnected and has no measurement here.

The benchmark also offers `--query-source independent` to remove the
self-neighbor bias. With the same mixed references and fine cell size, five
samples yielded:

| Independent mixed Q | Brick build | Brick query | Metal full scan query | cKDTree build | cKDTree query |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2.491 | 84.839 | 15.527 | 230.184 | 0.024 |
| 2,048 | 2.513 | 133.051 | 41.852 | 215.909 | 7.750 |
| 65,536 | 2.695 | 358.479 | 1,073.702 | 212.966 | 198.839 |

The first half of independent queries is sampled from the cluster's side-1
coordinate **range**, and the second half from the full background cube; no
query is intentionally copied from a reference row. All three rows matched
the native Metal index and selected-distance bits. SciPy differed by 0, 2,
and 107 slots; the benchmark's float32/float64 distance classifier found
every differing pair an exact equal-distance tie. This independent-query
measurement confirms the large-Q Metal-brute crossover for this fixture,
while cKDTree remains faster for repeated query-only use. Uniform independent
queries have so far been measured only at Q=16 (brick query 26.877 ms versus
Metal full scan 15.152 ms); the larger uniform independent matrix is still
unmeasured.

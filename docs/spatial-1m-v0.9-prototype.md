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
boundaries. The M1 check and an actual hierarchical acceleration structure
remain open before public routing.

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

### Development M5 Pro kNN scaling (pending clean-commit rerun)

The following values were collected from the current **dirty development
worktree**. Each JSON records the exact source-file hashes and five raw
synchronized samples, so these observations are reproducible against the
recorded files. They must be rerun from a clean code commit before publication.
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
| Uniform | 16 | 2.600 | 21.991 | 15.268 | 216.084 | 0.023 |
| Uniform | 256 | 2.486 | 51.332 | 14.524 | 211.354 | 0.401 |
| Uniform | 2,048 | 2.512 | 69.961 | 42.114 | 212.408 | 5.478 |
| Uniform | 4,096 | 2.634 | 58.520 | 81.469 | 241.715 | 12.424 |
| Uniform | 8,192 | 2.455 | 72.395 | 144.314 | 220.892 | 25.782 |
| Uniform | 65,536 | 2.506 | 205.629 | 1,115.544 | 258.446 | 223.324 |
| Mixed, fine cell | 16 | 2.450 | 61.486 | 14.886 | 207.364 | 0.026 |
| Mixed, fine cell | 256 | 2.585 | 132.148 | 15.110 | 214.254 | 0.630 |
| Mixed, fine cell | 2,048 | 2.495 | 133.226 | 43.277 | 208.068 | 5.802 |
| Mixed, fine cell | 4,096 | 2.548 | 129.620 | 77.355 | 213.087 | 13.094 |
| Mixed, fine cell | 8,192 | 2.420 | 136.420 | 145.398 | 223.686 | 32.785 |
| Mixed, fine cell | 65,536 | 2.546 | 380.810 | 1,092.182 | 221.667 | 205.372 |

For these sampled workloads, the brick **query** median first beats the
native Metal full scan at Q=4,096 for uniform data and Q=8,192 for the fine
mixed data. These are observed sampled-size crossovers, not a universal
threshold. The mixed Q=8,192 raw samples overlap between methods, so its
small median lead needs additional devices and runs. cKDTree query-only is
faster on every mixed row; at uniform Q=65,536 the GPU brick and cKDTree
query medians are close and their raw ranges overlap. The **sum of separately
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
have 14 and 120 differing slots versus SciPy respectively; a separate
diagnostic found all 14 Q=8,192 pairs are exact equal-distance ties in both
float32 and float64. SciPy's tie order is not the native Metal contract. The
benchmark now records such tie classification per run. In randomized small
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
| 16 | 2.609 | 87.759 | 15.641 | 245.210 | 0.025 |
| 2,048 | 2.536 | 125.569 | 41.107 | 206.996 | 6.578 |
| 65,536 | 2.532 | 343.532 | 1,075.874 | 208.388 | 240.307 |

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

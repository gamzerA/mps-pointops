# Morton-ordered AABB hierarchy: exact CPU contract and Metal gates

This design accompanies the **experimental** exact CPU
[`HierarchicalMortonReference`](../bench/spatial_hierarchy_reference.py) and
private Metal [`MortonTwoLevelBVH`](../bench/spatial_bvh.py). The Metal BVH is
now exposed through the unreleased, opt-in single-cloud
[`SpatialIndex`](spatial-api-v090.md); it does not replace the existing dense
or flat operators. M5 Pro results below are fixture-specific; the
physical M1 is currently unavailable and its gate remains pending.

## Structure and contracts

1. Validate finite XYZ coordinates. Quantize each axis to a configurable
   integer grid, interleave the bits into a Morton key, then sort by
   `(key, original_index)`. The key changes **only ordering**. It does not
   define the box used for pruning.
2. Pack adjacent Morton-sorted points into bricks of at most `leaf_size`.
   Compute each leaf AABB as the componentwise minimum and maximum of the
   actual stored coordinates. Build a balanced binary hierarchy over the
   bricks; an internal AABB is the union of its children. Each node also
   stores the minimum original point index in its subtree.
3. Radius query uses the strict geometric predicate `distance² < radius²`.
   Its limited result is the first `K` matching **original** indices, not
   Morton order or nearest-distance order. A bounded max-index heap retains
   those first `K`; a node with `min_index` no smaller than the current worst
   chosen index cannot improve the result.
4. kNN sorts by `(distance², original_index)`. Equal-distance candidates
   retain the smallest original index, regardless of tree/worker order.
   Equality of an AABB bound and the current Kth distance is **not** a reason
   to prune: a smaller original index might be inside that node.
5. After `max_node_visits`, the CPU reference visibly falls back to a full
   exact scan and increments a fallback counter. Empty input returns no
   matches; nonfinite coordinates and negative radius/K are rejected.

This is a balanced Morton-ordered brick tree, rather than a production
linear-BVH topology builder. Morton collisions and overlapping brick boxes
can still make a cluster approach a full scan. A two-level coarse/fine grid
is a candidate for scheduling, but its coarse keys must never be used as
under-approximated bounds.

## Why the CPU pruning is conservative

The CPU reference treats each finite binary floating-point input as its exact
dyadic rational value (`Fraction.from_float`). Let node box `A` contain every
descendant point `p`, with per-axis bounds `[lₐ,uₐ]`. For query `q`, define

`δₐ = max(lₐ − qₐ, 0, qₐ − uₐ)`, and `L(A,q) = Σₐ δₐ²`.

For every `p ∈ A`, `|pₐ − qₐ| ≥ δₐ`; consequently
`||p − q||² ≥ L(A,q)`. A leaf contains each stored point by actual-coordinate
min/max, and the child-union rule preserves containment by induction. The
quantized Morton key is absent from this inequality. The exact-rational CPU
implementation therefore can prune a radius node only if `L ≥ r²` under a
strict-radius contract. For kNN it prunes only if `L > τ`, where `τ` is the
current Kth distance. This also preserves exact-distance ties by index.

The exact CPU oracle is deliberately separate from the public Metal
float32 numerical contract. In particular, it does **not** establish
bitwise parity at a float32 distance/radius boundary merely by matching
Python binary64 brute force. The existing flat Metal kernels and their
documented rounding rules remain the GPU differential oracle.

### Float32 Metal gate before enabling AABB pruning

Actual-coordinate float32 min/max is necessary but insufficient. Rounding,
FMA contraction, overflow, and flush-to-zero can overestimate a computed
query-to-box lower bound or underestimate the candidate distance. An
overestimated lower bound can discard a true winner even when the box itself
contains every point. A production pruning predicate must use a proved
downward bound on the **same distance contract** as its candidate kernel,
including Safe/Fast compile settings and subnormals. If that certificate
cannot be established for an input (for example, an unresolved overflow,
nonfinite value, or an
unresolved underflow band), it must set the lower bound to zero or route to an
explicit full-scan path. An empirical epsilon alone is not a proof.

The acceptance test must compare full output indices and distances to flat
Metal brute force for uniform, clustered, sparse, duplicate, exact-tie,
near-tie, AABB-face, radius-boundary, subnormal and extreme finite inputs in
both math modes. Every prune can additionally be audited during development:
for each pruned node, brute-check all descendant points and assert that none
could change the result. This differential evidence can find faults but
cannot replace the conservative-bound proof for arbitrary inputs. The CPU
reference alone does not prove a Metal implementation. The private Metal BVH
uses the conditional Safe Math contract in the
[spatial prototype report](spatial-1m-v0.9-prototype.md) and rejects Fast
Math until that mode has an independently justified pruning bound.

For the currently supported `|coordinate|<=2**60` domain, every per-axis
difference is at most `2**61` and the squared-distance sum is bounded by
`3·2**122 < 2**124`, below float32's maximum finite value. Serial, split,
and public tests cover large finite distances and cross-microtree ties at
this bound. The shader also retains `+∞` candidates defensively if a later
version widens the domain; that branch is outside today's public contract.

## Work and memory

For `N` points, brick size `L`, and `M = ceil(N/L)` leaves, key generation
uses `O(N·bits)`, sorting `O(N log N)`, and AABB/tree build `O(N + M)` work.
The tree has at most `2M−1` nodes and `O(N+M)` storage. A query visiting `V`
nodes and examining `C` points costs `O(V log V + C log K)` for bounded
selection, plus its output; an unlimited radius output sorts its `E`
matches in `O(E log E)`. In the worst case `V=O(M)` and `C=N`; fallback is
another `O(N)` scan. Exact-rational CPU arithmetic is intentionally slow and
is excluded from GPU speed comparisons.

The Metal prototype implements Morton key generation, stable
key/index sorting, per-brick actual-coordinate AABB reduction, bottom-up
microtree and macro-tree union, and bounded per-query traversal. A stack
overflow returns a visible status and performs a full exact scan instead of
returning a partial row. The unreleased `SpatialIndex` routes explicit BVH
requests and a narrow measured M5 Pro kNN window to it; no flat/batched
public entry point or Fast Math path routes to this traversal.
Keep build, query, transfers and fallback count separate in measurements.

### Concrete two-level experiment

At one million points and 128 points per brick, the implementation pads the
7,813 bricks to 8,192 leaves. It groups them into 128 microtrees of 64 bricks
each and builds a macro tree over those roots. One Metal pass reduces
actual-coordinate leaf boxes and original-index minima within each
microtree; a second pass unions the macro nodes. Padded leaves are marked
empty. The tree has at most 16,383 nodes; its AABB and minimum-index buffers
occupy 458,752 bytes at this size.

The serial path assigns one query to one worker and visits nearer child boxes
first. Its strict `LB > Kth_distance` rule preserves exact-distance ties.
The optional split path supports `Q <= 256`: a seed leaf supplies a valid
Kth-distance upper bound, `Q × 128` microtree tasks each produce a local
top-K list, and a final kernel merges by `(distance, original index)`.
For `Q=16, K=16`, the local distance/index buffer is about 262 KiB before
other counters and alignment. The stronger equality prune
`LB == Kth_distance && min_original_index >= Kth_original_index` is safe if
the subtree minimum and bound are certified, but is not implemented here.

### M5 Pro clean-commit feasibility measurements

The [raw JSON matrix](../bench/results/spatial-bvh-v090-m5pro-clean/) was
generated from clean commit `5d2cf5986dcee43351e3c95d3a4a987a48ed3a4d`.
Each row is the median of five synchronized Safe Math runs at `N=1,000,000`,
`K=16`, seed `20261002`; inputs are resident before timing and shader
compilation is excluded. Uniform points span `[0,1024)^3` with Morton cell
size 16. Mixed data put 90% in a side-1 cube near `(512,512,512)` and 10%
uniformly in the larger cube, with cell size `1/64`. Sampled queries come
from references; independent mixed queries draw half from the cluster range
and half from the full cube. The collapsed fixture sets every coordinate to
the same value. All times below are **query-only milliseconds**.

| Input | Q | Split BVH | Serial BVH | Bricks | Metal full scan | cKDTree |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform, sampled | 16 | 6.14 | 18.04 | 20.59 | 15.06 | 0.02 |
| Uniform, sampled | 256 | 9.62 | 31.52 | 50.97 | 14.32 | 0.38 |
| Uniform, sampled | 2,048 | — | 29.99 | 67.66 | 40.19 | 6.15 |
| Uniform, sampled | 4,096 | — | 29.89 | 58.93 | 75.64 | 11.20 |
| Uniform, sampled | 8,192 | — | 31.98 | 70.45 | 143.62 | 26.87 |
| Uniform, sampled | 65,536 | — | 167.87 | 200.37 | 1,086.47 | 211.21 |
| Mixed, sampled | 16 | 5.97 | 30.87 | 61.94 | 15.48 | 0.02 |
| Mixed, sampled | 256 | 11.33 | 53.59 | 132.35 | 15.36 | 0.77 |
| Mixed, sampled | 2,048 | — | 54.01 | 132.95 | 42.36 | 5.48 |
| Mixed, sampled | 4,096 | — | 57.09 | 126.82 | 75.09 | 12.07 |
| Mixed, sampled | 8,192 | — | 53.44 | 132.01 | 145.49 | 23.37 |
| Mixed, sampled | 65,536 | — | 218.07 | 355.57 | 1,090.18 | 188.76 |
| Mixed, independent | 16 | 6.68 | 28.64 | 80.70 | 14.40 | 0.02 |
| Mixed, independent | 2,048 | — | 55.00 | 125.54 | 41.09 | 7.15 |
| Mixed, independent | 65,536 | — | 218.46 | 366.35 | 1,094.09 | 237.44 |
| Uniform, independent | 16 | 5.50 | 14.91 | 29.64 | 15.48 | 0.02 |
| Collapsed | 16 | 2.96 | 229.15 | 224.19 | 15.45 | 19.18 |
| Collapsed | 256 | 9.46 | 235.52 | 231.20 | 14.57 | 254.13 |

The split path reverses the low-Q brick regression on these M5 fixtures.
For the mixed sampled fixture, the **serial** hierarchy first beats Metal
full scan at Q=4,096, while it still loses at Q=2,048; the optional split
path currently stops at Q=256. cKDTree query-only remains faster for most
rows and can reuse a previously built tree. GPU build median is roughly
2.4–3.0 ms in this matrix; cKDTree build is recorded separately in each
JSON. A sum of build and query medians is not a directly timed end-to-end
sample, and CPU-to-MPS transfer is excluded. The mixed Q=65,536 sampled
query-only row still favors cKDTree over this BVH.

Every row has zero BVH/native and brick/native index mismatches, zero
BVH/brick squared-distance bit mismatches, and zero recorded stack
fallbacks; split rows also have zero split/native and split/brick mismatches.
These are fixture-specific differential checks. In the mixed Q=65,536 row,
serial traversal visits a median 7,936 points per query, p95 23,296 and
maximum 36,608, versus one million for full scan. The collapsed serial
counterexample visits all one million; splitting parallelizes that work but
does not reduce its total comparisons. The sampled MPS allocator high-water
at Q=65,536 is about 81.5 MB for live tensors and 1.08 GB for driver
allocation. It includes process caches, is **not** incremental BVH memory,
and is **not** a true transient GPU peak. Instruments profiling, M1, Fast
Math, and wider input-domain proof remain release gates.

## Measurements required for broader public kNN/radius routing

The [public dispatch matrix](spatial-dispatch-v090.md) already supports a
narrow M5 Pro Safe-Math kNN automatic rule. The [radius matrix](spatial-radius-dispatch-v090.md)
measures the opt-in BVH path, while automatic radius routing remains on scan.
The following axes remain useful before expanding either automatic policy.

At `N=1,000,000`, measure `Q={16,256,2048,8192,65536}` where feasible, with
uniform points, a dense cluster plus sparse background, and a collapsed or
near-collapsed cluster. Include queries inside the cluster, in the sparse
background and outside occupied space. Record distribution parameters,
radius/K, number of occupied cells, node/point visits per query (including
p50/p95/max), fallback counts, and full output mismatch counts. This reveals
the crossover as query count increases and the worst-case divergence hidden
by a uniform-only median.

Compare synchronized Metal index build, query and end-to-end times with
resident-data PyTorch brute force and `scipy.spatial.cKDTree` build/query.
Record input transfer separately. Report tensor allocation, driver
allocation and a measured peak where the platform exposes one; the latter
two are not interchangeable. A full `Q×N` brute run may exceed practical
time or memory, so cap it explicitly and use deterministic complete scans
for sampled queries plus smaller exhaustive fixtures. Never infer exact
correctness from cKDTree alone at float32 ties or radius boundaries.

The first M5 Pro matrix is recorded in the linked studies; extend it with
larger radii and boundary-heavy data. Repeat on the physical M1 when it reconnects;
do not substitute an emulator or project an M5 crossover onto M1.

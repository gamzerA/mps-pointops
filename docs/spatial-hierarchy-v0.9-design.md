# Morton-ordered AABB hierarchy: exact CPU contract and Metal gates

This design accompanies the **experimental**, CPU-only
[`HierarchicalMortonReference`](../bench/spatial_hierarchy_reference.py). It is
not wired into the public package and has no measured GPU speed claim. The
physical M1 is currently unavailable; its gate remains pending.

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
cannot be established for an input (for example, overflow, nonfinite or an
unresolved underflow band), it must set the lower bound to zero or route to an
explicit full-scan path. An empirical epsilon alone is not a proof.

The acceptance test must compare full output indices and distances to flat
Metal brute force for uniform, clustered, sparse, duplicate, exact-tie,
near-tie, AABB-face, radius-boundary, subnormal and extreme finite inputs in
both math modes. Every prune can additionally be audited during development:
for each pruned node, brute-check all descendant points and assert that none
could change the result. This differential evidence can find faults but
cannot replace the conservative-bound proof for arbitrary inputs. **Metal
AABB pruning is not enabled or claimed correct by this CPU prototype.**

## Work and memory

For `N` points, brick size `L`, and `M = ceil(N/L)` leaves, key generation
uses `O(N·bits)`, sorting `O(N log N)`, and AABB/tree build `O(N + M)` work.
The tree has at most `2M−1` nodes and `O(N+M)` storage. A query visiting `V`
nodes and examining `C` points costs `O(V log V + C log K)` for bounded
selection, plus its output; an unlimited radius output sorts its `E`
matches in `O(E log E)`. In the worst case `V=O(M)` and `C=N`; fallback is
another `O(N)` scan. Exact-rational CPU arithmetic is intentionally slow and
is excluded from GPU speed comparisons.

The Metal mapping is: Morton key generation; stable key/index sort; per-brick
actual-coordinate AABB reduction; bottom-up internal AABB reduction; then
query traversal with bounded per-query state. Queries that overflow the node
or candidate budget must return a visible status and enter an exact fallback
path. Large `Q` requires enough query parallelism; a single serial traversal
per query and a giant cluster may still cause SIMD divergence. Keep build,
query, transfers and fallback count separate in measurements.

### Concrete two-level experiment

At one million points and 128 points per brick, pad the 7,813 bricks to 8,192
leaves. Group them into 128 microtrees of 64 bricks each, and build a macro
tree over those 128 roots. A Metal build pass can reduce actual-coordinate
leaf boxes and original-index minima within each microtree; a second pass
unions the macro nodes. The padded leaves are marked empty. This gives at
most 16,383 nodes and avoids quantized cell faces as pruning bounds.

For `Q >= 256`, first measure one query per worker with near-child-first
depth-first traversal and a bounded local stack. A node can be skipped when
its certified float32 lower bound exceeds the current Kth distance. At equal
distance it can also be skipped when its minimum original index is no smaller
than the Kth chosen index; this rule requires a correct subtree index
minimum. Safe and Fast Math need separate proofs or a full-scan fallback.
For `Q = 16`, compare that serial traversal with a split by microtree:
`Q × 128` independent tasks produce local top-K lists, followed by one
deterministic `(distance, original index)` merge per query. The extra buffer
for `Q=16, K=16` is roughly 262 KiB for 64-bit pairs before alignment, and
its extra dispatch may outweigh the parallelism. These are proposed Metal
experiments, not results from the CPU reference or an enabled public path.

## Measurements required before routing public kNN/radius

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

Run the matrix on M5 Pro first. Repeat on the physical M1 when it reconnects;
do not substitute an emulator or project an M5 crossover onto M1.

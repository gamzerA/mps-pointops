# v0.9–v1.0 acceptance plan

This plan starts from the archived v0.8.0 tag. A target version is a proposed
scope, not a claim that its implementation, benchmark, or release exists. Mark
an item complete only with a source commit, raw test output, hardware record,
and a public artifact. Do not advance the package version merely because a
prototype passes a small fixture.

## Release disposition

The untagged v0.9.0 and v0.10.0 milestones remain proposed full scopes.
The combined v1.0.0 release freezes the documented, tested public API subset
in the [migration and version policy](../site-docs/migration.md), with the
[tested support matrix](../site-docs/support.md) defining device coverage.
Incomplete Chamfer combinations, the bounded opt-in spatial backend, and
private sparse research paths retain their stated limitations. Publishing
v1.0.0 does not mark all of the research gates below complete.

## v0.9.0: spatial search and Chamfer expansion

### Spatial index

The first experiment is a **sorted-Morton uniform grid** for bounded-radius
queries. Its Metal key stage, MPS stable sort, and Metal query stage are in
[`bench/spatial_keys.py`](../bench/spatial_keys.py) and
[`bench/spatial_radius.py`](../bench/spatial_radius.py); neither is connected
to the public API. It preserves the first-K original-index rule on tested
finite float32 inputs. Keep brute force as the oracle and an explicit
degenerate fallback. The final target is a **Morton-sorted linear BVH over
small point bricks**: leaf AABBs must use actual float32 coordinates rather
than treating quantized Morton cells as exact bounds. This avoids the grid's
unbounded same-cell candidate scan while supporting both radius and exact
kNN. Query pruning must use conservative AABB lower bounds; build and memory
costs need measurement before adoption. A private
[two-level Metal BVH](spatial-hierarchy-v0.9-design.md) now implements exact
kNN and first-K dense Ball Query for a bounded single-cloud Safe Math domain;
the [opt-in SpatialIndex facade](spatial-api-v090.md) keeps the established
dense and flat APIs unchanged. This is development scope, not v0.9 completion.

The proposed full pipeline is key generation on Metal, device sorting,
leaf/topology/AABB construction, then Metal query traversal. Sorting a million
arbitrary keys is not assumed to fit in one or two dispatches. Report every
dispatch and every PyTorch/MPS operation in build time. An index may be reused
across queries; publish both build-plus-one-query and amortized repeated-query
timings. The original grid prototype uses sorted-key binary searches and has
no CSR, batch layout, or kNN traversal. The separate BVH has Metal leaf/macro
builds and query traversal, but no public flat/batched routing. The physical
M1 results below cover measured fixtures, not a complete release gate.
The initial [CPU research grid](../bench/spatial_grid_reference.py) checks
candidate coverage, original-index ordering, exact kNN termination, and
explicit degenerate fallback. It uses binary64 distances and is **not** a
Metal speed result or the flat API's final float32 numerical contract.
The [clean-commit M5 Pro study](spatial-1m-v0.9-prototype.md) measured five
Safe/Fast samples at 1M points and found exact first-K agreement for its
random fixtures. Uniform 1M-point data with 1,024 queries took 4.080 ms to
build and 2.589 ms to query in Safe Math, compared with 252.358 ms plus
3.817 ms for cKDTree. With 1M points concentrated in one cell and 16 queries,
the grid query took 376.175 ms in Safe Math, exceeding cKDTree's 242.007 ms
build plus 0.019 ms query. These are prototype-specific workloads and do not
complete the radius/kNN or cross-device release gates; they motivated the BVH.

The [M5 Pro BVH study](spatial-hierarchy-v0.9-design.md) measures
`Q=16..65,536` on uniform, clustered-plus-sparse, and collapsed point
clouds. Exact native-Metal neighbor indices and selected squared-distance
bits agree on the recorded kNN fixtures, while the collapsed case shows why
a serial BVH cannot be dispatched by Q alone. The private BVH Ball Query
additionally preserves original-index first-K order under the dense radius
threshold, with explicit full-scan recovery on stack overflow. The
[memory study](spatial-memory-v090.md) records PyTorch allocator peaks that
include transient tensor allocations. The [M5 Pro Instruments follow-up](spatial-instruments-v090.md)
adds six target-process Metal allocation traces and a separate sampled
process-footprint pilot. These establish observed resource-allocation and
process-memory measurements, not an exact whole-device physical GPU peak.
A subsequent [physical M1 study](spatial-m1-v090.md)
records clean-source spatial parity, synchronized 1M-point timings, SciPy
`cKDTree` build/query comparisons, and allocator counters on 8 GiB Apple M1.
It does not provide an Instruments physical-memory trace or close the full
release gate.
The [M5 Pro branch regression logs](evidence/spatial-v090/README.md) preserve
separate Safe and Fast runs with CPU fallback disabled; Fast Math skips the
Safe-only BVH contract tests. The original
[nine M5 Pro radius-performance JSONs](../bench/results/spatial-radius-dispatch-v090-m5pro-clean/)
and [twelve M1 radius-dispatch JSONs](../bench/results/spatial-v090-m1/)
check output parity but predate per-query stack-fallback reporting. Four later
[M5 Pro radius-crossover records](../bench/results/spatial-radius-crossover-v090-m5pro/)
do report an untimed diagnostic count: zero fallback rows among 4,096 queries
at each tested radius. The [bounded-stack argument](spatial-radius-dispatch-v090.md#bounded-dfs-stack-and-fallback-evidence)
shows overflow is unreachable with the present balanced tree and `N≤1M`
limit; this is a source-constant-dependent claim, not a measurement for the
legacy fixtures. Archive their per-query diagnostic counts, including a
physical M1 run, before calling that measurement gate complete.

For the original grid radius path, inspect every cell intersecting the query
ball and check the original coordinates with the existing radius comparison policy. Cell
occupancy and hash collisions must never silently discard candidates. The
flat `radius` API returns the **first** `max_num_neighbors` matches in original
reference order, so a spatial traversal must select by original index before
writing its result. The set alone is insufficient for API parity. Dense Ball
Query and PyTorch3D have their own documented boundary and padding rules.
The BVH instead traverses actual-point node AABBs and preserves the same
original-index output rule for its bounded dense Ball Query path.

For exact grid kNN, expand cells until the lower bound on the distance to every
unvisited cell exceeds the current kth distance; BVH kNN applies the same
certificate to unvisited nodes. Resolve equal rounded
distances by the existing lowest-index rule. If this proof or a bounded-key
representation cannot be established, use an explicit brute-force fallback
and report its frequency. Nonfinite coordinates, duplicate points, zero or
very large radii, coordinates near cell boundaries, variable batches, and
grid-key overflow need separate tests. Grid quantization must conservatively
include possible boundary cells; candidate distances are evaluated in the
operator's actual floating-point precision.

The benchmark gate fixes `N=1,000,000`, several stated query counts rather
than silently treating `Q=N`, `k`, radius, distribution, dtype, batch layout,
input order, and cold/warm index use. Record synchronized median and raw
samples, peak tensor and driver allocations, build/query/compaction times,
chip, OS, PyTorch, math mode, and exact source hashes. Compare against SciPy
`cKDTree` including its tree build for the one-shot result. The CPU tree uses
different arithmetic; report exact index agreement on separated fixtures and
set agreement outside a documented floating-point boundary band. Ties must
be checked against this package's deterministic brute-force oracle rather
than treated as SciPy ordering errors. A speedup claim requires repeated M5 Pro
and physical M1 measurements with uncertainty intervals and no memory failure.
The physical M1 has now been measured for the stated fixtures; its
[raw records and limitations](spatial-m1-v090.md) are part of this plan.
The broader speedup and exact physical-memory research gates remain open;
the bounded v1.0 support scope is defined by the release disposition above.
For the profiling evidence, report supported GPU execution intervals, tensor
allocator peaks, observed Metal allocation maxima, and sampled process
footprint under their own definitions. Runtime Occupancy is optional when a
short counter-availability probe is rejected: preserve the failure and mark
it uncollected. Threadgroup size, SIMD width, and declared threadgroup storage
are source/resource descriptors, not measured Occupancy or a derived
Occupancy upper bound. A GPU pass interval must not be relabeled as an
individual shader time unless the trace establishes that correspondence.
On an 8 GB M1, verify the chip, memory, macOS and PyTorch versions before
testing. Run each Safe-Math, CPU-fallback-disabled fixture in a fresh child
process with a recorded wall-time limit and allocator/driver checkpoints.
Increase from 20k points and 256 queries through 100k points before trying
1M points; raise `Q` only after the preceding size finishes without a Metal
watchdog, out-of-memory error or excessive measured memory use. Keep the
M1 automatic policy on scan until its own crossover matrix is measured.
Record a failure as a failed gate, rather than dropping that fixture from the
published table. The M5 Pro timing or 48 GB memory result is not a substitute
for this run.

### Chamfer

Preserve the v0.8 squared-L2 CI gate. Add and independently gate:

1. L1 nearest-neighbor **ranking**, returned distance, and selected-pair
   backward. PyTorch3D's pinned L1 cusp convention chooses `-1` at equal
   coordinates; a generic `sign(0)=0` derivative is not equivalent.
2. Normal loss `1 - abs(cosine)` or `1 - cosine` according to `abs_cosine`,
   including zero vectors, gathered reference normals, masking, reductions,
   weights, and first-order normal gradients.
3. Optional PyTorch3D `Pointclouds` unpacking, including object/tensor mixes,
   ragged lengths, normal precedence, and gradients to the original leaves.

The pinned upstream source, exceptional zero-weight shapes, unsupported empty
clouds, and all acceptance cases are in
[the v0.8 scope decision](chamfer-api-scope-v0.8.md). Declare *full PyTorch3D
Chamfer compatibility* only if every supported upstream argument combination,
return structure, error path, and first-order derivative is tested. Otherwise
label the exact supported subset. Safe and Fast Math run in separate processes
with MPS CPU fallback disabled.
The L1 prototype has passed the [direct pinned-upstream parity matrix](chamfer-l1-v0.9-prototype.md)
on CPU and M5 Pro Safe/Fast. Normal loss and optional `Pointclouds` inputs
have a separate [development parity matrix](chamfer-normals-pointclouds-v0.9-prototype.md)
from a clean source commit. These finite float32 fixtures do not establish
every upstream argument/error combination or upstream parity on physical M1,
so the full-compatibility release gate remains open.
The [physical M1 internal Chamfer logs](evidence/chamfer-v090-m1/) record
76 passed and 2 skipped in each math mode. PyTorch3D was absent from that
initial environment, so its upstream parity cases were skipped. A subsequent
[direct M1 upstream archive](evidence/chamfer-v090-m1-upstream/README.md)
from the same clean source commit passed 480 base and 252 extended cases in
each Safe/Fast mode after a pinned CPU extension was built. This resolves the
physical M1 parity gap for those finite float32 matrices. It does not cover
every upstream argument and error combination or close the full-compatibility
gate.

### Optional optimal transport research

Specify entropic optimal transport separately from exact Earth Mover's
Distance. A log-domain Sinkhorn reference should fix the mass normalization,
ground cost, regularization strength, tolerance, maximum iterations, stopping
criterion, and differentiation mode before a Metal implementation. This is a
stretch research result; it does not block spatial or Chamfer release gates.

## v0.10.0: sparse 3D convolution

Use `spconv`'s actual distinctions: `SubMConv3d` preserves active coordinates;
`SparseConv3d` is the ordinary/strided sparse convolution;
`SparseInverseConv3d` reuses a saved `indice_key` to restore the prior active
set; `SparseConvTranspose3d` is a distinct generative transpose operator.
Do not use “inverse” and “transposed” as synonyms. The source-compatible
`SparseConvTensor` contract starts with features `[A,C]`, integer indices
`[A,4]` with batch index first, spatial shape, and batch size.

Implement and gate the sparse tensor validator and deterministic rulebook
first, then submanifold forward/backward, then strided and inverse paths.
Specify duplicate-coordinate policy, output ordering, indice-key reuse,
kernel-center convention, padding/stride/dilation, bias, dtype, and first-order
gradient semantics. An MPS-compatible dense PyTorch oracle checks every
small-coordinate mapping and gradient. A pinned `spconv` 2.x CUDA oracle on a
separate Linux/NVIDIA host checks source compatibility: its macOS CPU wheel is
not an assumed dependency. Report the actual tested settings alongside
`rtol=1e-4, atol=1e-5`. Use OpenPCDet's `VoxelBackBone8x` as the first
submanifold/strided integration target, and its `UNetV2` to exercise
`SparseInverseConv3d` before claiming the full milestone. Pin the OpenPCDet
commit, configuration, fixture, seed, dtype, and convolution algorithm;
compare coordinate sets before aligning feature rows and comparing outputs,
input gradients, and weight gradients. If the CUDA oracle or model is
unavailable, retain an experimental label.

The current bounded implementation has SubM, ordinary Strided and saved-key
Inverse paths. The [Windows RTX 2080 oracle](verification/spconv-toy-rtx2080.json)
records three tiny `spconv` 2.3.8 fixtures with 15 output/first-gradient
comparisons; the [physical M1 archive](evidence/m1-sparse-2026-10-03/README.md)
records 92 passing tests in each of Safe and Fast Math at source `6959fd59`,
without CPU fallback or skips. The [OpenPCDet integration](sparse-openpcdet-local-integration.md)
is a fixed synthetic local CPU/MPS comparison. These establish the stated
operator and fixture coverage. Whole-model upstream CUDA parity and a full
`spconv` API replacement remain outside the validated scope, so the private
sparse interfaces retain their experimental label.

## v1.0.0: API and support contract

Prepare upstream-ready patches and tests for the open
[`pyg-lib` discussion](https://github.com/pyg-team/pyg-lib/issues/733),
[`pytorch_cluster` discussion](https://github.com/rusty1s/pytorch_cluster/issues/172),
and [`pytorch3d` discussion](https://github.com/facebookresearch/pytorch3d/issues/2049).
Ask maintainers which integration point they prefer before duplicating the
same operators in both PyG projects. Submission and response history are under
this project's control; acceptance or merging by another project is not a
release gate.

Publish an operator/API and numerical-contract site, migration notes, and a
versioned device matrix. Freeze only public APIs whose behavior and supported
input ranges have tests. State a minimum macOS and an explicit tested PyTorch
version set after CI has exercised them; a broad `2.7–2.12+` promise would
include untested releases and omit newer ones. Treat M2–M4 as unverified until
physical devices are measured. The v1.0 tag requires green required CI,
reproducible release artifacts, source/version/DOI agreement, and no open
release-blocking correctness issue. Terms such as “production ready” and
“100% compatible” are not inferred from a passing synthetic model alone.

## Sources used for scope

- [PyTorch3D Chamfer source](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/loss/chamfer.py)
- [spconv usage and operator distinctions](https://github.com/traveller59/spconv/blob/master/docs/USAGE.md)
- [spconv 2.x CPU/platform notes](https://github.com/traveller59/spconv/blob/master/docs/SPCONV_2_BREAKING_CHANGEs.md)
- [OpenPCDet sparse backbone](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/spconv_backbone.py)
- [OpenPCDet sparse U-Net](https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/backbones_3d/spconv_unet.py)
- [PyTorch MPS documentation](https://docs.pytorch.org/docs/stable/notes/mps.html)

# Reusable spatial search API (v0.9 development)

`mps_pointops.SpatialIndex` is an **unreleased, opt-in** single-cloud API in
the v0.9 development branch. The existing dense `knn` and `ball_query` and
the flat/PyG adapters keep their signatures and default kernels. Building an
index does not register a global replacement or silently move an MPS tensor
to CPU.

```python
import torch
from mps_pointops import SpatialIndex

points = torch.rand(1_000_000, 3, device="mps")
query = torch.rand(8192, 3, device="mps")
index = SpatialIndex(points)  # auto; lazily builds an index only if selected
euclidean_distances, knn_indices = index.knn(query, 16)
squared_distances, first_indices = index.ball_query(query, 0.1, 16)
```

`points` has shape `[N,3]`; each query has shape `[Q,3]`. `knn` requires
`0 <= K <= N` and returns `[Q,K]` Euclidean distances and int64 reference
indices. The MPS path orders ties by original reference index; CPU tensors
retain the existing PyTorch `cdist`/`topk` reference behavior.
`ball_query` takes a Python scalar radius and `K>=0`; on MPS it returns
squared float32 distances and int64 indices in **first matching
original-reference index** order. The CPU scan retains the input dtype for
distances. It uses the strict PyTorch3D-style radius test
`distance² < fl32(fl32(radius) * fl32(radius))`; unused slots are `(0,-1)`.
Selection indices have no gradient. The BVH Ball Query path propagates
first-order squared-distance coordinate gradients to query and reference,
including repeated reference matches; radius has no gradient. Higher-order
gradients are outside this API's supported contract. Input points are
borrowed, and in-place mutation after construction raises an error.

| `backend` | Selection | Unsupported BVH input |
| --- | --- | --- |
| `"scan"` | Existing dense Metal full scan for MPS; existing PyTorch reference for CPU | N/A |
| `"bvh"` | Two-level Morton BVH for both kNN and Ball Query | Explicit error |
| `"auto"` (default) | Narrow measured M5 Pro kNN rule below; otherwise scan | Remains on the existing scan path |

The explicit BVH accepts one MPS float32 cloud, `N<=1,000,000`, `K<=32`,
finite normal-or-zero coordinates with `|coordinate|<=2**60`, a valid 21-bit
Morton domain, and `PYTORCH_MPS_FAST_MATH=0` **set before Python starts**.
Finite coordinates can still produce `+∞` float32 squared distance; kNN
ranks such candidates by original index, matching the dense Metal sentinel
rule, instead of treating them as absent.
These are research bounds, not hardware limits. The optional `origin` and
`cell_size` constructor arguments control Morton **ordering**, never the
distance predicate. By default the origin is the coordinate minimum and a
deterministic 4,096-point sample chooses a cell size; the full span reserves
space inside the 21-bit domain. A domain failure raises for explicit BVH and
keeps the scan path in automatic mode. The BVH reuses its sorted index across
queries; include construction cost for one-shot comparisons.

## Adaptive dispatch, narrowly scoped

For kNN only, `auto` selects BVH if all of these conditions hold: the chip is
the measured **Apple M5 Pro**; Safe Math is explicit; `N=1,000,000`, `K=16`,
and `8192<=Q<=65536`; both tensors are MPS float32; the bounded BVH input
checks pass; and no Morton cell contains at least 5% of a deterministic
4,096-point sample. The sample-key readback is at most 32 KiB; grid sizing,
domain checks, and key validation also synchronize the host. It is a speed
heuristic that rejects the measured all-coincident regression, **not** a proof
that the selected path is faster for every distribution. On M1 or any
unmeasured chip, Fast
Math, other sizes/K, and every Ball Query call, `auto` retains GPU full scan.
Use `backend="bvh"` to evaluate the hierarchy deliberately. For `Q<=256`,
the BVH kNN method uses the tested split-microtree query unless
`parallel_microtrees=False` is passed.

The [M5 Pro hierarchy matrix](spatial-hierarchy-v0.9-design.md) shows why
one `Q` threshold is insufficient: mixed `Q=2048` serial BVH loses to native
Metal full scan, while `Q=4096` wins; the all-coincident serial case loses
even at larger `Q`. The separate [public dispatch study](spatial-dispatch-v090.md)
includes the cell-size selection and dispatch overhead once its measurements
are frozen. This policy is deliberately narrower than the private prototype.

## Numerical and release boundaries

The AABB pruning proof is conditional on the documented float32 Safe Math
arithmetic and bounded input domain; see the [proof and adversarial tests](spatial-1m-v0.9-prototype.md).
The index stores actual-point AABBs rather than rounded Morton cell extents.
The BVH's first-K radius traversal preserves original index order and falls
back to a complete scan on bounded-stack overflow. Source-level reasoning
and differential tests do not certify every future Metal compiler or input.

This API does not route batched dense tensors, flat `torch_cluster`/PyG
edges, float16, or feature-space `D>3` through the BVH. Those callers keep
their established kernels and numerical contracts. In particular, flat
`radius` uses a different radius-square rounding rule. The [allocator memory
study](spatial-memory-v090.md) reports PyTorch tensor peaks, while a total
GPU physical-memory peak still requires an Instruments trace. M1 is
temporarily unavailable, so neither cross-chip automatic routing nor v0.9/
v1.0 release is justified by this M5-only study.

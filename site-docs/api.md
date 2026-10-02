# Public API and bounded research paths

Signatures below refer to the current repository source. Install a tagged release
for a frozen package snapshot. Unless stated otherwise, MPS coordinate kernels
use float32 inputs and keep their outputs on the input device.

## Dense tensors

| Call | Input | Result | Boundary |
| --- | --- | --- | --- |
| `furthest_point_sample(xyz, npoint, start_idx=0, skip_near_origin=False, *, strategy="auto")` | `[B,N,3]` | `[B,npoint]` int64 indices | Exact smaller-index tie rule. The opt-in `multigroup` path requires `B=1` when sampling more than one point; automatic use is limited to the measured M5 Pro policy. |
| `knn(query, ref, k)` | `[B,Q,D]`, `[B,N,D]`, `D>=1` | Euclidean float32 distances and int64 indices, each `[B,Q,k]` | MPS `k<=256`; ties use the smaller reference index. Neighbor selection has no coordinate gradient. |
| `ball_query(query, ref, radius, K)` | `[B,Q,3]`, `[B,N,3]` | Squared float32 distances and int64 indices, each `[B,Q,K]` | Strict radius, first `K` **in input order**, `(0,-1)` padding. Selected distances propagate coordinate gradients. |
| `three_nn(unknown, known)` | `[B,N,3]`, `[B,M,3]` | Euclidean float32 distances and int32 indices, each `[B,N,3]` | Requires at least three known points for nonempty queries. Selection and distances do not differentiate coordinates. |
| `three_interpolate(features, indices, weights)` | `[B,C,M]`, `[B,N,3]`, `[B,N,3]` | Float32 features `[B,C,N]` | Caller supplies weights. Backward accumulates feature gradients; requested weight gradient is zero by the PointNet++ wrapper contract. |
| `chamfer_distance(x, y, ..., norm=2, ...)` | `[B,P,D]`, `[B,Q,D]` | Loss and optional normal loss | Bounded PyTorch3D-style subset: `norm=1` or `2`, optional lengths/weights/reductions; normals only for `D=3`. MPS float32. Check the [contract](https://github.com/gamzerA/mps-pointops/blob/main/docs/chamfer-contract.md) before assuming a particular `Pointclouds` or reduction combination. |

The separate `mps_pointops.pytorch3d.ball_query` adapter accepts the documented
PyTorch3D argument names and returns a `KNN(dists, idx, knn)` tuple. Its input
and dimensional scope is narrower than all of PyTorch3D. The
[Ball Query contract](https://github.com/gamzerA/mps-pointops/blob/main/docs/ball-query-math.md)
specifies dtype, padding, lengths, backward, and the cube-filter hint.

## Flat tensors and graph adapters

`mps_pointops.flat.fps`, `.knn`, and `.radius` accept `[total_points,D]`
coordinates with batch vectors or offsets. Their edge result is `[2,E]` with
row 0 holding global query indices and row 1 holding global reference indices.
`flat.fps` instead returns global sample indices. `flat.knn` accepts feature
dimension `D>=1`, while `flat.radius` is 3D. MPS effective kNN width above 256
raises an error rather than copying the work to CPU. Flat radius uses the
`torch_cluster` threshold rule, which differs by one ULP from dense Ball Query
for some decimal radii. See [numerical contracts](numerical.md).

The opt-in `mps_pointops.compat` shim and `mps_pointops.pyg` registration cover
documented subsets of `torch_cluster` and PyG 2.8 operators. They are not a
promise that arbitrary imports, versions, or models are drop-in compatible.
The pinned [compatibility matrix](https://github.com/gamzerA/mps-pointops/blob/main/docs/phase3-compatibility-matrix.md)
states the tested calls.

## Voxel and spatial modules

`mps_pointops.voxel.voxelize(pos, size, batch=None, *, start=None, end=None)`
returns compact cell rows, inverse indices, and CSR mappings. The
`voxel_downsample(..., features=None, feature_reduce="mean",
pool_backend="index_add")` path aggregates positions and optional features;
`pool_backend="fused_csr"` is an **experimental opt-in** MPS reducer. Integer
cell maps have an exact ordering contract; floating-point reductions are
compared with tolerances. See the [voxel contract](https://github.com/gamzerA/mps-pointops/blob/main/docs/voxel-api-contract.md).

`SpatialIndex(points, backend="auto")` borrows one `[N,3]` cloud and exposes
`.knn(query, k)` and `.ball_query(query, radius, K)`. Its explicit BVH path is
**unreleased research scope**, limited to MPS float32, `N<=1,000,000`,
`K<=32`, bounded finite coordinates, and Safe Math explicitly set before
Python starts. Automatic routing uses the BVH only for a narrow measured M5
Pro kNN case; other calls retain the scan path. Read the
[spatial API contract](https://github.com/gamzerA/mps-pointops/blob/main/docs/spatial-api-v090.md).

The sparse convolution rulebook and SubM modules are **private prototypes**.
They are not exported as a public `spconv` replacement. Strided and inverse
Metal convolution, transpose convolution, and model-level equivalence remain
release gates. See [sparse convolution scope](https://github.com/gamzerA/mps-pointops/blob/main/docs/sparse-conv-foundation-v0.10.md).

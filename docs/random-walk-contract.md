# Experimental legacy random walks

The source comparison uses `torch-cluster` 1.6.3 at commit
`29cd22bf1a5b82fc06b108d6573f81302c5d6b12`.
The inspected SHA-256 digests are `rw.py`
`6a54c7940d4d969c8d8e9a354b85cff9001beb26d9267984e72c489c9c8478d8`,
`rw_cpu.cpp` `4dd3342f982de16deb3a9ba6238a94764c389fcdab3019c0e61f6877e06a52c2`,
and `rw_cuda.cu`
`0b2bc5d17d932b92d6dc2e3f9be992f22e5ee7c49f0e4458c6e306cf4fa4853c`.

`mps_pointops.random_walk.random_walk` accepts the `torch_cluster` 1.6.3
signature `random_walk(row, col, start, walk_length, p=1, q=1,
coalesced=True, num_nodes=None, return_edge_indices=False)`. It returns an
`(W, L + 1)` int64 node tensor for `W` starting nodes and `L` steps, or a pair
with an `(W, L)` int64 edge tensor. The first node is each input start. A
degree-zero node repeats itself and records edge `-1` at every subsequent step.
An edge ID indexes the edge sequence *after* optional coalescing, not
necessarily the input COO position.

The first supported path is finite graph topology on CPU or MPS with int64
`row`, `col`, and `start` on one device, valid node IDs, and a nonnegative walk
length. With `coalesced=True`, edges are sorted by `row * num_nodes + col`, as
in the pinned Python wrapper. With `coalesced=False`, the caller must have
already grouped edges by source row. `num_nodes` is inferred from all three
index tensors when none is supplied. It must be explicit if any are empty.
The output is integer valued; gradients are not defined.
For `walk_length=0`, this implementation returns the start nodes and an
empty edge tensor. The pinned CPU extension has an unconditional first-hop
write in its biased branch, so that case is a deliberate safe deviation.
The ratio between the smallest and largest of `1/p`, `1`, and `1/q` must
remain in the float32 normal range; otherwise the implementation raises to
avoid flush-to-zero in the stored weights. This check does not eliminate
float32 cumulative-sum rounding when an individual weight is tiny relative
to the total row weight; the biased path is an approximate sampling contract.

## Transition law

For `p = q = 1`, each outgoing edge is selected with equal probability. The
implementation generates one float32 `torch.rand((W, L))` matrix, then uses
`trunc(rand[w, step] * outdegree)` to choose an edge. This mirrors the
[upstream CPU sampler](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cpu/rw_cpu.cpp)
for the same CPU PyTorch RNG state, default float32 dtype, and duplicate-free
inputs. The dtype is explicit here, so changing the process-wide PyTorch
default dtype does not change this implementation's random precision. The
pinned extension creates its random matrix without an explicit dtype and
then accesses it as float32; parity is therefore scoped to the normal
float32 default. MPS and CUDA
have different RNG streams; exact sampled-walk equality across devices is
not a meaningful general requirement.

For positive `p, q` unequal to one, the first hop is uniform. The upstream
**CPU** source proposes a candidate outgoing edge using `rand() % degree`,
draws one variate `r`, and accepts it when *any* of three ordered conditions
holds. The modulo operation has a tiny proposal bias unless the random
integer range divides evenly by the degree. Ignoring that discrete bias,
write

```text
M  = max(1/p, 1, 1/q)
a0 = (1/p)/M,  a1 = 1/M,  a2 = (1/q)/M.
```

For a candidate next node `x`, current node `v`, and previous node `t`, the
effective acceptance threshold is

```text
A(x | t, v) = max(a2, a1 if edge x -> t exists, a0 if x == t).
```

Because proposals are uniform and independently retried, the probability of
selecting a particular outgoing *edge entry* `e` is
`A(col[e] | t, v) / sum_{f out of v} A(col[f] | t, v)`. The implementation
samples that distribution directly with a vectorized cumulative sum. Thus
it avoids an unbounded rejection loop while preserving the source transition
distribution for ideal uniform proposals, including directed adjacency and
duplicate edge entries. Float32 cumulative sums can perturb very small
probabilities. It does not reproduce the source's `rand()` or `curand`
bitstream or individual walk outcomes. The
[pinned Python wrapper](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/torch_cluster/rw.py),
[CPU sampler](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cpu/rw_cpu.cpp),
and [CUDA sampler](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cuda/rw_cuda.cu)
define the comparison. No upstream implementation is copied here.

The pinned CUDA biased kernel additionally overwrites `row_start` and
`row_end` with the candidate `x`'s adjacency range inside its rejection loop.
If that proposal is rejected, the next proposal reads the new range rather
than the current vertex `v`'s range. This is a source-level observation; no
CUDA binary was executed here. This implementation follows the CPU sampler's
current-vertex proposal rule and does **not** claim biased CUDA parity.

The biased implementation currently allocates temporary tensors proportional
to `W * d_max` per step, where `d_max` is the largest outdegree visited in
that step. Its first-hop random vector is `O(W)`; the result itself is
`O(W * L)`. It is an experimental correctness path, not a performance claim.
The categorical draw clamps its selected offset to each walker's actual
outdegree. This prevents a float32 draw rounded up to the cumulative total
from selecting another walker's padding in the same vectorized tile.
PyTorch's MPS random generator is seeded with `torch.manual_seed`, but its
draws need not match the CPU generator. Safe and Fast Math can differ at the
cumulative-sum boundary; tests check walk validity and distribution rather
than demanding cross-device bit identity.

## Verification

The committed tests check node/edge shapes and alignment, isolated vertices,
sorted and pre-coalesced inputs, invalid arguments, and two three-way biased
transitions with expected probabilities `(2/7, 4/7, 1/7)` and
`(4/7, 2/7, 1/7)`. When a real
`torch_cluster==1.6.3` CPU extension is importable, the test resets
`torch.manual_seed` and compares **all uniform node and edge IDs exactly**.
Run Safe and Fast Math in separate processes with
`PYTORCH_ENABLE_MPS_FALLBACK=0`.

The legacy shim and PyG 2.8's separate `pyg-lib` operator path have different
registration surfaces. This experimental result only covers the named legacy
API and does not establish universal PyG compatibility.

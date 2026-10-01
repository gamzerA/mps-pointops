# Experimental legacy Graclus matching on CPU and MPS

`mps_pointops.compat.install()` exposes the legacy
`torch_cluster.graclus_cluster(row, col, weight=None, num_nodes=None)` name.
This slice reproduces the observable **CPU** greedy matching contract of
[`torch_cluster` 1.6.3](https://github.com/rusty1s/pytorch_cluster/tree/1.6.3)
for int64 node indices and optional finite float32 weights. It is an
independently written implementation; no upstream source or compiled binary
is included in this repository. The native MPS matching decision runs in a
single Metal work item, so this is a correctness baseline for modest graphs,
not a speedup claim or a complete PyG compatibility claim.

## Contract fixed from the original source

The [Python wrapper](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/torch_cluster/graclus.py)
infers `num_nodes` from the maximum node ID when omitted, removes self-loops,
and randomly permutes the remaining **edges only when `weight is None`**.
It sorts edges by source row, builds CSR offsets, and calls the native op.
The [CPU implementation](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cpu/graclus_cpu.cpp)
then independently permutes the nodes. An unassigned node chooses its first
unassigned outgoing neighbor in CSR order. With weights, it starts with a
maximum weight of zero and updates on `>=`, so negative-only edges do not
match and the **last** eligible equal-weight neighbor wins. A pair receives
label `min(u, v)`; an unmatched or isolated node receives its own ID. Labels
are int64 and need not be consecutive. The
[CUDA implementation](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cuda/graclus_cuda.cu)
instead uses repeated random coloring and propose/respond passes; exact CPU
versus CUDA selection is not part of this MPS slice.

`num_nodes` may be larger than the largest edge endpoint, retaining isolated
nodes. The original wrapper raises during `max()` for an empty edge list with
omitted `num_nodes`; this implementation raises a direct `ValueError` and
requires `num_nodes` for that case. Invalid shapes, dtypes, devices, and node
indices receive explicit errors before native buffer access. MPS accepts
finite float32 weights; other weight dtypes and nonfinite weights are rejected
explicitly. These safety checks can synchronize scalar values to the CPU.
No gradient is defined through the integer cluster assignment.

CPU and MPS consume different device RNG streams, and their `argsort` ordering
among equal source rows can differ. **Matching `torch.manual_seed` does not
promise exact CPU/MPS labels** on a graph with several legal matchings.
Tests therefore use exact labels for disjoint pairs with a unique matching,
an explicit CSR/order probe for the last-equal-weight tie rule, and structural
invariants for general graphs: all vertices assigned, clusters of size at
most two, pair labels equal to the smaller endpoint, and every pair linked
by at least one input edge. The public API preserves the original stochastic
edge and node permutations rather than hiding them behind a deterministic
replacement.

## Pinned direct CPU comparison

The original checkout is tag `1.6.3`, commit
`29cd22bf1a5b82fc06b108d6573f81302c5d6b12`. Git blob SHA-1 values
for its wrapper, CPU, CUDA, and registration source are recorded in the
[raw comparison JSON](parity/graclus-163-cpu.json). Its unmodified
`graclus.cpp` and `graclus_cpu.cpp` were built against PyTorch 2.14.1 on
macOS arm64 into an external temporary directory. The extension binary's
SHA-256 is `c0c548694c29fcc1f7d793f224098675b293395198050f0ec77edac76f0da407`.
Eight graph/weight cases across five seeds each gave **40/40 exact label
arrays** between the original Python wrapper plus compiled CPU op and this
project's CPU path. This direct comparison exercises self-loops, isolated
nodes, weighted ties, zero/negative weights, and empty edges with explicit
`num_nodes`. It does not prove equality for unsupported dtypes or CUDA.

To reproduce without copying upstream code into the project:

```bash
git clone --branch 1.6.3 --depth 1 https://github.com/rusty1s/pytorch_cluster.git /tmp/pytorch_cluster-1.6.3
python tools/build_graclus_upstream_cpu.py \
  --upstream-checkout /tmp/pytorch_cluster-1.6.3 --build-dir /tmp/graclus-163-build
PYTHONPATH=. python tools/graclus_upstream_parity.py \
  --upstream-checkout /tmp/pytorch_cluster-1.6.3 \
  --extension /tmp/graclus-163-build/graclus_163_cpu.so \
  --output docs/parity/graclus-163-cpu.json
```

The [comparison script](../tools/graclus_upstream_parity.py) verifies the
source commit and refuses modified wrapper/kernel files. The build helper
requires a C++ compiler, Ninja, and matching PyTorch headers.

## MPS verification and limits

M5 Pro with PyTorch 2.14.1 and `PYTORCH_ENABLE_MPS_FALLBACK=0` passed the
focused [Safe](parity/graclus-pytest-safe.log) and
[Fast](parity/graclus-pytest-fast.log) suites: **20/20** in each separate
process. These include a native Metal test that passes fixed CSR data and
node order to verify weighted last-tie selection independently of random
permutation. On the branch rebased onto main `61a4a076a38147e237d1dec36283ec720d811d6e`,
the complete suite also passed: [Safe](parity/graclus-full-safe.log)
**331 passed, 22 skipped**, and [Fast](parity/graclus-full-fast.log)
**330 passed, 23 skipped**. The test code is
[`tests/test_graclus.py`](../tests/test_graclus.py).

The MPS path first uses native PyTorch masking, permutation, sorting, and CSR
operations, then launches one Metal kernel for the sequential greedy decision.
The kernel writes every output slot, including isolated nodes, before return;
reading results from the host or `torch.mps.synchronize()` waits for the
asynchronous queue. Performance and large-graph scalability remain
unmeasured. Further work should benchmark and design a parallel matching
schedule before presenting this path as an acceleration.

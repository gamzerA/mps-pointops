# Experimental sparse convolution coordinate oracle (v0.10 groundwork)

Status: **CPU rulebook, private bounded CPU stride/inverse reference,
experimental Metal SubM forward and first-order backward with an integrated
sorted MPS rulebook path.** These private modules are not exported by
`mps_pointops` and are not an `spconv` replacement. The v0.10 release gate
remains open. The [CPU stride/inverse contract](sparse-conv-stride-inverse-cpu.md)
documents the new reference and its remaining gaps.

## Source contract and our supported subset

The [spconv 2.3.8 usage guide](https://github.com/traveller59/spconv/blob/v2.3.8/docs/USAGE.md)
specifies features `[N, C]`, integer coordinates `[N, batch + three spatial
axes]`, batch in column 0, and exact alignment of coordinate axis order with
`spatial_shape` and convolution parameters. It defines `SubMConv3d` as
retaining active output coordinates, while ordinary `SparseConv3d` can change
the active coordinate set. The
[pinned index generator](https://github.com/traveller59/spconv/blob/v2.3.8/spconv/pytorch/ops.py)
uses the standard dense-convolution output-size formula, returns the original
indices for SubM, and passes only kernel size and dilation to the SubM pair
generator. The [guide](https://github.com/traveller59/spconv/blob/v2.3.8/docs/USAGE.md)
also distinguishes `SparseInverseConv3d` from `SparseConvTranspose3d`. Only a
bounded private CPU reference for the former is implemented here.

This independent oracle accepts only CPU `int32` indices and CPU `float32` or
`float64` features. It checks `[N, 4]` and `[N, C]` shapes, `C ≥ 1`, matching
row counts, spatial and batch bounds, and unique coordinates. Duplicate active
coordinates fail explicitly instead of silently merging features. Empty input
is accepted. The validator retains input row order and does not reinterpret
the spatial columns as a fixed XYZ or ZYX convention. The caller must keep all
three axes aligned. This is a deliberately narrower contract than spconv's
complete type and device support.

## Rulebook semantics

Each pair is `(kernel_offset, input_row, output_row)`. Kernel offsets use
row-major axis order with the final spatial axis fastest. Pairs are ordered
by kernel offset, then output row. This deterministic order is **our oracle
format**, not a claim about spconv's internal pair order.

For ordinary cross-correlation, each axis uses

```text
input_position = output_position * stride - padding + offset * dilation
output_size = floor((input_size + 2*padding - dilation*(kernel_size-1) - 1)
                    / stride) + 1
```

An output is active exactly when at least one active input lies in its kernel
footprint. Output coordinates are sorted lexicographically by batch and then
the three spatial columns. For SubM, output coordinates and row order equal
the input. The prototype restricts SubM to odd kernels, unit stride, and
implicit center padding. Its source position is
`output_position + (offset - floor(kernel_size/2))*dilation`.

The private `subm_conv3d_forward_mps(indices, features, weights, spatial_shape,
batch_size, dilation=1, bias=None, rulebook_backend="mps")` validates its CPU
coordinates, transfers them to MPS, constructs the sorted rulebook there,
and uses one Metal writer per output scalar to gather and reduce features.
`rulebook_backend="cpu"` retains CPU rulebook construction and CSR transfer as
a reference and small-input comparison path. Coordinates are
CPU int32 `[N,4]`; features and weights are MPS float32 `[N,Cin]` and
`[Cout,Cin,K0,K1,K2]`, with optional MPS float32 bias `[Cout]`. The output
is MPS float32 `[N,Cout]` in input coordinate row order. The weight layout is
the *dense PyTorch Conv3d* layout used by the oracle; it is not a claim about
the pinned spconv weight layout. Each output starts from bias (or zero) and
visits rulebook pairs in increasing kernel-offset order, then input-channel
order, applying `fma(feature, weight, total)` on each term. Both Safe and Fast
Math are exercised in separate processes. Sparse floating-point output need
only be close to dense PyTorch, since its convolution reduction order may
differ. For upstream output gradient `G[o,d]`, the supported first-order
backward is defined on the exact same integer rulebook pairs `(k,i,o)`:

```text
dX[i,c]     = sum_{(k,i,o)} sum_d G[o,d] * W[d,c,k]
dW[d,c,k]   = sum_{(k,i,o)} G[o,d] * X[i,c]
db[d]       = sum_o G[o,d]
```

The default MPS path uses the GPU-resident output CSR in reverse: for odd
centered kernels, `(k,i,o)` has a reverse pair `(V-1-k,o,i)`, where `V` is
kernel volume. One Metal thread owns each `dX[i,c]` and visits the reverse
output CSR, so no float atomics are needed. The GPU rulebook also provides an
offset-major `offset_ptr`; one thread owns each `dW[d,c,k]` and reduces its
valid pair range without reading `pair_count` on the host. The CPU baseline
continues to group pairs by kernel offset, use matrix products for `dW`, and
use native MPS `index_add_` for `dX`. Bias is
added once per output, including outputs with no active input pairs. Empty
input returns empty output and zero first-order weight/bias gradients. The
integer pair mapping and output coordinate row order are exact; floating-point
gradient bits are **not** claimed to match dense CPU `Conv3d` because its
reduction order can differ from Metal FMA; the optional CPU baseline's MPS
matrix reductions and `index_add_` can also differ. Tests use `rtol=1e-4,
atol=1e-5` for float32 forward and backward. The Metal forward and GPU
`dX`/`dW` shaders use explicit FMA and Safe math. `PYTORCH_MPS_FAST_MATH`
controls the remaining native PyTorch operations, including bias reduction
and the optional CPU-rulebook baseline backward, in separate processes.
Higher-order gradients,
coordinate gradients, GPU-side coordinate validation, Metal
strided/inverse/transpose convolution, and source-compatible `spconv` APIs
remain unsupported. The CPU stride/inverse reference has separate tests and
limits described [here](sparse-conv-stride-inverse-cpu.md).

## Private MPS-native sorted rulebook prototype

The private `generate_subm_rulebook_mps` takes prevalidated, unique MPS
`int32` coordinates `[N,4]` in input row order. The caller must guarantee
batch/spatial bounds and uniqueness; unlike the CPU oracle, this prototype
does not inspect coordinate values on the host or reject invalid/duplicate
coordinates. A contiguous MPS clone snapshots the input once for both lookup
and returned output coordinates, matching the CPU oracle's clone behavior
when callers mutate their input after the call. It accepts odd kernel
dimensions and positive dilation, each fitting `int32`. The former 1,024-row
and 27,648-slot caps are gone. The current wrapper checks a Metal 1-D dispatch
limit of `N * kernel_volume <= 2**32 - 2`; available device memory can limit
practical inputs earlier. These are private implementation limits, not
`spconv` limits.

1. Four stable MPS sorts of row IDs, from the last coordinate field to the
   first, produce a lexicographic index. PyTorch 2.7 MPS `index_select` and
   `gather` corrupt nearby large `int32` values in the tested environment;
   that version uses a Metal coordinate-field gather before each sort. A
   Metal thread for each
   `(kernel_offset, output_row)` computes the target and binary-searches that
   index. Lookup needs `O(log N)` comparisons per target; the previous
   prototype scanned all `N` rows, taking `O(kernel_volume * N²)` lookup
   work. Overall cost also includes the four sorts. PyTorch does not promise
   their sorting algorithm or asymptotic cost on MPS. Intermediate target
   arithmetic uses signed 64-bit integers, and out-of-int32 targets are
   rejected before lookup to prevent wraparound matches.
2. MPS-device integer prefix sums compute ranks in both offset-major and
   output-major order. A second Metal kernel writes each valid pair once to
   its unique rank. There are no atomics and no schedule-dependent ties.
3. The result keeps offset-major `(offset, input_row, output_row)` pairs and
   output-major CSR `ptr/sources/offsets` on MPS. Both match the CPU oracle's
   chosen orders bit-for-bit for valid coordinates. The pair count is a
   one-element MPS tensor; allocation uses a fixed padded capacity and fills
   every unused element with `-1`. Reading the count on CPU would synchronize,
   but construction and CSR consumption do not require such a readback.

No public SubM API is exported. The private wrapper now consumes the
GPU-generated CSR and GPU offset-major pair metadata in both forward and
first-order backward. The pair count stays on MPS; queue ordering carries
rulebook writes to the convolution and gradient kernels without an explicit
intermediate synchronization. CPU coordinate validation remains mandatory
because the GPU builder does not reject duplicate or out-of-range inputs.
The output-coordinate row order is unchanged. The `rulebook_backend="cpu"`
baseline remains available for a direct full-wrapper comparison. Upstream
`spconv` parity and a model-level result remain separate release gates.

## Reproducible local check

On the 2026-10-02 M5 Pro workspace with PyTorch 2.14.0, run each math mode
in a fresh process:

```bash
python -m pytest -q tests/test_sparse_rulebook.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest -q tests/test_sparse_rulebook.py tests/test_subm_metal.py tests/test_subm_rulebook_mps.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python -m pytest -q tests/test_sparse_rulebook.py tests/test_subm_metal.py tests/test_subm_rulebook_mps.py
```

The CPU oracle suite checks duplicate and out-of-range rejection, input row
order, batch separation, exact kernel-offset pairs, dilation, stride, empty
input, and dense occupancy/value agreement. The related CPU and Metal suites
passed **51/51** on M5 Pro in each mode with fallback disabled. New cases
include 1,025 and 10,000 unsorted rows, a 5×5×5 kernel above the old slot
cap, and int32 coordinate/dilation boundaries. The [M5 Pro evidence](evidence/subm-scalable-m5pro-2026-10-02/README.md)
contains the tested source hash, synchronized 1,024/10,000-row timings, and
[Safe](evidence/subm-scalable-m5pro-2026-10-02/pytest-safe.log) and
[Fast](evidence/subm-scalable-m5pro-2026-10-02/pytest-fast.log) raw pytest logs.
Direct `spconv` runtime parity has **not** been run because it is not installed
in this environment.

Before the sorted builder, the 1,024-row quadratic prototype was independently
run on a physical Apple M1 (8 GiB, macOS 26.5.2, PyTorch 2.14.1) with MPS CPU
fallback disabled; Safe and Fast each passed **47 tests**. The
[historical raw M1 logs and source provenance](evidence/subm-m1-prototype-2026-10-02/README.md)
remain archived. These operator tests do not prove performance of the full
convolution wrapper or upstream `spconv` parity.

## Remaining gates before any compatibility claim

1. Compare coordinates, feature values, and gradients against pinned spconv
   2.x CPU and CUDA runs, including its output-order mapping and weight layout.
2. Specify duplicate-input handling, empty outputs, missing cells, and
   `indice_key` reuse against upstream behavior instead of assuming parity.
3. Profile the integrated SubM path across dense and sparse large clouds and
   older Apple Silicon devices, including device-memory peaks. Implement
   strided and inverse Metal forward/backward and a separate transpose path.
4. Verify at least one pinned sparse 3D backbone end to end on supported
   Apple Silicon hardware. The physical M1 test above covers only small
   operator fixtures, not a backbone or large-cloud workload.

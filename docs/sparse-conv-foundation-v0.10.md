# Experimental sparse convolution coordinate oracle (v0.10 groundwork)

Status: **CPU rulebook plus experimental Metal SubM forward and first-order
backward.** These private
modules (`mps_pointops._sparse_rulebook` and `_subm_conv_mps`) are not exported
by `mps_pointops` and are not an `spconv` replacement. The v0.10 release gate
remains open.

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
also distinguishes `SparseInverseConv3d` from `SparseConvTranspose3d`; neither
is implemented here.

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

The rulebook is generated and sorted into output CSR on the CPU. The optional
private `subm_conv3d_forward_mps(indices, features, weights, spatial_shape,
batch_size, dilation=1, bias=None)` transfers that CSR to MPS and uses one
Metal writer per output scalar to gather and reduce features. Coordinates are
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

The implementation groups pairs by kernel offset, gathers the corresponding
source features and output gradients, computes matrix products for each
weight slice, and uses native MPS `index_add_` for feature gradients. Bias is
added once per output, including outputs with no active input pairs. Empty
input returns empty output and zero first-order weight/bias gradients. The
integer pair mapping and output coordinate row order are exact; floating-point
gradient bits are **not** claimed to match dense CPU `Conv3d` because MPS
matrix reduction and `index_add_` accumulation order can differ, especially
when one input contributes to several outputs. Tests use `rtol=1e-4,
atol=1e-5` for float32 forward and backward. The Metal shader's forward
uses explicit FMA and Safe math, while `PYTORCH_MPS_FAST_MATH` controls the
PyTorch backward operations in separate processes. Higher-order gradients,
coordinate gradients, Metal rulebook construction, strided/inverse/transpose
convolution, and source-compatible `spconv` APIs remain unsupported.

## Reproducible local check

On the 2026-10-02 workspace with PyTorch 2.14.1, run:

```bash
python -m pytest -q tests/test_sparse_rulebook.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest -q tests/test_sparse_rulebook.py tests/test_subm_metal.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python -m pytest -q tests/test_sparse_rulebook.py tests/test_subm_metal.py
```

The initial rulebook suite passed **20 tests**. It covers duplicate/out-of-range
rejection; non-sorted input, batch isolation, exact kernel-offset pair order,
dilation, padding, stride, empty input, and invalid parameters. For several
3D kernels, an independently built dense occupancy convolution checks the
active output set, and a dense feature convolution checks all rulebook
gather/weight/reduction values. Direct `spconv` runtime parity has **not** been
run because it is not installed in this environment. The Metal suite compares
small batched, unsorted, dilated, biased, empty, and noncontiguous cases with
dense PyTorch Conv3d at active output coordinates and compares first-order
feature, weight, and bias gradients for batched, dilated, empty, and
noncontiguous-view fixtures. On the local Apple Silicon GPU with
`PYTORCH_ENABLE_MPS_FALLBACK=0`, the combined suites passed **31 tests** in
each separately launched Safe and Fast Math process on 2026-10-02. These
tests establish a bounded forward/backward prototype, not performance or
upstream `spconv` parity.

## Remaining gates before any compatibility claim

1. Compare coordinates, feature values, and gradients against pinned spconv
   2.x CPU and CUDA runs, including its output-order mapping and weight layout.
2. Specify duplicate-input handling, empty outputs, missing cells, and
   `indice_key` reuse against upstream behavior instead of assuming parity.
3. Replace CPU rulebook generation with a scalable Metal path and profile the
   full call (including CSR transfer and native MPS backward). Implement
   strided backward and separate inverse and transpose paths.
4. Verify at least one pinned sparse 3D backbone end to end on supported
   Apple Silicon hardware. Physical M1 measurement is deferred while the
   remote device is available for later validation.

# Experimental sparse convolution coordinate oracle (v0.10 groundwork)

Status: **CPU-only design and test fixture.** This module is private
(`mps_pointops._sparse_rulebook`), is not exported by `mps_pointops`, and is
not an `spconv` replacement. The v0.10 release gate remains open.

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

The row-index rulebook specifies gathering and reduction but has no Metal
implementation, weights, bias, feature output, or backward path. It cannot
establish bitwise parity of floating-point accumulation order.

## Reproducible local check

On the 2026-10-02 workspace with PyTorch 2.14.1, run:

```bash
python -m pytest -q tests/test_sparse_rulebook.py
```

The initial suite passed **20 tests**. It covers duplicate/out-of-range
rejection; non-sorted input, batch isolation, exact kernel-offset pair order,
dilation, padding, stride, empty input, and invalid parameters. For several
3D kernels, an independently built dense occupancy convolution checks the
active output set, and a dense feature convolution checks all rulebook
gather/weight/reduction values. Direct `spconv` runtime parity has **not** been
run because it is not installed in this environment.

## Remaining gates before any compatibility claim

1. Compare coordinates, feature values, and gradients against pinned spconv
   2.x CPU and CUDA runs, including its output-order mapping and weight layout.
2. Specify duplicate-input handling, empty outputs, missing cells, and
   `indice_key` reuse against upstream behavior instead of assuming parity.
3. Implement and profile Metal rulebook generation and gather-GEMM-scatter,
   then SubM and strided backward. Add separate inverse and transpose paths.
4. Verify at least one pinned sparse 3D backbone end to end on supported
   Apple Silicon hardware. Physical M1 measurement is deferred while the
   remote device is disconnected.

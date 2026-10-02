# Private CPU stride and inverse sparse-convolution reference

Status: a **bounded, CPU-only, private reference** for ordinary `SparseConv3d`
coordinate changes and `SparseInverseConv3d`-style `indice_key` reuse. It is
not exported from `mps_pointops`, does not run on Metal, and is not a drop-in
`spconv` module. It does not change the package version or a release gate.

The reference is based on the existing coordinate rulebook and follows the
operator distinction in the [spconv 2.3.8 usage guide](https://github.com/traveller59/spconv/blob/v2.3.8/docs/USAGE.md): an inverse restores the *input active coordinates* of the ordinary convolution whose key it reuses. A transposed convolution independently generates output coordinates and can therefore have a different active set. The [pinned `conv.py` implementation](https://github.com/traveller59/spconv/blob/v2.3.8/spconv/pytorch/conv.py) retrieves saved `indices`, `out_indices`, and pairs under the key for inverse execution; its weight layout depends on the selected algorithm and build options. This CPU reference deliberately uses PyTorch's `[Cout, Cin, K0, K1, K2]` layout for both operations. No spconv CUDA value or gradient equivalence is asserted.

## Supported contract

`mps_pointops._sparse_conv_cpu.sparse_conv3d_forward_cpu` accepts a validated
`SparseTensor3D`, CPU float32 or float64 weights, optional same-dtype bias,
kernel shape from the weights, and integer stride, padding, and dilation. It
uses the ordinary rulebook's equation

```text
input_position = output_position * stride - padding + offset * dilation
```

and lexicographically sorted active outputs. With a nonempty `indice_key`, the
caller must pass a mutable `indice_cache` mapping. Forward saves independent
snapshots of the input and output coordinate rows, shapes, geometry, and
`(offset, old_row, downsampled_row)` pairs. A duplicate key fails.

`sparse_inverse_conv3d_forward_cpu` requires that key and cache. Its input
coordinate rows, row order, spatial shape, and batch size must match the saved
forward output; its kernel shape must match the saved kernel. It reverses each
saved pair's row direction without changing its offset and writes output in
the **original input row order**, including original points that had no
forward pair. Those unmatched points receive bias alone. A missing or stale
mapping fails explicitly. Inverse stride, padding, and dilation are taken
solely from the saved mapping; they are not regenerated from inverse-layer
arguments.

The functions use PyTorch CPU matrix products and `index_add` so first-order
feature, weight, and bias gradients are available through autograd. Empty
inputs retain defined zero feature and weight gradients. The reference
accepts at most 1024 active input and output points, kernel volume at most
125, and at most 32768 input-point/kernel-offset candidates when generating a
forward rulebook. Inverse consumes saved pairs and does not regenerate those
candidates. These are conservative Python-reference limits, not spconv limits.

## Local validation

With the project Python environment on 2026-10-02:

```bash
python -m pytest -q tests/test_sparse_conv_cpu.py tests/test_sparse_rulebook.py
```

The combined check passed **29 tests**. New tests compare ordinary stride
output coordinates and values with dense occupancy and `torch.nn.functional.conv3d`
across varied kernels, padding, and dilation. They compare first-order feature,
weight, and bias gradients with dense PyTorch. Inverse tests use the saved
mapping, compare restored values and gradients with `conv_transpose3d` sampled
*only at the original active coordinates*, check an original point absent
from the transposed active set, and reject missing, duplicate, stale, and
mismatched keys. The dense transpose is a numerical reference at selected
positions; it is not the inverse coordinate-generation algorithm.

## Gaps before spconv compatibility or a release claim

1. Run pinned spconv 2.x CPU and CUDA oracle comparisons for exact coordinate
   membership and ordering, values, first-order gradients, and actual weight
   layout conversions. spconv is not installed in this local environment.
2. Implement and validate scalable Metal ordinary stride and inverse paths,
   including device-resident rulebook generation and backward. This module
   performs all coordinate work and feature arithmetic on CPU.
3. Implement a **separate** sparse transpose operator. Inverse key reuse does
   not provide transpose output coordinates or output-padding behavior.
4. Resolve broader spconv contracts, including duplicate inputs, algorithm
   variants, groups, additional dtypes, mixed precision, cache propagation
   through a backbone, and large-cloud performance. The private reference
   rejects duplicate coordinates and does not provide a public sparse tensor
   or module API.

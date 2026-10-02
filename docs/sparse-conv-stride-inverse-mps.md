# Private bounded Strided/Inverse Metal prototype

Status: `mps_pointops._sparse_conv_mps` implements **float32 feature arithmetic**
for ordinary `SparseConv3d` and saved-key `SparseInverseConv3d` on MPS. It is a
private experimental module, with no `spconv` import shim or public release
promise. The existing [CPU contract](sparse-conv-stride-inverse-cpu.md) defines
coordinate order, geometry, cache ownership, and lineage. The new functions
take `SparseTensor3D` and return `SparseConvResult3D`, so a caller can use
`replace_features()` between a downsample and its inverse.

## Scope and data flow

- CPU `int32` coordinates `[N,4]` are checked for batch/spatial bounds and
  uniqueness. The bounded CPU builder constructs lexicographically sorted
  output coordinates and offset-major `(offset,input_row,output_row)` pairs.
  At most 1,024 active input/output rows, kernel volume 125, and 32,768
  input-offset candidates are accepted. Exceeding a limit raises; it does not
  silently fall back to another convolution implementation.
- Input features, weights, optional bias, outputs, and first gradients are
  **MPS float32**. Weights use PyTorch's `[Cout,Cin,K0,K1,K2]` layout. CSR and
  pair arrays are copied from CPU to MPS before Metal arithmetic. There is no
  MPS feature readback for rulebook generation.
- Each output scalar is written by one Metal thread using the saved rulebook.
  A separate input-row CSR computes `dX` using the *original* kernel offset;
  SubM's reverse-offset shortcut is invalid for stride. An offset-major CSR
  computes `dW`, one thread per weight scalar. Native MPS sums `dBias` over
  output rows. No float atomics are used. For pair `(k,i,o)`, the equations are

  ```text
  Y[o,d]  = bias[d] + sum_(k,i,o) sum_c X[i,c] W[d,c,k]
  dX[i,c] = sum_(k,i,o) sum_d dY[o,d] W[d,c,k]
  dW[d,c,k] = sum_(k,i,o) dY[o,d] X[i,c]
  dBias[d] = sum_o dY[o,d]
  ```

- A keyed ordinary convolution snapshots input/output indices and pairs.
  Inverse requires matching key, row order, shape, batch, kernel, and lineage;
  it reverses each saved pair's row direction and restores the *original*
  input active set and row order. The inverse is distinct from sparse
  transposed convolution, which generates its own active set. An original
  input point without a saved pair receives bias alone.
- Empty inputs and zero-pair results have defined zero first gradients.
  Higher derivatives and coordinate gradients are unsupported. Operations
  with invalid dtypes/devices fail before dispatch.

The Metal accumulation order is deterministic for fixed CSR rows and uses
`fma` for each feature-weight term. Dense PyTorch may reduce in another order,
so CPU/MPS float values and first gradients use `rtol=1e-4, atol=1e-5`.
Integer indices and shapes must match exactly. Safe and Fast math are tested
in distinct processes. CPU-to-MPS CSR transfer is blocking: in a PyTorch 2.7
experiment, setting `non_blocking=True` produced a hang at
`torch.mps.synchronize()` on the second fixture in one process; removing that
flag made the 40-case target suite pass. The runtime cause was not isolated
beyond that transfer change.

## Verification and remaining work

On the M5 Pro, the targeted suites
`tests/test_sparse_conv_mps.py`, `tests/test_sparse_conv_cpu.py`, and
`tests/test_sparse_rulebook.py` passed **42/42** with PyTorch 2.7.0 and
2.14.1, each in both Safe and Fast modes. The new Metal cases compare forward
values, output coordinates, and input/weight/bias first gradients to the CPU
reference for ragged batches, stride, padding, dilation, inverse key reuse,
two-layer downsample/inverse gradient propagation,
unreachable points, empty results, and noncontiguous feature/weight views.

On a physical Apple M1, the [source-pinned Safe/Fast validation](evidence/m1-sparse-2026-10-03/README.md)
at commit `6959fd590deecbefc98a163d46c23debaaa67c9e` passed **92 tests
per mode, zero skipped** with PyTorch 2.14.1 and fallback disabled. That is
the aggregate of six CPU rulebook, SubM, Strided/Inverse Metal, and adapter
suites; it is not 92 independent Strided/Inverse tests. The optional 1,025/
10,000-row timings measure **SubM rulebook backend choice** while both paths
run feature arithmetic on MPS, so they are not a Strided/Inverse performance
benchmark or a pure CPU-versus-GPU convolution comparison. The 10,000-row
MPS-rulebook backward median was slower than the CPU-rulebook path in both
math modes, despite the shorter forward median.

This prototype still uses CPU geometry generation and small fixed caps. A
device-resident ordinary rulebook builder, scalable CSR construction, a
separate sparse transpose operator, upstream `spconv 2.x` CUDA parity for
these paths, and backbone-level tests remain open. The current result does
not establish full `spconv` compatibility or a v1.0 release gate.

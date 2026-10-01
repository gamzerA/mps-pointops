# Direct PyTorch3D Chamfer parity for the v0.5.0 candidate

This experiment executes the official
[PyTorch3D `chamfer_distance`](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/pytorch3d/loss/chamfer.py)
with its compiled CPU kNN extension and compares the supported
`mps_pointops.chamfer_distance` subset on the same input tensors. The upstream
source is pinned to commit `88e182f989c80836f4bd744e0d9cb1852762ce01`
(package version `0.7.9`). No upstream algorithm or kernel source is included
in this repository.

## Results

On an Apple M5 Pro (macOS 26.5.2, Python 3.12.13), each run covered 160 cases
and checked 1,080 loss, output-shape, normal-output, gradient-presence, and
first-order gradient results **per port device**. The 160 cases comprise two
random seeds; `mean`, `sum`, `max`, and unreduced points; valid batch `mean`,
`sum`, and unreduced modes; one-way and two-way distance; and absent, positive,
partially zero, and all-zero weights. Every case used unequal valid lengths
`x_lengths=[7,4]` and `y_lengths=[6,3]` with finite padding. The tensors were
float32, with shapes `x=(2,7,3)` and `y=(2,6,3)`.

| Upstream CPU environment | Port device and math mode | Failed checks | Largest loss error | Largest x / y / weight gradient error |
| --- | --- | ---: | ---: | ---: |
| PyTorch 2.4.1, unmodified upstream source | CPU | 0 / 1,080 | 4.7684e-7 | 0 / 0 / 4.7684e-7 |
| PyTorch 2.14.1, C++20 build flag only | CPU | 0 / 1,080 | 4.7684e-7 | 0 / 0 / 4.7684e-7 |
| PyTorch 2.14.1, C++20 build flag only | MPS Safe | 0 / 1,080 | 9.5367e-7 | 2.9803e-8 / 2.3842e-7 / 4.7684e-7 |
| PyTorch 2.14.1, C++20 build flag only | MPS Fast | 0 / 1,080 | 9.5367e-7 | 2.9803e-8 / 2.3842e-7 / 4.7684e-7 |

The acceptance predicate is elementwise
`abs(port - upstream) <= 2e-5 + 2e-4 * abs(upstream)`. Loss shapes, normal
output type/shape, and whether each input has a connected first derivative
must match exactly. The raw case-by-case checks and source hashes are in
[PyTorch 2.4.1 CPU](results/2026-10-01-m5pro-chamfer-upstream-torch241-cpu.json),
[Safe Math](results/2026-10-01-m5pro-chamfer-upstream-safe.json), and
[Fast Math](results/2026-10-01-m5pro-chamfer-upstream-fast.json).

The all-zero-weight case exposed an unusual upstream API branch. The upstream
returns `(B,B)` zeros *before* batch reduction, including when point reduction
is disabled; a zero normal-loss tensor appears even when normals were absent.
In a single-directional call, its `y` input is disconnected from the graph
(`y.grad is None`) while `x` and `weights` have present zero derivatives.
The port now matches these signatures. The bidirectional `max` case retains
`normals=None`. The port still masks nonfinite padded coordinates in this zero
path; nonfinite padding is outside this direct upstream comparison.

## Reproduce

Use the official PyTorch3D checkout at the commit above. On this Mac,
PyTorch 2.4.1 needed `CXXFLAGS=-Wno-invalid-specialization` with the installed
Apple Clang; its upstream source files remained unmodified. In the
PyTorch 2.14.1 environment, the only change in a **temporary** upstream
checkout was both `-std=c++17` occurrences in `setup.py` to `-std=c++20`,
because PyTorch 2.14.1's C++ headers require C++20. No PyTorch3D loss,
kNN, or other algorithm file was changed. PyTorch 2.14.1 is outside the
[upstream documented PyTorch 2.1–2.4.1 install range](https://github.com/facebookresearch/pytorch3d/blob/88e182f989c80836f4bd744e0d9cb1852762ce01/INSTALL.md),
which is why the separate 2.4.1 CPU result is included.

After building the CPU extension in place and making PyTorch3D and its
`iopath`/`fvcore` dependencies importable, run the
[verifier](../tests/verify_upstream_chamfer.py). The following commands show
the invocation from the project root; replace the interpreter and upstream
checkout paths with local ones. Safe and Fast must run in separate processes.

```sh
export PYTHONPATH="/path/to/pytorch3d:/path/to/dependencies:$PWD"
PYTORCH_ENABLE_MPS_FALLBACK=0 python tests/verify_upstream_chamfer.py \
  --upstream-root /path/to/pytorch3d \
  --upstream-build-note 'CPU extension, C++20 flag-only compatibility build' \
  --out docs/results/2026-10-01-m5pro-chamfer-upstream-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python tests/verify_upstream_chamfer.py \
  --upstream-root /path/to/pytorch3d \
  --upstream-build-note 'CPU extension, C++20 flag-only compatibility build' \
  --out docs/results/2026-10-01-m5pro-chamfer-upstream-fast.json
```

For PyTorch 2.4.1, add `--devices cpu`, and identify the unmodified checkout
and local `CXXFLAGS` warning suppression in `--upstream-build-note`.

This matrix tests finite, randomly separated points. It does not establish
bitwise cross-backend equality, tied nearest neighbors, extreme float32
underflow/overflow, empty point sets, normals, L1 distance, second-order
gradients, or PyTorch3D CUDA parity. Those cases retain the limits in
[the Chamfer contract](chamfer-contract.md).

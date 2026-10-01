# PyTorch3D Chamfer MPS parity gate

The [dedicated CI workflow](../.github/workflows/chamfer-upstream-parity.yml)
executes the official PyTorch3D 0.7.9 CPU implementation at commit
`88e182f989c80836f4bd744e0d9cb1852762ce01` as an oracle. It builds the
CPU extension in a temporary checkout. The only source change is replacing the
two `-std=c++17` build flags in `setup.py` with `-std=c++20`, required by the
pinned PyTorch 2.14.1 headers. No PyTorch3D algorithm or kernel is copied into
this package or changed in the temporary checkout.

The job requires an available MPS device and sets
`PYTORCH_ENABLE_MPS_FALLBACK=0`. Safe and Fast Math run in separate processes.
Each process checks 160 finite float32 cases and 1,080 output or gradient
signatures: unequal valid lengths and padded coordinates, absent and three
forms of batch weights, both directional modes, all supported point and batch
reductions, and first derivatives with respect to both clouds and weights.
Output shapes, normal-slot shape, and gradient connectivity must agree exactly;
numeric tensors use `abs(actual-reference) <= 2e-5 + 2e-4*abs(reference)`.
The job uploads case-by-case JSON for both modes as run artifacts.

This gate covers the package's squared-L2 subset. It does not test L1,
normal-vector loss, `Pointclouds` inputs, second derivatives, CUDA parity,
near-ties, or nonfinite valid coordinates. Those are separate contracts or
future extensions. The previous local [M5 Pro direct comparison](chamfer-upstream-parity-0.5.0.md)
and its raw files remain the baseline; CI now repeats the supported matrix on
every pull request and `main` push. A green workflow means the supported
subset passed on that runner, not that the full PyTorch3D API is implemented.

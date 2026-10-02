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

The corrected workflow passed on [GitHub Actions run 36899829775](https://github.com/gamzerA/mps-pointops/actions/runs/36899829775)
for PR #52. The archived [Safe](results/2026-10-02-hosted-chamfer-upstream-safe.json)
and [Fast](results/2026-10-02-hosted-chamfer-upstream-fast.json) JSON each report
160 cases, 1,080 checked output or gradient tensors, zero failed elements,
zero failed tensors, and maximum absolute error `9.5367431640625e-7`.
Both used macOS 26.6.2, Python 3.12.10, PyTorch 2.14.1, an available MPS
device, and fallback disabled. The JSON's `port.commit` value
`a88a26fec7883d9e77e65d2f0958d7e8f9ca800e` is GitHub's temporary PR
merge commit for that run, while the source PR head was `a4091e8`; the files
also record hashes of the checked package kernel, wrapper, and verifier.
The artifact is `chamfer-upstream-parity-36899829775` (ID 11182240265).
SHA-256: Safe `b37c098ef7224ec2a5bd5e24bd5b376cd733ce56fe1e59dbf679ca8a461466a3`,
Fast `9ad69bd9aebad05fd13a24a5c8a25ef2af78474460cf0d70aa3959f2a4e3564b`.

This gate covers the package's squared-L2 subset. It does not test L1,
normal-vector loss, `Pointclouds` inputs, second derivatives, CUDA parity,
near-ties, or nonfinite valid coordinates. Those are separate contracts or
future extensions. The previous local [M5 Pro direct comparison](chamfer-upstream-parity-0.5.0.md)
and its raw files remain the baseline; CI now repeats the supported matrix on
every pull request and `main` push. A green workflow means the supported
subset passed on that runner, not that the full PyTorch3D API is implemented.

A separate [physical M1 direct-upstream archive](evidence/chamfer-v090-m1-upstream/README.md)
now records L1, squared-L2, normal loss, `Pointclouds`, and first derivatives
from clean development commit `ef07a933` against the same pinned PyTorch3D
CPU oracle. Safe and Fast Math each passed 480 base and 252 extended cases with
zero failed checks or elements. This is a wider opt-in experiment, not an
expansion of the required CI gate or a full-API compatibility declaration.

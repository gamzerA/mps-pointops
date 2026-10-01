# PointNet++ CUDA ↔ Apple MPS propagation parity

This record compares the independent `mps_pointops.pointnet2` implementation
with the **executed original PointNet++ C++/CUDA extension**. The complete
tensor results are in [the MPS Safe JSON](pointnet2-mps-safe.json),
[MPS Fast JSON](pointnet2-mps-fast.json), and
[upstream CUDA JSON](pointnet2-cuda.json). The machine-readable
[Safe](pointnet2-compare-safe.json) and [Fast](pointnet2-compare-fast.json)
comparison files name their inputs, input SHA-256 hashes, source hashes, and
individual pass/fail checks.

## Source and execution identity

| Item | Pinned observation |
| --- | --- |
| Project source | Commit `88cdc23` for the measured MPS files; `mps_pointops/pointnet2.py` SHA-256 `55d946e1c4bfe32755566ecf51f643b0a08df3f94317f8ee6f1e07adf685a2e7`; `mps_pointops/kernels/pointnet2.metal` SHA-256 `c46238511f5d36a0d3dea3bbdbe8efe571f692a5f82350ee92425ea57b6569c1` |
| Upstream source | [`erikwijmans/Pointnet2_PyTorch`](https://github.com/erikwijmans/Pointnet2_PyTorch) commit `b5ceb6d9ca0467ea34beb81023f96ee82228f626` |
| Upstream CUDA source | Git blob SHA-256 `e4a327f15d49bcefe62bd72adc407157f97039c3b34037f8fc767b9a15936173` for [`interpolate_gpu.cu`](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/b5ceb6d9ca0467ea34beb81023f96ee82228f626/pointnet2_ops_lib/pointnet2_ops/_ext-src/src/interpolate_gpu.cu), which contains both `three_nn` and `three_interpolate` kernels |
| Upstream imported wrapper | Installed `pointnet2_utils.py` SHA-256 `32dced7ce30daf3af52b16f73e95d22a477cb968a1f05af15b455e2635eef135`, equal to the checked-out wrapper |
| Upstream compiled extension | `_ext.cp311-win_amd64.pyd` SHA-256 `7db4927b3358c4ca5c212b582889588225cc4b6392f3a3939ca371ae237da86e` |
| Verifier | `tools/pointnet2parity.py` at commit `287ee89`, Git blob SHA-256 `3a251e4ef5523f1d4b949cbc5da6085f31ab5ceb010edb8fa8c4b35ea498d224`; fixture ID `pointnet2-propagation-v2` |
| Apple machine | M5 Pro, 48 GB, macOS 26.5.2, PyTorch 2.14.1; `PYTORCH_ENABLE_MPS_FALLBACK=0` |
| CUDA machine | NVIDIA GeForce RTX 2080, Windows, PyTorch `2.14.0+cu132`; GPU and environment recorded in CUDA JSON |

The Windows checkout used a build-only modification to the official
`pointnet2_ops_lib/setup.py`: CUDA architectures were narrowed from the
legacy list to `7.5` for this RTX 2080, and `-Xcompiler
/Zc:preprocessor` was added for the installed MSVC/CUDA toolchain. The exact
two-line diff is in the CUDA JSON. The Python wrapper and CUDA implementation
files were unchanged. The verifier checks the official Git HEAD, clean
implementation directory, pinned CUDA Git blob, installed wrapper bytes, and
loaded extension hash. The Windows working-tree CUDA file has a different byte
SHA because Git checked it out with CRLF line endings; the clean Git blob is
the cross-platform source identity.

## Cases and acceptance rules

Two deterministic float32 cases ran on both devices:

1. `ties-and-duplicates`: `B=2, N=4, M=6, C=2`, with exact neighbor ties,
   repeated interpolation indices, and requested feature and weight gradients.
2. `seeded-stride`: `B=1, N=73, M=257, C=5`, with fixed Python RNG seed
   `20261001` and binary-fraction inputs. This crosses the 256-point reference
   scan size and uses nontrivial channel and query dimensions.

The verifier requires exact nested shapes and indices. For floating outputs it
requires elementwise `|CUDA−MPS| ≤ 1e-5 + 2e-6·|MPS|`, finite values, and
matching absence of gradients for both coordinate tensors. A failed check
exits nonzero. This threshold is a test acceptance rule, not a guarantee for
all inputs.

| Array | Compared elements | Safe maximum absolute error | Fast maximum absolute error |
| --- | ---: | ---: | ---: |
| `three_nn` Euclidean distances | 243 | 0 | `2.384185791015625e-7` |
| `three_nn` selected indices | 243 | 0, exact | 0, exact |
| `three_interpolate` supplied indices | 243 | 0, exact | 0, exact |
| `three_interpolate` output | 381 | 0 | 0 |
| Feature gradient | 1,309 | 0 | 0 |
| Weight gradient | 243 | 0 | 0 |

Both comparison files report `passed: true`. The weight gradient is an
explicit zero tensor in the original PointNet++ autograd wrapper and in this
compatibility implementation. Mathematically, treating a supplied weight as
an independent variable gives
`∂Y[b,c,n]/∂W[b,n,t] = F[b,c,I[b,n,t]]`, which is generally nonzero.
The zero tensor is the observed backward behavior in the original wrapper; the
feature gradient follows the usual chain rule with repeated indices summed.
The full project test suite on this branch separately passed in Safe and Fast
Math processes: `244 passed, 12 skipped` in each mode.

## Reproduction

On an Apple Silicon Mac, from this checkout and its PyTorch environment:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/pointnet2parity.py --backend project --device mps --output docs/parity/pointnet2-mps-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 python tools/pointnet2parity.py --backend project --device mps --output docs/parity/pointnet2-mps-fast.json
```

The two MPS modes must run in separate processes. On the NVIDIA machine,
check out the pinned official upstream commit in a separate directory, apply
only the documented `setup.py` build changes if its toolchain requires them,
and install the extension from `pointnet2_ops_lib`:

```bash
python -m pip install . --no-build-isolation --no-deps
```

From this project's checkout on that machine:

```bash
python tools/pointnet2parity.py --backend upstream --device cuda --upstream-checkout ../Pointnet2_PyTorch --compare docs/parity/pointnet2-mps-safe.json --output docs/parity/pointnet2-cuda.json
```

To reproduce the Safe and Fast comparisons from the three raw JSON files
without either GPU, import `compare` from `tools/pointnet2parity.py` and
compare the CUDA result to each MPS result. The two comparison JSON files
include the raw input file SHA-256 hashes and full per-array checks.

## Scope

These fixtures cover finite coordinates, finite features and weights,
`M ≥ 3`, and distances whose float32 squared values do not overflow.
The original CUDA `three_nn` has special behavior below three reference
points and when distances overflow; this implementation deliberately validates
those cases differently. Near-tie selection and non-finite values were not
tested as cross-device parity claims. This is operator-level forward and
first-order backward evidence, not PointNet++ model-level accuracy or CUDA
performance evidence. The original extension build was adapted only to
compile on the available modern Windows toolchain; the kernel source and
algorithm remained unchanged.

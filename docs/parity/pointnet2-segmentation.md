# PointNet++ SSG segmentation: synthetic model-level validation

This check covers one fixed synthetic, finite float32 cloud (`B=1`, `N=1,152`,
six non-coordinate features, 13 synthetic classes). The model runs four set
abstraction stages (FPS → Ball Query → grouped MLP → max), four three-neighbor
feature propagation stages, a classification head, cross-entropy loss, and
backward through the full graph. The model is in `eval()` mode so BatchNorm
uses fixed state and dropout is disabled; backward still computes the input
and all parameter gradients. This is **numerical validation of one fixture**,
not segmentation accuracy on a labeled dataset or a training-convergence test.

## Provenance and execution

- Original source: [Pointnet2.PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch)
  at commit `b5ceb6d9ca0467ea34beb81023f96ee82228f626`.
  The harness imports its unchanged `pointnet2_modules.py` directly from an
  external pinned checkout. Its SA/FP stage dimensions mirror
  `pointnet2/models/pointnet2_ssg_sem.py` at that commit. The source files are
  **not copied** into this repository.
- Original model Git blobs: `pointnet2_ssg_sem.py`
  `d3f70bc0bfb5bad4e8cdca5bd3f58a8d8c8118d8`, `pointnet2_modules.py`
  `a0ad4f6bc23f54ca2d61454e657a6f533e9b875c`, `pointnet2_utils.py`
  `150fcccade21001971a76e6c3628963972305739`. The six checked source
  blob IDs and host file SHA-256 values are in each run's JSON. Git blob IDs
  are equal across hosts; checked-out bytes can differ because of LF/CRLF.
- CUDA: RTX 2080, PyTorch `2.14.0+cu132`, CUDA `13.2`, compiled original
  `pointnet2_ops._ext` on Windows. Only `setup.py` was adapted to target sm75
  and the MSVC conforming preprocessor; the model, wrapper, and CUDA kernel
  sources remained unchanged. The exact build-only diff and compiled extension
  SHA-256 are in [CUDA metadata](pointnet2-segmentation-cuda.json). The
  host-local extension directory was redacted from that metadata after
  capture; its filename, binary hash, and model-output NPZ bytes were kept.
- MPS: Apple M5 Pro, PyTorch `2.14.1`, `PYTORCH_ENABLE_MPS_FALLBACK=0`.
  [Safe](pointnet2-segmentation-mps-safe.json) and
  [Fast Math](pointnet2-segmentation-mps-fast.json) ran in separate processes.
  The MPS path uses the project's low-level operations and independently
  composed grouping classes with the original SA/FP Python layer classes.
- Fixed [fixture](pointnet2-segmentation-synthetic-fixture.npz) SHA-256:
  `e3998b2eb58c9b9a410ef2a081b539e65aca5d15fcdb72a203849ade0be511f1`.
  It fixes all coordinates, features, labels, parameters, and buffers.

The [external-checkout harness](../../tools/pointnet2_segmentation_parity.py)
reproduces the model-level run and stores all selected indices, logits, loss,
input gradient, and 68 parameter gradients in compressed NPZ files. The
[selection inspector](../../tools/analyze_pointnet2_selections.py) maps local
indices back to the original point IDs. The external upstream checkout and
original CUDA extension must be installed separately to rerun the CUDA side.

## Results and acceptance rule

All 71 floating-point arrays (logits, loss, input gradient, and 68 parameter
gradients) passed `numpy.allclose` with `atol=rtol=1e-4` in both modes. The
14,976 logits had maximum absolute error `1.431e-5` (Safe) or `1.526e-5`
(Fast). Loss differed by `5.245e-6`; the largest parameter-gradient error
was `6.676e-6`. Input-gradient maximum error was `1.639e-7` (Safe) or
`1.565e-7` (Fast). This is tolerance parity, not bitwise equality.

The 12 recorded selection arrays (four FPS, four Ball Query, four 3NN) were
compared **exactly** as local indices. Six passed and six failed in each mode,
so the overall raw comparison status is `77/83` passed, not `83/83`.
The full [Safe comparison](pointnet2-segmentation-compare-safe.json) and
[Fast comparison](pointnet2-segmentation-compare-fast.json) retain every
array's count and error. The paired raw tensors are
[MPS Safe](pointnet2-segmentation-mps-safe.npz),
[MPS Fast](pointnet2-segmentation-mps-fast.npz), and
[official CUDA](pointnet2-segmentation-cuda.npz).

### Why local indices differ

The [original-ID analysis](pointnet2-segmentation-selection-analysis.json)
finds the first difference at first-stage FPS selection 252. The previous 252
selected points are identical. MPS chooses original point 41 and CUDA chooses
752. Relative to that common prefix, both candidates have the same float64
minimum squared distance, `0.0033321380615234375`, and both maximize it over
the input. We did **not** inspect the two GPUs' internal distance register
bits, so this establishes a mathematical tie and an observed choice difference,
not an isolated proof of which device rule caused it.

The first-stage FPS sets contain the same 1,024 original points in a different
order. After mapping local indices and query rows to original point IDs, all
four 3NN stages select identical neighbor triples; Ball Query stages 1, 3,
and 4 also select identical rows. Stage 2 differs in one of 256 rows, at the
last of its 32 neighbors for original query point 148. Candidate original
IDs 33 and 216 are both within radius; the first-32-in-input-order contract
admits different final neighbors when the sampled source order differs.
Thus these observations do not identify an independent 3NN or Ball Query
kernel error. They also do not establish exact index parity for all inputs.

## CI reproduction and limits

[`tests/test_pointnet2_e2e.py`](../../tests/test_pointnet2_e2e.py) independently
builds a compact four-stage SSG graph with 96 synthetic points. It needs no
external upstream checkout. The fixed generator seed is `20261002`.
It checks CPU forward/backward and, when MPS is available, requires exact
CPU/MPS agreement for its 12 selected-index tensors and `atol=rtol=1e-4`
for logits, loss, input gradient, and every parameter gradient. This compact
test exercises the same operator chain, while the raw CUDA run above checks
the original layer classes and full stage sizes. These are separate claims.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python -m pytest -q tests/test_pointnet2_e2e.py
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 python -m pytest -q tests/test_pointnet2_e2e.py
```

On M5 Pro with PyTorch 2.14.1, each process reported `2 passed`; the
[Safe log](pointnet2-e2e-pytest-safe.log) and
[Fast log](pointnet2-e2e-pytest-fast.log) are preserved. A non-MPS CI runner
runs the CPU test and skips only the device comparison.
The complete test suite on the merged branch also passed with fallback
disabled: [Safe](pointnet2-e2e-full-safe.log) `286 passed, 13 skipped` and
[Fast](pointnet2-e2e-full-fast.log) `285 passed, 14 skipped`. The modes were
run in separate Python processes so the MPS shader cache used the requested
math setting in each run.

Remaining work: repeat full-model checks with labeled real clouds, measure
task accuracy/training convergence, and test more model/driver/hardware
combinations. The fixed synthetic result does not support a general PyG or
all-PointNet++ compatibility claim.

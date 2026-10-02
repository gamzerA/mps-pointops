# Bounded `spconv` 2.3.8 CUDA parity probe

On 2026-10-02, a Windows 10 (build 19045) machine with an NVIDIA GeForce
RTX 2080 ran the fixed fixtures in
[`tools/verify_spconv_toy_parity.py`](../../tools/verify_spconv_toy_parity.py)
against the private CPU sparse-convolution reference and `spconv` 2.3.8.
The source checkout was
`98ba436537e7905fc45c666ea8a7cf7b1a3878fb`. The exact command was:

```powershell
..\toyvenv\Scripts\python -m tools.verify_spconv_toy_parity --backend cuda --json toy.json
```

The [JSON result](spconv-toy-rtx2080.json) records three passing fixtures and
no failures. Two fixtures cover batched, unsorted, boundary, dilated, and
isolated SubM coordinates. The third covers a stride-2 convolution followed
by an inverse convolution with a reused `indice_key`. For each fixture,
output coordinates were compared as complete sorted keys before feature
values and first-order gradients were compared. The 15 recorded comparisons
cover forward features, input-feature gradients, kernel-weight gradients,
and bias gradients. All reported maximum absolute differences were zero;
the pass criterion was `rtol=1e-4, atol=1e-5`. The installed `spconv` weight
layout was `KRSC`, mapped to the private reference layout by the probe.

The Python environment used Python 3.11.16 and PyTorch 2.14.0+cu132. The
isolated `toyvenv` used the existing PyTorch installation through
`--system-site-packages`; `spconv-cu124==2.3.8` and the official CUDA 12.4
runtime/NVRTC wheels were installed only there. The existing research
environment was not modified. PyTorch's CUDA allocator was capped at 25% of
device memory. Its measured peak was 17,056,768 bytes allocated and
23,068,672 bytes reserved on an 8,589,475,840-byte GPU. That cap and peak
do not include native allocations made by `spconv`/`cumm`; they are not a
whole-process VRAM measurement. The fixtures contain at most eight input
points and are not performance benchmarks.

## Provenance and limits

The app host restarted after the Windows run and lost direct access to the
Windows `toy.json` file. The repository owner pasted the complete PowerShell
`Get-Content` output; the committed JSON reproduces its semantic content.
JSON number formatting was normalized (`0` versus `0.0`), so this repository
does not assert the original Windows file's byte hash. The committed JSON
has SHA-256
`c51b95426624e39877864dabcafe82547087905e4456609097bea4a`.

The probe source SHA-256 at that commit is
`aae487c13585c9b55c0c19bc6de3d8c83ae6d7b100584fcd728cbcd4a89d0f61`.
The CPU convolution reference is
`161b46a983d3b9cd043cc65489803a76f7b404ee75861ac219497a878f779f09`;
the rulebook source is
`75a30a5e754ba95f3cd69ca238e6dbd0b7712c7eea0df2873fae7075135b1384`.
The local CPU fixture smoke check at the same source commit independently
reproduced the two SubM pair counts (17 and 8).

This establishes direct CUDA-oracle agreement only for the listed small
fixed fixtures and the private CPU implementation. It does not establish
binary equivalence across arbitrary coordinates, algorithms, dtypes,
duplicate inputs, empty outputs, large clouds, or full models. It does not
exercise an MPS strided/inverse implementation or a public `spconv`-compatible
wrapper. The existing M1 and M5 Pro SubM checks are separate evidence.

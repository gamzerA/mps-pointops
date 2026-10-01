# Native MPS scatter backward: tied extrema and empty segments

Captured 2026-10-01T13:17:06.380964+00:00; `Apple M5 Pro`, macOS 26.5.2, PyTorch 2.14.1, Fast Math `1`.
Repository base `94398da77fb305039c9648380182d55747609ad1`; script SHA-256 `5bbea5d67a246ac25217ca0ff3105311dd249358efa780d615ad09e6ff31c37f`.
Reproduce: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 python3 bench/probe_scatter_backward_mps.py --output bench/results/2026-10-01-apple-m5-pro-scatter-backward-fast.json`

CPU and MPS forward were checked against an independent segment-wise Python oracle; MPS ran 20 complete forward/backward repetitions. The process disabled MPS CPU fallback before importing PyTorch.

| Operation | MPS↔CPU grad | Max MPS output error vs math | Max MPS grad error vs CPU | Max CPU grad error vs math | CPU grad matches math | MPS grad matches math | Output repeat bits | Gradient repeat bits |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| `scatter_add` | passed | 0 | 0 | 0 | True | True | True | True |
| `sum` | passed | 0 | 0 | 0 | True | True | True | True |
| `mean` | passed | 0 | 0 | 0 | True | True | True | True |
| `amax` | passed | 0 | 0 | 2 | False | False | True | True |
| `amin` | passed | 0 | 0 | 1.5 | False | False | True | True |

The source has shape `[6, 2]`; destinations are `[0, 0, 0, 2, 2, 2]` within a four-node output. Nodes 1 and 3 are empty but receive nonzero upstream gradients, testing that those gradients do not leak to source edges. Both channels contain exact tied extrema; node 2, channel 1 has a maximum of zero from a source edge, testing `include_self=False`.

The oracle splits `amax`/`amin` gradient among exact tied *source* extrema. PyTorch's observed native backward can differ when a nonempty segment's extremum equals the initial output zero, even with `include_self=False`; the table records that discrepancy rather than silently treating CPU as the mathematical oracle. Empty output segments stay zero. This establishes only this PyTorch version, dtype, fixture and device; it does not establish global bitwise determinism, NaN behavior, or `torch_scatter` arg-index compatibility. No kernel timing is included.

# Tested support matrix

This table is evidence, not an extrapolation to every Apple GPU or macOS
version. **macOS 26.5.2 is the oldest physical macOS version in the archived
project measurements; those measurements span different source commits.**
Hosted CI uses macOS 26.6.2. A lower project
support floor (such as macOS 14 or 15) requires a corresponding real test;
PyTorch's own MPS availability does not establish this package's Metal
kernel parity on that OS.

| Environment | Version and scope | What the evidence establishes |
| --- | --- | --- |
| Physical Apple M5 Pro, 48 GiB | macOS 26.5.2, PyTorch 2.7.0 and 2.14.1 for private sparse checks | Main operator, spatial, graph/voxel, and private SubM/strided/inverse correctness records; fixed synthetic OpenPCDet adapter forward/first-gradient parity. See each source-pinned record before using a performance number. |
| Physical Apple M1, 8 GiB | macOS 26.5.2; PyTorch 2.12.0 focused tests and 2.14.1 spatial/sparse runs | Safe/Fast operator, spatial, and Chamfer records, plus the clean-source `6959fd59` sparse runner: 92 tests passed per mode with no skips and synchronized SubM rulebook-backend benchmarks. This is targeted sparse validation, not full `spconv` compatibility. |
| GitHub-hosted Apple M1 Virtual | macOS 26.6.2; PyTorch 2.7.0 on Python 3.10, PyTorch 2.14.1 on Python 3.10/3.12 | Package suite in distinct Safe and Fast processes with MPS availability required. Virtual timing is not a physical-device benchmark. |
| GitHub-hosted PyG integration | PyTorch 2.12.0, PyG 2.8.0, pyg-lib 0.7.0 | Pinned operator subset on hosted MPS; it does not cover every PyG model or optional extension. |
| GitHub-hosted Linux | Python 3.12, CPU PyTorch | Import/package/reference checks; no Metal execution. |

The exact unpinned PyTorch version above comes from the
[passing CI job](https://github.com/gamzerA/mps-pointops/actions/runs/37015949889).
The version policy in `pyproject.toml` is Python `>=3.10` and PyTorch `>=2.7`,
without an upper bound. **A dependency range is not a continuous tested-version
claim.** Explicitly exercised versions include 2.7.0, 2.12.0, and 2.14.1.
Python 3.11 is declared but is not a separate CI row. The latest main commit
must pass its own CI before treating these older passing logs as its result.

## Device and feature limits

- M2, M3, and M4 have no physical-device parity or benchmark record here.
- M1 and M5 Pro do not share one automatic spatial routing policy. The BVH
  speed evidence from an explicit M1 call must not be reported as the M1
  `backend="auto"` speed.
- Flat and dense MPS kNN support at most 256 effective neighbors. The
  experimental explicit BVH is narrower (`K<=32`).
- Large single-cloud FPS multigroup selection is enabled automatically only
  for the measured M5 Pro case. Uneven batched multigroup FPS is unfinished.
- The MPS `chamfer_distance` subset uses float32 and has pinned upstream
  comparisons. Full PyTorch3D input coverage is a separate requirement.
- Private sparse convolution tests include three fixed `spconv` 2.3.8 CUDA
  toy fixtures against the CPU reference and a fixed synthetic M5 Pro
  OpenPCDet adapter pass. They do not establish public `spconv` 2.x
  compatibility, arbitrary-shape CUDA parity, trained-model accuracy, or a
  complete OpenPCDet pipeline. Ordinary and inverse rulebooks still build
  coordinates on CPU; feature arithmetic uses Metal within the documented
  bounds.

Read the [physical M1 scope](https://github.com/gamzerA/mps-pointops/blob/main/docs/spatial-m1-v090.md),
[M5 Pro private SubM evidence](https://github.com/gamzerA/mps-pointops/blob/main/docs/evidence/subm-integrated-m5pro-2026-10-02/README.md),
[current-source physical M1 sparse archive](https://github.com/gamzerA/mps-pointops/blob/main/docs/evidence/m1-sparse-2026-10-03/README.md),
[strided/inverse scope](https://github.com/gamzerA/mps-pointops/blob/main/docs/sparse-conv-stride-inverse-mps.md),
[CUDA toy comparison](https://github.com/gamzerA/mps-pointops/blob/main/docs/verification/spconv-toy-rtx2080.md),
[local OpenPCDet adapter check](https://github.com/gamzerA/mps-pointops/blob/main/docs/sparse-openpcdet-local-integration.md),
and [v0.9–v1.0 acceptance plan](https://github.com/gamzerA/mps-pointops/blob/main/docs/milestones-v0.9-v1.0.md)
for case lists and open gates.

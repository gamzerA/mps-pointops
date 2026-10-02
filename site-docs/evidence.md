# Benchmarks and evidence

Every number should carry a source commit, package/runtime version, device,
macOS version, input distribution, tensor shape and dtype, math mode, warmup,
number of timed samples, and whether `torch.mps.synchronize()` brackets the
measurement. A selected workload is not a device-wide speed promise.

| Topic | Source record | Read this before quoting |
| --- | --- | --- |
| Physical M1 1M-point spatial search | [M1 study](https://github.com/gamzerA/mps-pointops/blob/main/docs/spatial-m1-v090.md) | Explicit BVH versus scan, separate build and query; uniform, mixed, collapsed distributions. Automatic dispatch stayed on scan on M1. |
| M5 Pro BVH and radius dispatch | [kNN](https://github.com/gamzerA/mps-pointops/blob/main/docs/spatial-dispatch-v090.md) · [Ball Query](https://github.com/gamzerA/mps-pointops/blob/main/docs/spatial-radius-dispatch-v090.md) | Include the large-radius counterexample and one-shot index build cost. |
| M5 Pro memory and GPU activity | [allocator measurement](https://github.com/gamzerA/mps-pointops/blob/main/docs/spatial-memory-v090.md) · [Instruments study](https://github.com/gamzerA/mps-pointops/blob/main/docs/spatial-instruments-v090.md) | Six target-only Metal allocation captures and a separate seven-case GPU query trace. GPU Active time is a query aggregate, not named-kernel time; allocation maxima are not total physical GPU peak. Occupancy was not collected. |
| Flat API | [M5 Pro flat benchmark](https://github.com/gamzerA/mps-pointops/blob/main/docs/flat-api-bench-2026-10-02.md) | Read the stated point count, batch shape, and CPU baseline. |
| Voxel pooling | [M1/M5 stages](https://github.com/gamzerA/mps-pointops/blob/main/docs/voxel-api-benchmark.md) · [fused ablation](https://github.com/gamzerA/mps-pointops/blob/main/docs/voxel-fused-benchmark.md) | A fused kernel is opt-in; full-call results depend on density and batch layout. |
| Chamfer contention | [large bidirectional study](https://github.com/gamzerA/mps-pointops/blob/main/docs/chamfer-large-contention-2026-10-02.md) | Concentrated neighbor destinations expose a physical M1 backward penalty. |
| Private SubM | [integrated M5 Pro run](https://github.com/gamzerA/mps-pointops/blob/main/docs/evidence/subm-integrated-m5pro-2026-10-02/README.md) | Private prototype and CPU-rulebook comparison; not an upstream CUDA `spconv` or whole-model benchmark. |
| Physical M1 sparse operators | [raw M1 archive](https://github.com/gamzerA/mps-pointops/blob/main/docs/evidence/m1-sparse-2026-10-03/README.md) | Safe/Fast each 92 passed, zero skips, pinned to `6959fd59`. SubM benchmark compares CPU/MPS rulebook construction; both paths run feature arithmetic on MPS. |
| Windows CUDA sparse parity | [tiny `spconv` oracle](https://github.com/gamzerA/mps-pointops/blob/main/docs/verification/spconv-toy-rtx2080.json) | Three tiny fixtures and 15 output/first-gradient comparisons; not full upstream API or a CUDA model benchmark. |

Safe/Fast tests use separate processes with `PYTORCH_ENABLE_MPS_FALLBACK=0`.
Where JSON and SHA-256 manifests exist, cite the raw record alongside the
summary. Hosted CI is a correctness gate; physical M1/M5 Pro logs are the
performance evidence. The [acceptance plan](https://github.com/gamzerA/mps-pointops/blob/main/docs/milestones-v0.9-v1.0.md)
tracks missing cross-device, upstream, and physical-memory evidence.

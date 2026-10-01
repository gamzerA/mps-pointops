# Experimental PointNet++ and Chamfer MPS benchmark

- UTC: 2026-10-01T10:35:59.896479+00:00
- Device: Apple M5 Pro; macOS 26.5.2; PyTorch 2.14.1
- Source base commit: `471355bf6f341bbf9d83dfd1b1469f9159d55c68`; tracked operator sources clean: `True`
- Math mode: `0`; CPU fallback: `0`
- Reproduce: `PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 python bench/bench_v050_ops.py --warmup 4 --repeat 20 --seed 27001 --output bench/results/2026-10-01-apple-m5-pro-v050-experimental-safe.json`
- 4 warmups and 20 measured calls per case; seed 27001.

## Median synchronized wall time

| Operator | Shape | Forward (ms) | Backward (ms) |
| --- | --- | ---: | ---: |
| `three_nn` | B=2, N_unknown=512, M_known=1024, D=3 | 0.467 | — |
| `three_nn` | B=2, N_unknown=2048, M_known=4096, D=3 | 1.027 | — |
| `three_interpolate` | B=2, C=32, M_source=512, N_target=2048, neighbors=3 | 0.406 | 0.282 |
| `three_interpolate` | B=2, C=64, M_source=2048, N_target=8192, neighbors=3 | 0.592 | 0.812 |
| `chamfer_distance` | B=2, P=256, Q=256, D=3 | 0.922 | 0.507 |
| `chamfer_distance` | B=2, P=1024, Q=1024, D=3 | 1.312 | 0.814 |

## Method and scope

- Inputs are independently seeded float32 standard-normal coordinates/features; interpolation uses seeded random valid indices and normalized positive weights. Input construction, shader compilation, and initial finite-output checks precede timed calls.
- `three_nn` is forward-only by contract. `three_interpolate` backward uses a preallocated all-ones output gradient. Chamfer uses its default bidirectional squared-L2, point mean, and batch mean with no padding, normals, or weights.
- Each public call is bracketed by `torch.mps.synchronize()`. Python validation, allocation, dispatch, and autograd are included. These are not isolated shader timings. Cases run in table order, so thermal state may affect comparisons.
- No CPU baseline or cross-operator speedup claim is made. The JSON records every raw sample and exact SHA-256 hashes of the Python modules, Metal shaders, and this benchmark script.

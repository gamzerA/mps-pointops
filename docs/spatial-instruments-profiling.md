# Spatial query trace correlation harness

`bench/profile_spatial_instruments.py` prepares the six exact M5 Pro cases in
[`spatial-memory-v090-m5pro-clean`](../bench/results/spatial-memory-v090-m5pro-clean/)
for a **single external Instruments session**. It optionally adds one derived
uniform native-scan case paired with the archived uniform BVH input. It checks
the archived commit, six-case fixture matrix, and all archived implementation
SHA-256 values against the current checkout before any MPS dispatch. A change
to the kernels or fixture generator requires a new profile protocol and
explicit revalidation; the old records are never rewritten.

Each condition creates its input and builds its BVH **once**, warms the full
query twice, then performs five synchronized query repeats by default. The
process drops each condition's tensors and empties unused MPS cache before
the next condition. The run prints the PID and waits before dispatch so an
Instruments trace can be attached to that one process. `--mps-signposts`
enables PyTorch's MPS operation signposts with
`wait_until_completed=False`; this gives trace markers, not a GPU occupancy
measurement. It can perturb execution, so record whether it was enabled.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m bench.profile_spatial_instruments \
  --output-dir /tmp/mps-spatial-instruments-m5-20261003 \
  --repeats 5 --warmups 2 --include-uniform-scan \
  --mps-signposts --pause-before-s 45 --pause-after-s 20
```

Attach **Metal System Trace** (and any separately available GPU counter
instrument) to the printed Python PID before the pause expires. Save the
native `.trace` with the JSON output. If Instruments cannot expose a named
occupancy counter on the actual device, report that counter as *unavailable*;
GPU utilization is not interchangeable with shader occupancy. For the same
reason, PyTorch MPS signpost intervals and synchronized Python wall time must
not be relabeled as GPU kernel duration. Read kernel start/end or duration
from the GPU timeline and identify the actual kernel symbol:

| Path | Expected query kernels to find in the trace |
| --- | --- |
| Serial BVH | `bvh_knn_f32` |
| Split BVH | `bvh_seed_f32`, `bvh_search_micro_f32`, `bvh_merge_micro_f32` |
| Native full scan | `flat_knn_indices` |

The JSON records `host_unix_start_ns`/`host_unix_end_ns` and monotonic
`host_perf_start_ns`/`host_perf_end_ns` for each query repeat. These boundaries,
the PID, case name, repeat number, and printed markers make it possible to
correlate GPU trace rows. `host_synchronized_ns` includes Python dispatch,
allocation, queue submission, and `torch.mps.synchronize()`; it is **not** the
traversal kernel's GPU duration. `allocator_peak_during_query` is a PyTorch
allocator peak, `memory_after_query` is an endpoint, and
`archived_2026_10_02_memory` is copied from a prior run for the six original
cases and is `null` for the derived uniform scan. None measures whole-device
physical GPU memory. Preserve the Instruments trace or
export as separate evidence with its Xcode/Instruments version and precise
counter name; do not fill `gpu_kernel_duration_ns` or `gpu_occupancy` in the
manifest by inference from host or allocator numbers.

The script refuses an existing output directory and writes each case JSON
before moving to the next. This retains completed evidence if a later case
fails, but such a partial run is incomplete. The source checkout must be
clean. A seventh uniform native-scan case is generated only with
`--include-uniform-scan`; it cites the uniform BVH record as its **input
fixture**, not as a prior native-scan memory measurement.

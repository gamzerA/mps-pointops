# Phase 3 native scatter profiling decision, 2026-10-02

## Scope and evidence

This decision concerns whether the measured workloads justify replacing
PyTorch's native MPS scatter with a project Metal reduction kernel **now**.
It is not a claim that all PyG graphs run, or that a replacement is faster.
Each source below records its own commit, environment, inputs, validation,
and synchronized host-wall samples; the runs must not be combined into one
hardware or PyTorch-version comparison.

| Evidence | Tested workload | Observation |
| --- | --- | --- |
| [M5 Pro PyG 2.8 profile](2026-10-01-m5-pro-scatter-profile.md), [Safe](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-solo-safe.json) / [Fast](../../bench/results/2026-10-01-apple-m5-pro-pyg-scatter-solo-fast.json) | Physical M5 Pro, PyTorch 2.14.1, PyG 2.8.0, no optional scatter library. Two-layer GCN, GraphSAGE, and GAT on **synthetic** uniform and 80%-to-1%-hub graphs of 4,096 or 32,768 nodes; separate native scatter controls. | All 24 model/mode/graph executions completed forward/backward. CPU/MPS outputs and input gradients were compared within `rtol=atol=5e-4` **only at 4,096 nodes**; 32,768-node outputs/gradients were checked for finiteness. At 32,768 uniform nodes, Safe medians were 3.237 ms for C32 `scatter_add_`, 0.187 ms for scalar `amax`, and 46.137/25.345 ms for whole GAT forward/backward; Fast was 5.649, 0.197, and 34.644/27.830 ms. The controls differ from GAT's exact shapes and dispatch counts, so these numbers do not measure scatter's fraction of model time. GAT samples varied substantially. |
| [M5 Pro large graph fan-in probe](2026-10-01-m5-pro-scatter-fanin-large.md), [Safe](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-safe.json) / [Fast](../../bench/results/2026-10-01-apple-m5-pro-scatter-fanin-fast.json) | Physical M5 Pro, PyTorch 2.14.1 **without PyG**. Native scatter calls with synthetic uniform, hub, and all-to-one destinations; up to 2,097,152 edges. | At the largest size, all-to-one scalar `scatter_reduce_(amax)` forward was 2.76–4.42× uniform across original/reversed Safe/Fast runs. C32 `scatter_add_` had no stable fan-in penalty across sizes, modes, and orders. This is a native-call stress test, not a PyG model. |
| [M5 Pro C3 point-gradient probe](2026-10-01-m5-pro-scatter-xyz-fanin.md), [Safe](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-safe.json) / [Fast](../../bench/results/2026-10-01-apple-m5-pro-scatter-xyz-fast.json) | Same PyTorch 2.14.1 native-call environment without PyG; 1,048,576 synthetic three-channel contributions. | All-to-one `scatter_add_` forward was 2.55–2.97× uniform across original/reversed Safe/Fast runs. This isolates one possible Chamfer-gradient accumulation call; it is not full Chamfer backward. |
| [Physical M1 report](../phase3-physical-m1-2026-10-02.md#concentrated-destinations-are-a-severe-m1-counterexample), native [graph Safe](../../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-scatter-fanin.json) / [Fast](../../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-scatter-fanin.json) and [C3 Safe](../../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-scatter-xyz.json) / [Fast](../../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-scatter-xyz.json) | Physical M1, 8 GiB, separate PyTorch 2.14.1 environment **without PyG**; 1,048,576 synthetic contributions to uniform or one destination. | Safe C32 `scatter_add_` forward rose from 30.892 to 537.020 ms; C3 rose from 3.312 to 4,527.662 ms. Fast was 30.941 to 538.841 ms and 3.338 to 4,578.584 ms. Exact-value output and source-gradient checks passed. Memory pressure and thermal state were not controlled; one Fast backward result needs rerun. |
| [M1 forced-index Chamfer fixture](../phase3-physical-m1-2026-10-02.md#concentrated-destinations-are-a-severe-m1-counterexample), [Safe](../../bench/results/physical-m1-2026-10-02/mps-pointops-m1-safe-chamfer-contention.json) / [Fast](../../bench/results/physical-m1-2026-10-02/mps-pointops-m1-fast-chamfer-contention.json) | Same physical M1, but **PyTorch 2.12.0** package path; single-directional Chamfer, B=1 and N=2,048, with forced uniform/concentrated nearest indices. | Backward Safe medians were 0.968/3.814 ms and Fast 0.860/3.669 ms. Expected indices and analytic gradients passed. This is a full Chamfer backward for the stated fixture, not a representative PyG graph or the random bidirectional Chamfer benchmark. |

Every timed MPS call was bounded by `torch.mps.synchronize()` as described in
its source. The timings include host dispatch and runtime effects, not
isolated GPU-kernel duration. The all-to-one cases demonstrate distribution
sensitivity in these native calls; they **do not identify Metal atomics or
prove atomic contention**. The M5 PyG workload and the M1 stress probes use
different operators, input shapes, package environments, and chips.

## Decision

Keep native PyTorch scatter as the default for the tested PyG graph path.
The representative synthetic model runs establish bounded correctness and
native operator availability, but do not establish a model bottleneck or a
measured gain from a replacement. The severe M1 concentrated case and the
M5 scalar/C3 fan-in penalties justify a **targeted segmented reduction
candidate and ablation**, especially for heavily concentrated destinations;
they do not justify switching the public path before that comparison.

A candidate must first pin output shape/dtype, empty segments, index and tie
rules, zero-extremum backward behavior, and gradient parity against the
chosen PyTorch/PyG contract. Then compare identical shapes and destination
distributions against native calls on physical M1 and M5 Pro, including
grouping/sorting and buffer costs, Safe/Fast processes, correctness, and
synchronized complete GCN/GraphSAGE/GAT or Chamfer forward/backward. Include
real graph distributions, controlled repeats, and GPU-side tracing before
attributing a penalty to a particular GPU primitive. The proposed numerical
contract and zero-extremum counterexample are in the
[M5 profile](2026-10-01-m5-pro-scatter-profile.md#proposed-segmented-reduction-contract-design-only).

The **profiling and present-day adoption decision** can close for the pinned
synthetic workloads above: the decision is to retain native scatter and
measure a candidate where skew warrants it. The separate roadmap task to
**implement core Metal scatter reductions** remains open. No custom kernel,
model-level speedup, universal PyG coverage, or physical M2–M4 result is
established by this report.

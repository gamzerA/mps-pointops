# Untimed radius BVH fallback audit

`bench/probe_spatial_radius_fallback.py` fills one evidence gap in the v0.9
radius study. The original nine M5 Pro and twelve M1 public Ball Query
performance JSONs recorded output parity but not the BVH shader's per-query
stack-overflow flag. This runner **does not remeasure latency** or modify
those JSONs. It regenerates their exact fixtures, makes one private BVH
radius call and one native Metal full-scan call per fixture, then records the
overflow count, row indices, and output parity.

Run from a clean checkout with the package dependencies installed. Safe Math
and disabled MPS CPU fallback must be set **before** Python starts. Each
fixture runs in its own child process with a 180-second default wall limit;
the M1 matrix starts with 20k and 100k collapsed staging cases before any
one-million-point case. The output directory must be new and outside the
checkout. For the physical M1, one command runs all twelve fixtures:

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m bench.probe_spatial_radius_fallback \
  --matrix m1 --output-dir /tmp/mps-radius-overflow-m1-20261003
```

After other GPU experiments finish, run the nine-fixture M5 Pro matrix:

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m bench.probe_spatial_radius_fallback \
  --matrix m5 --output-dir /tmp/mps-radius-overflow-m5-20261003
```

Use a different output-directory name if one already exists. The command
refuses to overwrite results. On a failure or timeout, it stops the matrix,
retains completed case JSONs, and writes `manifest.json` with the failing
case. Do not describe a partial matrix as a passing one.

## Fixture and report contract

The runner reads the committed legacy JSONs as fixture specifications and
checks the historical source commit, clean-source flag, fixture generator
SHA-256, seed, float32 radius and radius-square arithmetic, cell size, query
source, and the complete expected case set. Uniform and cluster–sparse queries
are independent of reference points. Collapsed fixtures retain the original
half-coincident, half-displaced (`+1` on x) query construction. The shared
fixture generator has the same SHA-256 in both legacy matrices and this
checkout; a future change fails the preflight instead of silently changing
the sampled points. Each new JSON records hashes of the generated point and
query arrays as well as the legacy JSON file.

The private result's int64 indices must exactly match the native Metal scan.
Selected float32 squared distances must match **bitwise**; original-index
order and the old record's row-count pattern are checked. The shader's
`stats[:,4]` must contain only `0` or `1`; both its nonzero row indices and
their fraction of all queries are archived. A mismatch produces a nonzero
exit code and a failure record. The report also records the executing source
commit, hashes of the probe, fixture generator, wrappers and Metal shaders,
chip, RAM, macOS, Python, PyTorch, NumPy, and math-mode settings. No report
field is a performance measurement or total GPU-memory claim.

The source-level bound in the
[radius study](spatial-radius-dispatch-v090.md#bounded-dfs-stack-and-fallback-evidence)
shows a maximum DFS stack occupancy of 14 for the current `N≤1M`, 128-point
leaf, balanced-tree contract, below the capacity of 32. The new records are
still needed to distinguish observed frequencies from the legacy records'
missing fields and to catch any implementation error in that argument.

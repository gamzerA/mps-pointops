# M5 Pro BVH radius fallback diagnostic

These are the unedited JSON files emitted by
[`bench/probe_spatial_radius_fallback.py`](../../probe_spatial_radius_fallback.py)
on 2026-10-03 from clean commit
`4bfcb9192c468f0928842a7332fa84bde465e29c`.

Run command:

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m bench.probe_spatial_radius_fallback \
  --matrix m5 --output-dir /private/tmp/mps-radius-overflow-m5-20261003-run1
```

`manifest.json` contains the original per-file SHA-256 hashes. Nine of nine
cases passed. All 233,472 query rows had zero stack-overflow fallback flags;
BVH and native scan indices, selected float32 distance bits, original-index
order, and archived row-count categories agreed. These are untimed
diagnostics, not new performance results. The physical M1 matrix remains
pending.

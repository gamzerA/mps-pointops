# Physical M1 v0.9 development evidence

The 34 JSON files in [`bench/results/spatial-v090-m1`](../../../bench/results/spatial-v090-m1/)
were generated on a physical Apple M1 (8 GiB, macOS 26.5.2) from a clean
detached checkout at `ef07a9337b40065c18a6c950bda4240c789f31f1`. Each
JSON stores its own source commit, dirty flag, source-file SHA-256 map,
environment, fixture, raw timings and parity counts. Python was 3.10.6,
PyTorch 2.14.1, NumPy 2.2.6, and SciPy 1.15.3 for `cKDTree`. All performance
and memory commands set `PYTORCH_ENABLE_MPS_FALLBACK=0` and
`PYTORCH_MPS_FAST_MATH=0`. Runs were sequential, in fresh processes, with
per-process wall-time limits and outputs outside the checkout. The JSON
`timing_scope` and `memory_scope` fields define the reported counter meaning.

The spatial logs here and the [Chamfer logs](../chamfer-v090-m1/) use the
same clean source checkout with PyTorch 2.12.0. They are raw pytest stdout;
the logs alone do not embed a source hash or full command line. Equivalent
reproduction commands isolate Safe and Fast Math processes, disable MPS CPU
fallback, and avoid bytecode/cache writes:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=0 python -m pytest -q -rs -p no:cacheprovider \
  tests/test_spatial_public.py tests/test_spatial_bvh.py \
  tests/test_spatial_bvh_radius.py

PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
PYTORCH_MPS_FAST_MATH=1 python -m pytest -q -rs -p no:cacheprovider \
  tests/test_spatial_public.py tests/test_spatial_bvh.py \
  tests/test_spatial_bvh_radius.py
```

The companion [SHA-256 manifest](SHA256SUMS.txt) covers 34 JSONs and four
pytest logs. `python bench/validate_spatial_m1_evidence.py` verifies those
bytes, all recorded source hashes against the measured Git commit, and the
JSON parity counters. It does not rerun a GPU test. Interpretation and
limitations are in the [M1 report](../../spatial-m1-v090.md).

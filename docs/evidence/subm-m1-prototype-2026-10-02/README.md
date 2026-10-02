# Physical M1 SubM prototype verification — 2026-10-02

This archive records tests of **private, experimental** Submanifold Sparse
Convolution code. It is not a `spconv` compatibility or v1.0 release claim.

## Provenance

- Device: physical Apple M1, 8 GiB unified memory; macOS 26.5.2, arm64.
- Python: 3.10 virtual environment with PyTorch 2.14.1; MPS available.
- Source tested: commit `4946cd6c548b94300457bde856af99e0bf6442e1`
  (`codex/subm-gpu-rulebook`), tree
  `cf7cdbd5f0217748d381cf479f899eb129b196d5`. The same tree was
  cherry-picked into the draft PR branch as `8e984c5`.
- Source was exported with `git archive` into a fresh M1 temporary directory.
  The test runner was installed into a separate temporary directory; the
  existing PyTorch environment was not modified. NumPy was supplied from an
  existing temporary dependency directory.
- `PYTORCH_ENABLE_MPS_FALLBACK=0` in both fresh processes.

## Reproduction

From the exported source, run each mode in a fresh process:

```sh
PYTHONPATH="$PWD:/path/to/pytest:/path/to/numpy" \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
python -m pytest -q tests/test_sparse_rulebook.py \
  tests/test_subm_metal.py tests/test_subm_rulebook_mps.py

PYTHONPATH="$PWD:/path/to/pytest:/path/to/numpy" \
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
python -m pytest -q tests/test_sparse_rulebook.py \
  tests/test_subm_metal.py tests/test_subm_rulebook_mps.py
```

Safe and Fast each passed **47 tests**, with zero skips or failures. The tests
include SubM forward and first-order gradients for features, weights, and
bias; CPU-rulebook parity; the bounded MPS rulebook's pair/CSR parity; and a
rulebook-to-Metal-forward chain without an intermediate host sync. The raw
pytest output is in the two `.log` files. Verify their bytes with:

```sh
shasum -a 256 -c SHA256SUMS
```

The MPS rulebook uses an O(KN²) coordinate scan, requires validated unique
int32 coordinates, and is capped at 1,024 active points. The public sparse
convolution API, strided/inverse paths, large-cloud performance, and direct
`spconv 2.x` CUDA output/gradient parity remain unverified.

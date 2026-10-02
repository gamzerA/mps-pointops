# Bounded Strided/Inverse MPS regression evidence

Source commit: `029d4ff66441611095b220294539cfd6a4588c21` (including
the two-layer chain test). The Metal implementation commit is
`bffcddcf94e7263b03555c816e6a515fc16965fd`.
Hardware: Apple M5 Pro; macOS 26.5.2. This is an operator fixture check,
not a throughput or upstream `spconv` compatibility benchmark.

Each log is from a **separate process** with
`PYTORCH_ENABLE_MPS_FALLBACK=0`; `PYTORCH_MPS_FAST_MATH` was either `0`
(Safe) or `1` (Fast). The installed PyTorch versions were 2.7.0 and 2.14.1.
The command from the repository root was:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest -q tests/test_sparse_conv_mps.py \
  tests/test_sparse_conv_cpu.py tests/test_sparse_rulebook.py
```

Repeat with `PYTORCH_MPS_FAST_MATH=1` in a fresh process. All four cases
passed **42/42**; the raw output and SHA-256 digests are in this directory.
The target set contains eight new MPS tests plus one CPU CSR test and the
existing 33 CPU oracle/rulebook tests. Passing these small fixtures does not
close the device-resident geometry, sparse transpose, upstream CUDA parity,
or end-to-end sparse backbone gates.

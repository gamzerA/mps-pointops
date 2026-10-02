# Physical M1 sparse validation command

Run this after the Strided/Inverse Metal PR and the private `spconv` adapter
test suite have both been merged into a **clean, pinned checkout** on the M1
Mac. Use the existing local Python 3.10+ environment with PyTorch 2.7 or later
and pytest; the runner installs nothing and does not use SSH or global settings.

```bash
python tools/run_m1_sparse_validation.py \
  --commit "$(git rev-parse HEAD)" \
  --output ../m1-sparse-evidence-$(git rev-parse --short HEAD) \
  --bench-subm
```

`--bench-subm` is optional. It sequentially measures 1,025 and 10,000 SubM
rows through the existing CPU-rulebook and MPS-rulebook wrapper, with one
warmup and three synchronized repeats per case. Remove the flag to perform
only correctness checks. `--timeout-s` and `--bench-timeout-s` default to
900 seconds; both can be set explicitly. The output directory must be new
and outside the checkout so reruns cannot overwrite evidence.

The runner checks the full 40-character HEAD commit, a clean working tree,
Darwin/arm64 and an Apple M1-family CPU, PyTorch 2.7 or later, and an
available MPS backend. It runs the CPU sparse rulebook/reference, SubM GPU rulebook and
convolution, Strided/Inverse Metal, and adapter suites in separate Safe/Fast
processes with `PYTORCH_ENABLE_MPS_FALLBACK=0`. A failed test, skip, missing
suite, missing backend, timeout, or incomplete benchmark returns a nonzero
exit status. It writes raw pytest logs, environment summaries, relative-path
source SHA-256 values, optional benchmark JSON/logs, and a manifest with the
source commit and file hashes. The M1 hardware check identifies the chip
family; the operator running it confirms that the Mac is the intended physical
device. No M1 result is claimed until this command succeeds on that machine.

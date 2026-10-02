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
source commit and file hashes. It rechecks the clean commit and source hashes
after both modes complete. Older supported PyTorch versions may not expose an
MPS device-name API; the manifest then records a null device name alongside
the verified Apple M1 CPU brand. The M1 hardware check identifies the chip
family; the operator running it confirms that the Mac is the intended physical
device.

## Completed physical M1 run

On 2026-10-03, this command completed on an Apple M1/arm64 Mac with
macOS 26.5.2, Python 3.11.17, PyTorch 2.14.1, and clean source commit
`6959fd590deecbefc98a163d46c23debaaa67c9e`. The
[original logs and manifest](evidence/m1-sparse-2026-10-03/README.md) preserve
**92 passed, zero skipped** in each of Safe and Fast Math, with MPS CPU
fallback disabled. The manifest's eight artifact hashes and 93 source-file
hashes were independently checked against the received bytes and the pinned
Git commit. The optional 1,025/10,000-row benchmark finished in both modes.

For that benchmark, `cpu` and `mps` distinguish **rulebook generation** only;
both paths execute the SubM feature convolution and backward on MPS. The
forward timing includes validation and rulebook generation, while backward
starts after a synchronized forward. The evidence README reports all eight
three-repeat medians and the 10,000-row backward reversal. The result does
not validate pure CPU-convolution versus GPU-convolution speed, other Apple
chips, larger sparse tensors, or full `spconv` API compatibility.

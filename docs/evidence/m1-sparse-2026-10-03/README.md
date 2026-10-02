# Physical M1 sparse operator validation, 2026-10-03

The eight `.log`/benchmark `.json` files and `manifest.json` in this directory
are byte-for-byte copies of the received M1 output from
`tools/run_m1_sparse_validation.py`. This README was added after receipt and
is not part of the manifest. The manifest identifies clean source commit
`6959fd590deecbefc98a163d46c23debaaa67c9e`, Apple M1/arm64,
macOS 26.5.2, Python 3.11.17, PyTorch 2.14.1, and disabled MPS CPU fallback.
All eight artifact SHA-256 values match the files here. All 93 relative source
hashes were independently checked against `git show 6959fd59:<path>`.

The six pinned suites in the manifest passed **92/92 tests in Safe Math** and
**92/92 in Fast Math**, with no skips. The suites cover CPU sparse rulebooks,
SubM MPS rulebooks and arithmetic, ordinary Strided/Inverse MPS arithmetic,
and the bounded `spconv` adapter. These are correctness checks on one physical
M1 and the pinned source, not a claim of full upstream `spconv` compatibility.

The optional SubM benchmark completed 1,025 and 10,000 active rows in each
math mode, with one warmup and three synchronized repeats. Its `cpu` and
`mps` names refer **only to the rulebook construction backend**. Both paths
run SubM feature arithmetic on MPS. The `cpu` row is therefore **not** a
CPU-convolution baseline, and its forward time includes coordinate validation
and rulebook construction.

| Math | Rows | Rulebook backend | Forward median (ms) | Backward median (ms) |
| --- | ---: | --- | ---: | ---: |
| Safe | 1,025 | CPU | 17.105 | 3.348 |
| Safe | 1,025 | MPS | 1.940 | 0.822 |
| Safe | 10,000 | CPU | 249.601 | 4.041 |
| Safe | 10,000 | MPS | 15.060 | 6.489 |
| Fast | 1,025 | CPU | 17.298 | 3.046 |
| Fast | 1,025 | MPS | 1.882 | 0.794 |
| Fast | 10,000 | CPU | 250.200 | 3.774 |
| Fast | 10,000 | MPS | 14.185 | 6.572 |

The table reports medians from three samples per row and describes this
fixture only. At 10,000 rows the MPS-rulebook path has a slower **backward**
median than the CPU-rulebook path in both modes; the forward medians show the
opposite order. The raw per-repeat samples and exact values are preserved in
`subm-benchmark-safe.json` and `subm-benchmark-fast.json`.

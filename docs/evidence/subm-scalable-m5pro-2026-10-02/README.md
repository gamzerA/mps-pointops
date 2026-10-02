# Scalable private SubM rulebook prototype (2026-10-02)

Source baseline: `origin/main` at `d2efed964779cd9a6ce00f1992e828f213bc028d`.
Device: Apple M5 Pro, macOS 26.5.2, PyTorch 2.14.0. The private API and
package version remain unchanged.

The prior lookup compared each of `K*N` targets with up to `N` input rows and
rejected `N > 1024`. This prototype stably sorts row IDs by the four int32
coordinate fields, then performs a Metal binary search for each target. The
lookup work is `O(N log N + K*N log N)`; sorting and compaction stay on MPS.
The target arithmetic is signed int64, with explicit int32 range checks before
search, so an out-of-range neighbor cannot wrap to a valid coordinate. Input
coordinates still require uniqueness and valid range from the caller. The
new bound is the Metal 1-D dispatch of at most `2**32-2` slots; available
device memory will limit realistic inputs earlier.

The independent CPU oracle and Metal tests passed **51/51** in separate Safe
and Fast Math processes with CPU fallback disabled. Tests include unsorted
two-batch clouds at 1,025 and 10,000 rows, full int32 batch/coordinate and
dilation boundaries, a 1,024-row 5×5×5 kernel above the old slot limit,
preserved input snapshot, exact padded pair/CSR order, and chaining the CSR
into the Metal forward kernel.

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest tests/test_sparse_rulebook.py tests/test_subm_rulebook_mps.py tests/test_subm_metal.py -q
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python -m pytest tests/test_sparse_rulebook.py tests/test_subm_rulebook_mps.py tests/test_subm_metal.py -q
```

The benchmark times the complete private rulebook construction on already
resident MPS coordinates, including four sorts, lookup, scans, allocation,
and synchronization. Each case uses 20 warmups and 50 measured calls on a
fixed, shuffled, two-batch 64×32×16 coordinate universe with a
3×3×3 kernel. Medians are host wall time in milliseconds; the raw samples are
in the adjacent JSON files.

| Math | Rows | Original scan | Sorted lookup |
| --- | ---: | ---: | ---: |
| Safe | 1,024 | 0.801 | 0.329 |
| Fast | 1,024 | 0.774 | 0.433 |
| Safe | 10,000 | unsupported | 0.783 |
| Fast | 10,000 | unsupported | 0.786 |

The 1,024-row comparison uses the same fixed input in the `d2efed9` source
worktree. Benchmarks are single-device observations, and the original scan
could not run at 10,000 rows because of its explicit row limit.

Reproduce the new path with
`python bench/bench_subm_rulebook_mps.py --warmup 20 --repeats 50` under each
environment above. The original scan result used the same generator, kernel,
warmup, repetitions, and synchronization against the baseline source.

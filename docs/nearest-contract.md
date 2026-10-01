# Legacy `torch_cluster.nearest` MPS contract

This implementation targets the `nearest(x, y, batch_x=None, batch_y=None)`
entry point of [`torch-cluster` 1.6.3](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/torch_cluster/nearest.py),
at source commit `29cd22bf1a5b82fc06b108d6573f81302c5d6b12`. For each row of
`x`, it returns one **global index into `y`**. This direction differs from the
project's `flat.knn(x, y, k)` edge convention, in which `y` supplies queries.
Both inputs may be one-dimensional or shaped `(N, D)` and `(M, D)`, with
`D >= 1`. The result is an `int64` vector of length `N` on the input device.

## Batches and edge cases

`batch_x` and `batch_y` are optional sorted integer vectors. Missing IDs can
form empty gaps. The set of **occupied** IDs must agree, as checked by the
[upstream CUDA Python wrapper](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/torch_cluster/nearest.py#L49-L79).
A vector omitted on one side means one batch with ID zero. A batch present on
only one side raises `ValueError`. Empty `x` and `y` together return an empty
vector; an empty `y` for a nonempty `x` raises `ValueError`. These empty-input
results are explicit safety choices: upstream's SciPy CPU path can return an
undefined index for an empty `y`, and its CUDA branch has no defined zero-grid
launch result for empty `x`.

MPS supports matching float32 inputs. Other MPS dtypes raise `TypeError`;
CUDA tensors passed to the shim raise `NotImplementedError` so they cannot
silently run on CPU. CPU tensors use a direct PyTorch reference for the CUDA
threshold and lane tie rule. The upstream CPU implementation instead uses
SciPy `vq` and may diverge at ties, close floating-point decisions, or
nonfinite inputs. The integer result has no gradient.

## Selection rule and numerical limit

The [legacy CUDA kernel](https://github.com/rusty1s/pytorch_cluster/blob/1.6.3/csrc/cuda/nearest_cuda.cu)
starts each of 1024 CUDA lanes at squared distance `1e38` and global index
`0`. Lane `t` visits local indices `t, t + 1024, ...` and replaces its best
only for a strictly smaller squared distance. Its reduction retains the lower
lane on an equal distance. Accordingly, the MPS kernel scans only the matching
`y` batch and chooses the smallest
`(distance², local_index mod 1024, local_index)` among distances strictly less
than float32 `1e38`. For more than 1024 references, this can **differ from
the smallest global index**. If no candidate qualifies, it returns the legacy
global index `0`, even for later batches.
NaNs, infinity, and float32 overflow cannot beat the finite initializer.

Distances are accumulated directly in increasing feature order. Metal
contraction is disabled in this kernel, while the CUDA compiler may contract
multiply-adds. Near ties and underflow boundaries therefore have no bitwise
cross-backend guarantee. A fixed batch fixture and well-separated
feature inputs pass exact-index checks. An adversarial tie at local indices 1
and 1024 returns 1024 by the CUDA source rule while upstream SciPy CPU returns
1. No CUDA binary was executed on the Apple test machine; no general CUDA
binary parity or speedup claim follows from the source and CPU comparisons.

## Reproduction

`tools/verify_nearest_upstream.py` loads the **unmodified** upstream
`torch_cluster/nearest.py` from the pinned 1.6.3 checkout. It verifies its
SHA-256 before comparing the actual SciPy CPU output with this project. It
also records the known CPU/CUDA-style tie difference. The upstream source is
not redistributed here. Each of CPU, MPS Safe Math, and MPS
Fast Math matched all 408 indices across an eight-row batch fixture and
four 100-row feature cases (D=1, 3, 64, 128):

- [CPU comparison](parity/nearest-upstream-cpu-2026-10-01.json)
- [MPS Safe comparison](parity/nearest-upstream-mps-safe-2026-10-01.json)
- [MPS Fast comparison](parity/nearest-upstream-mps-fast-2026-10-01.json)

The latter two used M5 Pro, PyTorch 2.14.1, and
`PYTORCH_ENABLE_MPS_FALLBACK=0` in separate processes. The upstream CPU
function uses SciPy 1.18.1. To repeat a comparison from a local checkout of
the tagged upstream source:

```bash
PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python tools/verify_nearest_upstream.py \
  --upstream-file /path/to/pytorch_cluster/torch_cluster/nearest.py \
  --device mps
```

The native kernel's regression tests also exercise 1024-lane index ties, gaps, full
threadgroup output, empty and mismatched batches, float32 threshold equality,
nonfinite input, D=64/128 accumulation, and explicit dtype failure in both
Safe and Fast Math. On the branch rebased onto main `8b19dc84`, the full MPS
suite completed with [312 passed, 15 skipped in Safe Math](pytest-nearest-safe-torch214-2026-10-01.log)
and [311 passed, 16 skipped in Fast Math](pytest-nearest-fast-torch214-2026-10-01.log).
The PyG 2.8 optional integration tests were skipped in this environment;
their separate CI job remains the gate for that path. Only the local pytest
`rootdir` path was normalized in these logs. A local wheel and sdist build
both contained `nearest.py` and `nearest.metal`; the package CI job checks
those files explicitly.

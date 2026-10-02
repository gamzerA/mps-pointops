# Physical M1 direct PyTorch3D Chamfer parity

These are raw results from a physical Apple M1 MacBook Pro (8 GiB, macOS
26.5.2), not a release or a full PyTorch3D compatibility claim. The package
source was clean commit `ef07a9337b40065c18a6c950bda4240c789f31f1`.
The CPU oracle was the official PyTorch3D 0.7.9 source at
`88e182f989c80836f4bd744e0d9cb1852762ce01`. Only two C++ standard flags
in its temporary `setup.py` build were changed from C++17 to C++20 for
PyTorch 2.14.1; the algorithm was not edited or copied here.

`PYTORCH_ENABLE_MPS_FALLBACK=0` was set. Safe and Fast Math each ran in a
separate Python process. The base verifier checked L1 and squared-L2 loss,
first derivatives, ragged lengths, weights, reduction modes, ties, and
coincident points. The extended verifier additionally checked normal loss,
first normal gradients, `Pointclouds` inputs, and selected empty-cloud cases.

| Matrix, per math mode | Cases | Checked tensors | Failed checks | Failed elements | Maximum absolute error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 480 | 3,240 | 0 | 0 | 1.9073486328125e-6 |
| Normals and `Pointclouds` | 252 | 3,720 | 0 | 0 | 1.9073486328125e-6 |

The numerical pass rule is `abs(actual-reference) <= 2e-5 +
2e-4*abs(reference)`, applied elementwise. Output signatures and gradient
presence are compared exactly. The base JSON's large `max_rel_nonzero` comes
from division by a nearly zero reference value; it is not a failed check.

`chamfer-{safe,fast}.json` and `chamfer-extended-{safe,fast}.json` contain all
case records. Corresponding `.log` files record process summaries. The
The JSON and log bytes were copied from the M1 without alteration. The
build-flags diff was stripped of blank-line trailing spaces for the repository;
its SHA-256 in `SHA256SUMS` covers the archived form. The manifest covers ten
evidence files, while this README is an explanatory addition. Verify from this
directory with `shasum -a 256 -c SHA256SUMS`, or from the repository root
with `python3 bench/validate_chamfer_m1_evidence.py`. The latter also checks
all case statuses and measured source-file hashes against the Git commit.

This finite float32 matrix does not cover every PyTorch3D argument and error
path, second derivatives, CUDA parity, or nonfinite inputs. Its measured
source is development code, separate from the published v0.8.0 wheel.

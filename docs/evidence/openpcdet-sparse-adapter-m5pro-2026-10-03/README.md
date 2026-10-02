# Pinned OpenPCDet sparse adapter smoke checks — Apple M5 Pro

These records were produced on macOS 26.5.2, Apple M5 Pro, from adapter
commit `06a99838473f354ffbb3006661c91137e6ac59d9` and the unmodified
OpenPCDet model source at commit
`233f849829b6ac19afb8af8837a0246890908755`. The JSON files also
contain SHA-256 hashes of every participating local CPU/Metal source file and Git blob SHA-1
hashes of both upstream model files. The companion `.log` files are the
direct stdout of the verification command. `SHA256SUMS` covers both formats.

| Process | PyTorch | Fast Math | MPS fallback | Backbone max normalized error | UNet max normalized error |
| --- | --- | --- | --- | ---: | ---: |
| `torch27-safe` | 2.7.0 | 0 | 0 | 0.033291 | 0.028487 |
| `torch27-fast` | 2.7.0 | 1 | 0 | 0.033291 | 0.028487 |
| `torch214-safe` | 2.14.1 | 0 | 0 | 0.033291 | 0.028487 |
| `torch214-fast` | 2.14.1 | 1 | 0 | 0.033291 | 0.028487 |

The normalized error is `|MPS − CPU| / (1e−5 + 1e−4·|CPU|)`; a pass requires
every value to be at most 1. Both models matched every sparse module's exact
integer coordinate values and row order. The backbone compared 28 module
outputs and 36 nonzero parameter gradients; UNetV2 compared 57 module
outputs and 84 nonzero parameter gradients. Both also compared encoded
features, input gradients, and UNetV2 point features. The first SubM layer's
output, input gradient, and weight gradient matched a separate dense Conv3d
oracle exactly under this fixed diagnostic fixture.

The fixture has 13 active voxels, one batch, and deterministic diagnostic
weights. It is not a trained checkpoint or full OpenPCDet pipeline. The
verification process substitutes only the sparse utility import and voxel
center helper. It does not compare against official CUDA `spconv 2.x`; that
separate toy-case gate is tracked independently. See the
[scope and reproduction guide](../../sparse-openpcdet-local-integration.md).
The JSON smoke-run wall times include compilation and Python overhead and
are not performance measurements.

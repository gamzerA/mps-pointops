# Private sparse adapter: pinned OpenPCDet local check

The private `mps_pointops._spconv_compat` module is an opt-in, bounded adapter
for local experiments. It exposes the sparse tensor, sequential container,
SubM, strided, and inverse 3D convolution calls exercised by the pinned
OpenPCDet `VoxelBackBone8x` and `UNetV2` classes. It does not install a
`spconv` package, modify `sys.modules` outside the verification process,
provide a public drop-in API, or establish full OpenPCDet compatibility.
For SubM, only `padding=0` and the kernel's implicit centered padding are
accepted; both map to the centered active-site rulebook. Other padding values
raise an error rather than being ignored.

## Fixed source and fixture

The [verification script](../tools/verify_openpcdet_sparse_adapter.py) loads
unmodified `spconv_backbone.py` and `spconv_unet.py` from
[OpenPCDet commit `233f849829b6ac19afb8af8837a0246890908755`](https://github.com/open-mmlab/OpenPCDet/tree/233f849829b6ac19afb8af8837a0246890908755/pcdet/models/backbones_3d).
It checks their Git blob SHA-1 values before importing them. Upstream files
are intentionally absent from this repository. Only the two utility imports
used by these classes are substituted in the test process: `spconv_utils`
routes sparse calls to the private adapter, and `common_utils` supplies the
reported voxel-center coordinates for `UNetV2`. The latter substitution is
not an upstream `common_utils` parity test.

The fixture has one batch, 13 distinct active voxels, a `[z,y,x]` sparse
shape of `[33,8,8]`, and positive two-channel features. The models run in
evaluation mode with fixed diagnostic weights that preserve nonzero paths
through every sparse layer. These weights are not a trained checkpoint.
The scalar diagnostic loss includes terms from every sparse module output,
which ensures every model parameter has a nonzero first gradient; it is not
OpenPCDet's detection or segmentation training loss.
An independent dense PyTorch `Conv3d` check covers the first SubM layer's
selected output rows and its first input and weight gradients. The separate
unit suite uses nonzero off-center weights and checks coordinate rulebooks,
cache lineage, and first gradients for SubM → stride → inverse.

## Reproduce locally

Fetch exactly the two official files from the pinned commit, then run the
script with the source directory. It verifies the file hashes automatically.

```bash
commit=233f849829b6ac19afb8af8837a0246890908755
mkdir -p /tmp/mps-pointops-openpcdet
for file in spconv_backbone.py spconv_unet.py; do
  curl -fsSLo "/tmp/mps-pointops-openpcdet/$file" \
    "https://raw.githubusercontent.com/open-mmlab/OpenPCDet/$commit/pcdet/models/backbones_3d/$file"
done

PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python tools/verify_openpcdet_sparse_adapter.py \
  --upstream-dir /tmp/mps-pointops-openpcdet --device both \
  --json-out /tmp/openpcdet-safe.json
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1 \
  python tools/verify_openpcdet_sparse_adapter.py \
  --upstream-dir /tmp/mps-pointops-openpcdet --device both \
  --json-out /tmp/openpcdet-fast.json
```

Safe and Fast Math must run in separate Python processes because PyTorch MPS
settings are captured during process initialization. On a machine without
MPS, use `--device cpu` to run the upstream-source, first-layer dense-oracle,
and CPU model forward/backward checks. Run the adapter unit suite with
`python -m pytest -q tests/test_spconv_compat.py`.

## Evidence and limits

The four local JSON records in
[`docs/evidence/openpcdet-sparse-adapter-m5pro-2026-10-03/`](evidence/openpcdet-sparse-adapter-m5pro-2026-10-03/README.md)
cover PyTorch 2.7.0 and 2.14.1, Safe and Fast Math, on Apple M5 Pro with
macOS 26.5.2. PyTorch MPS CPU fallback was disabled for all four runs.

| Model | Sparse convolutions | Sparse module outputs compared | Nonzero parameter gradients | Largest normalized CPU–MPS error |
| --- | ---: | ---: | ---: | ---: |
| `VoxelBackBone8x` | 12 | 28 | 36 / 36 | 0.0333 |
| `UNetV2` | 28 | 57 | 84 / 84 | 0.0285 |

The normalized error is `abs(MPS − CPU) / (1e−5 + 1e−4·abs(CPU))`; every
compared value is below 1. Exact coordinate values and row order are required
at every sparse module. The comparison includes the encoded features,
`UNetV2` point features, input gradients, and every present parameter
gradient. In this diagnostic fixture, the first SubM dense-oracle output,
input gradient, and weight gradient had zero measured difference. Absolute
error can reach `0.0103` in the UNet because some synthetic activations and
gradients are large; the normalized bound gives the relevant scale.

This establishes a local fixed-fixture CPU–MPS integration check, not direct
`spconv 2.x` CUDA parity, trained model accuracy, arbitrary-shape support,
full model deployment, or a benchmark. The JSON wall times include first-run
compilation and Python overhead and must not be used as performance claims.
The adapter validates and stores coordinates on CPU. Ordinary and inverse
convolution rulebooks are built there; the SubM path uses the existing GPU
rulebook by default. Coordinate transfer, host rulebook work, and launch
overheads remain to be profiled before public use.

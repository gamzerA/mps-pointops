# mps-pointops reference

Metal point-cloud, graph, and geometry operators for PyTorch tensors on Apple Silicon.

This reference describes the **repository source** and the bounded v1.0.0
release scope. The latest published package and source on `main` may differ
during release preparation. Check [PyPI](https://pypi.org/project/mps-pointops/)
and the [release tags](https://github.com/gamzerA/mps-pointops/releases)
before attributing a feature to an installed version. Opt-in research paths
and private modules do not imply complete upstream-library compatibility.

<div class="reference-actions">
  <a href="https://gamzerA.github.io/mps-pointops/">Interactive explainer</a>
  <a href="https://github.com/gamzerA/mps-pointops">Source and issues</a>
  <a href="https://doi.org/10.5281/zenodo.23076057">Version archive series</a>
</div>

## Start here

```bash
python -m pip install mps-pointops
```

Native Metal calls require an Apple Silicon Mac, Python 3.10 or later, an MPS-capable
PyTorch build (package metadata requires `torch>=2.7`), and
`torch.backends.mps.is_available() == True`. CPU tensors use the PyTorch reference
path where the public function documents one; an MPS request does not silently
move to CPU. See the [tested combinations and limits](support.md) before deploying
on another OS or GPU generation.

The [API index](api.md) distinguishes public, opt-in, and private modules. The
[numerical contracts](numerical.md) explain ordering, rounding, and gradients.
The [evidence guide](evidence.md) links raw logs and synchronized measurements
to the exact hardware and workload they cover.

## Reproduce a Safe Math run

Set the mode **before** starting Python. The MPS compiler caches the setting
inside a process.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
  python -m pytest -ra tests
```

Run Fast Math in a fresh process with `PYTORCH_MPS_FAST_MATH=1`. Fast Math is a
separate observed configuration, not an extension of the Safe Math numerical
guarantee.

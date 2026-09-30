# Ball Query provenance

The Ball Query Metal kernel, its Python autograd wrapper, contract tests,
numerical analysis and floating-point probe were developed in a separate
local `opensouce_mps` prototype and ported into this repository. They retain
the prototype's MIT notice in
[LICENSES/MIT-ball-query.txt](../LICENSES/MIT-ball-query.txt); the repository's
other files are under Apache-2.0.

The implementation follows the publicly documented mathematical behavior of
[PyTorch3D Ball Query](https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/ops/ball_query.py):
strict radius comparison, first K matches in source order, and `-1`/zero
padding. The first-K approach is also present in
[PointNet++](https://github.com/charlesq34/pointnet2/blob/master/tf_ops/grouping/tf_grouping_g.cu).
Those projects' source files were used to compare behavior; they were not
copied into the Metal or Python implementation. The implementation is not
claimed to be the first or only MPS point-cloud project.

See [the numerical contract](ball-query-math.md),
[the probe](../tools/metal_fp_probe.py), and the preserved
[Safe](metal-fp-probe-safe.json) and [Fast](metal-fp-probe-fast.json) results.
The probe records observed behavior on the listed host and compiler; it does
not certify behavior on all Apple GPUs.

import sys

import numpy as np
import pytest
import torch

from mps_pointops import compat, knn, reference

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
NAMES = ["pointnet2_ops", "pointnet2_ops.pointnet2_utils", "knn_cuda"]


@pytest.fixture
def installed():
    saved = {n: sys.modules.pop(n) for n in NAMES if n in sys.modules}
    compat.install(force=True)
    yield
    for n in NAMES:
        sys.modules.pop(n, None)
    sys.modules.update(saved)


def cloud(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, 3, generator=g)


def test_install_serves_imports(installed):
    from knn_cuda import KNN
    from pointnet2_ops import pointnet2_utils

    import pointnet2_ops.pointnet2_utils as direct

    assert direct is pointnet2_utils
    assert pointnet2_utils.furthest_point_sample is compat.furthest_point_sample
    assert KNN is compat.KNN


def test_install_leaves_existing_modules_alone(installed):
    marker = object()
    sys.modules["knn_cuda"] = marker
    assert "knn_cuda" not in compat.install()
    assert sys.modules["knn_cuda"] is marker


@mps
def test_fps_skips_points_near_origin_like_pointnet2():
    xyz = cloud(1, 2000)
    # 1e-3 is the squared-magnitude cutoff; put points just inside and outside it.
    inside = np.float32(np.sqrt(np.float32(1e-3))) * np.float32(0.999)
    outside = np.float32(np.sqrt(np.float32(1e-3))) * np.float32(1.001)
    xyz[0, 1:40] = torch.tensor([inside, 0.0, 0.0]) * torch.rand(39, 1)
    xyz[0, 40] = torch.tensor([float(inside), 0.0, 0.0])
    xyz[0, 41] = torch.tensor([float(outside), 0.0, 0.0])

    got = compat.furthest_point_sample(xyz.to("mps"), 2000).cpu()
    want = reference.furthest_point_sample(xyz, 2000, skip_near_origin=True).int()
    assert got.dtype == torch.int32
    assert torch.equal(got, want)
    picked = set(got[0].tolist())
    assert picked.isdisjoint(range(1, 41))  # near-origin points are never picked
    assert 41 in picked


@mps
def test_fps_all_points_near_origin_repeats_start():
    xyz = torch.zeros(1, 50, 3)
    got = compat.furthest_point_sample(xyz.to("mps"), 5).cpu()
    assert got.tolist() == [[0, 0, 0, 0, 0]]


def test_gather_and_grouping_operations():
    features = torch.randn(2, 4, 30)
    idx = torch.randint(0, 30, (2, 7), dtype=torch.int32)
    out = compat.gather_operation(features, idx)
    assert out.shape == (2, 4, 7)
    assert torch.equal(out[1, :, 3], features[1, :, idx[1, 3]])

    gidx = torch.randint(0, 30, (2, 7, 5), dtype=torch.int32)
    out = compat.grouping_operation(features, gidx)
    assert out.shape == (2, 4, 7, 5)
    assert torch.equal(out[0, :, 6, 2], features[0, :, gidx[0, 6, 2]])


def test_gather_operation_backpropagates():
    features = torch.randn(1, 3, 10, requires_grad=True)
    compat.gather_operation(features, torch.tensor([[2, 2, 5]])).sum().backward()
    assert features.grad[0, 0].tolist() == [0, 0, 2, 0, 0, 1, 0, 0, 0, 0]


def test_ball_query_pads_with_first_neighbor_like_pointnet2():
    xyz = torch.tensor([[[5.0, 0, 0], [0.1, 0, 0], [9, 9, 9], [0.2, 0, 0]]])
    new_xyz = torch.tensor([[[0.0, 0, 0], [50.0, 50, 50]]])
    idx = compat.ball_query(0.5, 4, xyz, new_xyz)
    assert idx.dtype == torch.int32
    assert idx.tolist() == [[[1, 3, 1, 1], [0, 0, 0, 0]]]


@mps
@pytest.mark.parametrize("transpose_mode", [True, False])
def test_knn_module_layouts(transpose_mode):
    ref = cloud(2, 3000).to("mps")
    query = cloud(2, 100, seed=1).to("mps")
    want_d, want_i = knn(query, ref, 16)
    module = compat.KNN(16, transpose_mode=transpose_mode)
    if transpose_mode:
        d, i = module(ref, query)
    else:
        d, i = module(ref.transpose(1, 2), query.transpose(1, 2))
        d, i = d.transpose(1, 2), i.transpose(1, 2)
    assert torch.equal(i, want_i) and torch.equal(d, want_d)

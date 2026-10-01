"""Focused Pointcept v1.2.1 PTv1 pointops subset checks."""

import sys

import pytest
import torch

from mps_pointops import compat, pointcept

DEVICES = [
    "cpu",
    pytest.param(
        "mps", marks=pytest.mark.skipif(
            not torch.backends.mps.is_available(), reason="MPS unavailable"
        ),
    ),
]


@pytest.mark.parametrize("device", DEVICES)
def test_fps_uses_cumulative_offsets_and_global_int32_indices(device):
    xyz = torch.tensor(
        [[0., 0, 0], [1., 0, 0], [4., 0, 0], [9., 0, 0],
         [10., 0, 0], [11., 0, 0], [15., 0, 0]], device=device,
    )
    offset = torch.tensor([4, 7], dtype=torch.int32, device=device)
    new_offset = torch.tensor([2, 4], dtype=torch.int32, device=device)
    result = pointcept.farthest_point_sampling(xyz, offset, new_offset)
    assert result.dtype == torch.int32 and result.device == xyz.device
    assert result.cpu().tolist() == [0, 3, 4, 6]


@pytest.mark.parametrize("device", DEVICES)
def test_knn_euclidean_distances_and_minus_one_padding(device):
    xyz = torch.tensor(
        [[0., 0, 0], [2., 0, 0], [10., 0, 0], [12., 0, 0], [14., 0, 0]],
        device=device, requires_grad=True,
    )
    query = torch.tensor([[0.5, 0, 0], [11., 0, 0]], device=device, requires_grad=True)
    offset = torch.tensor([2, 5], dtype=torch.int64, device=device)
    new_offset = torch.tensor([1, 2], dtype=torch.int32, device=device)
    idx, dist = pointcept.knn_query(4, xyz, offset, query, new_offset)
    assert idx.dtype == torch.int32 and idx.cpu().tolist() == [[0, 1, -1, -1], [2, 3, 4, -1]]
    torch.testing.assert_close(
        dist.cpu(), torch.tensor([[0.5, 1.5, 1e5, 1e5], [1., 1., 3., 1e5]]),
        rtol=1e-5, atol=1e-5,
    )
    assert not dist.requires_grad


@pytest.mark.parametrize("device", DEVICES)
def test_knn_equal_distances_use_smaller_reference_index(device):
    x = torch.arange(16, dtype=torch.float32, device=device) * 0.25
    xyz = torch.stack((x, torch.zeros_like(x), torch.zeros_like(x)), dim=1)
    offset = torch.tensor([16], dtype=torch.int32, device=device)
    idx, dist = pointcept.knn_query(8, xyz, offset)
    assert idx[2].cpu().tolist() == [2, 1, 3, 0, 4, 5, 6, 7]
    torch.testing.assert_close(
        dist[2].cpu(), torch.tensor([0., 0.25, 0.25, 0.5, 0.5, 0.75, 1., 1.25]),
    )


@pytest.mark.parametrize("device", DEVICES)
def test_knn_strict_upstream_distance_ceiling(device):
    xyz = torch.tensor(
        [[99_999., 0, 0], [100_000., 0, 0], [100_001., 0, 0]], device=device,
    )
    query = torch.zeros((1, 3), device=device)
    offset = torch.tensor([3], dtype=torch.int32, device=device)
    new_offset = torch.tensor([1], dtype=torch.int32, device=device)
    idx, dist = pointcept.knn_query(3, xyz, offset, query, new_offset)
    assert idx.cpu().tolist() == [[0, -1, -1]]
    torch.testing.assert_close(
        dist.cpu(), torch.tensor([[99_999., 100_000., 100_000.]]), rtol=0, atol=0,
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_knn_nonfinite_reference_keeps_missing_slot_outside_other_batches(device, bad):
    xyz = torch.tensor([[0., 0, 0], [bad, 0, 0]], device=device)
    query = torch.tensor([[0., 0, 0]], device=device)
    offset = torch.tensor([1, 2], dtype=torch.int32, device=device)
    new_offset = torch.tensor([0, 1], dtype=torch.int32, device=device)
    idx, dist = pointcept.knn_query(1, xyz, offset, query, new_offset)
    assert idx.cpu().tolist() == [[-1]]
    torch.testing.assert_close(dist.cpu(), torch.tensor([[100_000.]]), rtol=0, atol=0)

    xyz = torch.tensor([[0., 0, 0], [bad, 0, 0], [1., 0, 0]], device=device)
    offset = torch.tensor([1, 3], dtype=torch.int32, device=device)
    idx, dist = pointcept.knn_query(2, xyz, offset, query, new_offset)
    assert idx.cpu().tolist() == [[2, -1]]
    torch.testing.assert_close(dist.cpu(), torch.tensor([[1., 100_000.]]), rtol=0, atol=0)


@pytest.mark.parametrize("device", DEVICES)
def test_grouping_zero_padding_relative_xyz_and_gradients(device):
    xyz = torch.tensor([[0., 0, 0], [1., 0, 0], [2., 0, 0]], device=device, requires_grad=True)
    feat = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], device=device, requires_grad=True)
    query = torch.tensor([[1., 0, 0]], device=device, requires_grad=True)
    idx = torch.tensor([[2, -1, 2]], dtype=torch.int32, device=device)
    result = pointcept.grouping(idx, feat, xyz, query, with_xyz=True)
    assert result.shape == (1, 3, 5)
    torch.testing.assert_close(result[0, 1], torch.zeros(5, device=device))
    torch.testing.assert_close(result[0, 0], torch.tensor([1., 0, 0, 5, 6], device=device))
    result.sum().backward()
    torch.testing.assert_close(feat.grad.cpu(), torch.tensor([[0., 0], [0, 0], [2, 2.]]))
    torch.testing.assert_close(xyz.grad.cpu(), torch.tensor([[0., 0, 0], [0, 0, 0], [2, 2, 2.]]))
    torch.testing.assert_close(query.grad.cpu(), torch.tensor([[-2., -2, -2.]]))


@pytest.mark.parametrize("device", DEVICES)
def test_query_and_group_reuses_indices_and_interpolation_gradient(device):
    xyz = torch.tensor([[0., 0, 0], [2., 0, 0]], device=device, requires_grad=True)
    feat = torch.tensor([[2.], [10.]], device=device, requires_grad=True)
    query = torch.tensor([[0.5, 0, 0]], device=device, requires_grad=True)
    offset = torch.tensor([2], dtype=torch.int32, device=device)
    new_offset = torch.tensor([1], dtype=torch.int32, device=device)
    grouped, idx = pointcept.knn_query_and_group(
        feat, xyz, offset, query, new_offset, nsample=3, with_xyz=True,
    )
    assert idx.cpu().tolist() == [[0, 1, -1]]
    reused, same = pointcept.knn_query_and_group(feat, xyz, idx=idx, new_xyz=query)
    assert same is idx
    torch.testing.assert_close(reused, grouped[..., 3:])
    result = pointcept.interpolation(xyz, query, feat, offset, new_offset, k=3)
    torch.testing.assert_close(result.cpu(), torch.tensor([[4.]]), atol=1e-5, rtol=1e-5)
    result.sum().backward()
    torch.testing.assert_close(feat.grad.cpu(), torch.tensor([[0.75], [0.25]]), atol=1e-5, rtol=1e-5)
    assert xyz.grad is None and query.grad is None


@pytest.mark.parametrize("device", DEVICES)
def test_empty_reference_batch_interpolates_zero(device):
    xyz = torch.tensor([[10., 0, 0]], device=device)
    feat = torch.tensor([[3.]], device=device, requires_grad=True)
    query = torch.tensor([[0., 0, 0], [10., 0, 0]], device=device)
    offset = torch.tensor([0, 1], dtype=torch.int32, device=device)
    new_offset = torch.tensor([1, 2], dtype=torch.int32, device=device)
    output = pointcept.interpolation(xyz, query, feat, offset, new_offset)
    torch.testing.assert_close(output.cpu(), torch.tensor([[0.], [3.]]))
    output.sum().backward()
    torch.testing.assert_close(feat.grad.cpu(), torch.tensor([[1.]]))


def test_bad_offsets_and_explicit_optional_registration():
    xyz = torch.zeros((3, 3))
    with pytest.raises(ValueError, match="end at"):
        pointcept.farthest_point_sampling(xyz, torch.tensor([2]), torch.tensor([1]))
    with pytest.raises(ValueError, match="more FPS"):
        pointcept.farthest_point_sampling(xyz, torch.tensor([3]), torch.tensor([4]))
    names = (
        "pointnet2_ops", "pointnet2_ops.pointnet2_utils", "knn_cuda",
        "torch_cluster", "pointops",
    )
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    try:
        assert "pointops" not in compat.install()
        assert "pointops" in compat.install(force=True, pointcept=True)
        import pointops

        assert pointops.knn_query is pointcept.knn_query
        assert pointops.interpolation is pointcept.interpolation
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)

"""Legacy torch_cluster.nearest orientation, batching, and MPS contracts."""

import sys

import pytest
import torch

from mps_pointops import compat
from mps_pointops.nearest import nearest


MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")


def _on(device, *items):
    return [item.to(device) for item in items]


def test_batch_assignment_fixture_and_global_indices():
    x = torch.tensor([[-3., -2.], [-3., 2.], [4., 2.], [4., -2.],
                      [-7., -1.], [-7., 1.], [9., 1.], [9., -1.]])
    y = torch.tensor([[-3., 0.], [4., 0.], [-7., 0.], [9., 0.]])
    bx = torch.tensor([0] * 4 + [2] * 4)
    by = torch.tensor([0] * 2 + [2] * 2)
    assert nearest(x, y, bx, by).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert nearest(x, y).tolist() == [0, 0, 1, 1, 2, 2, 3, 3]


def test_one_dimensional_input_tie_and_single_optional_batch():
    x = torch.tensor([0., 2., 9.])
    y = torch.tensor([-1., 1., 10.])
    assert nearest(x, y).tolist() == [0, 1, 2]
    assert nearest(x, y, batch_x=torch.zeros(3, dtype=torch.long)).tolist() == [0, 1, 2]
    assert nearest(x, y, batch_y=torch.zeros(3, dtype=torch.long)).tolist() == [0, 1, 2]


def test_empty_and_unmatched_batches_are_explicit():
    empty = torch.empty(0, 3)
    assert nearest(empty, empty).shape == (0,)
    with pytest.raises(ValueError, match="batch indices"):
        nearest(empty, torch.zeros(1, 3))
    with pytest.raises(ValueError, match="batch indices"):
        nearest(torch.zeros(1, 3), empty)
    x = torch.zeros(2, 3)
    y = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="batch indices"):
        nearest(x, y, torch.tensor([0, 2]), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="batch indices"):
        nearest(x, y, torch.tensor([0, 2]), None)
    with pytest.raises(ValueError, match="sorted"):
        nearest(x, y, torch.tensor([1, 0]), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="non-negative"):
        nearest(x, y, torch.tensor([-1, 0]), torch.tensor([0, 1]))


def test_input_shape_dtype_and_device_validation():
    x = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="dimensions differ"):
        nearest(x, torch.zeros(2, 4))
    with pytest.raises(TypeError, match="same dtype"):
        nearest(x, x.double())
    with pytest.raises(ValueError, match="shape"):
        nearest(torch.zeros(2, 1, 3), x)
    with pytest.raises(ValueError, match="shape"):
        nearest(x, torch.empty(2, 0))
    with pytest.raises(ValueError, match="batch_x must have shape"):
        nearest(x, x, torch.tensor([0]))


def test_cuda_finite_threshold_and_nonfinite_fallback_policy_on_cpu():
    # In the upstream CUDA kernel, best starts at 1e38 and index at 0.
    x = torch.tensor([[0.], [1e20], [float("nan")], [float("inf")]])
    y = torch.tensor([[2.], [3.]])
    assert nearest(x, y).tolist() == [0, 0, 0, 0]
    # The fallback is global index zero even when a later batch owns the query.
    bx = torch.tensor([0, 1])
    by = torch.tensor([0, 1])
    assert nearest(torch.tensor([[0.], [1e20]]), y, bx, by).tolist() == [0, 0]
    # fl32((1e19)^2) equals the rounded 1e38 initializer. The first
    # candidate is rejected, while the second is strictly closer.
    assert nearest(torch.tensor([[1e19]]), torch.tensor([[0.], [1e18]])).tolist() == [1]


def test_cuda_1024_lane_tie_order_is_not_global_index_order():
    y = torch.full((1030, 1), 10.0)
    y[1, 0] = 0.0
    y[1024, 0] = 0.0
    # CUDA lane 0 owns local 1024 and beats lane 1, which owns local 1.
    assert nearest(torch.zeros(1, 1), y).tolist() == [1024]
    # Each batch resets the CUDA lane numbering at its own y offset.
    shifted_y = torch.cat((torch.tensor([[20.], [21.], [22.]]), y))
    bx = torch.tensor([0, 2])
    by = torch.tensor([0] * 3 + [2] * len(y))
    x = torch.tensor([[20.], [0.]])
    assert nearest(x, shifted_y, bx, by).tolist() == [0, 1027]


def test_compat_installs_nearest_and_restores_modules():
    names = ("pointnet2_ops", "pointnet2_ops.pointnet2_utils", "knn_cuda", "torch_cluster")
    saved = {name: sys.modules[name] for name in names if name in sys.modules}
    try:
        compat.install(force=True)
        from torch_cluster import nearest as shim_nearest

        assert shim_nearest is nearest
        assert shim_nearest(torch.tensor([[0.], [2.]]), torch.tensor([[1.]])).tolist() == [0, 0]
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)


@MPS
@pytest.mark.parametrize("dim", [1, 2, 3, 64, 128])
def test_mps_matches_cpu_for_ragged_ties_and_full_output(dim):
    # Nine queries cross a threadgroup boundary. Batch ID 1 is intentionally
    # empty, and the second batch has only two unequal reference points.
    x = torch.zeros(9, dim)
    x[:8, 0] = torch.tensor([0., 1., 2., 3., 4., 5., 6., 7.])
    x[8, 0] = 101.
    y = torch.zeros(5, dim)
    y[:, 0] = torch.tensor([-1., 1., 7., 100., 102.])
    bx = torch.tensor([0] * 8 + [2])
    by = torch.tensor([0] * 3 + [2] * 2)
    want = nearest(x, y, bx, by)
    assert want.tolist() == [0, 1, 1, 1, 1, 2, 2, 2, 3]
    got = nearest(*_on("mps", x, y, bx, by))
    assert got.device.type == "mps" and got.dtype == torch.long
    assert torch.equal(got.cpu(), want)


@MPS
@pytest.mark.parametrize("dim", [1, 3, 64, 128])
def test_mps_feature_distance_uses_all_dimensions(dim):
    generator = torch.Generator().manual_seed(3107 + dim)
    y = torch.randn(60, dim, generator=generator) * 5
    chosen = torch.randint(0, 60, (41,), generator=generator)
    x = y[chosen] + torch.randn(41, dim, generator=generator) * 0.001
    got = nearest(*_on("mps", x, y)).cpu()
    assert torch.equal(got, chosen)
    assert torch.equal(got, nearest(x, y))


@MPS
def test_mps_nonfinite_and_overflow_threshold_match_cuda_source():
    x = torch.tensor([[0.], [1e20], [float("nan")], [float("inf")], [1.]])
    y = torch.tensor([[2.], [3.]])
    got = nearest(*_on("mps", x, y))
    assert got.cpu().tolist() == [0, 0, 0, 0, 0]
    bx = torch.tensor([0, 1])
    by = torch.tensor([0, 1])
    assert nearest(*_on("mps", torch.tensor([[0.], [1e20]]), y, bx, by)).cpu().tolist() == [0, 0]
    threshold_x = torch.tensor([[1e19]])
    threshold_y = torch.tensor([[0.], [1e18]])
    assert nearest(*_on("mps", threshold_x, threshold_y)).cpu().tolist() == [1]


@MPS
def test_mps_preserves_original_cuda_1024_lane_tie_order():
    y = torch.full((1030, 1), 10.0)
    y[1, 0] = 0.0
    y[1024, 0] = 0.0
    x = torch.zeros(1, 1)
    assert nearest(*_on("mps", x, y)).cpu().tolist() == [1024]


@MPS
def test_mps_unsupported_dtype_fails_without_cpu_fallback():
    x = torch.zeros(1, 3, dtype=torch.float16, device="mps")
    with pytest.raises(TypeError, match="float32"):
        nearest(x, x)

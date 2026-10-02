import pytest
import torch

from mps_pointops import chamfer as chamfer_module
from mps_pointops.chamfer import chamfer_distance


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_bidirectional_loss_and_reverse_gradient_with_tie(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device=device, requires_grad=True)
    y = torch.tensor([[[1.0, 0.0, 0.0]]], device=device, requires_grad=True)
    loss, normals = chamfer_distance(x, y, batch_reduction="sum")
    assert normals is None
    assert loss.item() == 2.0
    loss.backward()
    # y chooses x[0] on a tie. Its reverse contribution belongs only to x[0].
    torch.testing.assert_close(x.grad.cpu(), torch.tensor([[[-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]))
    torch.testing.assert_close(y.grad.cpu(), torch.tensor([[[2.0, 0.0, 0.0]]]))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_lengths_hide_nonfinite_padding_and_zero_its_gradient(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.tensor([[[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]], device=device, requires_grad=True)
    y = torch.tensor([[[1.0, 0.0, 0.0], [float("inf"), 0.0, 0.0]]], device=device, requires_grad=True)
    lengths = torch.tensor([1], dtype=torch.long, device=device)
    loss, _ = chamfer_distance(x, y, lengths, lengths)
    assert loss.item() == 2.0
    loss.backward()
    torch.testing.assert_close(x.grad[0, 0].cpu(), torch.tensor([-4.0, 0.0, 0.0]))
    torch.testing.assert_close(y.grad[0, 0].cpu(), torch.tensor([4.0, 0.0, 0.0]))
    assert torch.equal(x.grad[0, 1].cpu(), torch.zeros(3))
    assert torch.equal(y.grad[0, 1].cpu(), torch.zeros(3))


def test_reductions_weights_and_single_direction():
    x = torch.zeros((2, 1, 3))
    y = torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]])
    weights = torch.tensor([0.1, 0.2])
    mean, _ = chamfer_distance(x, y, weights=weights)
    torch.testing.assert_close(mean, torch.tensor(6.0))
    summed, _ = chamfer_distance(x, y, weights=weights, point_reduction="sum", batch_reduction="sum")
    torch.testing.assert_close(summed, torch.tensor(1.8))
    per_batch, _ = chamfer_distance(x, y, point_reduction="max", batch_reduction=None)
    torch.testing.assert_close(per_batch, torch.tensor([1.0, 4.0]))
    one_way, _ = chamfer_distance(x, y, single_directional=True, batch_reduction=None)
    torch.testing.assert_close(one_way, torch.tensor([1.0, 4.0]))
    zero_weight, _ = chamfer_distance(x, y, weights=torch.zeros(2))
    assert zero_weight.item() == 0.0


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_weighted_bidirectional_backward_matches_closed_form(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.zeros((2, 1, 3), device=device, requires_grad=True)
    y = torch.tensor([[[1.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], device=device, requires_grad=True)
    weights = torch.tensor([1.0, 3.0], device=device)
    loss, _ = chamfer_distance(x, y, weights=weights)
    loss.backward()
    torch.testing.assert_close(loss.cpu(), torch.tensor(6.5), rtol=0, atol=0)
    torch.testing.assert_close(x.grad[:, 0, 0].cpu(), torch.tensor([-1.0, -6.0]), rtol=0, atol=0)
    torch.testing.assert_close(y.grad[:, 0, 0].cpu(), torch.tensor([1.0, 6.0]), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("norm", [1, 2])
def test_all_zero_weights_have_zero_gradient_on_every_input(device, norm):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.zeros((1, 1, 3), device=device, requires_grad=True)
    y = torch.tensor([[[2.0, 0.0, 0.0]]], device=device, requires_grad=True)
    weights = torch.zeros(1, device=device, requires_grad=True)
    loss, _ = chamfer_distance(x, y, weights=weights, norm=norm)
    loss.backward()
    assert loss.item() == 0.0
    torch.testing.assert_close(x.grad.cpu(), torch.zeros((1, 1, 3)), rtol=0, atol=0)
    torch.testing.assert_close(y.grad.cpu(), torch.zeros((1, 1, 3)), rtol=0, atol=0)
    torch.testing.assert_close(weights.grad.cpu(), torch.zeros(1), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("norm", [1, 2])
@pytest.mark.parametrize("point_reduction", ["mean", "sum", "max", None])
@pytest.mark.parametrize("single_directional", [False, True])
def test_all_zero_weights_match_upstream_return_shape_and_gradient_presence(
    device, norm, point_reduction, single_directional
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
         [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]],
        device=device, requires_grad=True,
    )
    y = torch.tensor(
        [[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
         [[3.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
        device=device, requires_grad=True,
    )
    weights = torch.zeros(2, device=device, requires_grad=True)
    loss, normals = chamfer_distance(
        x, y, weights=weights, point_reduction=point_reduction,
        batch_reduction=None, single_directional=single_directional, norm=norm,
    )
    parts = loss if isinstance(loss, tuple) else (loss,)
    assert len(parts) == (1 if single_directional or point_reduction is not None else 2)
    for part in parts:
        assert part.shape == (2, 2)
        torch.testing.assert_close(part.cpu(), torch.zeros(2, 2), rtol=0, atol=0)
    if point_reduction == "max" and not single_directional:
        assert normals is None
    elif isinstance(loss, tuple):
        assert isinstance(normals, tuple) and len(normals) == 2
        assert all(part.shape == (2, 2) for part in normals)
    else:
        assert isinstance(normals, torch.Tensor) and normals.shape == (2, 2)
    grads = torch.autograd.grad(sum(part.sum() for part in parts), (x, y, weights), allow_unused=True)
    assert grads[0] is not None and grads[2] is not None
    assert (grads[1] is None) == single_directional
    for grad in grads:
        if grad is not None:
            torch.testing.assert_close(grad.cpu(), torch.zeros_like(grad.cpu()), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize(
    ("point_reduction", "expected_loss", "expected_x_grad", "expected_y_grad"),
    [
        ("sum", 6.0, [-4.0, 4.0], 0.0),
        ("max", 4.0, [0.0, 4.0], -4.0),
    ],
)
def test_point_reduction_backward_scaling(
    device, point_reduction, expected_loss, expected_x_grad, expected_y_grad
):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]], device=device, requires_grad=True)
    y = torch.tensor([[[1.0, 0.0, 0.0]]], device=device, requires_grad=True)
    loss, _ = chamfer_distance(x, y, point_reduction=point_reduction)
    loss.backward()
    assert loss.item() == expected_loss
    torch.testing.assert_close(
        x.grad[0, :, 0].cpu(), torch.tensor(expected_x_grad), rtol=0, atol=0
    )
    assert y.grad[0, 0, 0].item() == expected_y_grad


def test_unreduced_distances_have_zero_padding():
    x = torch.tensor([[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]])
    y = torch.tensor([[[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    lengths = torch.tensor([1])
    pair, normals = chamfer_distance(
        x, y, lengths, lengths, point_reduction=None, batch_reduction=None
    )
    assert normals is None and isinstance(pair, tuple)
    torch.testing.assert_close(pair[0], torch.tensor([[1.0, 0.0]]))
    torch.testing.assert_close(pair[1], torch.tensor([[1.0, 0.0]]))


def test_reference_tiles_preserve_lowest_index_tie(monkeypatch):
    monkeypatch.setattr(chamfer_module, "_PAIRS_PER_TILE", 2)
    x = torch.tensor([[[0.0, 0.0, 0.0]]], requires_grad=True)
    y = torch.tensor([[[8.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [9.0, 0.0, 0.0], [10.0, 0.0, 0.0],
                       [-1.0, 0.0, 0.0]]], requires_grad=True)
    loss, _ = chamfer_distance(x, y, single_directional=True)
    assert loss.item() == 1.0
    loss.backward()
    assert y.grad[0, 1, 0].item() == 2.0
    assert y.grad[0, 4, 0].item() == 0.0


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_overflowed_squared_distances_keep_a_valid_first_index(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    # The coordinates are finite, but every squared distance overflows to inf.
    # The first reference must still be selected for a safe backward gather.
    x = torch.tensor([[[1e20, 0.0, 0.0]]], device=device, requires_grad=True)
    y = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device=device, requires_grad=True)
    if device == "mps":
        distances, indices = chamfer_module._nearest_mps(
            x, y, torch.tensor([1], device=device), torch.tensor([2], device=device)
        )
        torch.mps.synchronize()
        assert torch.isinf(distances).all().item()
        torch.testing.assert_close(indices.cpu(), torch.tensor([[0]]), rtol=0, atol=0)
    loss, _ = chamfer_distance(x, y, single_directional=True)
    assert torch.isinf(loss).item()
    loss.backward()
    torch.testing.assert_close(x.grad.cpu(), torch.tensor([[[2e20, 0.0, 0.0]]]))
    torch.testing.assert_close(y.grad.cpu(), torch.tensor([[[-2e20, 0.0, 0.0], [0.0, 0.0, 0.0]]]))


def test_double_precision_gradcheck_away_from_ties():
    x = torch.tensor([[[0.1, 0.2, 0.3], [2.1, 0.4, 0.5]]], dtype=torch.double, requires_grad=True)
    y = torch.tensor([[[1.0, 0.1, 0.2], [4.0, 0.8, 0.7]]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: chamfer_distance(a, b)[0], (x, y))


def test_l1_double_precision_gradcheck_away_from_ties_and_cusps():
    x = torch.tensor([[[0.1, 0.2, 0.3], [2.1, 0.4, 0.5]]], dtype=torch.double, requires_grad=True)
    y = torch.tensor([[[1.0, 0.11, 0.21], [4.0, 0.8, 0.7]]], dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: chamfer_distance(a, b, norm=1)[0], (x, y))


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_l1_ranks_by_absolute_sum_and_matches_upstream_equal_coordinate_gradient(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.zeros((1, 1, 3), device=device, requires_grad=True)
    y = torch.tensor([[[3.0, 0.0, 0.0], [2.0, 2.0, 0.0]]], device=device, requires_grad=True)
    l1, _ = chamfer_distance(x, y, norm=1, single_directional=True)
    l2, _ = chamfer_distance(x, y, norm=2, single_directional=True)
    assert l1.item() == 3.0
    assert l2.item() == 8.0
    l1.backward()
    torch.testing.assert_close(x.grad.cpu(), torch.tensor([[[-1.0, -1.0, -1.0]]]), rtol=0, atol=0)
    torch.testing.assert_close(
        y.grad.cpu(), torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]]), rtol=0, atol=0
    )


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_l1_tie_uses_first_reference_and_masks_padding(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.tensor([[[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]], device=device, requires_grad=True)
    y = torch.tensor([[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]], device=device, requires_grad=True)
    lx = torch.tensor([1], device=device)
    ly = torch.tensor([2], device=device)
    if device == "mps":
        distances, indices = chamfer_module._nearest_mps(x, y, lx, ly, norm=1)
        torch.mps.synchronize()
        torch.testing.assert_close(distances.cpu(), torch.tensor([[1.0, 0.0]]), rtol=0, atol=0)
        torch.testing.assert_close(indices.cpu(), torch.tensor([[0, -1]]), rtol=0, atol=0)
    loss, _ = chamfer_distance(x, y, lx, ly, norm=1, single_directional=True)
    assert loss.item() == 1.0
    loss.backward()
    torch.testing.assert_close(x.grad[0, 0].cpu(), torch.tensor([-1.0, -1.0, -1.0]), rtol=0, atol=0)
    torch.testing.assert_close(x.grad[0, 1].cpu(), torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(y.grad[0, 0].cpu(), torch.tensor([1.0, 1.0, 1.0]), rtol=0, atol=0)
    torch.testing.assert_close(y.grad[0, 1].cpu(), torch.zeros(3), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_l1_exact_coincidence_uses_upstream_minus_one_subgradient(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    query = torch.zeros((1, 1, 3), device=device, requires_grad=True)
    ref = torch.zeros((1, 1, 3), device=device, requires_grad=True)
    loss, _ = chamfer_distance(
        query, ref, norm=1, single_directional=True,
        point_reduction="sum", batch_reduction="sum",
    )
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(query.grad.cpu(), -torch.ones((1, 1, 3)), rtol=0, atol=0)
    torch.testing.assert_close(ref.grad.cpu(), torch.ones((1, 1, 3)), rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_l1_metal_tie_is_stable_across_lanes_groups_and_repeated_dispatches():
    query = torch.zeros((2, 17, 3), device="mps")
    ref = torch.full((2, 65, 3), 100.0, device="mps")
    ref[:, 0] = torch.tensor([1.0, 0.0, 0.0], device="mps")
    ref[:, 31] = torch.tensor([-1.0, 0.0, 0.0], device="mps")
    ref[:, 32] = torch.tensor([0.0, 1.0, 0.0], device="mps")
    ref[:, 64] = torch.tensor([0.0, -1.0, 0.0], device="mps")
    query_lengths = torch.tensor([17, 17], device="mps")
    ref_lengths = torch.tensor([65, 65], device="mps")
    outputs = [chamfer_module._nearest_mps(query, ref, query_lengths, ref_lengths, norm=1)
               for _ in range(32)]
    torch.mps.synchronize()
    for distances, indices in outputs:
        torch.testing.assert_close(distances.cpu(), torch.ones((2, 17)), rtol=0, atol=0)
        torch.testing.assert_close(indices.cpu(), torch.zeros((2, 17), dtype=torch.long), rtol=0, atol=0)


def test_rejects_unsupported_or_invalid_inputs():
    x = torch.zeros((1, 2, 3))
    y = torch.ones((1, 2, 3))
    with pytest.raises(ValueError, match="batch_reduction"):
        chamfer_distance(x, y, point_reduction=None)
    with pytest.raises(ValueError, match="1 or 2 norm"):
        chamfer_distance(x, y, norm=3)
    with pytest.raises(NotImplementedError, match="normal"):
        chamfer_distance(x, y, x_normals=x)
    with pytest.raises(ValueError, match="at least one"):
        chamfer_distance(x, y, torch.tensor([0]), torch.tensor([2]))
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        chamfer_distance(x, y, torch.tensor([3]), torch.tensor([2]))
    with pytest.raises(ValueError, match="finite"):
        chamfer_distance(x.clone().fill_(float("nan")), y)
    with pytest.raises(ValueError, match="nonnegative"):
        chamfer_distance(x, y, weights=torch.tensor([-1.0]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_mps_matches_cpu_forward_and_backward_with_uneven_lengths():
    generator = torch.Generator().manual_seed(2026)
    x_cpu = torch.randn((2, 7, 3), generator=generator, requires_grad=True)
    y_cpu = torch.randn((2, 5, 3), generator=generator, requires_grad=True)
    lx_cpu = torch.tensor([7, 3], dtype=torch.long)
    ly_cpu = torch.tensor([5, 2], dtype=torch.long)
    cpu_loss, _ = chamfer_distance(x_cpu, y_cpu, lx_cpu, ly_cpu)
    cpu_loss.backward()

    x_mps = x_cpu.detach().to("mps").requires_grad_()
    y_mps = y_cpu.detach().to("mps").requires_grad_()
    mps_loss, _ = chamfer_distance(x_mps, y_mps, lx_cpu.to("mps"), ly_cpu.to("mps"))
    assert mps_loss.device.type == "mps"
    mps_loss.backward()
    torch.mps.synchronize()
    torch.testing.assert_close(mps_loss.cpu(), cpu_loss.detach(), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_mps.grad.cpu(), x_cpu.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(y_mps.grad.cpu(), y_cpu.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_native_nearest_writes_every_row_and_preserves_first_tie():
    # Nine queries force a second threadgroup; 37 references cross a SIMD
    # boundary. Equal distances at references 0, 31 and 36 choose index 0.
    query_cpu = torch.zeros((2, 9, 3), dtype=torch.float32)
    ref_cpu = torch.full((2, 37, 3), 9.0, dtype=torch.float32)
    ref_cpu[:, 0] = torch.tensor([1.0, 0.0, 0.0])
    ref_cpu[:, 31] = torch.tensor([-1.0, 0.0, 0.0])
    ref_cpu[:, 36] = torch.tensor([0.0, 1.0, 0.0])
    query_cpu[0, 8] = torch.tensor([float("nan"), 0.0, 0.0])
    query_cpu[1, 7:] = torch.tensor([float("nan"), 0.0, 0.0])
    query_lengths = torch.tensor([8, 7], dtype=torch.long, device="mps")
    ref_lengths = torch.tensor([37, 32], dtype=torch.long, device="mps")

    distance, index = chamfer_module._nearest_mps(
        query_cpu.to("mps"), ref_cpu.to("mps"), query_lengths, ref_lengths
    )
    torch.mps.synchronize()
    expected_distance = torch.ones((2, 9))
    expected_distance[0, 8] = 0.0
    expected_distance[1, 7:] = 0.0
    expected_index = torch.zeros((2, 9), dtype=torch.long)
    expected_index[0, 8] = -1
    expected_index[1, 7:] = -1
    torch.testing.assert_close(distance.cpu(), expected_distance, rtol=0, atol=0)
    torch.testing.assert_close(index.cpu(), expected_index, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_native_nearest_matches_independent_small_oracle():
    generator = torch.Generator().manual_seed(137)
    query_cpu = torch.randn((3, 13, 3), generator=generator)
    ref_cpu = torch.randn((3, 67, 3), generator=generator)
    query_lengths_cpu = torch.tensor([13, 4, 9], dtype=torch.long)
    ref_lengths_cpu = torch.tensor([67, 39, 3], dtype=torch.long)
    # A strided view exercises host-side contiguous packing.
    query_cpu = query_cpu.repeat_interleave(2, dim=1)[:, ::2, :]
    ref_cpu = ref_cpu.repeat_interleave(2, dim=1)[:, ::2, :]
    distance, index = chamfer_module._nearest_mps(
        query_cpu.to("mps"), ref_cpu.to("mps"),
        query_lengths_cpu.to("mps"), ref_lengths_cpu.to("mps"),
    )
    torch.mps.synchronize()
    actual_distance = distance.cpu()
    actual_index = index.cpu()
    for batch in range(3):
        for qi in range(13):
            if qi >= query_lengths_cpu[batch]:
                assert actual_index[batch, qi] == -1
                assert actual_distance[batch, qi] == 0
                continue
            candidates = []
            for ri in range(ref_lengths_cpu[batch]):
                delta = query_cpu[batch, qi] - ref_cpu[batch, ri]
                squared = (delta[0] * delta[0] + delta[1] * delta[1]) + delta[2] * delta[2]
                candidates.append((float(squared), ri))
            expected_distance, expected_index = min(candidates)
            assert actual_index[batch, qi] == expected_index
            torch.testing.assert_close(
                actual_distance[batch, qi], torch.tensor(expected_distance), rtol=1e-6, atol=1e-7
            )

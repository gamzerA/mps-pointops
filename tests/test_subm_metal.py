"""Small independent dense-convolution checks for the private Metal prototype."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mps_pointops._subm_conv_mps import _output_csr, subm_conv3d_forward_mps


def test_output_csr_keeps_output_and_offset_order():
    pairs = torch.tensor(
        [[2, 7, 1], [0, 5, 1], [1, 9, 0], [0, 4, 0]], dtype=torch.int64
    )
    ptr, sources, offsets = _output_csr(pairs, 3)
    assert ptr.tolist() == [0, 2, 4, 4]
    assert sources.tolist() == [4, 9, 5, 7]
    assert offsets.tolist() == [0, 1, 0, 2]


MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")


@MPS
@pytest.mark.parametrize(
    ("coordinates", "shape", "batch", "kernel", "dilation", "bias"),
    [
        (
            [[0, 2, 1, 1], [0, 1, 1, 1], [1, 2, 1, 1], [0, 3, 1, 1]],
            (5, 3, 3), 2, (3, 3, 3), (1, 1, 1), True,
        ),
        (
            [[0, 4, 2, 1], [0, 0, 2, 1], [0, 2, 2, 1]],
            (5, 5, 3), 1, (3, 1, 1), (2, 1, 1), False,
        ),
        (
            [[0, 0, 0, 0], [0, 3, 3, 3]],
            (4, 4, 4), 1, (1, 1, 1), (1, 1, 1), True,
        ),
    ],
)
def test_metal_subm_forward_matches_dense_at_active_coordinates(
    coordinates, shape, batch, kernel, dilation, bias
):
    cpu_indices = torch.tensor(coordinates, dtype=torch.int32)
    generator = torch.Generator().manual_seed(3205)
    cpu_features = torch.randn(len(coordinates), 3, generator=generator)
    cpu_weights = torch.randn((2, 3, *kernel), generator=generator)
    cpu_bias = torch.randn(2, generator=generator) if bias else None
    actual = subm_conv3d_forward_mps(
        cpu_indices, cpu_features.to("mps"), cpu_weights.to("mps"), shape, batch,
        dilation=dilation, bias=None if cpu_bias is None else cpu_bias.to("mps"),
    )
    torch.mps.synchronize()
    dense = torch.zeros((batch, 3, *shape), dtype=torch.float32)
    for row, (b, a0, a1, a2) in enumerate(coordinates):
        dense[b, :, a0, a1, a2] = cpu_features[row]
    expected_dense = F.conv3d(
        dense, cpu_weights, cpu_bias, padding=tuple(
            dilation[axis] * (kernel[axis] // 2) for axis in range(3)
        ), dilation=dilation,
    )
    expected = torch.stack([
        expected_dense[b, :, a0, a1, a2] for b, a0, a1, a2 in coordinates
    ])
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-4, atol=1e-5)


@MPS
def test_metal_subm_empty_has_zero_first_order_gradients():
    indices = torch.empty((0, 4), dtype=torch.int32)
    features = torch.empty((0, 2), dtype=torch.float32, device="mps", requires_grad=True)
    weights = torch.ones((3, 2, 3, 3, 3), dtype=torch.float32, device="mps", requires_grad=True)
    bias = torch.ones(3, dtype=torch.float32, device="mps", requires_grad=True)
    empty = subm_conv3d_forward_mps(indices, features, weights, (3, 3, 3), 0, bias=bias)
    assert empty.shape == (0, 3) and empty.device.type == "mps"
    empty.sum().backward()
    assert features.grad.shape == features.shape
    assert torch.count_nonzero(weights.grad).item() == 0
    assert torch.count_nonzero(bias.grad).item() == 0


@MPS
def test_metal_subm_bias_only_and_second_order_contract():
    indices = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    features = torch.ones((2, 1), device="mps")
    weights = torch.ones((1, 1, 3, 1, 1), device="mps")
    bias = torch.tensor([0.5], device="mps", requires_grad=True)
    output = subm_conv3d_forward_mps(indices, features, weights, (2, 1, 1), 1, bias=bias)
    output.sum().backward()
    torch.testing.assert_close(bias.grad.cpu(), torch.tensor([2.]))

    features = features.detach().requires_grad_()
    output = subm_conv3d_forward_mps(indices, features, weights, (2, 1, 1), 1)
    first = torch.autograd.grad(output.sum(), features, create_graph=True)[0]
    assert not first.requires_grad
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(first.sum(), features)


@MPS
@pytest.mark.parametrize(
    ("coordinates", "shape", "batch", "kernel", "dilation", "with_bias"),
    [
        (
            [[0, 2, 1, 1], [0, 1, 1, 1], [1, 2, 1, 1], [0, 3, 1, 1]],
            (5, 3, 3), 2, (3, 3, 3), (1, 1, 1), True,
        ),
        (
            [[0, 4, 2, 1], [0, 0, 2, 1], [0, 2, 2, 1]],
            (5, 5, 3), 1, (3, 1, 1), (2, 1, 1), False,
        ),
        (
            [[0, 0, 0, 0], [0, 3, 3, 3]],
            (4, 4, 4), 1, (1, 1, 1), (1, 1, 1), True,
        ),
    ],
)
def test_metal_subm_first_order_gradients_match_dense_conv3d(
    coordinates, shape, batch, kernel, dilation, with_bias
):
    generator = torch.Generator().manual_seed(8207)
    cpu_indices = torch.tensor(coordinates, dtype=torch.int32)
    cpu_features = torch.randn(len(coordinates), 3, generator=generator).requires_grad_()
    cpu_weights = torch.randn((2, 3, *kernel), generator=generator).requires_grad_()
    cpu_bias = torch.randn(2, generator=generator).requires_grad_() if with_bias else None
    upstream = torch.randn(len(coordinates), 2, generator=generator)

    dense = torch.zeros((batch, 3, *shape), dtype=torch.float32)
    for row, (b, a0, a1, a2) in enumerate(coordinates):
        dense[b, :, a0, a1, a2] = cpu_features[row]
    dense_out = F.conv3d(
        dense, cpu_weights, cpu_bias,
        padding=tuple(dilation[axis] * (kernel[axis] // 2) for axis in range(3)),
        dilation=dilation,
    )
    expected = torch.stack([
        dense_out[b, :, a0, a1, a2] for b, a0, a1, a2 in coordinates
    ])
    expected_grads = torch.autograd.grad(
        (expected * upstream).sum(),
        (cpu_features, cpu_weights) if cpu_bias is None
        else (cpu_features, cpu_weights, cpu_bias),
    )

    mps_features = cpu_features.detach().to("mps").requires_grad_()
    mps_weights = cpu_weights.detach().to("mps").requires_grad_()
    mps_bias = cpu_bias.detach().to("mps").requires_grad_() if cpu_bias is not None else None
    actual = subm_conv3d_forward_mps(
        cpu_indices, mps_features, mps_weights, shape, batch,
        dilation=dilation, bias=mps_bias,
    )
    (actual * upstream.to("mps")).sum().backward()
    torch.mps.synchronize()
    torch.testing.assert_close(actual.cpu(), expected.detach(), rtol=1e-4, atol=1e-5)
    actual_grads = (mps_features.grad, mps_weights.grad)
    if mps_bias is not None:
        actual_grads += (mps_bias.grad,)
    for observed, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(observed.cpu(), reference, rtol=1e-4, atol=1e-5)


@MPS
def test_metal_subm_rejects_duplicate_coordinates_and_weight_mismatch():
    features = torch.ones((2, 2), device="mps")
    weights = torch.ones((3, 2, 3, 3, 3), device="mps")
    with pytest.raises(ValueError, match="duplicate"):
        subm_conv3d_forward_mps(
            torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=torch.int32),
            features, weights, (3, 3, 3), 1,
        )
    with pytest.raises(ValueError, match="expected features"):
        subm_conv3d_forward_mps(
            torch.tensor([[0, 0, 0, 0], [0, 0, 0, 1]], dtype=torch.int32),
            features, torch.ones((3, 1, 3, 3, 3), device="mps"), (3, 3, 3), 1,
        )


@MPS
def test_metal_subm_accepts_noncontiguous_feature_weight_and_bias_views():
    indices = torch.tensor([[0, 2, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    base_features = torch.tensor(
        [[1., 99., 2., 99.], [3., 99., 4., 99.]], device="mps", requires_grad=True
    )
    features = base_features[:, ::2]
    base_weights = torch.arange(
        1, 2 * 2 * 3 * 2 + 1, dtype=torch.float32, device="mps", requires_grad=True
    )
    weights = base_weights.reshape(2, 2, 3, 1, 2)[..., ::2]
    base_bias = torch.tensor([0.5, 99., -0.25, 99.], device="mps", requires_grad=True)
    bias = base_bias[::2]
    assert not features.is_contiguous() and not weights.is_contiguous()
    assert not bias.is_contiguous()
    actual = subm_conv3d_forward_mps(
        indices, features, weights, (4, 1, 1), 1, bias=bias
    )
    cpu_base_features = base_features.detach().cpu().requires_grad_()
    cpu_base_weights = base_weights.detach().cpu().requires_grad_()
    cpu_base_bias = base_bias.detach().cpu().requires_grad_()
    cpu_features = cpu_base_features[:, ::2]
    cpu_weights = cpu_base_weights.reshape(2, 2, 3, 1, 2)[..., ::2]
    cpu_bias = cpu_base_bias[::2]
    dense = torch.zeros((1, 2, 4, 1, 1), dtype=torch.float32)
    dense[0, :, 2, 0, 0] = cpu_features[0]
    dense[0, :, 1, 0, 0] = cpu_features[1]
    expected_dense = F.conv3d(
        dense, cpu_weights, cpu_bias, padding=(1, 0, 0)
    )
    expected = torch.stack([expected_dense[0, :, 2, 0, 0], expected_dense[0, :, 1, 0, 0]])
    torch.testing.assert_close(actual.detach().cpu(), expected.detach(), rtol=1e-4, atol=1e-5)
    upstream = torch.tensor([[0.75, -1.25], [1.5, 0.5]])
    (actual * upstream.to("mps")).sum().backward()
    (expected * upstream).sum().backward()
    torch.mps.synchronize()
    for observed, reference in (
        (base_features.grad, cpu_base_features.grad),
        (base_weights.grad, cpu_base_weights.grad),
        (base_bias.grad, cpu_base_bias.grad),
    ):
        torch.testing.assert_close(observed.cpu(), reference, rtol=1e-4, atol=1e-5)

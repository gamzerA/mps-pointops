"""Private bounded Metal ordinary/inverse sparse-convolution parity."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from mps_pointops._sparse_conv_cpu import (
    sparse_conv3d_forward_cpu, sparse_inverse_conv3d_forward_cpu,
)
from mps_pointops._sparse_conv_mps import (
    _csr, _offset_major, sparse_conv3d_forward_mps,
    sparse_inverse_conv3d_forward_mps,
)
from mps_pointops._sparse_rulebook import SparseTensor3D


MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")


def test_directional_csr_and_offset_major_keep_actual_offset():
    pairs = torch.tensor([[2, 1, 0], [0, 2, 0], [1, 1, 1]], dtype=torch.int64)
    ptr, outputs, offsets = _csr(pairs, 2, 1, 3)
    assert ptr.tolist() == [0, 0, 2, 3]
    assert outputs.tolist() == [1, 0, 0]
    assert offsets.tolist() == [1, 2, 0]
    rows, offset_ptr = _offset_major(pairs, 2, 1, 3)
    assert rows.tolist() == [[0, 0, 2], [1, 1, 1], [2, 0, 1]]
    assert offset_ptr.tolist() == [0, 1, 2, 3]


@MPS
@pytest.mark.parametrize(
    ("coords", "shape", "batch", "kernel", "stride", "padding", "dilation"),
    [
        (
            [[1, 4, 2, 2], [0, 1, 2, 2], [1, 0, 0, 0],
             [0, 5, 3, 2], [0, 2, 1, 1], [1, 3, 3, 1]],
            (6, 4, 3), 2, (3, 2, 1), (2, 1, 1), (1, 0, 0), (1, 1, 1),
        ),
        (
            [[0, 3, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
            (5, 1, 1), 1, (1, 1, 1), (2, 1, 1), (0, 0, 0), (1, 1, 1),
        ),
        (
            [[0, 1, 1, 0], [1, 2, 0, 1], [0, 0, 0, 0], [1, 3, 1, 0]],
            (4, 3, 2), 2, (2, 3, 2), (2, 2, 1), (0, 1, 1), (2, 1, 1),
        ),
    ],
)
def test_ordinary_mps_matches_cpu_reference_forward_and_gradients(
    coords, shape, batch, kernel, stride, padding, dilation,
):
    rng = torch.Generator().manual_seed(5128)
    indices = torch.tensor(coords, dtype=torch.int32)
    values = torch.randn((len(coords), 2), generator=rng)
    weight_values = torch.randn((3, 2, *kernel), generator=rng)
    bias_values = torch.randn((3,), generator=rng)
    x_cpu, w_cpu, b_cpu = (
        tensor.clone().requires_grad_() for tensor in (values, weight_values, bias_values)
    )
    reference = sparse_conv3d_forward_cpu(
        SparseTensor3D(indices, x_cpu, shape, batch), w_cpu,
        stride=stride, padding=padding, dilation=dilation, bias=b_cpu,
    )
    x_mps, w_mps, b_mps = (
        tensor.to("mps").requires_grad_() for tensor in (values, weight_values, bias_values)
    )
    actual = sparse_conv3d_forward_mps(
        SparseTensor3D(indices, x_mps, shape, batch), w_mps,
        stride=stride, padding=padding, dilation=dilation, bias=b_mps,
    )
    assert actual.features.device.type == "mps"
    assert torch.equal(actual.indices, reference.indices)
    assert actual.spatial_shape == reference.spatial_shape
    upstream = torch.randn(reference.features.shape, generator=rng)
    reference.features.backward(upstream)
    actual.features.backward(upstream.to("mps"))
    torch.mps.synchronize()
    torch.testing.assert_close(actual.features.detach().cpu(), reference.features.detach(), rtol=1e-4, atol=1e-5)
    for observed, expected in (
        (x_mps.grad, x_cpu.grad), (w_mps.grad, w_cpu.grad), (b_mps.grad, b_cpu.grad),
    ):
        torch.testing.assert_close(observed.cpu(), expected, rtol=1e-4, atol=1e-5)


@MPS
def test_keyed_inverse_mps_matches_cpu_reference_and_restores_original_rows():
    rng = torch.Generator().manual_seed(8821)
    indices = torch.tensor(
        [[1, 5, 0, 0], [0, 1, 0, 0], [1, 2, 0, 0],
         [0, 4, 0, 0], [0, 0, 0, 0], [1, 3, 0, 0]], dtype=torch.int32,
    )
    shape, batch = (6, 1, 1), 2
    input_values = torch.randn((6, 2), generator=rng)
    down_weights = torch.randn((3, 2, 3, 1, 1), generator=rng)
    inverse_weights = torch.randn((2, 3, 3, 1, 1), generator=rng)
    inverse_bias = torch.randn((2,), generator=rng)
    cpu_cache, mps_cache = {}, {}
    down_args = dict(stride=(2, 1, 1), padding=(1, 0, 0),
                     indice_key="down")
    cpu_down = sparse_conv3d_forward_cpu(
        SparseTensor3D(indices, input_values, shape, batch), down_weights,
        indice_cache=cpu_cache, **down_args,
    )
    mps_down = sparse_conv3d_forward_mps(
        SparseTensor3D(indices, input_values.to("mps"), shape, batch),
        down_weights.to("mps"), indice_cache=mps_cache, **down_args,
    )
    assert torch.equal(mps_down.indices, cpu_down.indices)
    values = torch.randn(cpu_down.features.shape, generator=rng)
    x_cpu, w_cpu, b_cpu = (
        tensor.clone().requires_grad_() for tensor in (values, inverse_weights, inverse_bias)
    )
    reference = sparse_inverse_conv3d_forward_cpu(
        cpu_down.replace_features(x_cpu), w_cpu,
        bias=b_cpu, indice_key="down", indice_cache=cpu_cache,
    )
    x_mps, w_mps, b_mps = (
        tensor.to("mps").requires_grad_() for tensor in (values, inverse_weights, inverse_bias)
    )
    actual = sparse_inverse_conv3d_forward_mps(
        mps_down.replace_features(x_mps), w_mps,
        bias=b_mps, indice_key="down", indice_cache=mps_cache,
    )
    assert actual.features.device.type == "mps"
    assert torch.equal(actual.indices, indices)
    assert actual.spatial_shape == shape
    upstream = torch.randn(reference.features.shape, generator=rng)
    reference.features.backward(upstream)
    actual.features.backward(upstream.to("mps"))
    torch.mps.synchronize()
    torch.testing.assert_close(actual.features.detach().cpu(), reference.features.detach(), rtol=1e-4, atol=1e-5)
    for observed, expected in (
        (x_mps.grad, x_cpu.grad), (w_mps.grad, w_cpu.grad), (b_mps.grad, b_cpu.grad),
    ):
        torch.testing.assert_close(observed.cpu(), expected, rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="lineage"):
        sparse_inverse_conv3d_forward_mps(
            SparseTensor3D(mps_down.indices, x_mps, mps_down.spatial_shape, batch),
            w_mps, indice_key="down", indice_cache=mps_cache,
        )
    with pytest.raises(ValueError, match="row order"):
        sparse_inverse_conv3d_forward_mps(
            replace(mps_down, indices=mps_down.indices.flip(0),
                    features=mps_down.features.flip(0)),
            w_mps, indice_key="down", indice_cache=mps_cache,
        )


@MPS
def test_inverse_unmatched_original_row_is_bias_only():
    indices = torch.tensor(
        [[0, 3, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
        dtype=torch.int32,
    )
    cache = {}
    down = sparse_conv3d_forward_mps(
        SparseTensor3D(indices, torch.ones((4, 1), device="mps"), (5, 1, 1), 1),
        torch.ones((1, 1, 1, 1, 1), device="mps"),
        stride=(2, 1, 1), indice_key="down", indice_cache=cache,
    )
    inverse_features = torch.ones(down.features.shape, device="mps", requires_grad=True)
    weight = torch.ones((1, 1, 1, 1, 1), device="mps", requires_grad=True)
    bias = torch.tensor([7.], device="mps", requires_grad=True)
    restored = sparse_inverse_conv3d_forward_mps(
        down.replace_features(inverse_features), weight, bias=bias,
        indice_key="down", indice_cache=cache,
    )
    assert torch.equal(restored.indices, indices)
    torch.testing.assert_close(
        restored.features[:, 0].cpu(), torch.tensor([7., 8., 8., 8.])
    )
    restored.features.sum().backward()
    torch.testing.assert_close(inverse_features.grad.cpu(), torch.ones((3, 1)))
    torch.testing.assert_close(weight.grad.cpu(), torch.tensor([[[[[3.]]]]]))
    torch.testing.assert_close(bias.grad.cpu(), torch.tensor([4.]))


@MPS
def test_downsample_inverse_chain_propagates_first_gradients():
    rng = torch.Generator().manual_seed(4815)
    indices = torch.tensor(
        [[1, 4, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0],
         [0, 3, 0, 0], [1, 5, 0, 0], [0, 2, 0, 0]], dtype=torch.int32,
    )
    values = torch.randn((6, 2), generator=rng)
    down_values = torch.randn((3, 2, 3, 1, 1), generator=rng)
    up_values = torch.randn((2, 3, 3, 1, 1), generator=rng)
    bias_values = torch.randn((2,), generator=rng)

    def run(device, down_op, inverse_op):
        x, down_w, up_w, bias = (
            value.detach().to(device).requires_grad_()
            for value in (values, down_values, up_values, bias_values)
        )
        cache = {}
        down = down_op(
            SparseTensor3D(indices, x, (6, 1, 1), 2), down_w,
            stride=(2, 1, 1), padding=(1, 0, 0),
            indice_key="down", indice_cache=cache,
        )
        up = inverse_op(
            down, up_w, bias=bias, indice_key="down", indice_cache=cache,
        )
        (up.features.square().sum()).backward()
        return up, (x.grad, down_w.grad, up_w.grad, bias.grad)

    reference, cpu_grads = run("cpu", sparse_conv3d_forward_cpu,
                               sparse_inverse_conv3d_forward_cpu)
    actual, mps_grads = run("mps", sparse_conv3d_forward_mps,
                            sparse_inverse_conv3d_forward_mps)
    torch.mps.synchronize()
    assert torch.equal(actual.indices, reference.indices)
    torch.testing.assert_close(actual.features.detach().cpu(), reference.features.detach(),
                               rtol=1e-4, atol=1e-5)
    for observed, expected in zip(mps_grads, cpu_grads):
        torch.testing.assert_close(observed.cpu(), expected, rtol=1e-4, atol=1e-5)


@MPS
def test_empty_and_unreachable_inputs_have_zero_gradients():
    indices = torch.empty((0, 4), dtype=torch.int32)
    features = torch.empty((0, 2), device="mps", requires_grad=True)
    weights = torch.ones((3, 2, 1, 1, 1), device="mps", requires_grad=True)
    bias = torch.ones((3,), device="mps", requires_grad=True)
    output = sparse_conv3d_forward_mps(
        SparseTensor3D(indices, features, (5, 1, 1), 1), weights,
        stride=(2, 1, 1), bias=bias,
    )
    assert output.features.shape == (0, 3)
    output.features.sum().backward()
    assert features.grad.shape == features.shape
    assert torch.count_nonzero(weights.grad).item() == 0
    assert torch.count_nonzero(bias.grad).item() == 0

    lone = torch.tensor([[0, 1, 0, 0]], dtype=torch.int32)
    x = torch.tensor([[5.]], device="mps", requires_grad=True)
    w = torch.ones((1, 1, 1, 1, 1), device="mps", requires_grad=True)
    b = torch.tensor([7.], device="mps", requires_grad=True)
    missing = sparse_conv3d_forward_mps(
        SparseTensor3D(lone, x, (3, 1, 1), 1), w,
        stride=(2, 1, 1), bias=b,
    )
    assert missing.features.shape == (0, 1)
    missing.features.sum().backward()
    assert torch.count_nonzero(x.grad).item() == 0
    assert torch.count_nonzero(w.grad).item() == 0
    assert torch.count_nonzero(b.grad).item() == 0


@MPS
def test_noncontiguous_feature_and_weight_gradients():
    indices = torch.tensor([[0, 2, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    base_x = torch.tensor([[1., 99., 2., 99.], [3., 99., 4., 99.]],
                          device="mps", requires_grad=True)
    x = base_x[:, ::2]
    base_w = torch.arange(1, 25, dtype=torch.float32, device="mps", requires_grad=True)
    w = base_w.reshape(2, 2, 3, 1, 2)[..., ::2]
    assert not x.is_contiguous() and not w.is_contiguous()
    actual = sparse_conv3d_forward_mps(
        SparseTensor3D(indices, x, (4, 1, 1), 1), w,
        stride=1, padding=(1, 0, 0),
    )
    cpu_x = base_x.detach().cpu().requires_grad_()
    cpu_w = base_w.detach().cpu().requires_grad_()
    reference = sparse_conv3d_forward_cpu(
        SparseTensor3D(indices, cpu_x[:, ::2], (4, 1, 1), 1),
        cpu_w.reshape(2, 2, 3, 1, 2)[..., ::2],
        stride=1, padding=(1, 0, 0),
    )
    upstream = torch.tensor([[0.75, -1.25], [1.5, 0.5], [-0.3, 0.8], [0.4, -0.2]])
    (actual.features * upstream.to("mps")).sum().backward()
    (reference.features * upstream).sum().backward()
    torch.mps.synchronize()
    torch.testing.assert_close(actual.features.detach().cpu(), reference.features.detach(), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(base_x.grad.cpu(), cpu_x.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(base_w.grad.cpu(), cpu_w.grad, rtol=1e-4, atol=1e-5)

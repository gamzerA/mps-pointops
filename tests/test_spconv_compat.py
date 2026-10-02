"""Bounded CPU checks for the private sparse-nn adapter, not spconv parity."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mps_pointops._sparse_conv_cpu import (
    _evaluate_pairs,
    sparse_conv3d_forward_cpu,
    sparse_inverse_conv3d_forward_cpu,
)
from mps_pointops._sparse_rulebook import generate_subm_rulebook, validate_sparse_tensor_3d
from mps_pointops._spconv_compat import (
    SparseConv3d,
    SparseConvTensor,
    SparseInverseConv3d,
    SparseSequential,
    SubMConv3d,
)


COORDS = torch.tensor(
    [[0, 4, 3, 2], [0, 0, 0, 0], [0, 3, 2, 2], [0, 2, 2, 1], [0, 1, 3, 0]],
    dtype=torch.int32,
)
SHAPE = (6, 5, 4)


def _features():
    torch.manual_seed(730)
    return torch.randn(len(COORDS), 2, dtype=torch.float64)


def _weight_for_oracle(layer):
    weight = layer.weight.detach().clone().requires_grad_()
    bias = None if layer.bias is None else layer.bias.detach().clone().requires_grad_()
    return weight, bias


@pytest.mark.parametrize("shape", [(6.0, 5, 4), (True, 5, 4), (0, 5, 4), (2**31, 5, 4)])
def test_shape_rejects_silent_integer_conversion(shape):
    with pytest.raises(ValueError, match="spatial_shape"):
        SparseConvTensor(_features(), COORDS, shape, 1)


def test_numpy_integer_shape_and_dense_scatter_preserve_row_gradients():
    values = _features().requires_grad_()
    sparse = SparseConvTensor(values, COORDS, np.array(SHAPE, dtype=np.int64), np.int64(1))
    dense = sparse.dense(channels_first=False)
    assert dense.shape == (1, *SHAPE, 2)
    torch.testing.assert_close(dense[0, 4, 3, 2], values[0])
    torch.testing.assert_close(dense[0, 0, 0, 0], values[1])
    dense.square().sum().backward()
    torch.testing.assert_close(values.grad, 2 * values.detach())


def test_constructor_rejects_invalid_indices_and_groups():
    with pytest.raises(ValueError, match="int32"):
        SparseConvTensor(_features(), COORDS.long(), SHAPE, 1)
    with pytest.raises(ValueError, match="groups=1"):
        SubMConv3d(2, 3, 3, groups=True)
    with pytest.raises(ValueError, match="Native"):
        SubMConv3d(2, 3, 3, algo=0)
    with pytest.raises(ValueError, match="SubM padding"):
        SubMConv3d(2, 3, 3, padding=2)
    with pytest.raises(ValueError, match="int32"):
        SparseConvTensor(torch.empty((0, 2)), torch.empty((0, 4), dtype=torch.int32), SHAPE, 2**31)


def test_dense_conversion_rejects_oversized_allocation_before_allocating():
    sparse = SparseConvTensor(
        torch.empty((0, 2)), torch.empty((0, 4), dtype=torch.int32),
        (2**20, 2**10, 2**10), 1,
    )
    with pytest.raises(ValueError, match="256 MiB"):
        sparse.dense()


def test_subm_krsc_adapter_output_and_first_gradients_match_coordinate_oracle():
    torch.manual_seed(731)
    layer = SubMConv3d(2, 3, 3, padding=1, bias=True, indice_key="subm").double()
    input_features = _features().requires_grad_()
    sparse = SparseConvTensor(input_features, COORDS, SHAPE, 1)
    output = layer(sparse)

    reference_features = _features().requires_grad_()
    reference = validate_sparse_tensor_3d(COORDS, reference_features, SHAPE, 1)
    ref_weight, ref_bias = _weight_for_oracle(layer)
    pairs = generate_subm_rulebook(reference, kernel_size=(3, 3, 3)).pairs
    expected = _evaluate_pairs(
        reference_features, ref_weight.permute(0, 4, 1, 2, 3),
        ref_bias, pairs, len(COORDS), inverse=False,
    )
    assert torch.equal(output.indices, COORDS)
    torch.testing.assert_close(output.features, expected, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(output.features)
    output.features.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(input_features.grad, reference_features.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer.weight.grad, ref_weight.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(layer.bias.grad, ref_bias.grad, rtol=1e-12, atol=1e-12)


def test_stride_inverse_preserve_cache_lineage_and_first_gradients():
    torch.manual_seed(732)
    down = SparseConv3d(2, 3, 3, stride=2, padding=1, bias=True, indice_key="down").double()
    up = SparseInverseConv3d(3, 2, 3, indice_key="down", bias=True).double()
    features = _features().requires_grad_()
    sparse = SparseConvTensor(features, COORDS, SHAPE, 1)
    reduced = down(sparse)
    assert "down" in reduced.indice_dict
    middle = reduced.replace_feature(torch.relu(reduced.features))
    restored = up(middle)
    assert torch.equal(restored.indices, COORDS)
    assert restored.spatial_shape == list(SHAPE)

    ref_features = _features().requires_grad_()
    ref_sparse = validate_sparse_tensor_3d(COORDS, ref_features, SHAPE, 1)
    ref_down_weight, ref_down_bias = _weight_for_oracle(down)
    ref_up_weight, ref_up_bias = _weight_for_oracle(up)
    cache = {}
    ref_reduced = sparse_conv3d_forward_cpu(
        ref_sparse, ref_down_weight.permute(0, 4, 1, 2, 3),
        stride=2, padding=1, bias=ref_down_bias, indice_key="down", indice_cache=cache,
    )
    ref_restored = sparse_inverse_conv3d_forward_cpu(
        ref_reduced.replace_features(torch.relu(ref_reduced.features)),
        ref_up_weight.permute(0, 4, 1, 2, 3),
        indice_key="down", indice_cache=cache, bias=ref_up_bias,
    )
    assert torch.equal(reduced.indices, ref_reduced.indices)
    torch.testing.assert_close(restored.features, ref_restored.features, rtol=1e-12, atol=1e-12)
    upstream = torch.randn_like(restored.features)
    restored.features.backward(upstream)
    ref_restored.features.backward(upstream)
    for actual, expected in (
        (features.grad, ref_features.grad),
        (down.weight.grad, ref_down_weight.grad),
        (down.bias.grad, ref_down_bias.grad),
        (up.weight.grad, ref_up_weight.grad),
        (up.bias.grad, ref_up_bias.grad),
    ):
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    forged = SparseConvTensor(middle.features.detach(), middle.indices, middle.spatial_shape,
                              middle.batch_size, indice_dict=middle.indice_dict)
    with pytest.raises(ValueError, match="lineage"):
        up(forged)


def test_sparse_sequential_applies_feature_modules_without_losing_indices():
    model = SparseSequential(SubMConv3d(2, 2, 1, bias=False), torch.nn.ReLU())
    sparse = SparseConvTensor(_features().float(), COORDS, SHAPE, 1)
    result = model(sparse)
    assert torch.equal(result.indices, COORDS)
    assert torch.all(result.features >= 0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_mps_adapter_subm_stride_inverse_matches_cpu_first_gradients():
    torch.manual_seed(733)

    def new_layers(device):
        return torch.nn.ModuleList([
            SubMConv3d(2, 3, 3, padding=1, bias=True, indice_key="subm"),
            SparseConv3d(3, 3, 3, stride=2, padding=1, bias=True, indice_key="down"),
            SparseInverseConv3d(3, 2, 3, indice_key="down", bias=True),
        ]).to(device)

    cpu_layers = new_layers("cpu")
    mps_layers = new_layers("mps")
    mps_layers.load_state_dict(cpu_layers.state_dict())

    def evaluate(layers, device):
        features = _features().float().to(device).requires_grad_()
        sparse = SparseConvTensor(features, COORDS, SHAPE, 1)
        first = layers[0](sparse)
        down = layers[1](first)
        restored = layers[2](down.replace_feature(torch.relu(down.features)))
        loss = first.features.square().sum() + down.features.square().sum() + restored.features.square().sum()
        loss.backward()
        if device == "mps":
            torch.mps.synchronize()
        return first, down, restored, features.grad, [parameter.grad for parameter in layers.parameters()]

    expected = evaluate(cpu_layers, "cpu")
    actual = evaluate(mps_layers, "mps")
    for position in range(3):
        assert torch.equal(actual[position].indices, expected[position].indices)
        torch.testing.assert_close(actual[position].features.cpu(), expected[position].features,
                                   rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual[3].cpu(), expected[3], rtol=1e-4, atol=1e-5)
    for got_grad, expected_grad in zip(actual[4], expected[4]):
        assert got_grad is not None and expected_grad is not None
        torch.testing.assert_close(got_grad.cpu(), expected_grad, rtol=1e-4, atol=1e-5)

"""Dense-PyTorch checks for the private ordinary/inverse CPU reference."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mps_pointops._sparse_conv_cpu import (
    MAX_CPU_POINTS,
    sparse_conv3d_forward_cpu,
    sparse_inverse_conv3d_forward_cpu,
)
from mps_pointops._sparse_rulebook import SparseTensor3D, validate_sparse_tensor_3d


def _sparse(coords, features, shape=(6, 4, 3), batch_size=2):
    indices = torch.tensor(coords, dtype=torch.int32).reshape(-1, 4)
    return validate_sparse_tensor_3d(indices, features, shape, batch_size)


def _dense_features(sparse):
    shape = sparse.spatial_shape
    rows = sparse.indices.to(torch.int64)
    positions = (((rows[:, 0] * shape[0] + rows[:, 1]) * shape[1] + rows[:, 2])
                 * shape[2] + rows[:, 3])
    dense = sparse.features.new_zeros(
        (sparse.batch_size * shape[0] * shape[1] * shape[2], sparse.features.shape[1])
    ).index_copy(0, positions, sparse.features)
    return dense.reshape(sparse.batch_size, *shape, -1).permute(0, 4, 1, 2, 3)


def _at_indices(dense, indices):
    rows = indices.to(torch.int64)
    return dense.permute(0, 2, 3, 4, 1)[
        rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    ]


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation"),
    [
        ((3, 2, 1), (2, 1, 1), (1, 0, 0), (1, 1, 1)),
        ((2, 3, 2), (2, 2, 1), (0, 1, 1), (2, 1, 1)),
        ((1, 1, 1), (3, 2, 2), (0, 0, 0), (1, 1, 1)),
    ],
)
def test_ordinary_stride_output_and_first_order_gradients_match_dense(
    kernel, stride, padding, dilation
):
    torch.manual_seed(11)
    coords = [
        [1, 4, 2, 2], [0, 1, 2, 2], [1, 0, 0, 0],
        [0, 5, 3, 2], [0, 2, 1, 1], [1, 3, 3, 1],
    ]
    features = torch.randn(len(coords), 2, dtype=torch.float64)
    weights = torch.randn(3, 2, *kernel, dtype=torch.float64)
    bias = torch.randn(3, dtype=torch.float64)
    sparse = _sparse(coords, features.clone().requires_grad_())
    sparse_weights = weights.clone().requires_grad_()
    sparse_bias = bias.clone().requires_grad_()
    cache = {}
    output = sparse_conv3d_forward_cpu(
        sparse, sparse_weights, stride=stride, padding=padding,
        dilation=dilation, bias=sparse_bias, indice_key="down", indice_cache=cache,
    )

    dense_sparse = _sparse(coords, features.clone().requires_grad_())
    dense_weights = weights.clone().requires_grad_()
    dense_bias = bias.clone().requires_grad_()
    occupancy = _dense_features(
        _sparse(coords, torch.ones(len(coords), 1, dtype=torch.float64))
    )
    counts = F.conv3d(
        occupancy, torch.ones((1, 1, *kernel), dtype=torch.float64),
        stride=stride, padding=padding, dilation=dilation,
    )
    expected_indices = torch.nonzero(counts[:, 0] > 0).to(torch.int32)
    assert torch.equal(output.indices, expected_indices)
    assert output.spatial_shape == tuple(counts.shape[2:])
    dense_output = F.conv3d(
        _dense_features(dense_sparse), dense_weights, dense_bias,
        stride=stride, padding=padding, dilation=dilation,
    )
    expected_values = _at_indices(dense_output, expected_indices)
    torch.testing.assert_close(output.features, expected_values, rtol=1e-12, atol=1e-12)

    upstream = torch.randn_like(output.features)
    output.features.backward(upstream)
    expected_values.backward(upstream)
    for got, expected in (
        (sparse.features.grad, dense_sparse.features.grad),
        (sparse_weights.grad, dense_weights.grad),
        (sparse_bias.grad, dense_bias.grad),
    ):
        torch.testing.assert_close(got, expected, rtol=1e-12, atol=1e-12)
    assert torch.equal(cache["down"].input_indices, sparse.indices)
    assert torch.equal(cache["down"].output_indices, output.indices)


@pytest.mark.parametrize(
    ("coords", "shape", "kernel", "stride", "padding"),
    [
        ([[0, 3, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
         (5, 1, 1), (1, 1, 1), (2, 1, 1), (0, 0, 0)),
        ([[1, 5, 0, 0], [0, 1, 0, 0], [1, 2, 0, 0], [0, 4, 0, 0],
          [0, 0, 0, 0], [1, 3, 0, 0]],
         (6, 1, 1), (3, 1, 1), (2, 1, 1), (1, 0, 0)),
    ],
)
def test_inverse_restores_saved_rows_and_matches_dense_transpose_gradients(
    coords, shape, kernel, stride, padding
):
    torch.manual_seed(31)
    original = _sparse(
        coords, torch.randn(len(coords), 2, dtype=torch.float64),
        shape=shape, batch_size=max(coord[0] for coord in coords) + 1,
    )
    cache = {}
    down = sparse_conv3d_forward_cpu(
        original, torch.randn(3, 2, *kernel, dtype=torch.float64),
        stride=stride, padding=padding, indice_key="down", indice_cache=cache,
    )
    inverse_features = torch.randn_like(down.features)
    inverse_weights = torch.randn(2, 3, *kernel, dtype=torch.float64)
    inverse_bias = torch.randn(2, dtype=torch.float64)
    inverse_input = SparseTensor3D(
        down.indices.clone(), inverse_features.clone().requires_grad_(),
        down.spatial_shape, down.batch_size,
    )
    sparse_weights = inverse_weights.clone().requires_grad_()
    sparse_bias = inverse_bias.clone().requires_grad_()
    restored = sparse_inverse_conv3d_forward_cpu(
        inverse_input, sparse_weights, bias=sparse_bias,
        indice_key="down", indice_cache=cache,
    )
    assert torch.equal(restored.indices, original.indices)
    assert restored.spatial_shape == original.spatial_shape

    dense_input = SparseTensor3D(
        down.indices.clone(), inverse_features.clone().requires_grad_(),
        down.spatial_shape, down.batch_size,
    )
    dense_weights = inverse_weights.clone().requires_grad_()
    dense_bias = inverse_bias.clone().requires_grad_()
    output_padding = tuple(
        shape[axis] - ((down.spatial_shape[axis] - 1) * stride[axis]
                       - 2 * padding[axis] + kernel[axis])
        for axis in range(3)
    )
    dense_restored = F.conv_transpose3d(
        _dense_features(dense_input), dense_weights.transpose(0, 1),
        dense_bias, stride=stride, padding=padding, output_padding=output_padding,
    )
    expected = _at_indices(dense_restored, original.indices)
    torch.testing.assert_close(restored.features, expected, rtol=1e-12, atol=1e-12)

    upstream = torch.randn_like(expected)
    restored.features.backward(upstream)
    expected.backward(upstream)
    for got, want in (
        (inverse_input.features.grad, dense_input.features.grad),
        (sparse_weights.grad, dense_weights.grad),
        (sparse_bias.grad, dense_bias.grad),
    ):
        torch.testing.assert_close(got, want, rtol=1e-12, atol=1e-12)


def test_inverse_is_not_sparse_transpose_coordinate_generation():
    sparse = _sparse(
        [[0, 3, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]],
        torch.ones(4, 1, dtype=torch.float64), shape=(5, 1, 1), batch_size=1,
    )
    cache = {}
    down = sparse_conv3d_forward_cpu(
        sparse, torch.ones((1, 1, 1, 1, 1), dtype=torch.float64),
        stride=(2, 1, 1), indice_key="down", indice_cache=cache,
    )
    restored = sparse_inverse_conv3d_forward_cpu(
        down, torch.ones((1, 1, 1, 1, 1), dtype=torch.float64),
        bias=torch.tensor([7.0], dtype=torch.float64),
        indice_key="down", indice_cache=cache,
    )
    assert restored.indices.tolist() == sparse.indices.tolist()
    assert restored.features[:, 0].tolist() == [7.0, 8.0, 8.0, 8.0]
    # A transposed convolution can reach only positions 0, 2 and 4 here;
    # the saved original active position 3 remains present with bias only.
    occupancy = F.conv_transpose3d(
        _dense_features(_sparse(
            down.indices.tolist(), torch.ones((3, 1), dtype=torch.float64),
            shape=down.spatial_shape, batch_size=1,
        )),
        torch.ones((1, 1, 1, 1, 1), dtype=torch.float64), stride=(2, 1, 1),
    )
    assert torch.nonzero(occupancy[:, 0] > 0).tolist() == [
        [0, 0, 0, 0], [0, 2, 0, 0], [0, 4, 0, 0]
    ]


def test_inverse_rejects_missing_stale_or_mismatched_cache():
    sparse = _sparse(
        [[0, 4, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]],
        torch.ones(3, 1, dtype=torch.float64), shape=(5, 1, 1), batch_size=1,
    )
    weight = torch.ones((1, 1, 1, 1, 1), dtype=torch.float64)
    with pytest.raises(KeyError, match="no saved"):
        sparse_inverse_conv3d_forward_cpu(
            sparse, weight, indice_key="missing", indice_cache={}
        )
    cache = {}
    down = sparse_conv3d_forward_cpu(
        sparse, weight, stride=(2, 1, 1), indice_key="down", indice_cache=cache,
    )
    with pytest.raises(ValueError, match="already exists"):
        sparse_conv3d_forward_cpu(
            sparse, weight, stride=(2, 1, 1), indice_key="down", indice_cache=cache,
        )
    wrong_order = SparseTensor3D(
        down.indices.flip(0), down.features.flip(0), down.spatial_shape, down.batch_size,
    )
    with pytest.raises(ValueError, match="row order"):
        sparse_inverse_conv3d_forward_cpu(
            wrong_order, weight, indice_key="down", indice_cache=cache,
        )
    wrong_shape = SparseTensor3D(
        down.indices, down.features, (4, 1, 1), down.batch_size,
    )
    with pytest.raises(ValueError, match="spatial shape"):
        sparse_inverse_conv3d_forward_cpu(
            wrong_shape, weight, indice_key="down", indice_cache=cache,
        )
    with pytest.raises(ValueError, match="kernel size"):
        sparse_inverse_conv3d_forward_cpu(
            down, torch.ones((1, 1, 2, 1, 1), dtype=torch.float64),
            indice_key="down", indice_cache=cache,
        )
    # Output tensor mutation cannot rewrite the cache's saved mapping.
    down.indices[0, 1] = 1
    assert cache["down"].output_indices[0, 1].item() == 0
    with pytest.raises(ValueError, match="duplicate"):
        sparse_inverse_conv3d_forward_cpu(
            down, weight, indice_key="down", indice_cache=cache,
        )


def test_empty_pair_gradients_and_explicit_cpu_bounds():
    features = torch.empty((0, 2), dtype=torch.float64, requires_grad=True)
    empty = _sparse([], features, shape=(3, 1, 1), batch_size=0)
    forward_weight = torch.ones((3, 2, 1, 1, 1), dtype=torch.float64, requires_grad=True)
    cache = {}
    down = sparse_conv3d_forward_cpu(
        empty, forward_weight, indice_key="empty", indice_cache=cache,
    )
    assert down.features.shape == (0, 3)
    down.features.sum().backward()
    assert torch.equal(forward_weight.grad, torch.zeros_like(forward_weight))
    assert features.grad.shape == features.shape

    inverse_features = torch.empty((0, 3), dtype=torch.float64, requires_grad=True)
    down = SparseTensor3D(down.indices, inverse_features, down.spatial_shape, down.batch_size)
    inverse_weight = torch.ones((2, 3, 1, 1, 1), dtype=torch.float64, requires_grad=True)
    restored = sparse_inverse_conv3d_forward_cpu(
        down, inverse_weight, indice_key="empty", indice_cache=cache,
    )
    restored.features.sum().backward()
    assert restored.features.shape == (0, 2)
    assert torch.equal(inverse_weight.grad, torch.zeros_like(inverse_weight))
    assert inverse_features.grad.shape == inverse_features.shape

    too_many = _sparse(
        [[0, row, 0, 0] for row in range(MAX_CPU_POINTS + 1)],
        torch.ones((MAX_CPU_POINTS + 1, 1), dtype=torch.float64),
        shape=(MAX_CPU_POINTS + 1, 1, 1), batch_size=1,
    )
    with pytest.raises(ValueError, match="active points"):
        sparse_conv3d_forward_cpu(
            too_many, torch.ones((1, 1, 1, 1, 1), dtype=torch.float64)
        )


def test_inverse_reuses_saved_pairs_without_reapplying_generation_candidate_limit():
    sparse = _sparse(
        [[0, 2, 2, 2], [0, 10, 2, 2], [0, 18, 2, 2]],
        torch.ones((3, 1), dtype=torch.float64),
        shape=(22, 5, 5), batch_size=1,
    )
    weight = torch.ones((1, 1, 5, 5, 5), dtype=torch.float64)
    cache = {}
    down = sparse_conv3d_forward_cpu(
        sparse, weight, padding=2, indice_key="expanded", indice_cache=cache,
    )
    assert len(down.indices) == 375
    # 375 * 125 would exceed the forward generator's candidate budget, but
    # inverse only consumes the 375 saved pairs.
    restored = sparse_inverse_conv3d_forward_cpu(
        down, weight, indice_key="expanded", indice_cache=cache,
    )
    assert torch.equal(restored.indices, sparse.indices)
    assert restored.features[:, 0].tolist() == [125.0, 125.0, 125.0]

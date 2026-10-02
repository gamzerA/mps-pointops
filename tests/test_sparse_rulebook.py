"""Independent CPU checks for the experimental sparse rulebook oracle."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mps_pointops._sparse_rulebook import (
    generate_sparse_conv_rulebook,
    generate_subm_rulebook,
    validate_sparse_tensor_3d,
)


def _sparse(indices, shape=(5, 5, 5), batch_size=1, channels=2):
    coordinates = torch.tensor(indices, dtype=torch.int32).reshape(-1, 4)
    features = torch.arange(
        1, coordinates.shape[0] * channels + 1, dtype=torch.float64
    ).reshape(-1, channels)
    return validate_sparse_tensor_3d(coordinates, features, shape, batch_size)


@pytest.mark.parametrize(
    ("indices", "features", "shape", "batch", "message"),
    [
        (torch.zeros((1, 3), dtype=torch.int32), torch.ones((1, 1)), (2, 2, 2), 1, "indices"),
        (torch.zeros((1, 4), dtype=torch.int64), torch.ones((1, 1)), (2, 2, 2), 1, "indices"),
        (torch.zeros((1, 4), dtype=torch.int32), torch.ones((2, 1)), (2, 2, 2), 1, "features"),
        (torch.zeros((1, 4), dtype=torch.int32), torch.ones((1, 0)), (2, 2, 2), 1, "features"),
        (torch.zeros((1, 4), dtype=torch.int32), torch.ones((1, 1), dtype=torch.int32), (2, 2, 2), 1, "features"),
        (torch.tensor([[0, 2, 0, 0]], dtype=torch.int32), torch.ones((1, 1)), (2, 2, 2), 1, "outside"),
        (torch.tensor([[1, 0, 0, 0]], dtype=torch.int32), torch.ones((1, 1)), (2, 2, 2), 1, "outside"),
        (torch.tensor([[0, -1, 0, 0]], dtype=torch.int32), torch.ones((1, 1)), (2, 2, 2), 1, "outside"),
        (torch.zeros((2, 4), dtype=torch.int32), torch.ones((2, 1)), (2, 2, 2), 1, "duplicate"),
        (torch.empty((0, 4), dtype=torch.int32), torch.empty((0, 1)), (2, 0, 2), 0, "spatial_shape"),
    ],
)
def test_sparse_tensor_contract_rejects_invalid_input(indices, features, shape, batch, message):
    with pytest.raises(ValueError, match=message):
        validate_sparse_tensor_3d(indices, features, shape, batch)


def test_subm_preserves_unsorted_input_rows_and_batch_separation():
    sparse = _sparse(
        [[0, 2, 1, 1], [0, 1, 1, 1], [1, 2, 1, 1]], batch_size=2
    )
    rulebook = generate_subm_rulebook(sparse, kernel_size=3)
    assert torch.equal(rulebook.output_indices, sparse.indices)
    assert rulebook.output_spatial_shape == sparse.spatial_shape
    # 3x3x3 offset flattening: left=4, center=13, right=22.
    assert rulebook.pairs.tolist() == [
        [4, 1, 0],
        [13, 0, 0],
        [13, 1, 1],
        [13, 2, 2],
        [22, 0, 1],
    ]


def test_subm_dilation_and_even_kernel_rejection():
    sparse = _sparse([[0, 3, 0, 0], [0, 1, 0, 0]], shape=(6, 1, 1))
    rulebook = generate_subm_rulebook(sparse, kernel_size=(3, 1, 1), dilation=(2, 1, 1))
    assert rulebook.pairs.tolist() == [
        [0, 1, 0], [1, 0, 0], [1, 1, 1], [2, 0, 1]
    ]
    with pytest.raises(ValueError, match="odd kernel"):
        generate_subm_rulebook(sparse, kernel_size=(2, 1, 1))


def test_strided_conv_padding_exact_pair_order():
    sparse = _sparse(
        [[0, 4, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0]],
        shape=(5, 1, 1),
    )
    rulebook = generate_sparse_conv_rulebook(
        sparse, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)
    )
    assert rulebook.output_spatial_shape == (3, 1, 1)
    assert rulebook.output_indices.tolist() == [
        [0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]
    ]
    assert rulebook.pairs.tolist() == [
        [0, 1, 1],
        [1, 2, 0], [1, 3, 1], [1, 0, 2],
        [2, 1, 0],
    ]


@pytest.mark.parametrize(
    ("kernel", "stride", "padding", "dilation"),
    [
        ((3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1)),
        ((3, 2, 1), (2, 1, 1), (1, 0, 0), (1, 1, 1)),
        ((2, 3, 2), (2, 2, 1), (0, 1, 1), (2, 1, 1)),
        ((1, 1, 1), (3, 2, 2), (0, 0, 0), (1, 1, 1)),
    ],
)
def test_regular_rulebook_matches_independent_dense_conv(kernel, stride, padding, dilation):
    shape = (6, 5, 4)
    # Deliberately unsorted and split across two independent batches.
    sparse = _sparse(
        [[1, 4, 2, 2], [0, 1, 2, 3], [1, 0, 0, 0],
         [0, 5, 4, 3], [0, 2, 2, 2], [1, 3, 4, 1]],
        shape=shape, batch_size=2,
    )
    rulebook = generate_sparse_conv_rulebook(
        sparse, kernel_size=kernel, stride=stride, padding=padding, dilation=dilation
    )
    occupancy = torch.zeros((2, 1, *shape), dtype=torch.float64)
    dense_features = torch.zeros((2, sparse.features.shape[1], *shape), dtype=torch.float64)
    for row, (batch, a, b, c) in enumerate(sparse.indices.tolist()):
        occupancy[batch, 0, a, b, c] = 1
        dense_features[batch, :, a, b, c] = sparse.features[row]

    dense_count = F.conv3d(
        occupancy, torch.ones((1, 1, *kernel), dtype=torch.float64),
        stride=stride, padding=padding, dilation=dilation,
    )
    expected_coords = torch.nonzero(dense_count[:, 0] > 0).to(torch.int32)
    assert torch.equal(rulebook.output_indices, expected_coords)
    assert rulebook.output_spatial_shape == tuple(dense_count.shape[2:])

    weights = torch.arange(
        1, 3 * sparse.features.shape[1] * (kernel[0] * kernel[1] * kernel[2]) + 1,
        dtype=torch.float64,
    ).reshape(3, sparse.features.shape[1], *kernel) / 13.0
    dense_output = F.conv3d(
        dense_features, weights, stride=stride, padding=padding, dilation=dilation
    )
    oracle_values = torch.zeros((len(expected_coords), 3), dtype=torch.float64)
    weight_by_offset = weights.flatten(2)
    for offset, in_row, out_row in rulebook.pairs.tolist():
        oracle_values[out_row] += weight_by_offset[:, :, offset] @ sparse.features[in_row]
    expected_values = torch.stack(
        [dense_output[batch, :, a, b, c] for batch, a, b, c in expected_coords.tolist()]
    ) if len(expected_coords) else torch.empty((0, 3), dtype=torch.float64)
    torch.testing.assert_close(oracle_values, expected_values, rtol=1e-12, atol=1e-12)


def test_subm_rulebook_matches_dense_conv_only_at_active_outputs():
    sparse = _sparse(
        [[0, 2, 1, 1], [0, 0, 1, 1], [0, 4, 1, 1]], shape=(5, 3, 3)
    )
    kernel = (3, 1, 1)
    dilation = (2, 1, 1)
    rulebook = generate_subm_rulebook(sparse, kernel_size=kernel, dilation=dilation)
    dense_features = torch.zeros((1, 2, 5, 3, 3), dtype=torch.float64)
    for row, (batch, a, b, c) in enumerate(sparse.indices.tolist()):
        dense_features[batch, :, a, b, c] = sparse.features[row]
    weights = torch.arange(1, 13, dtype=torch.float64).reshape(2, 2, *kernel)
    dense_output = F.conv3d(
        dense_features, weights, stride=1, padding=(2, 0, 0), dilation=dilation
    )
    oracle_values = torch.zeros((len(sparse.indices), 2), dtype=torch.float64)
    weight_by_offset = weights.flatten(2)
    for offset, in_row, out_row in rulebook.pairs.tolist():
        oracle_values[out_row] += weight_by_offset[:, :, offset] @ sparse.features[in_row]
    expected = torch.stack([
        dense_output[batch, :, a, b, c]
        for batch, a, b, c in sparse.indices.tolist()
    ])
    torch.testing.assert_close(oracle_values, expected, rtol=1e-12, atol=1e-12)


def test_regular_output_is_independent_of_input_row_order():
    coords = [[0, 3, 0, 0], [0, 1, 0, 0], [0, 5, 0, 0]]
    first = _sparse(coords, shape=(6, 1, 1))
    second = _sparse(list(reversed(coords)), shape=(6, 1, 1))
    a = generate_sparse_conv_rulebook(first, kernel_size=(3, 1, 1), stride=(2, 1, 1))
    b = generate_sparse_conv_rulebook(second, kernel_size=(3, 1, 1), stride=(2, 1, 1))
    assert torch.equal(a.output_indices, b.output_indices)
    def by_coordinate(sparse, rulebook):
        return [
            (offset, tuple(sparse.indices[in_row].tolist()), tuple(rulebook.output_indices[out_row].tolist()))
            for offset, in_row, out_row in rulebook.pairs.tolist()
        ]
    assert by_coordinate(first, a) == by_coordinate(second, b)


def test_empty_input_and_parameter_rejection():
    sparse = _sparse([], shape=(4, 4, 4), batch_size=0)
    subm = generate_subm_rulebook(sparse, kernel_size=3)
    regular = generate_sparse_conv_rulebook(sparse, kernel_size=3, padding=1)
    assert subm.output_indices.shape == regular.output_indices.shape == (0, 4)
    assert subm.pairs.shape == regular.pairs.shape == (0, 3)
    for kwargs in ({"kernel_size": 0}, {"kernel_size": 3, "stride": 0},
                   {"kernel_size": 3, "padding": -1}, {"kernel_size": 3, "dilation": (1, 0, 1)}):
        with pytest.raises(ValueError):
            generate_sparse_conv_rulebook(sparse, **kwargs)
    with pytest.raises(ValueError, match="output spatial shape"):
        generate_sparse_conv_rulebook(sparse, kernel_size=(9, 1, 1))

"""Local checks for the portable parts of the tiny spconv parity probe."""

import pytest
import torch

from tools.verify_spconv_toy_parity import (
    _aligned,
    _from_native_layout,
    _native_layout,
    _values,
)


@pytest.mark.parametrize("channels", [(4, 2), (2, 4)])
def test_spconv_weight_layouts_round_trip(channels):
    weight = _values((channels[0], channels[1], 3, 3, 3), scale=64, phase=7)
    for shape, expected_layout in (
        ((channels[0], 3, 3, 3, channels[1]), "KRSC"),
        ((3, 3, 3, channels[0], channels[1]), "RSKC"),
        ((3, 3, 3, channels[1], channels[0]), "RSCK"),
    ):
        actual_layout, mapped = _native_layout(weight, torch.Size(shape))
        assert actual_layout == expected_layout
        assert torch.equal(_from_native_layout(mapped, actual_layout), weight)


def test_coordinate_alignment_ignores_row_order_but_not_membership():
    expected_indices = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.int32)
    observed_indices = expected_indices.flip(0)
    expected = torch.tensor([[10.], [20.]])
    observed = expected.flip(0)
    keys, left, right = _aligned(
        expected_indices, expected, observed_indices, observed
    )
    assert keys == [(0, 1, 0, 0), (1, 0, 0, 0)]
    assert torch.equal(left, right)
    bad_indices = torch.tensor([[1, 0, 0, 0], [0, 2, 0, 0]], dtype=torch.int32)
    with pytest.raises(AssertionError, match="coordinate mismatch"):
        _aligned(expected_indices, expected, bad_indices, observed)

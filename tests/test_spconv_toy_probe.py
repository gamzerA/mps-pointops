"""Local checks for the portable parts of the tiny spconv parity probe."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


# The probe is a standalone source-tree tool, deliberately absent from the
# installed wheel. CI invokes the ``pytest`` executable, whose sys.path need
# not include the repository root, so load the file by its actual location.
_PROBE_PATH = Path(__file__).resolve().parents[1] / "tools" / "verify_spconv_toy_parity.py"
_SPEC = importlib.util.spec_from_file_location("verify_spconv_toy_parity", _PROBE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _PROBE
_SPEC.loader.exec_module(_PROBE)
_aligned = _PROBE._aligned
_from_native_layout = _PROBE._from_native_layout
_native_layout = _PROBE._native_layout
_values = _PROBE._values


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

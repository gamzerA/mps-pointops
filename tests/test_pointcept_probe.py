"""Acceptance rules for the pinned Pointcept model probe."""

import pytest
import torch

from bench.probe_pointcept_ptv1 import _comparison


@pytest.mark.parametrize("scale", [2e-5, 6e-4])
def test_zero_mps_gradient_fails_despite_small_absolute_scale(scale):
    cpu = torch.tensor([scale, -scale / 2, 0.0])
    result = _comparison(cpu, torch.zeros_like(cpu), 1e-7, 5e-3, 1e-2)
    assert not result["passed"]
    assert not result["nonzero_gradient"]
    assert result["relative_l2_error"] == pytest.approx(1.0)


def test_small_nonzero_gradient_difference_passes():
    cpu = torch.tensor([2e-5, -1e-5, 0.0])
    mps = cpu + torch.tensor([2e-11, -2e-11, 0.0])
    result = _comparison(cpu, mps, 1e-7, 5e-3, 1e-2)
    assert result["passed"]
    assert result["nonzero_gradient"]
    assert result["relative_l2_error"] < 1e-2


def test_gradient_shape_mismatch_fails():
    with pytest.raises(AssertionError, match="shapes differ"):
        _comparison(torch.ones(2), torch.ones(1), 1e-7, 5e-3, 1e-2)

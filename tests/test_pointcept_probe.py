"""Acceptance rules for the pinned Pointcept model probe."""

import importlib.util
from pathlib import Path

import pytest
import torch

PROBE_FILE = Path(__file__).resolve().parents[1] / "bench/probe_pointcept_ptv1.py"
SPEC = importlib.util.spec_from_file_location("pointcept_probe", PROBE_FILE)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
_comparison = PROBE._comparison


@pytest.mark.parametrize("scale", [2e-5, 6e-4])
def test_zero_mps_gradient_fails_despite_small_absolute_scale(scale):
    cpu = torch.tensor([scale, -scale / 2, 0.0])
    old_rule = _comparison(cpu, torch.zeros_like(cpu), 2e-3, 5e-3)
    result = _comparison(cpu, torch.zeros_like(cpu), 1e-7, 5e-3, 1e-2)
    assert old_rule["passed"]
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

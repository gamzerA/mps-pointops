"""Focused contracts for tensor-coordinate Chamfer outside three dimensions."""

from importlib import import_module

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from mps_pointops.chamfer import _nearest_mps, chamfer_distance


class _DeltaAllocationObserver(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.max_bytes = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func == torch.ops.aten.sub.Tensor and isinstance(result, torch.Tensor) and result.ndim == 4:
            self.max_bytes = max(self.max_bytes, result.numel() * result.element_size())
        return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_high_dimensional_delta_budget_and_batch_tiling(monkeypatch, dtype) -> None:
    chamfer = import_module("mps_pointops.chamfer")
    # At D=128, only four float32 or two float64 pairs fit. Five batches
    # therefore require batch chunks, not only smaller query/reference tiles.
    byte_budget = 2048
    monkeypatch.setattr(chamfer, "_DELTA_BYTES_PER_TILE", byte_budget)
    x = torch.arange(5 * 3 * 128, dtype=dtype).reshape(5, 3, 128) / 7
    y = x.flip(1).contiguous()
    lengths = torch.full((5,), 3, dtype=torch.long)
    observer = _DeltaAllocationObserver()
    with observer:
        indices = chamfer._nearest_indices(x, y, lengths, lengths, norm=2)
    assert indices.tolist() == [[2, 1, 0]] * 5
    assert 0 < observer.max_bytes <= byte_budget


def test_cpu_d4096_shape_and_first_index() -> None:
    chamfer = import_module("mps_pointops.chamfer")
    x = torch.zeros((1, 2, 4096), dtype=torch.float32)
    x[0, 1, 0] = 2
    y = x.flip(1).contiguous()
    lengths = torch.tensor([2], dtype=torch.long)
    assert chamfer._nearest_indices(x, y, lengths, lengths, norm=2).tolist() == [[1, 0]]


@pytest.mark.parametrize("device", ["cpu", "mps"])
@pytest.mark.parametrize("dimension", [1, 2, 4, 64, 128])
@pytest.mark.parametrize("norm", [1, 2])
def test_first_index_tie_and_first_gradients(device: str, dimension: int, norm: int) -> None:
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    x = torch.zeros((1, 1, dimension), device=device, requires_grad=True)
    y = torch.zeros((1, 2, dimension), device=device)
    y[0, 0, 0] = 1
    y[0, 1, 0] = -1
    y.requires_grad_()
    loss, normals = chamfer_distance(
        x, y, norm=norm, single_directional=True,
        point_reduction="sum", batch_reduction="sum",
    )
    assert normals is None
    loss.backward()
    assert loss.item() == 1
    expected_x = torch.full((dimension,), -1.0 if norm == 1 else 0.0)
    if norm == 2:
        expected_x[0] = -2.0
    torch.testing.assert_close(x.grad[0, 0].cpu(), expected_x, rtol=0, atol=0)
    torch.testing.assert_close(y.grad[0, 0].cpu(), -expected_x, rtol=0, atol=0)
    torch.testing.assert_close(y.grad[0, 1].cpu(), torch.zeros(dimension), rtol=0, atol=0)


@pytest.mark.parametrize("dimension", [1, 2, 4, 64, 128])
@pytest.mark.parametrize("norm", [1, 2])
def test_mps_writes_padded_rows_and_retains_first_overflowed_reference(
    dimension: int, norm: int,
) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    x = torch.zeros((1, 2, dimension), device="mps")
    y = torch.zeros((1, 2, dimension), device="mps")
    y[0, :, 0] = torch.tensor([1e30, -1e30], device="mps")
    length_x = torch.tensor([1], dtype=torch.long, device="mps")
    length_y = torch.tensor([2], dtype=torch.long, device="mps")
    distances, indices = _nearest_mps(x, y, length_x, length_y, norm=norm)
    torch.mps.synchronize()
    assert indices.cpu().tolist() == [[0, -1]]
    assert distances[0, 1].item() == 0.0
    if norm == 1:
        assert distances[0, 0].item() == torch.tensor(1e30, dtype=torch.float32).item()
    else:
        assert torch.isinf(distances[0, 0]).item()


@pytest.mark.parametrize("dimension", [1, 2, 4])
def test_high_dimensional_normal_loss_remains_explicitly_unsupported(dimension: int) -> None:
    x = torch.zeros((1, 2, dimension))
    with pytest.raises(NotImplementedError, match="normal loss currently requires D == 3"):
        chamfer_distance(x, x, x_normals=x, y_normals=x)


def test_empty_coordinate_dimension_is_rejected() -> None:
    x = torch.empty((1, 2, 0))
    with pytest.raises(ValueError, match="D >= 1"):
        chamfer_distance(x, x)

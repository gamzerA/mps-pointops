"""MPS fused CSR pooling parity with the original index_add_ aggregation."""

from __future__ import annotations

import pytest
import torch

from mps_pointops.voxel import voxel_downsample, voxelize


def _case(name: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    generator = torch.Generator().manual_seed(701)
    if name == "dense":
        pos = torch.randint(0, 4, (2048, 3), generator=generator).float() / 8 + 0.125
        features = torch.randint(-32, 33, (2048, 7), generator=generator).float() / 16
        batch = torch.zeros(2048, dtype=torch.int64)
        size = 1.0
    elif name == "sparse":
        index = torch.randperm(257, generator=generator)
        pos = torch.stack((index.float() * 2 + 0.25, -index.float() * 2 + 0.25), 1)
        features = torch.randint(-32, 33, (257, 9), generator=generator).float() / 16
        batch = torch.randint(0, 5, (257,), generator=generator)
        size = 1.0
    elif name == "cancellation":
        # These large opposing terms and small residues are all exactly
        # representable, so differing addition orders have a stable target.
        pos = torch.tensor([[-1024.0], [1024.0], [0.25]]).repeat(341, 1)
        features = torch.tensor([
            [4096.0, -2048.0, 1024.0],
            [-4096.0, 2048.0, -1024.0],
            [0.25, 0.5, -0.125],
        ]).repeat(341, 1)
        batch = torch.zeros(len(pos), dtype=torch.int64)
        size = 4096.0
    elif name == "mixed_noncontiguous":
        pos = torch.randint(-8, 9, (1001, 3), generator=generator).float() + 0.25
        # Preserve a genuine noncontiguous (N, C) input on device.
        features = torch.randint(-32, 33, (17, 1001), generator=generator).float().T / 16
        batch = torch.randint(0, 4, (1001,), generator=generator)
        size = 4.0
    else:
        raise AssertionError(name)
    return pos, features, batch, size


@pytest.mark.mps
@pytest.mark.parametrize("name", ["dense", "sparse", "cancellation", "mixed_noncontiguous"])
@pytest.mark.parametrize("reduce", ["mean", "sum"])
def test_fused_matches_index_add_public_api(name: str, reduce: str) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")

    pos_cpu, features_cpu, batch_cpu, size = _case(name)
    pos = pos_cpu.to("mps").requires_grad_()
    features = features_cpu.to("mps").requires_grad_()
    batch = batch_cpu.to("mps")
    start = -2048.0 if name == "cancellation" else None
    result = voxel_downsample(pos, size, batch, features,
                              start=start, feature_reduce=reduce,
                              pool_backend="fused_csr")
    baseline = voxel_downsample(pos, size, batch, features,
                                start=start, feature_reduce=reduce)
    cpu_maps = voxelize(pos_cpu, size, batch_cpu, start=start)
    for field in ("voxel_coords", "batch", "inverse", "counts", "point_order", "ptr"):
        torch.testing.assert_close(getattr(result.voxels, field).cpu(),
                                   getattr(cpu_maps, field), rtol=0, atol=0)
        torch.testing.assert_close(getattr(result.voxels, field),
                                   getattr(baseline.voxels, field), rtol=0, atol=0)

    rows = result.pos.shape[0]
    assert result.features is not None
    assert baseline.features is not None
    torch.testing.assert_close(result.pos, baseline.pos, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(result.features, baseline.features, rtol=1e-4, atol=1e-5)

    pos_weight = torch.linspace(-1, 1, rows * pos.shape[1], device="mps").reshape_as(result.pos)
    feature_weight = torch.linspace(-2, 2, rows * features.shape[1], device="mps").reshape_as(result.features)
    fused_loss = (result.pos * pos_weight).sum() + (result.features * feature_weight).sum()
    reference_loss = (baseline.pos * pos_weight).sum() + (baseline.features * feature_weight).sum()
    actual_grad = torch.autograd.grad(fused_loss, (pos, features))
    expected_grad = torch.autograd.grad(reference_loss, (pos, features))
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.mps
def test_fused_without_features_matches_index_add_and_gather_gradient() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    pos_cpu, _, batch_cpu, size = _case("mixed_noncontiguous")
    pos = pos_cpu.to("mps").requires_grad_()
    result = voxel_downsample(pos, size, batch_cpu.to("mps"), pool_backend="fused_csr")
    baseline = voxel_downsample(pos, size, batch_cpu.to("mps"))
    assert result.features is None
    assert baseline.features is None
    torch.testing.assert_close(result.pos, baseline.pos, rtol=1e-4, atol=1e-5)
    weights = torch.arange(result.pos.numel(), dtype=torch.float32, device="mps").reshape_as(result.pos)
    (actual_grad,) = torch.autograd.grad((result.pos * weights).sum(), (pos,))
    (expected_grad,) = torch.autograd.grad((baseline.pos * weights).sum(), (pos,))
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-5)


def test_fused_backend_requires_mps_and_known_name() -> None:
    pos = torch.tensor([[0.25]])
    with pytest.raises(ValueError, match="requires MPS"):
        voxel_downsample(pos, 1.0, pool_backend="fused_csr")
    with pytest.raises(ValueError, match="pool_backend"):
        voxel_downsample(pos, 1.0, pool_backend="unknown")


def _severe_cancellation(amplitude: float) -> tuple[torch.Tensor, torch.Tensor]:
    features_cpu = torch.tensor([[amplitude], [-amplitude], [0.1]]).repeat(341, 1)
    pos = torch.full((len(features_cpu), 1), 0.25, device="mps", requires_grad=True)
    features = features_cpu.to("mps").requires_grad_()
    return pos, features


@pytest.mark.mps
@pytest.mark.parametrize("amplitude", [1e4, 1e6])
def test_fused_severe_cancellation_matches_float64_oracle(amplitude: float) -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    pos, features = _severe_cancellation(amplitude)
    fused = voxel_downsample(pos, 1.0, features=features, feature_reduce="sum",
                             pool_backend="fused_csr")
    baseline = voxel_downsample(pos, 1.0, features=features, feature_reduce="sum")
    oracle = features.detach().cpu().double().sum(dim=0)
    assert fused.features is not None and baseline.features is not None
    torch.testing.assert_close(fused.features.cpu().double().flatten(), oracle,
                               rtol=1e-4, atol=1e-5)
    for field in ("voxel_coords", "batch", "inverse", "counts", "point_order", "ptr"):
        torch.testing.assert_close(getattr(fused.voxels, field),
                                   getattr(baseline.voxels, field), rtol=0, atol=0)
    fused_grad = torch.autograd.grad(fused.features.sum(), features)[0]
    baseline_grad = torch.autograd.grad(baseline.features.sum(), features)[0]
    torch.testing.assert_close(fused_grad, baseline_grad, rtol=0, atol=0)


@pytest.mark.mps
@pytest.mark.xfail(reason="MPS index_add_ atomic order differs under severe cancellation")
def test_severe_cancellation_exceeds_index_add_float_tolerance() -> None:
    if not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    pos, features = _severe_cancellation(1e4)
    fused = voxel_downsample(pos, 1.0, features=features, feature_reduce="sum",
                             pool_backend="fused_csr")
    baseline = voxel_downsample(pos, 1.0, features=features, feature_reduce="sum")
    assert fused.features is not None and baseline.features is not None
    torch.testing.assert_close(fused.features, baseline.features, rtol=1e-4, atol=1e-5)

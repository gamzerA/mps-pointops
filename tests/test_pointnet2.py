"""PointNet++ feature propagation contracts and native MPS differential tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mps_pointops import reference, three_interpolate, three_nn


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
def test_three_nn_distances_order_ties_and_no_coordinate_grad(device: str) -> None:
    unknown = torch.tensor([[[0., 0., 0.], [2., 0., 0.]]], device=device, requires_grad=True)
    known = torch.tensor(
        [[[1., 0., 0.], [-1., 0., 0.], [0., 2., 0.], [1., 0., 0.]]],
        device=device, requires_grad=True,
    )
    distances, indices = three_nn(unknown, known)
    assert distances.shape == indices.shape == (1, 2, 3)
    assert distances.dtype == torch.float32 and indices.dtype == torch.int32
    assert not distances.requires_grad and not indices.requires_grad
    np.testing.assert_array_equal(indices.cpu().numpy(), [[[0, 1, 3], [0, 3, 2]]])
    torch.testing.assert_close(
        distances.cpu(), torch.tensor([[[1., 1., 1.], [1., 1., 2.8284271247461903]]]),
        rtol=1e-6, atol=1e-6,
    )


@pytest.mark.parametrize("device", DEVICES)
def test_three_nn_matches_independent_numpy_oracle_and_noncontiguous(device: str) -> None:
    rng = np.random.default_rng(59)
    raw_unknown = torch.from_numpy(rng.normal(size=(2, 37, 6)).astype("float32"))
    raw_known = torch.from_numpy(rng.normal(size=(2, 113, 6)).astype("float32"))
    unknown = raw_unknown[..., ::2]
    known = raw_known[..., ::2]
    assert not unknown.is_contiguous() and not known.is_contiguous()
    diffs = unknown.numpy()[:, :, None] - known.numpy()[:, None]
    d2 = diffs[..., 0] * diffs[..., 0]
    d2 = d2 + diffs[..., 1] * diffs[..., 1]
    d2 = d2 + diffs[..., 2] * diffs[..., 2]
    expected_indices = np.argsort(d2, axis=-1, kind="stable")[..., :3]
    expected_distances = np.sqrt(np.take_along_axis(d2, expected_indices, axis=-1))
    distances, indices = three_nn(unknown.to(device), known.to(device))
    np.testing.assert_array_equal(indices.cpu().numpy(), expected_indices)
    np.testing.assert_allclose(distances.cpu().numpy(), expected_distances, rtol=2e-6, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(0, 7, 0), (2, 0, 0), (2, 0, 5)])
def test_three_nn_empty_queries(device: str, shape: tuple[int, int, int]) -> None:
    batch, queries, points = shape
    distances, indices = three_nn(
        torch.empty((batch, queries, 3), device=device),
        torch.empty((batch, points, 3), device=device),
    )
    assert distances.shape == indices.shape == (batch, queries, 3)


@pytest.mark.parametrize("device", DEVICES)
def test_three_interpolate_forward_backward_repeated_indices(device: str) -> None:
    features = torch.tensor(
        [[[2., 3., 5., 7.], [11., 13., 17., 19.]]],
        device=device, requires_grad=True,
    )
    indices = torch.tensor([[[0, 1, 1], [3, 1, 0]]], dtype=torch.int32, device=device)
    weights = torch.tensor(
        [[[0.5, 0.25, 0.25], [1., 0.5, -0.5]]],
        device=device, requires_grad=True,
    )
    output = three_interpolate(features, indices, weights)
    assert output.shape == (1, 2, 2)
    expected = torch.tensor([[[2.5, 7.5], [12., 20.]]])
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    output.sum().backward()
    expected_grad = torch.tensor([[[0., 1., 0., 1.], [0., 1., 0., 1.]]])
    torch.testing.assert_close(features.grad.cpu(), expected_grad, rtol=0, atol=0)
    assert weights.grad is None


@pytest.mark.parametrize("device", DEVICES)
def test_three_interpolate_matches_reference_on_noncontiguous_inputs(device: str) -> None:
    rng = np.random.default_rng(77)
    features = torch.from_numpy(rng.normal(size=(2, 5, 23)).astype("float32"))[:, :, ::2]
    indices = torch.from_numpy(rng.integers(0, features.shape[-1], size=(2, 7, 3), dtype="int64"))
    weights = torch.from_numpy(rng.normal(size=(2, 7, 3)).astype("float32"))
    want = reference.three_interpolate(features, indices, weights)
    out = three_interpolate(features.to(device), indices.to(device), weights.to(device))
    torch.testing.assert_close(out.cpu(), want, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_three_interpolate_backward_random_oracle(device: str) -> None:
    generator = torch.Generator().manual_seed(109)
    features = torch.randn(2, 4, 17, generator=generator, device="cpu")
    indices = torch.randint(17, (2, 11, 3), generator=generator, dtype=torch.int32)
    weights = torch.randn(2, 11, 3, generator=generator)
    upstream = torch.randn(2, 4, 11, generator=generator)
    oracle_features = features.clone().requires_grad_()
    reference.three_interpolate(oracle_features, indices, weights).backward(upstream)
    actual_features = features.to(device).requires_grad_()
    three_interpolate(actual_features, indices.to(device), weights.to(device)).backward(upstream.to(device))
    torch.testing.assert_close(actual_features.grad.cpu(), oracle_features.grad, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_three_interpolate_empty_output_and_bad_indices(device: str) -> None:
    features = torch.randn(1, 2, 4, device=device, requires_grad=True)
    empty = three_interpolate(
        features, torch.empty((1, 0, 3), dtype=torch.int32, device=device),
        torch.empty((1, 0, 3), device=device),
    )
    assert empty.shape == (1, 2, 0)
    empty.sum().backward()
    assert features.grad is not None and torch.count_nonzero(features.grad).item() == 0
    with pytest.raises(IndexError):
        three_interpolate(
            features, torch.tensor([[[0, -1, 1]]], dtype=torch.int32, device=device),
            torch.ones((1, 1, 3), device=device),
        )
    with pytest.raises(IndexError):
        three_interpolate(
            features, torch.tensor([[[0, 4, 1]]], dtype=torch.int64, device=device),
            torch.ones((1, 1, 3), device=device),
        )


def test_invalid_shapes_and_dtypes() -> None:
    points = torch.zeros((1, 2, 3))
    with pytest.raises(ValueError, match="at least 3"):
        three_nn(points, points)
    with pytest.raises(TypeError, match="float32"):
        three_nn(points.half(), torch.zeros((1, 3, 3)).half())
    with pytest.raises(ValueError, match="batch"):
        three_nn(points, torch.zeros((2, 3, 3)))
    with pytest.raises(TypeError, match="indices"):
        three_interpolate(torch.zeros((1, 2, 3)), torch.zeros((1, 2, 3)), torch.ones((1, 2, 3)))
    with pytest.raises(ValueError, match="source point"):
        three_interpolate(
            torch.zeros((1, 2, 0)), torch.zeros((1, 2, 3), dtype=torch.int32),
            torch.ones((1, 2, 3)),
        )

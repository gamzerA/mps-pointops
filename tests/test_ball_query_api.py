"""Integration checks for the public Ball Query API and compatibility shim."""

from __future__ import annotations

import pytest
import torch

from mps_pointops import ball_query, compat, reference


def test_cpu_fallback_matches_reference() -> None:
    queries = torch.zeros((1, 1, 3))
    points = torch.tensor([[[0.25, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    got = ball_query(queries, points, 1.0, 3)
    want = reference.ball_query(queries, points, 1.0, 3)
    for actual, expected in zip(got, want):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_public_metal_and_pointnet_padding() -> None:
    queries = torch.zeros((1, 2, 3), device="mps")
    points = torch.tensor([[[0.25, 0.0, 0.0], [2.0, 0.0, 0.0]]], device="mps")
    distances, indices = ball_query(queries, points, 1.0, 3)
    assert indices.cpu().tolist() == [[[0, -1, -1], [0, -1, -1]]]
    torch.testing.assert_close(
        distances.cpu(), torch.tensor([[[0.0625, 0.0, 0.0]] * 2]), rtol=0, atol=0
    )
    padded = compat.ball_query(1.0, 3, points, queries)
    assert padded.dtype == torch.int32
    assert padded.cpu().tolist() == [[[0, 0, 0], [0, 0, 0]]]


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("radius", [-0.1, float("inf"), float("nan"), 1e-40, 1e39, True])
def test_public_api_rejects_bad_radius_on_every_device(device: str, radius: float) -> None:
    xyz = torch.rand(1, 20, 3, device=device)
    with pytest.raises(ValueError):
        ball_query(xyz[:, :4], xyz, radius, 4)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("K", [-1, 1.5, True])
def test_public_api_rejects_bad_k_on_every_device(device: str, K: int) -> None:
    xyz = torch.rand(1, 20, 3, device=device)
    with pytest.raises(ValueError):
        ball_query(xyz[:, :4], xyz, 0.5, K)


@pytest.mark.parametrize("device", DEVICES)
def test_public_api_zero_radius_has_no_neighbors(device: str) -> None:
    xyz = torch.rand(1, 20, 3, device=device)
    dist2, idx = ball_query(xyz[:, :4], xyz, 0.0, 3)
    assert (idx == -1).all() and (dist2 == 0).all()

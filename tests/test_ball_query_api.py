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

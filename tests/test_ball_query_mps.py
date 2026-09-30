# SPDX-License-Identifier: MIT
"""Independent contract tests for the Metal radius search."""

from __future__ import annotations

import math
import struct

import pytest
import torch

from mps_pointops._ball_query_mps import ball_query


def reference(
    queries: torch.Tensor,
    points: torch.Tensor,
    query_lengths: list[int],
    point_lengths: list[int],
    radius: float,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """CPU oracle for cases away from a radius boundary.

    Python arithmetic here uses wider intermediates than the Metal float32
    shader, so this is not a bitwise oracle for near-boundary decisions.
    """
    batch, query_count, _ = queries.shape
    indices = torch.full((batch, query_count, k), -1, dtype=torch.int64)
    distances = torch.zeros((batch, query_count, k), dtype=torch.float32)
    if k == 0:
        return distances, indices
    for b in range(batch):
        for i in range(query_lengths[b]):
            query = queries[b, i].tolist()
            if not all(math.isfinite(v) for v in query):
                continue
            found = 0
            for j in range(point_lengths[b]):
                point = points[b, j].tolist()
                if not all(math.isfinite(v) for v in point):
                    continue
                dist = sum((float(a) - float(c)) ** 2 for a, c in zip(query, point))
                if dist < radius * radius:
                    indices[b, i, found] = j
                    distances[b, i, found] = dist
                    found += 1
                    if found == k:
                        break
    return distances, indices


requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="PyTorch MPS device is unavailable"
)


@requires_mps
@pytest.mark.mps
@pytest.mark.parametrize("dtype,atol", [(torch.float32, 1e-5), (torch.float16, 3e-3)])
def test_matches_cpu_oracle(dtype: torch.dtype, atol: float) -> None:
    torch.manual_seed(9)
    queries_cpu = torch.randn(2, 7, 3, dtype=dtype)
    points_cpu = torch.randn(2, 19, 3, dtype=dtype)
    query_lengths = [7, 4]
    point_lengths = [19, 11]
    expected_d, expected_i = reference(
        queries_cpu, points_cpu, query_lengths, point_lengths, 2.0, 5
    )
    result = ball_query(
        queries_cpu.to("mps"),
        points_cpu.to("mps"),
        radius=2.0,
        k=5,
        query_lengths=torch.tensor(query_lengths, device="mps"),
        point_lengths=torch.tensor(point_lengths, device="mps"),
    )
    torch.testing.assert_close(result.indices.cpu(), expected_i, rtol=0, atol=0)
    torch.testing.assert_close(result.distances.cpu(), expected_d, rtol=0, atol=atol)


@requires_mps
@pytest.mark.mps
def test_first_k_not_nearest_and_strict_boundary() -> None:
    queries = torch.zeros((1, 1, 3), device="mps")
    points = torch.tensor(
        [[[0.8, 0, 0], [0.1, 0, 0], [1.0, 0, 0], [0.2, 0, 0]]],
        device="mps",
    )
    result = ball_query(queries, points, radius=1.0, k=2)
    assert result.indices.cpu().tolist() == [[[0, 1]]]
    torch.testing.assert_close(
        result.distances.cpu(), torch.tensor([[[0.64, 0.01]]]), atol=1e-6, rtol=0
    )


@requires_mps
@pytest.mark.mps
def test_float32_neighbors_of_an_exact_boundary() -> None:
    # At radius 1, the threshold and its two adjacent float32 coordinates are
    # representable. This checks a stable strict-boundary case without assuming
    # that normalized and unnormalized float32 expressions agree near every r.
    one = torch.tensor(1.0, dtype=torch.float32)
    just_inside = torch.nextafter(one, torch.tensor(0.0)).item()
    just_outside = torch.nextafter(one, torch.tensor(float("inf"))).item()
    queries = torch.zeros((1, 1, 3), device="mps")
    points = torch.tensor(
        [[[just_inside, 0.0, 0.0], [1.0, 0.0, 0.0], [just_outside, 0.0, 0.0]]],
        device="mps",
    )
    result = ball_query(queries, points, radius=1.0, k=3)
    assert result.indices.cpu().tolist() == [[[0, -1, -1]]]


@requires_mps
@pytest.mark.mps
def test_radius_square_uses_float32_radius_before_multiplication() -> None:
    # For radius=0.1, fl32(fl32(r)*fl32(r)) is 0x3c23d70b, whereas
    # fl32(real(0.1)*real(0.1)) is 0x3c23d70a. The chosen point has
    # squared distance equal to the lower value, so only the former includes it.
    queries = torch.zeros((1, 1, 3), device="mps")
    points = torch.tensor(
        [[[0.09999999403953552, 3.112176273134537e-05, 0.0]]], device="mps"
    )
    result = ball_query(queries, points, radius=0.1, k=1)
    assert result.indices.cpu().tolist() == [[[0]]]
    actual_bits = struct.unpack("<I", struct.pack("<f", result.distances.item()))[0]
    assert actual_bits == 0x3C23D70A


@requires_mps
@pytest.mark.mps
def test_small_normal_radius_component_order_does_not_change_membership() -> None:
    # r^2 is normal, but (r/4)^2 is subnormal and may flush if formed as a
    # separate product. The real squared norm is (0.25^2 + 0.98^2)*r^2 > r^2.
    # All three component positions must therefore be excluded.
    radius = 2.0**-62
    small, large = 0.25 * radius, 0.98 * radius
    queries = torch.zeros((1, 1, 3), device="mps")
    points = torch.tensor(
        [[
            [small, large, 0.0],
            [0.0, small, large],
            [large, 0.0, small],
            [small, 0.8 * radius, 0.0],
            [0.0, small, 0.8 * radius],
            [0.8 * radius, 0.0, small],
        ]],
        device="mps",
    )
    result = ball_query(queries, points, radius=radius, k=6)
    assert result.indices.cpu().tolist() == [[[3, 4, 5, -1, -1, -1]]]
    for slot, point_index in enumerate((3, 4, 5)):
        expected_f64 = sum(float(v) ** 2 for v in points[0, point_index].cpu().tolist())
        assert math.isclose(result.distances[0, 0, slot].item(), expected_f64, rel_tol=1e-6)


@requires_mps
@pytest.mark.mps
@pytest.mark.parametrize("radius", [0.1, 0.3, 1.0, 3.0])
def test_membership_outside_four_float32_ulp_boundary_band(radius: float) -> None:
    # The oracle evaluates distances in float64 *after* all coordinates have
    # been rounded to float32. It compares them with PyTorch3D's float32 r^2.
    # Within four float32 ulps of that threshold, FMA and division can change
    # membership, so this test intentionally accepts either result there.
    radius_f32 = struct.unpack("<f", struct.pack("<f", radius))[0]
    radius_sq_f32 = struct.unpack("<f", struct.pack("<f", radius_f32 * radius_f32))[0]
    radius_sq_bits = struct.unpack("<I", struct.pack("<f", radius_sq_f32))[0]
    next_sq = struct.unpack("<f", struct.pack("<I", radius_sq_bits + 1))[0]
    band = 4.0 * (next_sq - radius_sq_f32)

    radius_tensor = torch.tensor(radius_f32, dtype=torch.float32)
    lower = radius_tensor
    upper = radius_tensor
    vectors = [[0.0, 0.0, 0.0], [0.5 * radius_f32, 0.0, 0.0]]
    for _ in range(12):
        lower = torch.nextafter(lower, torch.tensor(0.0))
        upper = torch.nextafter(upper, torch.tensor(float("inf")))
        vectors.extend(([lower.item(), 0.0, 0.0], [upper.item(), 0.0, 0.0]))
    vectors.extend(
        ([0.6 * radius_f32, 0.8 * radius_f32, 0.0], [1.5 * radius_f32, 0.0, 0.0])
    )
    points_cpu = torch.tensor([vectors], dtype=torch.float32)
    queries = torch.zeros((1, 1, 3), dtype=torch.float32, device="mps")
    result = ball_query(queries, points_cpu.to("mps"), radius=radius, k=len(vectors))
    actual = [j for j in result.indices.cpu().flatten().tolist() if j >= 0]
    assert actual == sorted(actual) and len(actual) == len(set(actual))

    ambiguous = clear_inside = clear_outside = 0
    actual_set = set(actual)
    for j, xyz in enumerate(points_cpu[0].tolist()):
        squared_distance_f64 = sum(float(component) ** 2 for component in xyz)
        if abs(squared_distance_f64 - radius_sq_f32) <= band:
            ambiguous += 1
            continue
        expected = squared_distance_f64 < radius_sq_f32
        assert (j in actual_set) == expected, (radius, j, squared_distance_f64, band)
        clear_inside += int(expected)
        clear_outside += int(not expected)
    assert ambiguous > 0 and clear_inside > 0 and clear_outside > 0


@requires_mps
@pytest.mark.mps
@pytest.mark.parametrize("radius", [2.0**-62, 1e-30, 1e20])
def test_normalized_membership_outside_dimensionless_boundary_band(radius: float) -> None:
    # The two tiny radii take the r < 2^-50 path; 1e20 takes the r^2 = Inf
    # path. Compare with binary64 arithmetic applied to the *float32 inputs*.
    # Four float32 ulps at one cover boundary rounding in division and FMA;
    # decisions outside that band must agree exactly with the binary64 oracle.
    radius_f32 = struct.unpack("<f", struct.pack("<f", radius))[0]
    if radius == 1e20:
        assert float(radius_f32) ** 2 > torch.finfo(torch.float32).max
    else:
        assert radius_f32 < 2.0**-50
    dimensionless_band = 4.0 * 2.0**-23
    radius_tensor = torch.tensor(radius_f32, dtype=torch.float32)
    just_below = torch.nextafter(radius_tensor, torch.tensor(0.0)).item()
    just_above = torch.nextafter(radius_tensor, torch.tensor(float("inf"))).item()
    vectors = [
        [0.0, 0.0, 0.0],
        [0.5 * radius_f32, 0.0, 0.0],
        [radius_f32, 0.0, 0.0],
        [just_below, 0.0, 0.0],
        [just_above, 0.0, 0.0],
        [1.5 * radius_f32, 0.0, 0.0],
        [0.6 * radius_f32, 0.8 * radius_f32, 0.0],
        [0.25 * radius_f32, 0.98 * radius_f32, 0.0],
        [0.25 * radius_f32, 0.8 * radius_f32, 0.0],
    ]
    points_cpu = torch.tensor([vectors], dtype=torch.float32)
    queries = torch.zeros((1, 1, 3), dtype=torch.float32, device="mps")
    result = ball_query(queries, points_cpu.to("mps"), radius=radius, k=len(vectors))
    actual = [j for j in result.indices.cpu().flatten().tolist() if j >= 0]
    assert actual == sorted(actual) and len(actual) == len(set(actual))

    actual_set = set(actual)
    counts = {"inside": 0, "outside": 0, "ambiguous": 0}
    for j, xyz in enumerate(points_cpu[0].tolist()):
        squared_distance_f64 = sum(float(component) ** 2 for component in xyz)
        ratio_f64 = squared_distance_f64 / (float(radius_f32) ** 2)
        if abs(ratio_f64 - 1.0) <= dimensionless_band:
            counts["ambiguous"] += 1
            continue
        if ratio_f64 < 1.0:
            counts["inside"] += 1
            assert j in actual_set, (radius, j, ratio_f64)
        else:
            counts["outside"] += 1
            assert j not in actual_set, (radius, j, ratio_f64)
    assert all(counts.values()), (radius, counts)


@requires_mps
@pytest.mark.mps
def test_extreme_finite_radii() -> None:
    queries = torch.zeros((1, 1, 3), device="mps")
    tiny_points = torch.tensor([[[0.0, 0.0, 0.0], [1e-20, 0.0, 0.0]]], device="mps")
    tiny = ball_query(queries, tiny_points, radius=1e-30, k=2)
    assert tiny.indices.cpu().tolist() == [[[0, -1]]]
    assert tiny.distances.cpu().tolist() == [[[0.0, 0.0]]]
    zero = ball_query(queries, tiny_points, radius=0.0, k=1)
    assert zero.indices.cpu().tolist() == [[[-1]]]

    # The documented lower bound is checked after float32 rounding.
    minimum = 2.0**-112
    identical = ball_query(queries, tiny_points[:, :1], radius=minimum, k=1)
    assert identical.indices.cpu().tolist() == [[[0]]]
    just_below = torch.nextafter(
        torch.tensor(minimum, dtype=torch.float32), torch.tensor(0.0)
    ).item()
    with pytest.raises(ValueError, match="2\\*\\*-112"):
        ball_query(queries, tiny_points[:, :1], radius=just_below, k=1)

    # The radius is representable in float32, but its square is not.
    large_points = torch.tensor(
        [[[1e19, 0.0, 0.0], [5e19, 0.0, 0.0], [2e20, 0.0, 0.0]]], device="mps"
    )
    large = ball_query(queries, large_points, radius=1e20, k=3)
    # The second hit has a squared distance beyond float32, but is inside the radius.
    assert large.indices.cpu().tolist() == [[[0, 1, -1]]]
    assert torch.isfinite(large.distances[0, 0, 0]).item()


@requires_mps
@pytest.mark.mps
def test_normalized_branch_uses_stable_cases_away_from_boundary() -> None:
    # The tiny radius forces normalized comparison. Adjacent float32 values
    # around this boundary are deliberately not compared to a real-arithmetic
    # oracle: division and explicit FMA operations can round differently.
    radius = torch.tensor(1e-30, dtype=torch.float32).item()
    queries = torch.zeros((1, 1, 3), device="mps")
    points = torch.tensor(
        [[[0.5 * radius, 0.0, 0.0], [2.0 * radius, 0.0, 0.0]]], device="mps"
    )
    first = ball_query(queries, points, radius=radius, k=2)
    second = ball_query(queries, points, radius=radius, k=2)
    assert first.indices.cpu().tolist() == [[[0, -1]]]
    torch.testing.assert_close(first.indices, second.indices, rtol=0, atol=0)


@requires_mps
@pytest.mark.mps
def test_empty_missing_nonfinite_and_noncontiguous() -> None:
    base = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], device="mps")
    points = torch.tensor([[[float("nan"), 0, 0], [0.1, 0, 0]]], device="mps")
    result = ball_query(base, points, radius=0.5, k=3)
    assert result.indices.cpu().tolist() == [[[1, -1, -1], [-1, -1, -1]]]
    assert result.distances.cpu()[0, 1].tolist() == [0.0, 0.0, 0.0]

    no_points = ball_query(base, points[:, :0], radius=1.0, k=2)
    assert no_points.indices.cpu().tolist() == [[[-1, -1], [-1, -1]]]
    assert no_points.distances.cpu().sum().item() == 0.0

    no_neighbors = ball_query(base, points, radius=1.0, k=0)
    assert no_neighbors.indices.shape == (1, 2, 0)
    empty_queries = ball_query(base[:, :0], points, radius=1.0, k=2)
    assert empty_queries.indices.shape == (1, 0, 2)

    # A strided view is copied on the GPU without changing its logical values.
    expanded = torch.stack([base, base + 10], dim=2)
    strided = expanded[:, :, 0, :]
    assert not strided.is_contiguous()
    torch.testing.assert_close(
        ball_query(strided, points, radius=0.5, k=2).indices,
        ball_query(base, points, radius=0.5, k=2).indices,
        rtol=0,
        atol=0,
    )


@requires_mps
@pytest.mark.mps
def test_nan_and_infinite_coordinates_never_match_or_receive_gradients() -> None:
    queries = torch.tensor(
        [[[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0],
          [float("inf"), 0.0, 0.0], [-float("inf"), 0.0, 0.0]]],
        device="mps",
        requires_grad=True,
    )
    points = torch.tensor(
        [[[float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0],
          [-float("inf"), 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]],
        device="mps",
        requires_grad=True,
    )
    result = ball_query(queries, points, radius=1.0, k=3)
    assert result.indices.cpu().tolist() == [
        [[3, 4, -1], [-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]
    ]
    torch.testing.assert_close(
        result.distances.cpu()[0, 0], torch.tensor([0.0625, 0.25, 0.0])
    )
    assert torch.all(result.distances[0, 1:] == 0).item()

    result.distances.sum().backward()
    torch.testing.assert_close(queries.grad.cpu()[0, 0], torch.tensor([-1.5, 0.0, 0.0]))
    assert torch.all(queries.grad[0, 1:] == 0).item()
    assert torch.all(points.grad[0, :3] == 0).item()
    torch.testing.assert_close(
        points.grad.cpu()[0, 3:], torch.tensor([[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    )


@requires_mps
@pytest.mark.mps
def test_distance_gradients_and_padding() -> None:
    q = torch.tensor([[[0.0, 0.0, 0.0]]], device="mps", requires_grad=True)
    p = torch.tensor(
        [[[0.2, 0.0, 0.0], [0.5, 0.0, 0.0], [4.0, 0.0, 0.0]]],
        device="mps",
        requires_grad=True,
    )
    result = ball_query(q, p, radius=1.0, k=3)
    assert result.indices.cpu().tolist() == [[[0, 1, -1]]]
    result.distances.backward(torch.tensor([[[2.0, 3.0, 99.0]]], device="mps"))
    torch.testing.assert_close(q.grad.cpu(), torch.tensor([[[-3.8, 0.0, 0.0]]]))
    torch.testing.assert_close(
        p.grad.cpu(), torch.tensor([[[0.8, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    )


@requires_mps
@pytest.mark.mps
def test_second_derivatives_with_fixed_selected_index() -> None:
    # This differentiates the chosen pair's squared distance. It does not
    # differentiate the discrete neighbor selection at a radius boundary.
    q = torch.zeros((1, 1, 3), device="mps", requires_grad=True)
    p = torch.tensor([[[0.25, 0.0, 0.0]]], device="mps", requires_grad=True)
    loss = ball_query(q, p, radius=1.0, k=1).distances.sum()
    grad_q, grad_p = torch.autograd.grad(loss, (q, p), create_graph=True)
    hessian_qq = torch.autograd.grad(grad_q.sum(), q, retain_graph=True)[0]
    hessian_qp = torch.autograd.grad(grad_q.sum(), p, retain_graph=True)[0]
    hessian_pp = torch.autograd.grad(grad_p.sum(), p)[0]
    torch.testing.assert_close(hessian_qq, torch.full_like(q, 2.0))
    torch.testing.assert_close(hessian_qp, torch.full_like(p, -2.0))
    torch.testing.assert_close(hessian_pp, torch.full_like(p, 2.0))


@requires_mps
@pytest.mark.mps
def test_invalid_slots_do_not_poison_gradients() -> None:
    q = torch.tensor([[[0.0, 0.0, 0.0]]], device="mps", requires_grad=True)
    p = torch.tensor(
        [[[float("nan"), 0.0, 0.0], [0.25, 0.0, 0.0]]],
        device="mps",
        requires_grad=True,
    )
    result = ball_query(q, p, radius=1.0, k=3)
    assert result.indices.cpu().tolist() == [[[1, -1, -1]]]
    result.distances.backward(torch.tensor([[[1.0, 99.0, float("nan")]]], device="mps"))
    torch.testing.assert_close(q.grad.cpu(), torch.tensor([[[-0.5, 0.0, 0.0]]]))
    torch.testing.assert_close(p.grad.cpu()[0, 1], torch.tensor([0.5, 0.0, 0.0]))
    assert torch.all(p.grad[0, 0] == 0).item()


@requires_mps
@pytest.mark.mps
def test_repeated_point_accumulates_gradients() -> None:
    queries = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], device="mps", requires_grad=True
    )
    points = torch.tensor([[[0.25, 0.0, 0.0]]], device="mps", requires_grad=True)
    result = ball_query(queries, points, radius=2.0, k=1)
    assert result.indices.cpu().tolist() == [[[0], [0]]]
    result.distances.backward(torch.tensor([[[2.0], [3.0]]], device="mps"))
    torch.testing.assert_close(
        queries.grad.cpu(), torch.tensor([[[-1.0, 0.0, 0.0], [4.5, 0.0, 0.0]]])
    )
    torch.testing.assert_close(points.grad.cpu(), torch.tensor([[[-3.5, 0.0, 0.0]]]))


@requires_mps
@pytest.mark.mps
def test_float16_distance_backward() -> None:
    queries = torch.tensor([[[0.0, 0.0, 0.0]]], device="mps", dtype=torch.float16, requires_grad=True)
    points = torch.tensor(
        [[[0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]],
        device="mps",
        dtype=torch.float16,
        requires_grad=True,
    )
    result = ball_query(queries, points, radius=1.0, k=2)
    assert result.distances.dtype == torch.float32
    assert result.indices.cpu().tolist() == [[[0, 1]]]
    result.distances.backward(torch.tensor([[[1.0, 2.0]]], device="mps"))
    torch.testing.assert_close(queries.grad.cpu(), torch.tensor([[[-2.5, 0.0, 0.0]]], dtype=torch.float16))
    torch.testing.assert_close(
        points.grad.cpu(),
        torch.tensor([[[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float16),
    )


@requires_mps
@pytest.mark.mps
def test_bad_lengths_raise() -> None:
    q = torch.zeros((1, 2, 3), device="mps")
    p = torch.zeros((1, 3, 3), device="mps")
    with pytest.raises(ValueError, match="query_lengths"):
        ball_query(q, p, radius=1.0, k=1, query_lengths=torch.tensor([3], device="mps"))


def test_invalid_inputs_raise_before_dispatch() -> None:
    q = torch.zeros((1, 2, 3))
    p = torch.zeros((1, 3, 3))
    with pytest.raises(ValueError, match="MPS"):
        ball_query(q, p, radius=1.0, k=1)
    if not torch.backends.mps.is_available():
        return
    q, p = q.to("mps"), p.to("mps")
    with pytest.raises(ValueError, match="radius"):
        ball_query(q, p, radius=-1.0, k=1)
    with pytest.raises(ValueError, match="radius"):
        ball_query(q, p, radius=10**1000, k=1)
    with pytest.raises(ValueError, match="float32"):
        ball_query(q, p, radius=1e-100, k=1)
    with pytest.raises(ValueError, match="normal float32"):
        ball_query(q, p, radius=1e-40, k=1)
    radius_with_grad = torch.tensor(1.0, device="mps", requires_grad=True)
    with pytest.raises(ValueError, match="radius"):
        ball_query(q, p, radius=radius_with_grad, k=1)
    with pytest.raises(ValueError, match="k"):
        ball_query(q, p, radius=1.0, k=-1)
    with pytest.raises(ValueError, match="k"):
        ball_query(q, p, radius=1.0, k=1 << 63)

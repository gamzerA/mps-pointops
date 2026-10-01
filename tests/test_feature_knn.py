"""Direct feature-space kNN oracle and an EdgeConv forward/backward check."""

import os

import numpy as np
import pytest
import torch

from mps_pointops import flat, knn
from mps_pointops.pyg import _knn_mps


MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")


def _oracle(query: torch.Tensor, ref: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Independent scalar float32 L2 loop, ordered by distance then index."""
    q = query.detach().cpu().numpy()
    x = ref.detach().cpu().numpy()
    distances = np.empty((len(q), k), dtype=np.float32)
    indices = np.empty((len(q), k), dtype=np.int64)
    for qi, row in enumerate(q):
        ranked = []
        for j, point in enumerate(x):
            delta = np.float32(point[0] - row[0])
            score = np.float32(delta * delta)
            for d in range(1, len(row)):
                delta = np.float32(point[d] - row[d])
                score = np.float32(score + np.float32(delta * delta))
            ranked.append((float(score), j))
        ranked.sort()
        for slot, (score, j) in enumerate(ranked[:k]):
            distances[qi, slot] = np.sqrt(np.float32(score))
            indices[qi, slot] = j
    return distances, indices


@MPS
@pytest.mark.parametrize("dim", [1, 2, 4, 64, 128])
@pytest.mark.parametrize("k", [1, 5, 33])
def test_dense_feature_knn_matches_scalar_oracle(dim: int, k: int) -> None:
    gen = torch.Generator().manual_seed(107 + dim)
    query = torch.randn((2, 9, dim), generator=gen)
    ref = torch.randn((2, 97, dim), generator=gen)
    # These are genuine ties, including a query matching duplicate references.
    ref[:, 20] = ref[:, 3]
    query[:, 0] = ref[:, 3]
    q_view = torch.cat((query, query), dim=1)[:, ::2]
    x_view = torch.cat((ref, ref), dim=1)[:, ::2]
    assert not q_view.is_contiguous() and not x_view.is_contiguous()

    got_dist, got_idx = knn(q_view.to("mps"), x_view.to("mps"), k)
    assert got_idx.dtype == torch.int64 and got_dist.dtype == torch.float32
    assert got_idx.shape == got_dist.shape == (2, 9, k)
    for batch in range(2):
        want_dist, want_idx = _oracle(q_view[batch], x_view[batch], k)
        np.testing.assert_array_equal(got_idx[batch].cpu().numpy(), want_idx)
        np.testing.assert_allclose(got_dist[batch].cpu().numpy(), want_dist, rtol=3e-6, atol=2e-6)


@MPS
def test_feature_knn_exact_maximum_k_256_and_replacement() -> None:
    """Fill every lane slot, then replace entries beyond the first 256 refs."""
    gen = torch.Generator().manual_seed(2026)
    ref = torch.randn((1, 300, 64), generator=gen)
    query = torch.randn((1, 4, 64), generator=gen)
    ref[0, 1] = ref[0, 0]
    query[0, 0] = ref[0, 0]
    distance, index = knn(query.to("mps"), ref.to("mps"), 256)
    expected_distance, expected_index = _oracle(query[0], ref[0], 256)
    assert index.shape == distance.shape == (1, 4, 256)
    np.testing.assert_array_equal(index[0].cpu().numpy(), expected_index)
    np.testing.assert_allclose(distance[0].cpu().numpy(), expected_distance, rtol=3e-6, atol=2e-6)


@MPS
def test_flat_feature_knn_ragged_batches_and_empty_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    gen = torch.Generator().manual_seed(551)
    x = torch.randn((15, 128), generator=gen)
    y = torch.randn((7, 128), generator=gen)
    # Three batches: lengths 2, 0, 13 for x and 3, 1, 3 for y. The first
    # cloud has fewer than k references, so internal -1 slots are compacted.
    batch_x = torch.tensor([0] * 2 + [2] * 13)
    batch_y = torch.tensor([0] * 3 + [1] + [2] * 3)
    expected_query = []
    expected_ref = []
    for batch, (x_lo, x_hi) in enumerate(((0, 2), (2, 2), (2, 15))):
        for qi in (batch_y == batch).nonzero().flatten().tolist():
            if x_hi == x_lo:
                continue
            take = min(5, x_hi - x_lo)
            _, local = _oracle(y[qi:qi + 1], x[x_lo:x_hi], take)
            expected_query.extend([qi] * take)
            expected_ref.extend((local[0] + x_lo).tolist())

    # Fail if this test accidentally takes the PyTorch reference path on MPS.
    def forbidden(*args, **kwargs):
        raise AssertionError("feature-space kNN must use the native Metal path")

    monkeypatch.setattr(flat, "_reference_search", forbidden)
    got = flat.knn(x.to("mps"), y.to("mps"), 5, batch_x.to("mps"), batch_y.to("mps"))
    assert got.dtype == torch.int64 and got.shape == (2, len(expected_query))
    assert got.cpu().tolist() == [expected_query, expected_ref]


@MPS
def test_feature_knn_limits_are_explicit() -> None:
    x = torch.zeros((1, 300, 64), device="mps")
    with pytest.raises(ValueError, match="at most 256"):
        knn(x, x, 257)
    with pytest.raises(ValueError, match="at most 256"):
        flat.knn(x[0], x[0], 257)
    with pytest.raises(ValueError, match="ref must have shape"):
        knn(x, torch.zeros((1, 300, 128), device="mps"), 2)
    with pytest.raises(ValueError, match="dimensions differ"):
        flat.knn(x[0], torch.zeros((3, 128), device="mps"), 2)
    with pytest.raises(ValueError, match="D >= 1"):
        knn(x[:, :, :0], x[:, :, :0], 0)


@MPS
@pytest.mark.skipif(os.getenv("PYTORCH_MPS_FAST_MATH", "0") != "0", reason="non-finite Safe Math policy only")
def test_safe_math_nonfinite_and_overflow_candidates_leave_padded_slots() -> None:
    x = torch.zeros((1, 3, 64), device="mps")
    x[0, 0, 0] = 1.0
    x[0, 1, 0] = 1e30  # finite coordinate, squared distance overflows float32
    x[0, 2, 0] = 2.0
    y = torch.zeros((1, 2, 64), device="mps")
    y[0, 1, 0] = float("nan")
    dist, idx = knn(y, x, 3)
    torch.mps.synchronize()
    assert idx.cpu().tolist() == [[[0, 2, -1], [-1, -1, -1]]]
    assert dist[0, 0, :2].cpu().tolist() == [1.0, 2.0]
    assert torch.isinf(dist[:, :, 2]).all().item()
    edges = flat.knn(x[0], y[0], 3)
    assert edges.cpu().tolist() == [[0, 0], [0, 2]]


@MPS
def test_pyg_knn_wrapper_and_graph_accept_features() -> None:
    x = torch.zeros((5, 64), device="mps")
    x[:, 0] = torch.tensor([0.0, 1.0, 4.0, 10.0, 11.0], device="mps")
    y = x[[0, 3]] + 0.1
    bx = torch.tensor([0, 0, 0, 1, 1], device="mps")
    by = torch.tensor([0, 1], device="mps")
    ptr_x = torch.tensor([0, 3, 5], device="mps")
    ptr_y = torch.tensor([0, 1, 2], device="mps")
    assert _knn_mps(x, y, ptr_x, ptr_y, 2).cpu().tolist() == [[0, 0, 1, 1], [0, 1, 3, 4]]
    assert flat.knn(x, y, 2, bx, by).cpu().tolist() == [[0, 0, 1, 1], [0, 1, 3, 4]]
    graph = flat.knn_graph(x, 1, bx, loop=False)
    assert graph.shape == (2, 5)
    assert all(a != b for a, b in zip(*graph.cpu().tolist()))


def _edgeconv(
    features: torch.Tensor, weight: torch.Tensor, k: int,
    neighbor_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """One DGCNN-style EdgeConv: h_i=max_j ReLU(W [f_i, f_j-f_i])."""
    if neighbor_idx is None:
        _, neighbor_idx = knn(features.detach(), features.detach(), k)
    batch, points, dim = features.shape
    neighbors = features.gather(
        1, neighbor_idx.reshape(batch, -1).unsqueeze(-1).expand(-1, -1, dim)
    ).reshape(batch, points, k, dim)
    center = features.unsqueeze(2).expand_as(neighbors)
    edge = torch.cat((center, neighbors - center), dim=-1)
    return torch.nn.functional.relu(torch.nn.functional.linear(edge, weight)).amax(dim=2)


@MPS
@pytest.mark.parametrize("dim", [64, 128])
def test_dgcnn_edgeconv_forward_backward_cpu_mps(dim: int) -> None:
    gen = torch.Generator().manual_seed(90 + dim)
    source = torch.randn((1, 32, dim), generator=gen)
    weights = torch.randn((16, 2 * dim), generator=gen) / (2 * dim) ** 0.5
    source_cpu = source.clone().requires_grad_()
    weight_cpu = weights.clone().requires_grad_()
    out_cpu = _edgeconv(source_cpu, weight_cpu, k=6)
    out_cpu.square().mean().backward()

    source_mps = source.to("mps").requires_grad_()
    weight_mps = weights.to("mps").requires_grad_()
    out_mps = _edgeconv(source_mps, weight_mps, k=6)
    out_mps.square().mean().backward()
    torch.mps.synchronize()

    torch.testing.assert_close(out_mps.cpu(), out_cpu.detach(), rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(source_mps.grad.cpu(), source_cpu.grad, rtol=5e-5, atol=5e-5)
    torch.testing.assert_close(weight_mps.grad.cpu(), weight_cpu.grad, rtol=5e-5, atol=5e-5)


def _dynamic_graph_classifier(
    features: torch.Tensor, weights: list[torch.Tensor], head: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Four self-contained dynamic EdgeConv stages plus global max pooling."""
    stages = []
    neighbors = []
    for weight in weights:
        _, ids = knn(features.detach(), features.detach(), 6)
        neighbors.append(ids)
        features = _edgeconv(features, weight, 6, ids)
        stages.append(features)
    pooled = torch.cat(stages, dim=-1).amax(dim=1)
    return torch.nn.functional.linear(pooled, head), neighbors


@MPS
def test_four_stage_dynamic_graph_classifier_forward_backward() -> None:
    """Exercise 3D -> 64D -> 64D -> 128D graph recomputation.

    This is an independently written DGCNN-style integration smoke test, not
    parity against a pinned upstream full DGCNN implementation or dataset.
    """
    gen = torch.Generator().manual_seed(330)
    feature = torch.randn((2, 24, 3), generator=gen)
    matrices = [
        torch.randn((out, 2 * inp), generator=gen) / (2 * inp) ** 0.5
        for inp, out in ((3, 64), (64, 64), (64, 128), (128, 256))
    ]
    head = torch.randn((10, 512), generator=gen) / 512 ** 0.5

    def run(device: str):
        x = feature.to(device).detach().requires_grad_()
        w = [m.to(device).detach().requires_grad_() for m in matrices]
        h = head.to(device).detach().requires_grad_()
        logits, ids = _dynamic_graph_classifier(x, w, h)
        logits.square().mean().backward()
        return logits.detach().cpu(), [v.cpu() for v in ids], x.grad.cpu(), [v.grad.cpu() for v in w], h.grad.cpu()

    cpu = run("cpu")
    mps = run("mps")
    for a, b in zip(cpu[1], mps[1]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(cpu[0], mps[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cpu[2], mps[2], rtol=1e-4, atol=1e-4)
    for a, b in zip(cpu[3], mps[3]):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cpu[4], mps[4], rtol=1e-4, atol=1e-4)


def test_cpu_feature_knn_shapes() -> None:
    x = torch.arange(40, dtype=torch.float32).reshape(5, 8)
    d, i = knn(x[None, :2], x[None], 3)
    assert d.shape == i.shape == (1, 2, 3)
    assert flat.knn(x, x[:2], 3).shape == (2, 6)

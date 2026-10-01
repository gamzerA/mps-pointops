"""Legacy random-walk contract checks on CPU and MPS."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from mps_pointops.random_walk import random_walk


@pytest.fixture(params=["cpu", "mps"])
def device(request):
    if request.param == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    return torch.device(request.param)


def _graph(device):
    # The input is intentionally unsorted.  Its sorted edge IDs are used for
    # the return_edge_indices contract.
    row = torch.tensor([1, 0, 2, 1, 3, 1, 2], device=device)
    col = torch.tensor([2, 1, 1, 0, 4, 3, 0], device=device)
    return row, col


@pytest.mark.parametrize("length", [0, 1, 5])
def test_walk_shapes_edges_and_isolated_node(device, length):
    row, col = _graph(device)
    starts = torch.tensor([0, 1, 4, 5], device=device)
    nodes, edges = random_walk(row, col, starts, length, num_nodes=6,
                               return_edge_indices=True)
    assert nodes.shape == (4, length + 1)
    assert edges.shape == (4, length)
    assert nodes.device == edges.device
    assert nodes.device.type == device.type
    assert nodes.dtype == edges.dtype == torch.long
    assert torch.equal(nodes[:, 0], starts)
    assert torch.equal(nodes[2], torch.full((length + 1,), 4, device=device))
    assert torch.equal(nodes[3], torch.full((length + 1,), 5, device=device))
    assert torch.equal(edges[2:], torch.full((2, length), -1, device=device))

    perm = torch.argsort(row * 6 + col)
    sorted_row, sorted_col = row[perm], col[perm]
    for step in range(length):
        valid = edges[:, step] >= 0
        assert torch.equal(sorted_row[edges[valid, step]], nodes[valid, step])
        assert torch.equal(sorted_col[edges[valid, step]], nodes[valid, step + 1])


@pytest.mark.parametrize("p,q,expected", [
    (2, 4, [2 / 7, 0, 4 / 7, 1 / 7]),
    (0.5, 2, [4 / 7, 0, 2 / 7, 1 / 7]),
])
def test_biased_transition_distribution(device, p, q, expected):
    # Every walk starts 0 -> 1.  At 1, candidate 0 is a return, 2 has a
    # directed link 2 -> 0, and 3 is outward.  The upstream acceptance
    # thresholds change order as p and q change; the normalized values test
    # the union/max rule in both directions.
    row = torch.tensor([0, 1, 1, 1, 2], device=device)
    col = torch.tensor([1, 0, 2, 3, 0], device=device)
    starts = torch.zeros(12_000, dtype=torch.long, device=device)
    torch.manual_seed(7719)
    nodes, edges = random_walk(row, col, starts, 2, p=p, q=q,
                               num_nodes=4, return_edge_indices=True)
    assert torch.equal(nodes[:, 1], torch.ones_like(starts))
    assert torch.equal(edges[:, 0], torch.zeros_like(starts))
    count = torch.bincount(nodes[:, 2].cpu(), minlength=4).float() / len(starts)
    torch.testing.assert_close(count, torch.tensor(expected), atol=0.025, rtol=0)


def test_empty_graph_and_empty_starts(device):
    empty = torch.empty(0, dtype=torch.long, device=device)
    starts = torch.tensor([0, 2], dtype=torch.long, device=device)
    nodes, edges = random_walk(empty, empty, starts, 4, p=2, q=4,
                               num_nodes=3, return_edge_indices=True)
    assert nodes.tolist() == [[0] * 5, [2] * 5]
    assert edges.tolist() == [[-1] * 4, [-1] * 4]
    empty_nodes, empty_edges = random_walk(
        empty, empty, empty, 4, num_nodes=0, return_edge_indices=True)
    assert empty_nodes.shape == (0, 5)
    assert empty_edges.shape == (0, 4)


def test_random_precision_is_independent_of_default_dtype(device):
    row, col = _graph(device)
    starts = torch.tensor([0, 1, 2], device=device)
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        uniform_nodes, uniform_edges = random_walk(
            row, col, starts, 3, num_nodes=6, return_edge_indices=True)
        biased_nodes, biased_edges = random_walk(
            row, col, starts, 3, p=2, q=4, num_nodes=6,
            return_edge_indices=True)
    finally:
        torch.set_default_dtype(previous)
    for nodes, edges in ((uniform_nodes, uniform_edges),
                         (biased_nodes, biased_edges)):
        assert nodes.shape == (3, 4)
        assert edges.shape == (3, 3)
        assert nodes.dtype == edges.dtype == torch.int64


def test_biased_boundary_draw_does_not_select_another_rows_padding(
    device, monkeypatch,
):
    row = torch.tensor([0, 1, 3, 4, 4, 4], device=device)
    col = torch.tensor([1, 0, 4, 0, 2, 3], device=device)
    starts = torch.tensor([0, 3], device=device)

    def boundary_rand(shape, *, device, dtype):
        return torch.ones(shape, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "rand", boundary_rand)
    nodes, edges = random_walk(row, col, starts, 2, p=2, q=4,
                               num_nodes=5, return_edge_indices=True)
    assert nodes.tolist() == [[0, 1, 0], [3, 4, 3]]
    assert edges.tolist() == [[0, 1], [2, 5]]


def test_upstream_163_cpu_uniform_exact_if_available():
    spec = importlib.util.find_spec("torch_cluster")
    if spec is None or spec.origin is None:
        pytest.skip("real torch_cluster is not installed")
    import torch_cluster

    if getattr(torch_cluster, "__version__", "") != "1.6.3":
        pytest.skip("requires real torch_cluster 1.6.3")
    row, col = _graph(torch.device("cpu"))
    starts = torch.tensor([0, 1, 2, 3, 4, 5] * 4)
    for coalesced in (True, False):
        if not coalesced:
            perm = torch.argsort(row * 6 + col)
            row_in, col_in = row[perm], col[perm]
        else:
            row_in, col_in = row, col
        torch.manual_seed(416)
        expected = torch_cluster.random_walk(
            row_in, col_in, starts, 6, coalesced=coalesced,
            num_nodes=6, return_edge_indices=True)
        torch.manual_seed(416)
        actual = random_walk(row_in, col_in, starts, 6,
                             coalesced=coalesced, num_nodes=6,
                             return_edge_indices=True)
        assert all(torch.equal(a, e) for a, e in zip(actual, expected))

    generator = torch.Generator().manual_seed(281)
    random_row = torch.randint(0, 31, (97,), generator=generator)
    random_col = torch.randint(0, 32, (97,), generator=generator)
    random_starts = torch.randint(0, 32, (40,), generator=generator)
    for seed in range(20):
        torch.manual_seed(seed)
        expected = torch_cluster.random_walk(
            random_row, random_col, random_starts, 12,
            num_nodes=32, return_edge_indices=True)
        torch.manual_seed(seed)
        actual = random_walk(
            random_row, random_col, random_starts, 12,
            num_nodes=32, return_edge_indices=True)
        assert all(torch.equal(a, e) for a, e in zip(actual, expected))


def test_upstream_163_cpu_biased_transition_if_available():
    spec = importlib.util.find_spec("torch_cluster")
    if spec is None or spec.origin is None:
        pytest.skip("real torch_cluster is not installed")
    import torch_cluster

    if getattr(torch_cluster, "__version__", "") != "1.6.3":
        pytest.skip("requires real torch_cluster 1.6.3")
    row = torch.tensor([0, 1, 1, 1, 2])
    col = torch.tensor([1, 0, 2, 3, 0])
    starts = torch.zeros(12_000, dtype=torch.long)
    upstream = torch_cluster.random_walk(row, col, starts, 2, p=2, q=4,
                                         num_nodes=4)
    ours = random_walk(row, col, starts, 2, p=2, q=4, num_nodes=4)
    for walks in (upstream, ours):
        frequency = torch.bincount(walks[:, 2], minlength=4).float() / len(starts)
        torch.testing.assert_close(
            frequency, torch.tensor([2 / 7, 0, 4 / 7, 1 / 7]),
            atol=0.025, rtol=0)


def test_invalid_inputs_are_rejected():
    row = torch.tensor([0, 1])
    col = torch.tensor([1, 0])
    starts = torch.tensor([0])
    with pytest.raises(ValueError, match="nonnegative"):
        random_walk(row, col, starts, -1)
    with pytest.raises(ValueError, match="finite and positive"):
        random_walk(row, col, starts, 2, p=0)
    with pytest.raises(ValueError, match="float32 normal range"):
        random_walk(row, col, starts, 2, p=1e-100)
    with pytest.raises(ValueError, match="grouped by row"):
        random_walk(row.flip(0), col.flip(0), starts, 2, coalesced=False)
    with pytest.raises(ValueError, match="num_nodes is required"):
        random_walk(row, col, torch.empty(0, dtype=torch.long), 2)

"""Legacy torch_cluster 1.6.3 Graclus subset and partition invariants."""

from __future__ import annotations

import pytest
import torch

from mps_pointops.graclus import _greedy_cpu, _library, graclus_cluster


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


def assert_pair_partition(row: torch.Tensor, col: torch.Tensor, labels: torch.Tensor) -> None:
    row, col, labels = row.cpu().tolist(), col.cpu().tolist(), labels.cpu().tolist()
    groups: dict[int, list[int]] = {}
    for vertex, label in enumerate(labels):
        assert 0 <= label <= vertex
        groups.setdefault(label, []).append(vertex)
    edges = set(zip(row, col))
    for label, members in groups.items():
        assert len(members) <= 2
        assert label == min(members)
        if len(members) == 2:
            u, v = members
            assert (u, v) in edges or (v, u) in edges


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weighted", [False, True])
def test_disjoint_pairs_self_loops_and_isolated_nodes(device: str, weighted: bool) -> None:
    # Every choice is unique, so different CPU/MPS random permutations cannot
    # change the partition. The self-loop must not make node 4 join a pair.
    row = torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64, device=device)
    col = torch.tensor([1, 0, 3, 2, 4], dtype=torch.int64, device=device)
    weights = torch.tensor([2, 2, 3, 3, 100], dtype=torch.float32, device=device) if weighted else None
    result = graclus_cluster(row, col, weights, num_nodes=6)
    assert result.device.type == device and result.dtype == torch.int64
    torch.testing.assert_close(result.cpu(), torch.tensor([0, 0, 2, 2, 4, 5]), atol=0, rtol=0)
    assert_pair_partition(row, col, result)


@pytest.mark.parametrize("device", DEVICES)
def test_empty_edges_with_explicit_num_nodes(device: str) -> None:
    empty = torch.empty(0, dtype=torch.int64, device=device)
    actual = graclus_cluster(empty, empty, num_nodes=4)
    torch.testing.assert_close(actual.cpu(), torch.arange(4), atol=0, rtol=0)
    assert graclus_cluster(empty, empty, num_nodes=0).shape == (0,)


@pytest.mark.parametrize("device", DEVICES)
def test_weight_zero_and_negative_edges(device: str) -> None:
    # Zero is an eligible weight, while a negative-only neighbor is ignored.
    row = torch.tensor([0, 0, 1, 2], dtype=torch.int64, device=device)
    col = torch.tensor([1, 2, 0, 0], dtype=torch.int64, device=device)
    weight = torch.tensor([-1., 0., -1., 0.], dtype=torch.float32, device=device)
    out = graclus_cluster(row, col, weight)
    torch.testing.assert_close(out.cpu(), torch.tensor([0, 1, 0]), atol=0, rtol=0)


def test_weighted_equal_maximum_uses_last_available_csr_neighbor() -> None:
    # Explicit CSR and visit order isolate the CPU kernel's >= tie rule from
    # the public wrapper's node permutation and equal-key sort order.
    rowptr = torch.tensor([0, 2, 2, 2], dtype=torch.int64)
    col = torch.tensor([1, 2], dtype=torch.int64)
    weights = torch.tensor([2., 2.])
    order = torch.tensor([0, 1, 2], dtype=torch.int64)
    actual = _greedy_cpu(rowptr, col, weights, order)
    torch.testing.assert_close(actual, torch.tensor([0, 1, 0]), atol=0, rtol=0)


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_metal_weighted_tie_uses_last_available_csr_neighbor() -> None:
    rowptr = torch.tensor([0, 2, 2, 2], dtype=torch.int64, device="mps")
    col = torch.tensor([1, 2], dtype=torch.int64, device="mps")
    weights = torch.tensor([2., 2.], device="mps")
    order = torch.tensor([0, 1, 2], dtype=torch.int64, device="mps")
    out = torch.empty(3, dtype=torch.int64, device="mps")
    _library().graclus_greedy(rowptr, col, weights, order, out, 3, 1, threads=1, group_size=1)
    torch.mps.synchronize()
    torch.testing.assert_close(out.cpu(), torch.tensor([0, 1, 0]), atol=0, rtol=0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("weighted", [False, True])
def test_general_graph_preserves_matching_structure(device: str, weighted: bool) -> None:
    row = torch.tensor([0, 0, 1, 1, 1, 2, 2, 3, 4, 4, 5, 6, 7, 8], device=device)
    col = torch.tensor([1, 2, 0, 2, 3, 0, 1, 1, 5, 6, 4, 4, 8, 7], device=device)
    weight = torch.tensor([0., 3., 0., 2., 1., 3., 2., 1., 4., 2., 4., 2., -1., -1.], device=device) if weighted else None
    for seed in (0, 7, 19):
        torch.manual_seed(seed)
        actual = graclus_cluster(row, col, weight, num_nodes=10)
        assert_pair_partition(row, col, actual)


def test_cpu_repeatability_with_seed() -> None:
    row = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    col = torch.tensor([1, 2, 0, 3, 0, 3, 1, 2], dtype=torch.int64)
    torch.manual_seed(42)
    first = graclus_cluster(row, col)
    torch.manual_seed(42)
    second = graclus_cluster(row, col)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_input_contract_rejects_unsafe_native_access() -> None:
    row = torch.tensor([0, 1], dtype=torch.int64)
    col = torch.tensor([1, 0], dtype=torch.int64)
    with pytest.raises(ValueError, match="num_nodes is required"):
        graclus_cluster(row[:0], col[:0])
    with pytest.raises(TypeError, match="int64"):
        graclus_cluster(row.int(), col)
    with pytest.raises(ValueError, match="equal length"):
        graclus_cluster(row, col[:1])
    with pytest.raises(IndexError, match="node IDs"):
        graclus_cluster(torch.tensor([-1]), torch.tensor([0]), num_nodes=2)
    with pytest.raises(IndexError, match="node IDs"):
        graclus_cluster(row, col, num_nodes=1)
    with pytest.raises(TypeError, match="float32"):
        graclus_cluster(row, col, torch.ones(2, dtype=torch.float64))
    with pytest.raises(ValueError, match="finite"):
        graclus_cluster(row, col, torch.tensor([float("nan"), 1.]))


def test_compat_exposes_legacy_name() -> None:
    from mps_pointops import compat

    assert callable(compat.graclus.graclus_cluster)

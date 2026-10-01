"""GPU regression cases for ragged search and ordered radius compaction."""

import pytest
import torch

from mps_pointops import flat


pytestmark = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


def _both_devices_equal(fn, *values, **kwargs):
    expected = fn(*values, **kwargs)
    moved = [v.to("mps") if isinstance(v, torch.Tensor) else v for v in values]
    actual = fn(*moved, **kwargs)
    assert torch.equal(actual.cpu(), expected)
    return actual.cpu()


def test_radius_first_k_across_simd_blocks_and_empty_batch():
    refs = torch.full((75, 3), 20.0)
    matches = [3, 31, 32, 33, 63, 64, 69]
    refs[matches[:]] = torch.tensor([0.25, 0.0, 0.0])
    refs[70:] = torch.tensor([10.0, 0.0, 0.0])
    queries = torch.cat((torch.zeros(9, 3), torch.tensor([[10.0, 0.0, 0.0]])))
    batch_x = torch.tensor([0] * 70 + [2] * 5)
    batch_y = torch.tensor([0] * 9 + [2])

    edges = _both_devices_equal(
        flat.radius, refs, queries, 1.0, batch_x, batch_y, max_num_neighbors=4
    )
    assert edges.shape == (2, 40)
    assert edges[1, :4].tolist() == [3, 31, 32, 33]
    assert edges[1, -4:].tolist() == [70, 71, 72, 73]


def test_radius_discards_nonfinite_queries_and_references():
    refs = torch.tensor(
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0],
         [0.25, 0.0, 0.0], [float("inf"), 0.0, 0.0]]
    )
    queries = torch.tensor(
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0],
         [float("inf"), 0.0, 0.0]]
    )
    edges = _both_devices_equal(flat.radius, refs, queries, 1.0, max_num_neighbors=4)
    assert edges.tolist() == [[0, 0], [0, 2]]


def test_knn_ragged_batches_ties_and_missing_neighbors():
    refs = torch.zeros(69, 3)
    refs[:66, 0] = torch.arange(66, dtype=torch.float32) + 100.0
    refs[0, 0], refs[1, 0] = -1.0, 1.0
    refs[66:, 0] = torch.tensor([7.0, 8.0, 9.0])
    queries = torch.zeros(10, 3)
    queries[-1, 0] = 8.0
    batch_x = torch.tensor([0] * 66 + [2] * 3)
    batch_y = torch.tensor([0] * 9 + [2])

    edges = _both_devices_equal(flat.knn, refs, queries, 4, batch_x, batch_y)
    assert edges[1, :2].tolist() == [0, 1]
    assert edges[1, -3:].tolist() == [67, 66, 68]
    assert edges.shape == (2, 9 * 4 + 3)


def test_fps_uneven_batch_lengths_and_missing_batch():
    torch.manual_seed(7)
    x = torch.rand(87, 3)
    batch = torch.tensor([0] * 2 + [1] * 80 + [3] * 5)
    selected = _both_devices_equal(flat.fps, x, batch, ratio=0.25, random_start=False)
    assert selected.shape == (23,)
    assert selected[0].item() == 0
    assert selected[1].item() == 2
    assert selected[-2].item() >= 82

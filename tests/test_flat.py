import pytest
import torch

from mps_pointops import flat


mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")


def bipartite(device="cpu"):
    x = torch.tensor([[0., 0, 0], [2., 0, 0], [5., 0, 0],
                      [10., 0, 0], [12., 0, 0]], device=device)
    y = torch.tensor([[1., 0, 0], [0., 0, 0], [11., 0, 0],
                      [15., 0, 0]], device=device)
    batch_x = torch.tensor([0, 0, 0, 2, 2], dtype=torch.long, device=device)
    batch_y = torch.tensor([0, 1, 2, 2], dtype=torch.long, device=device)
    return x, y, batch_x, batch_y


def test_fps_global_indices_and_empty_batch():
    x, _, batch_x, _ = bipartite()
    assert flat.fps(x, batch_x, ratio=0.5, random_start=False).tolist() == [0, 2, 3]
    assert flat.fps(x, ptr=[0, 3, 3, 5], ratio=0.5, random_start=False).tolist() == [0, 2, 3]
    assert flat.fps(x, batch_x, ratio=0, random_start=False).shape == (0,)


def test_fps_default_ratio_and_random_start():
    x = torch.tensor([[0., 0, 0], [1., 0, 0], [4., 0, 0]])
    assert flat.fps(x, random_start=False).tolist() == [0, 2]
    torch.manual_seed(42)
    sample = flat.fps(x, ratio=1 / 3, random_start=True)
    assert sample.dtype == torch.long and sample.shape == (1,)
    assert 0 <= sample.item() < 3


def test_fps_empty_input_and_batch_validation():
    empty = torch.empty(0, 3)
    assert flat.fps(empty).shape == (0,)
    x, _, _, _ = bipartite()
    with pytest.raises(ValueError, match="sorted"):
        flat.fps(x, torch.tensor([0, 1, 0, 2, 2]))
    with pytest.raises(ValueError, match="ptr"):
        flat.fps(x, ptr=[0, 4, 3, 5])
    with pytest.raises(ValueError, match="ratio"):
        flat.fps(x, ratio=-0.1)
    assert flat.fps(x, ratio=1.1, random_start=False).shape == (6,)


def test_knn_bipartite_global_edges_and_missing_batch():
    x, y, batch_x, batch_y = bipartite()
    edge = flat.knn(x, y, 2, batch_x, batch_y)
    assert edge.dtype == torch.long and edge.shape == (2, 6)
    assert edge.tolist() == [[0, 0, 2, 2, 3, 3], [0, 1, 3, 4, 4, 3]]
    assert flat.knn(x, y, 5, batch_x, batch_y).shape == (2, 7)


def test_radius_compacts_first_neighbors_and_strict_boundary():
    x, y, batch_x, batch_y = bipartite()
    edge = flat.radius(x, y, 1.1, batch_x, batch_y, max_num_neighbors=2)
    assert edge.dtype == torch.long and edge.shape == (2, 4)
    assert edge.tolist() == [[0, 0, 2, 2], [0, 1, 3, 4]]
    assert flat.radius(x[:2], x[:1], 1.0).tolist() == [[0], [0]]
    assert flat.radius(x, y, 100.0, batch_x, batch_y, max_num_neighbors=1).tolist() == [
        [0, 2, 3], [0, 3, 3]
    ]


@pytest.mark.parametrize("method", ["knn", "radius"])
def test_search_empty_result(method):
    x = torch.empty(0, 3)
    y = torch.zeros(2, 3)
    result = flat.knn(x, y, 4) if method == "knn" else flat.radius(x, y, 1)
    assert result.shape == (2, 0) and result.dtype == torch.long
    batch_y = torch.tensor([0, 1])
    result = (
        flat.knn(x, y, 4, batch_y=batch_y)
        if method == "knn"
        else flat.radius(x, y, 1, batch_y=batch_y)
    )
    assert result.shape == (2, 0)


def test_rejects_unsupported_options_and_malformed_batches():
    x, y, batch_x, batch_y = bipartite()
    with pytest.raises(NotImplementedError, match="cosine"):
        flat.knn(x, y, 2, cosine=True)
    with pytest.raises(NotImplementedError, match="ignore_same_index"):
        flat.radius(x, y, 1, ignore_same_index=True)
    with pytest.raises(ValueError, match="both required"):
        flat.knn(x, y, 1, batch_x=batch_x)
    with pytest.raises(ValueError, match="sorted"):
        flat.radius(x, y, 1, batch_x=batch_x, batch_y=batch_y.flip(0))


@mps
def test_mps_flat_fps_matches_cpu():
    x, _, batch_x, _ = bipartite()
    expected = flat.fps(x, batch_x, ratio=0.5, random_start=False)
    got = flat.fps(x.to("mps"), batch_x.to("mps"), ratio=0.5, random_start=False)
    assert torch.equal(got.cpu(), expected)


@mps
def test_mps_bipartite_search_matches_cpu():
    x, y, batch_x, batch_y = bipartite()
    xm, ym, bxm, bym = (t.to("mps") for t in (x, y, batch_x, batch_y))
    assert torch.equal(flat.knn(xm, ym, 2, bxm, bym).cpu(), flat.knn(x, y, 2, batch_x, batch_y))
    assert torch.equal(
        flat.radius(xm, ym, 1.1, bxm, bym, max_num_neighbors=2).cpu(),
        flat.radius(x, y, 1.1, batch_x, batch_y, max_num_neighbors=2),
    )


@mps
def test_mps_large_k_uses_torch_fallback():
    x = torch.arange(300, dtype=torch.float32).unsqueeze(1).expand(-1, 3).contiguous()
    y = x[:1]
    result = flat.knn(x.to("mps"), y.to("mps"), 300)
    assert result.shape == (2, 300)
    assert torch.equal(result[1].cpu(), torch.arange(300))

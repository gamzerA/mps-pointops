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


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_knn_excludes_nonfinite_coordinates(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.tensor(
        [[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0], [float("inf"), 0.0, 0.0]],
        device=device,
    )
    y = torch.tensor([[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]], device=device)
    assert flat.knn(x, y, 3).cpu().tolist() == [[0], [0]]


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
@pytest.mark.parametrize("dimensions", [3, 64])
def test_mps_large_effective_k_is_explicit_error(dimensions):
    x = torch.arange(300, dtype=torch.float32).unsqueeze(1).expand(-1, 3).contiguous()
    y = x[:1]
    x = x[:, :1].expand(-1, dimensions).contiguous().to("mps")
    y = y[:, :1].expand(-1, dimensions).contiguous().to("mps")
    with pytest.raises(ValueError, match="effective k must be at most 256 on MPS"):
        flat.knn(x, y, 257)
    with pytest.raises(ValueError, match="effective k must be at most 256 on MPS"):
        flat.knn_graph(x, 256, loop=False)
    from mps_pointops import pyg

    with pytest.raises(ValueError, match="effective k must be at most 256 on MPS"):
        pyg._knn_mps(x, y, k=257)
    assert flat.knn(x[:256], y, 300).shape == (2, 256)
    assert pyg._knn_mps(x[:256], y, k=300).shape == (2, 256)


def test_cpu_large_k_remains_supported():
    x = torch.arange(257, dtype=torch.float32).unsqueeze(1).expand(-1, 3).contiguous()
    result = flat.knn(x, x[:1], 257)
    assert result.shape == (2, 257)
    assert torch.equal(result[1], torch.arange(257))


DEVICES = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "n, ratio, expected",
    [
        # torch_cluster computes ceil(float32(N) * float32(ratio)). These products
        # round just above an integer in float32, so the count is one more than
        # ceil of the exact product.
        (25, 0.6, 16),
        (45, 0.6, 28),
        (50, 0.3, 16),
        (90, 0.3, 28),
        # Ordinary cases.
        (10, 0.5, 5),
        (7, 0.5, 4),
        (9, 1.0, 9),
    ],
)
def test_fps_sample_count_uses_torch_cluster_float32_arithmetic(device, n, ratio, expected):
    x = torch.rand(n, 3, device=device)
    assert len(flat.fps(x, ratio=ratio, random_start=False)) == expected


def test_fps_sample_count_matches_float32_formula_over_a_sweep():
    for ratio in (0.05, 0.1, 0.3, 1 / 3, 0.6, 0.7, 0.95):
        lengths = list(range(1, 300))
        want = torch.ceil(torch.tensor(lengths, dtype=torch.float32) * torch.tensor(ratio, dtype=torch.float32)).long()
        ptr = [0]
        for n in lengths:
            ptr.append(ptr[-1] + n)
        x = torch.rand(ptr[-1], 3)
        assert len(flat.fps(x, ptr=ptr, ratio=ratio, random_start=False)) == int(want.sum())


@pytest.mark.parametrize("device", DEVICES)
def test_fps_float64_tensor_ratio_preserves_shape_and_device_rule(device):
    x = torch.rand(25, 3, device=device)
    scalar_ratio = torch.tensor(0.6, dtype=torch.float64)
    vector_ratio = torch.tensor([0.6], dtype=torch.float64)

    # torch_cluster CPU uses deg.float() * ratio. A float64 scalar is weakly
    # promoted, while a [1] tensor promotes the degree vector to float64.
    # Its CUDA path uses deg.to(ratio.dtype) * ratio for either ratio shape.
    expected_scalar = 16 if device == "cpu" else 15
    assert len(flat.fps(x, ratio=scalar_ratio, random_start=False)) == expected_scalar
    assert len(flat.fps(x, ratio=vector_ratio, random_start=False)) == 15


def test_fps_float64_coordinates_keep_original_cpu_degree_precision():
    x = torch.rand(25, 3, dtype=torch.float64)
    assert len(flat.fps(x, ratio=0.6, random_start=False)) == 16


@pytest.mark.parametrize("device", DEVICES)
def test_radius_threshold_is_double_product_rounded_to_float32(device):
    # torch_cluster compares against fl32(r * r) with r * r computed in double.
    # For r = 0.21 that is one float32 ULP above fl32(r)^2, so a point exactly
    # fl32(r) away from the query is a neighbor.
    r = 0.21
    r32 = torch.tensor(r, dtype=torch.float32)
    assert float(r32 * r32) < float(torch.tensor(r * r, dtype=torch.float32))  # the premise
    x = torch.tensor([[float(r32), 0.0, 0.0]], device=device)
    y = torch.zeros(1, 3, device=device)
    assert flat.radius(x, y, r).tolist() == [[0], [0]]

    # For r = 0.1, fl32(r * r) is below fl32(r)^2, so the same construction is outside.
    r = 0.1
    r32 = torch.tensor(r, dtype=torch.float32)
    assert float(r32 * r32) > float(torch.tensor(r * r, dtype=torch.float32))
    x = torch.tensor([[float(r32), 0.0, 0.0]], device=device)
    assert flat.radius(x, y, r).shape == (2, 0)

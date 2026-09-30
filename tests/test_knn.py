import numpy as np
import pytest
import torch

from mps_pointops import knn, reference

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")


def cloud(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, 3, generator=g)


def exact(query, ref, k):
    """Oracle: float32 squared distances rounded like the kernel, sorted by (distance, index)."""
    diff = query.numpy()[:, :, None, :] - ref.numpy()[:, None, :, :]
    d2 = diff[..., 0] * diff[..., 0]
    d2 = d2 + diff[..., 1] * diff[..., 1]
    d2 = d2 + diff[..., 2] * diff[..., 2]
    index = np.broadcast_to(np.arange(d2.shape[-1]), d2.shape)
    order = np.lexsort((index, d2), axis=-1)[..., :k]
    return np.sqrt(np.take_along_axis(d2, order, -1)), order


def check(query, ref, k):
    want_d, want_i = exact(query, ref, k)
    d, i = knn(query.to("mps"), ref.to("mps"), k)
    assert d.dtype == torch.float32 and i.dtype == torch.long
    assert d.shape == i.shape == (query.shape[0], query.shape[1], k)
    np.testing.assert_array_equal(i.cpu().numpy(), want_i)
    np.testing.assert_allclose(d.cpu().numpy(), want_d, rtol=1e-6, atol=0)


@mps
@pytest.mark.parametrize("n", [1, 31, 32, 33, 1000, 5000])
@pytest.mark.parametrize("k", [1, 5, 32, 128])
def test_matches_exact_search(n, k):
    if k > n:
        pytest.skip("k > n")
    ref = cloud(1, n)
    check(ref[:, :50] + 0.01, ref, k)


@mps
@pytest.mark.parametrize("m", [1, 7, 8, 9, 100])
def test_query_counts_around_group_size(m):
    check(cloud(1, m, seed=1), cloud(1, 2000), 16)


@mps
def test_k_equals_n_and_max_k():
    ref = cloud(1, 256)
    check(ref[:, :20], ref, 256)


@mps
def test_batches_are_independent():
    check(cloud(3, 40, seed=1), cloud(3, 3000), 32)


@mps
@pytest.mark.parametrize("descending", [False, True])
def test_spatially_sorted_reference(descending):
    ref = cloud(1, 20000)
    ref = ref[:, ref[0, :, 0].argsort(descending=descending)]
    check(cloud(1, 64, seed=1), ref, 128)


@mps
def test_ties_break_by_smaller_index():
    ref = torch.zeros(1, 300, 3)
    ref[0, ::3] = 1.0  # every third point is farther away
    check(torch.zeros(1, 2, 3), ref, 150)


@mps
def test_noncontiguous_input():
    ref = cloud(1, 4000)[:, ::2]
    query = cloud(1, 40, seed=1)[:, ::2]
    assert not ref.is_contiguous() and not query.is_contiguous()
    check(query, ref, 10)


@mps
def test_empty_outputs():
    ref = torch.randn(2, 50, 3, device="mps")
    d, i = knn(ref[:, :0], ref, 5)
    assert d.shape == i.shape == (2, 0, 5)
    d, i = knn(ref, ref, 0)
    assert d.shape == i.shape == (2, 50, 0)


@mps
def test_rejects_bad_input():
    ref = torch.randn(1, 300, 3, device="mps")
    with pytest.raises(ValueError):
        knn(ref, ref, 301)  # k > N
    with pytest.raises(ValueError):
        knn(ref, ref, 257)  # above the MPS maximum
    with pytest.raises(ValueError):
        knn(ref, ref.cpu(), 4)  # device mismatch
    with pytest.raises(ValueError):
        knn(ref, torch.randn(2, 300, 3, device="mps"), 4)  # batch mismatch
    with pytest.raises(TypeError):
        knn(ref.half(), ref.half(), 4)


def test_cpu_falls_back_to_reference():
    ref = cloud(1, 500)
    query = ref[:, :20]
    d, i = knn(query, ref, 8)
    want_d, want_i = reference.knn(query, ref, 8)
    assert torch.equal(i, want_i) and torch.equal(d, want_d)

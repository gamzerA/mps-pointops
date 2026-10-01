import pytest
import torch

from mps_pointops import furthest_point_sample, ops, reference

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")


def cloud(b, n, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, n, 3, generator=g)


def check(xyz, npoint, start_idx=0):
    want = reference.furthest_point_sample(xyz, npoint, start_idx)
    got = furthest_point_sample(xyz.to("mps"), npoint, start_idx).cpu()
    assert got.dtype == torch.long
    assert got.shape == want.shape
    assert torch.equal(got, want)


@mps
@pytest.mark.parametrize("n", [1, 2, 31, 32, 33, 1023, 1024, 1025, 5000, 100_000])
def test_matches_reference(n):
    check(cloud(1, n), min(n, 256))


@mps
@pytest.mark.parametrize("b", [2, 5])
def test_batches_are_independent(b):
    check(cloud(b, 3000), 128)


@mps
def test_start_idx():
    check(cloud(2, 1000), 64, start_idx=777)


@mps
def test_more_samples_than_points():
    # After every point is taken, all distances are 0 and index 0 repeats.
    check(cloud(1, 10), 25)


@mps
def test_duplicate_points_break_ties_by_smaller_index():
    xyz = torch.tensor([[[0.0, 0, 0], [1, 0, 0], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, 1, 0]]])
    check(xyz, 6)


@mps
def test_noncontiguous_input():
    xyz = cloud(1, 2000 * 2)[:, ::2]
    assert not xyz.is_contiguous()
    check(xyz, 100)


@mps
def test_empty_outputs():
    xyz = torch.randn(3, 50, 3, device="mps")
    assert furthest_point_sample(xyz, 0).shape == (3, 0)
    assert furthest_point_sample(xyz[:0], 5).shape == (0, 5)
    assert furthest_point_sample(xyz, 0, strategy="multigroup").shape == (3, 0)
    assert furthest_point_sample(xyz[:0], 5, strategy="multigroup").shape == (0, 5)
    assert torch.equal(
        furthest_point_sample(xyz, 1, strategy="multigroup"),
        furthest_point_sample(xyz, 1, strategy="single"),
    )


@mps
def test_rejects_bad_input():
    with pytest.raises(ValueError):
        furthest_point_sample(torch.randn(10, 3, device="mps"), 4)
    with pytest.raises(ValueError):
        furthest_point_sample(torch.randn(1, 0, 3, device="mps"), 4)
    with pytest.raises(IndexError):
        furthest_point_sample(torch.randn(1, 10, 3, device="mps"), 4, start_idx=10)
    with pytest.raises(TypeError):
        furthest_point_sample(torch.randn(1, 10, 3, device="mps", dtype=torch.float16), 4)


def test_cpu_falls_back_to_reference():
    xyz = cloud(1, 500)
    assert torch.equal(furthest_point_sample(xyz, 32), reference.furthest_point_sample(xyz, 32))


@mps
@pytest.mark.parametrize("n", [1, 31, 32, 33, 1025, 4095, 4096, 4097, 8193])
def test_multigroup_matches_reference_at_chunk_boundaries(n):
    xyz = cloud(1, n, seed=27)
    npoint = min(n + 2, 20)
    start_idx = n // 2
    want = reference.furthest_point_sample(xyz, npoint, start_idx)
    got = furthest_point_sample(
        xyz.to("mps"), npoint, start_idx, strategy="multigroup",
    ).cpu()
    assert got.dtype == torch.long
    assert got.shape == (1, npoint)
    assert torch.equal(got, want)


@mps
def test_multigroup_ties_skips_and_noncontiguous_input():
    xyz = torch.zeros((1, 129, 3))
    xyz[0, 1::2, 0] = 1
    xyz[0, 2::4, 1] = -1
    xyz = xyz.repeat_interleave(2, dim=1)[:, ::2]
    assert not xyz.is_contiguous()
    for skip_near_origin in (False, True):
        want = reference.furthest_point_sample(
            xyz, 150, start_idx=3, skip_near_origin=skip_near_origin,
        )
        got = furthest_point_sample(
            xyz.to("mps"), 150, start_idx=3,
            skip_near_origin=skip_near_origin, strategy="multigroup",
        ).cpu()
        assert torch.equal(got, want)


@mps
def test_large_multigroup_matches_single():
    xyz = cloud(1, 500_000, seed=19).to("mps")
    single = furthest_point_sample(xyz, 128, start_idx=11, strategy="single")
    multigroup = furthest_point_sample(xyz, 128, start_idx=11, strategy="multigroup")
    assert torch.equal(single.cpu(), multigroup.cpu())


@mps
def test_multigroup_auto_policy_and_batch_fallback(monkeypatch):
    real_library = ops._library
    libraries = []

    def record_library(name):
        libraries.append(name)
        return real_library(name)

    monkeypatch.setattr(ops, "_library", record_library)
    monkeypatch.setattr(ops, "_FPS_MULTIGROUP_MIN_N", 64)
    monkeypatch.setattr(ops, "_fps_auto_multigroup_supported", lambda: True)
    xyz = cloud(1, 128).to("mps")
    furthest_point_sample(xyz, 8)
    assert libraries == ["fps_multigroup"]

    libraries.clear()
    furthest_point_sample(xyz.repeat(2, 1, 1), 8)
    assert libraries == ["fps"]

    libraries.clear()
    monkeypatch.setattr(ops, "_fps_auto_multigroup_supported", lambda: False)
    furthest_point_sample(xyz, 8)
    assert libraries == ["fps"]
    with pytest.raises(ValueError, match="batch size 1"):
        furthest_point_sample(xyz.repeat(2, 1, 1), 8, strategy="multigroup")


def test_invalid_fps_strategy():
    with pytest.raises(ValueError, match="strategy"):
        furthest_point_sample(cloud(1, 8), 2, strategy="unknown")

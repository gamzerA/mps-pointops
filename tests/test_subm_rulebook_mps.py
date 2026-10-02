"""Bounded Metal rulebook parity against the independent CPU coordinate oracle."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from mps_pointops._sparse_rulebook import (
    generate_subm_rulebook,
    validate_sparse_tensor_3d,
)
from mps_pointops._subm_conv_mps import _library as _forward_library
from mps_pointops._subm_conv_mps import _output_csr
from mps_pointops._subm_rulebook_mps import generate_subm_rulebook_mps


MPS = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")


def _oracle(coordinates, shape, batches, kernel, dilation):
    indices = torch.tensor(coordinates, dtype=torch.int32).reshape(-1, 4)
    features = torch.empty((len(indices), 1), dtype=torch.float32)
    sparse = validate_sparse_tensor_3d(indices, features, shape, batches)
    return indices, generate_subm_rulebook(
        sparse, kernel_size=kernel, dilation=dilation
    )


def _check_rulebook(indices, cpu_rulebook, kernel, dilation):
    gpu = generate_subm_rulebook_mps(
        indices.to("mps"), kernel_size=kernel, dilation=dilation
    )
    torch.mps.synchronize()
    assert gpu.output_indices.device.type == "mps"
    assert gpu.pairs.device.type == "mps"
    assert gpu.pair_count.device.type == "mps"
    assert gpu.output_ptr.device.type == "mps"
    count = int(gpu.pair_count.cpu()[0])
    assert count == len(cpu_rulebook.pairs)
    assert torch.equal(gpu.output_indices.cpu(), cpu_rulebook.output_indices)
    assert torch.equal(gpu.pairs[:count].cpu(), cpu_rulebook.pairs)
    assert torch.all(gpu.pairs[count:] == -1).item()
    cpu_ptr, cpu_sources, cpu_offsets = _output_csr(cpu_rulebook.pairs, len(indices))
    assert torch.equal(gpu.output_ptr.cpu(), cpu_ptr)
    assert torch.equal(gpu.output_sources[:count].cpu(), cpu_sources)
    assert torch.equal(gpu.output_offsets[:count].cpu(), cpu_offsets)
    assert torch.all(gpu.output_sources[count:] == -1).item()
    assert torch.all(gpu.output_offsets[count:] == -1).item()
    return gpu


@MPS
@pytest.mark.parametrize(
    ("coordinates", "shape", "batches", "kernel", "dilation"),
    [
        (
            [[0, 2, 1, 1], [0, 1, 1, 1], [1, 2, 1, 1], [0, 3, 1, 1]],
            (5, 3, 3), 2, (3, 3, 3), (1, 1, 1),
        ),
        (
            [[0, 4, 2, 1], [0, 0, 2, 1], [0, 2, 2, 1]],
            (5, 5, 3), 1, (3, 1, 1), (2, 1, 1),
        ),
        (
            [[0, 0, 0, 0], [0, 3, 3, 3], [1, 0, 0, 0]],
            (4, 4, 4), 2, (1, 1, 1), (1, 1, 1),
        ),
        (
            [[1, 2, 2, 3], [0, 1, 2, 3], [1, 2, 2, 1], [0, 1, 2, 1]],
            (4, 4, 5), 2, (3, 3, 3), (1, 1, 2),
        ),
        ([], (4, 4, 4), 0, (3, 3, 3), (1, 1, 1)),
    ],
)
def test_mps_pairs_and_output_csr_are_bit_exact(
    coordinates, shape, batches, kernel, dilation
):
    indices, cpu = _oracle(coordinates, shape, batches, kernel, dilation)
    _check_rulebook(indices, cpu, kernel, dilation)


@MPS
@pytest.mark.parametrize("rows", [1, 7, 64, 256, 1024])
def test_mps_rulebook_random_unsorted_clouds(rows):
    # A shuffled, two-batch set catches mistakes that sorted coordinates or
    # one batch would hide. Fixed seeds keep failures reproducible.
    generator = torch.Generator().manual_seed(4812 + rows)
    universe = torch.randperm(2 * 16 * 8 * 8, generator=generator)[:rows]
    coordinates = []
    for flat in universe.tolist():
        batch, rest = divmod(flat, 16 * 8 * 8)
        axis0, rest = divmod(rest, 8 * 8)
        axis1, axis2 = divmod(rest, 8)
        coordinates.append([batch, axis0, axis1, axis2])
    kernel, dilation = (3, 3, 3), (1, 2, 1)
    indices, cpu = _oracle(coordinates, (16, 8, 8), 2, kernel, dilation)
    _check_rulebook(indices, cpu, kernel, dilation)


@MPS
def test_mps_rulebook_accepts_a_noncontiguous_coordinate_view():
    indices, cpu = _oracle(
        [[0, 1, 1, 1], [0, 2, 1, 1], [0, 3, 1, 1]],
        (5, 3, 3), 1, (3, 1, 1), (1, 1, 1),
    )
    padded = torch.full((len(indices), 8), -9, dtype=torch.int32, device="mps")
    padded[:, ::2] = indices.to("mps")
    view = padded[:, ::2]
    assert not view.is_contiguous()
    gpu = generate_subm_rulebook_mps(view, kernel_size=(3, 1, 1))
    count = int(gpu.pair_count.cpu()[0])
    assert torch.equal(gpu.pairs[:count].cpu(), cpu.pairs)


@MPS
def test_mps_rulebook_keeps_a_snapshot_when_caller_mutates_contiguous_indices():
    indices, cpu = _oracle(
        [[0, 1, 0, 0], [0, 2, 0, 0], [0, 3, 0, 0]],
        (5, 1, 1), 1, (3, 1, 1), (1, 1, 1),
    )
    caller = indices.to("mps")
    assert caller.is_contiguous()
    gpu = generate_subm_rulebook_mps(caller, kernel_size=(3, 1, 1))
    assert gpu.output_indices.data_ptr() != caller.data_ptr()
    # The input write is enqueued immediately after the call. The lookup and
    # output coordinates must both use the call-time snapshot.
    caller[0, 1] = 4
    torch.mps.synchronize()
    count = int(gpu.pair_count.cpu()[0])
    assert torch.equal(gpu.output_indices.cpu(), indices)
    assert torch.equal(gpu.pairs[:count].cpu(), cpu.pairs)


@MPS
def test_mps_target_arithmetic_does_not_wrap_int32_coordinate_boundary():
    indices, cpu = _oracle(
        [[0, 2**31 - 1, 0, 0], [0, 2**31 - 2, 0, 0]],
        (2**31, 1, 1), 1, (3, 1, 1), (1, 1, 1),
    )
    _check_rulebook(indices, cpu, (3, 1, 1), (1, 1, 1))


@MPS
def test_mps_generated_csr_chains_into_metal_forward_without_intermediate_sync():
    coordinates = [[0, 2, 1, 1], [0, 1, 1, 1], [1, 2, 1, 1], [0, 3, 1, 1]]
    shape, kernel = (5, 3, 3), (3, 3, 3)
    indices, _ = _oracle(coordinates, shape, 2, kernel, (1, 1, 1))
    generator = torch.Generator().manual_seed(8127)
    features = torch.randn((len(indices), 3), generator=generator)
    weights = torch.randn((2, 3, *kernel), generator=generator)
    bias = torch.randn((2,), generator=generator)
    forward_kernel = _forward_library().subm_conv3d_f32
    device_features = features.to("mps")
    device_weights = weights.to("mps")
    device_bias = bias.to("mps")
    gpu = generate_subm_rulebook_mps(
        indices.to("mps"), kernel_size=kernel
    )
    output = torch.empty((len(indices), 2), dtype=torch.float32, device="mps")
    forward_kernel(
        device_features, device_weights, gpu.output_ptr,
        gpu.output_sources, gpu.output_offsets, device_bias, output,
        len(indices), 3, 2, 27, 1,
        threads=output.numel(), group_size=output.numel(),
    )
    dense = torch.zeros((2, 3, *shape), dtype=torch.float32)
    for row, (batch, a0, a1, a2) in enumerate(coordinates):
        dense[batch, :, a0, a1, a2] = features[row]
    dense_result = F.conv3d(dense, weights, bias, padding=1)
    expected = torch.stack([
        dense_result[batch, :, a0, a1, a2]
        for batch, a0, a1, a2 in coordinates
    ])
    torch.mps.synchronize()
    torch.testing.assert_close(output.cpu(), expected, rtol=1e-4, atol=1e-5)


@MPS
def test_mps_rulebook_rejects_invalid_shapes_and_bounded_limit():
    coordinates = torch.zeros((2, 4), dtype=torch.int32, device="mps")
    with pytest.raises(ValueError, match="MPS int32"):
        generate_subm_rulebook_mps(coordinates.cpu(), kernel_size=3)
    with pytest.raises(ValueError, match="MPS int32"):
        generate_subm_rulebook_mps(coordinates.to(torch.int64), kernel_size=3)
    with pytest.raises(ValueError, match="MPS int32"):
        generate_subm_rulebook_mps(coordinates[:, :3], kernel_size=3)
    with pytest.raises(ValueError, match="odd kernel"):
        generate_subm_rulebook_mps(coordinates, kernel_size=(2, 1, 1))
    with pytest.raises(ValueError, match="dilation"):
        generate_subm_rulebook_mps(coordinates, kernel_size=3, dilation=0)
    with pytest.raises(ValueError, match="int32"):
        generate_subm_rulebook_mps(coordinates, kernel_size=3, dilation=2**31)
    with pytest.raises(ValueError, match="bounded"):
        generate_subm_rulebook_mps(
            torch.empty((1025, 4), dtype=torch.int32, device="mps"), kernel_size=3
        )
    with pytest.raises(ValueError, match="bounded"):
        generate_subm_rulebook_mps(
            torch.empty((1024, 4), dtype=torch.int32, device="mps"), kernel_size=5
        )

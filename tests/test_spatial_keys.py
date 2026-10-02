"""Stage-1 Metal spatial-index key contract, without claiming search speed."""

import pytest
import torch

from bench.spatial_keys import morton_key_cpu, morton_keys_f32


def test_morton_integer_oracle_layout():
    assert [morton_key_cpu(*xyz) for xyz in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))] == [0, 1, 2, 4]
    assert morton_key_cpu(2**21 - 1, 2**21 - 1, 2**21 - 1) == 2**63 - 1
    with pytest.raises(ValueError, match="21 bits"):
        morton_key_cpu(2**21, 0, 0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_metal_key_stage_writes_every_row_and_marks_bad_coordinates():
    cell_rows = [(0, 0, 0), (1, 2, 3), (31, 2, 9), (2**21 - 1, 0, 0)]
    points = torch.tensor([[float(x), float(y), float(z)] for x, y, z in cell_rows]
                          + [[-1.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]],
                          dtype=torch.float32, device="mps")
    origin = torch.zeros(3, dtype=torch.float32, device="mps")
    keys, invalid = morton_keys_f32(points, origin, 1.0)
    torch.mps.synchronize()
    assert keys.dtype == torch.int64 and invalid.dtype == torch.uint8
    assert keys.cpu().tolist()[:4] == [morton_key_cpu(*cell) for cell in cell_rows]
    assert invalid.cpu().tolist() == [0, 0, 0, 0, 1, 1]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")
def test_metal_key_stage_crosses_threadgroups_and_handles_empty():
    points = torch.arange(513, dtype=torch.float32).to("mps")
    points = torch.stack((points, torch.zeros_like(points), torch.zeros_like(points)), dim=1)
    keys, invalid = morton_keys_f32(points, torch.zeros(3, device="mps"), 1.0)
    empty_keys, empty_invalid = morton_keys_f32(points[:0], torch.zeros(3, device="mps"), 1.0)
    torch.mps.synchronize()
    assert keys.cpu().tolist() == [morton_key_cpu(i, 0, 0) for i in range(513)]
    assert not invalid.any().item()
    assert empty_keys.shape == empty_invalid.shape == (0,)

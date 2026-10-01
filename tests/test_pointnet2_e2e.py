"""Independent, compact PointNet++ SSG segmentation integration test.

The topology exercises four FPS/Ball Query set-abstraction blocks, four
three-neighbor feature-propagation blocks, a class head, and loss backward.
It is authored here rather than copied from the upstream PointNet++ model.
The separate pinned-upstream harness in tools/ validates the full original
layer implementation and architecture against its compiled CUDA extension.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mps_pointops import compat
from mps_pointops.pointnet2 import three_interpolate, three_nn


class SetAbstraction(nn.Module):
    def __init__(self, npoint: int, radius: float, in_channels: int, out_channels: int):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels + 3, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, xyz: torch.Tensor, features: torch.Tensor):
        sampled = compat.furthest_point_sample(xyz, self.npoint)
        new_xyz = compat.gather_operation(xyz.transpose(1, 2).contiguous(), sampled)
        new_xyz = new_xyz.transpose(1, 2).contiguous()
        neighbors = compat.ball_query(self.radius, 12, xyz, new_xyz)
        grouped_xyz = compat.grouping_operation(xyz.transpose(1, 2).contiguous(), neighbors)
        grouped_xyz = grouped_xyz - new_xyz.transpose(1, 2).unsqueeze(-1)
        grouped_features = compat.grouping_operation(features, neighbors)
        grouped = torch.cat((grouped_xyz, grouped_features), dim=1)
        return new_xyz, self.mlp(grouped).max(dim=-1).values, sampled, neighbors


class FeaturePropagation(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, target_xyz, source_xyz, target_features, source_features):
        distances, neighbors = three_nn(target_xyz, source_xyz)
        inverse = 1.0 / (distances + 1e-8)
        weights = inverse / inverse.sum(dim=-1, keepdim=True)
        interpolated = three_interpolate(source_features, neighbors, weights)
        joined = torch.cat((target_features, interpolated), dim=1)
        return self.mlp(joined), neighbors


class CompactSSGSegmentation(nn.Module):
    """Four-level SSG topology with small CI-friendly point/channel counts."""

    def __init__(self):
        super().__init__()
        self.sa = nn.ModuleList([
            SetAbstraction(64, 0.10, 6, 16),
            SetAbstraction(32, 0.20, 16, 24),
            SetAbstraction(16, 0.40, 24, 32),
            SetAbstraction(8, 0.80, 32, 48),
        ])
        self.fp = nn.ModuleList([
            FeaturePropagation(48 + 32, 32),
            FeaturePropagation(32 + 24, 24),
            FeaturePropagation(24 + 16, 16),
            FeaturePropagation(16 + 6, 16),
        ])
        self.head = nn.Conv1d(16, 13, 1)

    def forward(self, points: torch.Tensor):
        xyz = [points[..., :3].contiguous()]
        features = [points[..., 3:].transpose(1, 2).contiguous()]
        selections = []
        for layer in self.sa:
            new_xyz, new_features, sampled, neighbors = layer(xyz[-1], features[-1])
            xyz.append(new_xyz)
            features.append(new_features)
            selections.extend((sampled, neighbors))
        for i, layer in enumerate(self.fp):
            target = 3 - i
            propagated, neighbors = layer(
                xyz[target], xyz[target + 1], features[target], features[target + 1]
            )
            features[target] = propagated
            selections.append(neighbors)
        return self.head(features[0]), selections


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(20261002)
    xyz = rng.uniform(-0.15, 0.15, size=(1, 96, 3)).astype("float32")
    clusters = rng.integers(0, 3, size=(1, 96))
    offsets = np.array(((1.5, 1.5, 1.5), (1.9, 1.5, 1.5), (1.5, 1.9, 1.5)), dtype="float32")
    xyz += offsets[clusters]
    values = rng.uniform(-0.5, 0.5, size=(1, 96, 6)).astype("float32")
    labels = rng.integers(0, 13, size=(1, 96), dtype=np.int64)
    return torch.from_numpy(np.concatenate((xyz, values), axis=-1)), torch.from_numpy(labels)


def _run(device: str):
    # Model initialization must be identical on CPU and MPS. Eval mode fixes
    # BatchNorm and dropout policy while retaining the full loss backward.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261002)
        model = CompactSSGSegmentation().eval().to(device)
    points, labels = _fixture()
    points = points.to(device).requires_grad_(True)
    logits, selections = model(points)
    loss = F.cross_entropy(logits, labels.to(device))
    loss.backward()
    if device == "mps":
        torch.mps.synchronize()
    gradients = {name: p.grad.detach().cpu() for name, p in model.named_parameters()}
    assert len(gradients) > 0 and all(torch.isfinite(g).all() for g in gradients.values())
    return (
        logits.detach().cpu(), loss.detach().cpu(), points.grad.detach().cpu(),
        gradients, [s.detach().cpu() for s in selections],
    )


def test_compact_ssg_cpu_forward_and_loss_backward():
    logits, loss, input_grad, gradients, selections = _run("cpu")
    assert logits.shape == (1, 13, 96)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert input_grad.shape == (1, 96, 9) and torch.isfinite(input_grad).all()
    assert all(any(name.startswith(f"sa.{i}.") for name in gradients) for i in range(4))
    assert all(any(name.startswith(f"fp.{i}.") for name in gradients) for i in range(4))
    assert "head.weight" in gradients
    assert len(selections) == 12  # Four FPS, four Ball Query, four 3NN.


@pytest.mark.mps
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is unavailable")
def test_compact_ssg_mps_matches_cpu_forward_and_backward():
    cpu = _run("cpu")
    mps = _run("mps")
    for actual, expected in zip(mps[:3], cpu[:3]):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert mps[3].keys() == cpu[3].keys()
    for name in cpu[3]:
        torch.testing.assert_close(mps[3][name], cpu[3][name], atol=1e-4, rtol=1e-4, msg=name)
    for actual, expected in zip(mps[4], cpu[4]):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

import pytest
import torch

from mps_pointops import compat, flat


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_graph_wrappers_edge_orientation_and_self_loop(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
    x = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], device=device)
    forward = torch.tensor([[1, 0, 1], [0, 1, 2]])
    reverse = forward.flip(0)
    assert torch.equal(flat.knn_graph(x, 1).cpu(), forward)
    assert torch.equal(flat.knn_graph(x, 1, flow="target_to_source").cpu(), reverse)
    assert torch.equal(flat.radius_graph(x, 1.1, max_num_neighbors=1).cpu(), forward)
    assert torch.equal(
        flat.radius_graph(x, 1.1, max_num_neighbors=1, flow="target_to_source").cpu(),
        reverse,
    )
    assert flat.knn_graph(x, 1, loop=True).shape == (2, len(x))
    assert flat.radius_graph(x, 0.5, loop=True).shape == (2, len(x))


def test_pyg_import_only_stubs_report_unsupported_operations():
    import sys

    original = {name: sys.modules.get(name) for name in (
        "pointnet2_ops", "pointnet2_ops.pointnet2_utils", "knn_cuda", "torch_cluster"
    )}
    names = compat.install(force=True)
    try:
        from torch_cluster import graclus_cluster, grid_cluster, knn_graph

        assert knn_graph is flat.knn_graph
        assert torch.equal(
            grid_cluster(torch.zeros((1, 3)), torch.ones(3)),
            torch.zeros(1, dtype=torch.long),
        )
        with pytest.raises(NotImplementedError, match="graclus_cluster"):
            graclus_cluster(torch.empty(2, 0, dtype=torch.long))
    finally:
        for name in names:
            if original[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original[name]

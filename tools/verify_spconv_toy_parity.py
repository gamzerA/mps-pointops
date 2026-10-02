"""Bounded direct parity probe against spconv 2.3.8 on a CUDA machine.

This is an experimental operator probe, not a package compatibility test. It
compares two tiny SubM fixtures and one keyed stride/inverse chain with the
private CPU references. Coordinates are matched by key, independent of row
order. No full point cloud, model, or benchmark is allocated.

Run from the repository root in an environment with CUDA PyTorch and the
matching spconv CUDA wheel::

    python -m tools.verify_spconv_toy_parity --backend cuda --json toy.json

The ``mps`` backend tests only the already implemented SubM Metal operation
against the same CPU fixture. ``cpu`` checks fixture/reference construction
without importing spconv. A failure exits nonzero and remains visible in the
JSON report; it is never reclassified as a tolerated numerical difference.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mps_pointops._sparse_conv_cpu import (
    _evaluate_pairs,
    sparse_conv3d_forward_cpu,
    sparse_inverse_conv3d_forward_cpu,
)
from mps_pointops._sparse_rulebook import (
    generate_subm_rulebook,
    validate_sparse_tensor_3d,
)


RTOL = 1e-4
ATOL = 1e-5
SPCONV_VERSION = "2.3.8"


@dataclass(frozen=True)
class Fixture:
    name: str
    coordinates: tuple[tuple[int, int, int, int], ...]
    spatial_shape: tuple[int, int, int]
    batch_size: int
    kernel_size: tuple[int, int, int] = (3, 3, 3)
    dilation: tuple[int, int, int] = (1, 1, 1)


SUBM_FIXTURES = (
    Fixture(
        "subm_unsorted_batched_boundary",
        ((1, 3, 1, 1), (0, 0, 0, 0), (0, 1, 1, 1),
         (1, 1, 1, 1), (0, 2, 1, 1), (1, 2, 1, 1), (0, 3, 1, 1)),
        (4, 3, 3), 2,
    ),
    Fixture(
        "subm_dilated_isolated",
        ((0, 4, 2, 1), (0, 0, 2, 1), (0, 2, 2, 1), (1, 0, 0, 0)),
        (5, 3, 3), 2, (3, 1, 1), (2, 1, 1),
    ),
)

STRIDE_FIXTURE = Fixture(
    "stride2_inverse_batched",
    ((1, 5, 2, 2), (0, 0, 0, 0), (0, 1, 1, 1),
     (1, 1, 1, 1), (0, 2, 1, 1), (1, 2, 1, 1),
     (0, 5, 2, 2), (0, 3, 1, 1)),
    (6, 4, 4), 2,
)


def _values(shape: tuple[int, ...], *, scale: int, phase: int) -> torch.Tensor:
    count = 1
    for dim in shape:
        count *= dim
    # Values are small signed dyadic fractions. They exercise nonzero bias and
    # multiple contributing pairs without large reductions or cancellation.
    return ((torch.arange(count, dtype=torch.int32) * 7 + phase) % 29 - 14).to(
        torch.float32
    ).reshape(shape) / scale


def _canonical(indices: torch.Tensor, features: torch.Tensor):
    coordinates = [tuple(int(v) for v in row) for row in indices.cpu().tolist()]
    if len(set(coordinates)) != len(coordinates):
        raise AssertionError("the operator returned duplicate output coordinates")
    order = sorted(range(len(coordinates)), key=coordinates.__getitem__)
    permutation = torch.tensor(order, dtype=torch.int64, device=features.device)
    return [coordinates[i] for i in order], features.index_select(0, permutation)


def _aligned(reference_indices, reference_values, observed_indices, observed_values):
    reference_keys, reference = _canonical(reference_indices, reference_values)
    observed_keys, observed = _canonical(observed_indices, observed_values)
    if reference_keys != observed_keys:
        missing = sorted(set(reference_keys) - set(observed_keys))
        extra = sorted(set(observed_keys) - set(reference_keys))
        raise AssertionError(f"coordinate mismatch: missing={missing}, extra={extra}")
    return reference_keys, reference, observed


def _upstream(rows: int, channels: int) -> torch.Tensor:
    return _values((rows, channels), scale=16, phase=11)


def _compare(name: str, observed: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    observed = observed.detach().cpu()
    expected = expected.detach().cpu()
    if observed.shape != expected.shape:
        raise AssertionError(f"{name}: shape {tuple(observed.shape)} != {tuple(expected.shape)}")
    difference = (observed - expected).abs()
    maximum = float(difference.max()) if difference.numel() else 0.0
    torch.testing.assert_close(observed, expected, rtol=RTOL, atol=ATOL, msg=name)
    return {"name": name, "shape": list(expected.shape), "max_abs_error": maximum}


def _native_layout(weight: torch.Tensor, target_shape: torch.Size):
    # spconv 2.3.8 selects layout by algorithm/build. Shape detects the actual
    # installed binary instead of assuming a wheel-wide layout. Fixtures use
    # Cin=2, Cout=4 (inverse: Cin=4, Cout=2), making all three shapes distinct.
    candidates = {
        "KRSC": weight.permute(0, 2, 3, 4, 1),
        "RSKC": weight.permute(2, 3, 4, 0, 1),
        "RSCK": weight.permute(2, 3, 4, 1, 0),
    }
    matches = [(name, value) for name, value in candidates.items()
               if tuple(value.shape) == tuple(target_shape)]
    if len(matches) != 1:
        raise AssertionError(
            f"cannot uniquely map Conv3d weights {tuple(weight.shape)} "
            f"to spconv {tuple(target_shape)}; matches={[name for name, _ in matches]}"
        )
    name, value = matches[0]
    return name, value.contiguous()


def _from_native_layout(value: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "KRSC":
        return value.permute(0, 4, 1, 2, 3)
    if layout == "RSKC":
        return value.permute(3, 4, 0, 1, 2)
    if layout == "RSCK":
        return value.permute(4, 3, 0, 1, 2)
    raise AssertionError(f"unknown weight layout {layout}")


def _install_weights(module, weight: torch.Tensor, bias: torch.Tensor) -> str:
    layout, mapped = _native_layout(weight, module.weight.shape)
    with torch.no_grad():
        module.weight.copy_(mapped.to(module.weight.device))
        module.bias.copy_(bias.to(module.bias.device))
    return layout


def _run_subm(fixture: Fixture, backend: str, spconv=None, conv_algo=None):
    indices = torch.tensor(fixture.coordinates, dtype=torch.int32)
    features = _values((len(indices), 2), scale=8, phase=3).requires_grad_()
    weights = _values((4, 2, *fixture.kernel_size), scale=64, phase=5).requires_grad_()
    bias = _values((4,), scale=8, phase=13).requires_grad_()
    sparse = validate_sparse_tensor_3d(
        indices, features, fixture.spatial_shape, fixture.batch_size
    )
    rulebook = generate_subm_rulebook(
        sparse, kernel_size=fixture.kernel_size, dilation=fixture.dilation
    )
    expected = _evaluate_pairs(
        features, weights, bias, rulebook.pairs, len(indices), inverse=False
    )
    device = torch.device(backend)
    observed_features = features.detach().to(device).requires_grad_()
    results = []
    if backend == "mps":
        from mps_pointops._subm_conv_mps import subm_conv3d_forward_mps

        observed_weights = weights.detach().to(device).requires_grad_()
        observed_bias = bias.detach().to(device).requires_grad_()
        observed = subm_conv3d_forward_mps(
            indices, observed_features, observed_weights,
            fixture.spatial_shape, fixture.batch_size,
            dilation=fixture.dilation, bias=observed_bias,
        )
        observed_indices = indices
        layout = "PyTorch Conv3d"
    else:
        layer = spconv.SubMConv3d(
            2, 4, fixture.kernel_size, dilation=fixture.dilation,
            bias=True, algo=conv_algo,
        ).to(device)
        layout = _install_weights(layer, weights, bias)
        observed = layer(spconv.SparseConvTensor(
            observed_features, indices.to(device),
            fixture.spatial_shape, fixture.batch_size,
        ))
        observed_indices, observed = observed.indices, observed.features
        observed_weights, observed_bias = layer.weight, layer.bias

    keys, expected_sorted, observed_sorted = _aligned(
        indices, expected, observed_indices, observed
    )
    results.append(_compare("forward", observed_sorted, expected_sorted))
    upstream = _upstream(len(keys), 4)
    cpu_grads = torch.autograd.grad(
        (expected_sorted * upstream).sum(), (features, weights, bias)
    )
    (observed_sorted * upstream.to(device)).sum().backward()
    if backend == "cuda":
        torch.cuda.synchronize()
        weight_grad = _from_native_layout(observed_weights.grad, layout)
    else:
        torch.mps.synchronize()
        weight_grad = observed_weights.grad
    for name, observed_grad, expected_grad in zip(
        ("features_grad", "weights_grad", "bias_grad"),
        (observed_features.grad, weight_grad, observed_bias.grad),
        cpu_grads,
    ):
        results.append(_compare(name, observed_grad, expected_grad))
    return {"fixture": fixture.name, "indices": len(indices),
            "pairs": len(rulebook.pairs), "weight_layout": layout,
            "comparisons": results}


def _run_stride_inverse(spconv, conv_algo):
    fixture = STRIDE_FIXTURE
    indices = torch.tensor(fixture.coordinates, dtype=torch.int32)
    features = _values((len(indices), 2), scale=8, phase=9).requires_grad_()
    down_weight = _values((4, 2, 3, 3, 3), scale=64, phase=15).requires_grad_()
    down_bias = _values((4,), scale=8, phase=17).requires_grad_()
    up_weight = _values((2, 4, 3, 3, 3), scale=64, phase=19).requires_grad_()
    up_bias = _values((2,), scale=8, phase=21).requires_grad_()
    sparse = validate_sparse_tensor_3d(
        indices, features, fixture.spatial_shape, fixture.batch_size
    )
    cache = {}
    down = sparse_conv3d_forward_cpu(
        sparse, down_weight, stride=2, padding=1, bias=down_bias,
        indice_key="toy_down", indice_cache=cache,
    )
    up = sparse_inverse_conv3d_forward_cpu(
        down, up_weight, bias=up_bias,
        indice_key="toy_down", indice_cache=cache,
    )
    device = torch.device("cuda")
    observed_features = features.detach().to(device).requires_grad_()
    down_layer = spconv.SparseConv3d(
        2, 4, 3, stride=2, padding=1, bias=True,
        indice_key="toy_down", algo=conv_algo,
    ).to(device)
    up_layer = spconv.SparseInverseConv3d(
        4, 2, 3, bias=True, indice_key="toy_down", algo=conv_algo,
    ).to(device)
    down_layout = _install_weights(down_layer, down_weight, down_bias)
    up_layout = _install_weights(up_layer, up_weight, up_bias)
    observed_down = down_layer(spconv.SparseConvTensor(
        observed_features, indices.to(device),
        fixture.spatial_shape, fixture.batch_size,
    ))
    observed_up = up_layer(observed_down)
    if tuple(observed_down.spatial_shape) != down.spatial_shape:
        raise AssertionError("ordinary output spatial shape differs")
    if tuple(observed_up.spatial_shape) != up.spatial_shape:
        raise AssertionError("inverse output spatial shape differs")
    down_keys, expected_down, actual_down = _aligned(
        down.indices, down.features, observed_down.indices, observed_down.features
    )
    up_keys, expected_up, actual_up = _aligned(
        up.indices, up.features, observed_up.indices, observed_up.features
    )
    comparisons = [
        _compare("stride_forward", actual_down, expected_down),
        _compare("inverse_forward", actual_up, expected_up),
    ]
    # A combined loss exercises both the direct downsample path and the
    # inverse path, including all five first-order gradient tensors.
    down_upstream = _upstream(len(down_keys), 4)
    up_upstream = _upstream(len(up_keys), 2)
    cpu_loss = (expected_down * down_upstream).sum() + (
        expected_up * up_upstream
    ).sum()
    cpu_grads = torch.autograd.grad(
        cpu_loss, (features, down_weight, down_bias, up_weight, up_bias)
    )
    gpu_loss = (actual_down * down_upstream.to(device)).sum() + (
        actual_up * up_upstream.to(device)
    ).sum()
    gpu_loss.backward()
    torch.cuda.synchronize()
    observed_grads = (
        observed_features.grad,
        _from_native_layout(down_layer.weight.grad, down_layout),
        down_layer.bias.grad,
        _from_native_layout(up_layer.weight.grad, up_layout),
        up_layer.bias.grad,
    )
    for name, actual, expected in zip(
        ("features_grad", "stride_weights_grad", "stride_bias_grad",
         "inverse_weights_grad", "inverse_bias_grad"),
        observed_grads, cpu_grads,
    ):
        comparisons.append(_compare(name, actual, expected))
    return {"fixture": fixture.name, "input_indices": len(indices),
            "down_indices": len(down_keys), "inverse_indices": len(up_keys),
            "down_weight_layout": down_layout, "inverse_weight_layout": up_layout,
            "comparisons": comparisons}


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cuda", "mps", "cpu"), default="cuda")
    parser.add_argument("--json", type=Path, help="write results, including failures")
    parser.add_argument("--cuda-memory-fraction", type=float, default=0.25)
    args = parser.parse_args()
    if not 0 < args.cuda_memory_fraction <= 0.5:
        parser.error("--cuda-memory-fraction must be in (0, 0.5]")
    if args.backend == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA PyTorch is required for the spconv comparison")
    if args.backend == "mps" and not torch.backends.mps.is_available():
        parser.error("an available MPS backend is required")

    report: dict[str, Any] = {
        "source_commit": _git_head(), "backend": args.backend,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "python": sys.version.split()[0], "platform": platform.platform(),
        "tolerances": {"rtol": RTOL, "atol": ATOL},
        "scope": "two tiny SubM fixtures plus one keyed stride/inverse chain on CUDA; "
                 "SubM only on MPS; fixture construction only on CPU",
        "cases": [], "failures": [],
    }
    spconv = None
    conv_algo = None
    if args.backend == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction)
        import spconv as spconv_root
        import spconv.pytorch as spconv
        from spconv.core import ConvAlgo

        report["spconv"] = spconv_root.__version__
        report["gpu"] = torch.cuda.get_device_name()
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["cuda_memory_fraction_limit"] = args.cuda_memory_fraction
        if spconv_root.__version__ != SPCONV_VERSION:
            raise RuntimeError(
                f"expected pinned spconv {SPCONV_VERSION}, found {spconv_root.__version__}"
            )
        conv_algo = ConvAlgo.Native
        torch.cuda.reset_peak_memory_stats()
    elif args.backend == "mps":
        report["gpu"] = "Apple MPS"

    for fixture in SUBM_FIXTURES:
        try:
            if args.backend == "cpu":
                indices = torch.tensor(fixture.coordinates, dtype=torch.int32)
                features = _values((len(indices), 2), scale=8, phase=3)
                sparse = validate_sparse_tensor_3d(
                    indices, features, fixture.spatial_shape, fixture.batch_size
                )
                pairs = generate_subm_rulebook(
                    sparse, kernel_size=fixture.kernel_size,
                    dilation=fixture.dilation,
                ).pairs
                case = {"fixture": fixture.name, "indices": len(indices),
                        "pairs": len(pairs), "status": "fixture_only"}
            else:
                case = _run_subm(fixture, args.backend, spconv, conv_algo)
                case["status"] = "pass"
            report["cases"].append(case)
        except Exception as exc:
            report["failures"].append({"fixture": fixture.name,
                                       "error": f"{type(exc).__name__}: {exc}"})
    if args.backend == "cuda":
        try:
            case = _run_stride_inverse(spconv, conv_algo)
            case["status"] = "pass"
            report["cases"].append(case)
        except Exception as exc:
            report["failures"].append({"fixture": STRIDE_FIXTURE.name,
                                       "error": f"{type(exc).__name__}: {exc}"})
        report["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())

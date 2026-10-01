"""Standalone, non-asserting probe for an observed MPS Linear bias discrepancy.

This file intentionally imports only PyTorch and Python's standard library. It
compares a fixed 4x4 input, 3x4 weight, and length-3 bias on CPU and MPS in a
fresh process. A numerical discrepancy is diagnostic output, not a test failure.

Run with ``PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/diagnose_mps_linear.py``.
Use ``--output report.json`` to save the same JSON that is printed to stdout.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def _cpu_brand() -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        return subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_snapshot(value: torch.Tensor) -> torch.Tensor:
    # The MPS command queue is asynchronous. Synchronize before every readback
    # so the report cannot accidentally compare unfinished work.
    if value.device.type == "mps":
        torch.mps.synchronize()
    return value.detach().cpu().clone()


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _tensor(value: torch.Tensor) -> dict[str, object]:
    cpu = _cpu_snapshot(value)
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "norm_l2": float(torch.linalg.vector_norm(cpu).item()),
        "values": cpu.tolist(),
    }


def _evaluate(device: str, model_cpu: torch.nn.Linear, x_cpu: torch.Tensor) -> dict[str, object]:
    model = copy.deepcopy(model_cpu).to(device)
    x = x_cpu.to(device)
    weight = model.weight
    bias = model.bias
    assert bias is not None

    mm = x @ weight.T
    explicit = mm + bias
    addmm = torch.addmm(bias, x, weight.T)
    functional = F.linear(x, weight, bias)
    module = model(x)

    values = {
        "x": _cpu_snapshot(x),
        "weight": _cpu_snapshot(weight),
        "bias": _cpu_snapshot(bias),
        "mm": _cpu_snapshot(mm),
        "explicit_mm_plus_bias": _cpu_snapshot(explicit),
        "addmm": _cpu_snapshot(addmm),
        "functional_linear": _cpu_snapshot(functional),
        "module_linear": _cpu_snapshot(module),
    }
    return {
        "device": device,
        "tensors": {key: _tensor(value) for key, value in values.items()},
        "max_abs": {
            "addmm_minus_explicit": _max_abs(values["addmm"], values["explicit_mm_plus_bias"]),
            "functional_minus_explicit": _max_abs(
                values["functional_linear"], values["explicit_mm_plus_bias"]
            ),
            "functional_minus_mm": _max_abs(values["functional_linear"], values["mm"]),
            "module_minus_explicit": _max_abs(
                values["module_linear"], values["explicit_mm_plus_bias"]
            ),
            "module_minus_mm": _max_abs(values["module_linear"], values["mm"]),
            "module_minus_functional": _max_abs(
                values["module_linear"], values["functional_linear"]
            ),
        },
        "module_minus_explicit_elements": (
            values["module_linear"] - values["explicit_mm_plus_bias"]
        ).tolist(),
        "functional_minus_explicit_elements": (
            values["functional_linear"] - values["explicit_mm_plus_bias"]
        ).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write a copy of the JSON report")
    args = parser.parse_args()

    report: dict[str, object] = {
        "probe": "fixed-linear-bias-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "macos_version": platform.mac_ver()[0],
            "machine": platform.machine(),
            "cpu_brand": _cpu_brand(),
            "torch": torch.__version__,
            "mps_built": torch.backends.mps.is_built(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "mps_fast_math": os.environ.get("PYTORCH_MPS_FAST_MATH"),
            "github_runner_name": os.environ.get("RUNNER_NAME"),
            "github_runner_os": os.environ.get("RUNNER_OS"),
            "github_runner_arch": os.environ.get("RUNNER_ARCH"),
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "source_commit": os.environ.get("MPS_POINTOPS_SOURCE_SHA", os.environ.get("GITHUB_SHA")),
        },
    }

    x = torch.arange(16, dtype=torch.float32).reshape(4, 4) / 7
    model = torch.nn.Linear(4, 3)
    with torch.no_grad():
        model.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 19)
        model.bias.copy_(torch.tensor([-0.25, 0.5, 0.75], dtype=torch.float32))

    cpu = _evaluate("cpu", model, x)
    report["cpu"] = cpu
    if torch.backends.mps.is_available():
        mps = _evaluate("mps", model, x)
        report["mps"] = mps
        cpu_tensors = cpu["tensors"]
        mps_tensors = mps["tensors"]
        assert isinstance(cpu_tensors, dict) and isinstance(mps_tensors, dict)
        report["cpu_mps_max_abs"] = {
            key: _max_abs(
                torch.tensor(cpu_tensors[key]["values"]),
                torch.tensor(mps_tensors[key]["values"]),
            )
            for key in cpu_tensors
        }
        report["status"] = "completed"
    else:
        report["status"] = "mps_unavailable"

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()

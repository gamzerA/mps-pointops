"""Survey PyG GCN, GraphSAGE, and GAT on MPS without PyTorch CPU fallback.

Run in a process whose environment already contains
``PYTORCH_ENABLE_MPS_FALLBACK=0``. Each model/device pair runs in a fresh child
process so that a failed operator cannot contaminate the next observation.
No dataset download or optional torch-cluster/pyg-lib operator is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MODELS = ("gcn", "graphsage", "gat")
DEVICES = ("cpu", "mps")
RESULT_MARKER = "@@PYG_MPS_SURVEY_RESULT@@"
OPERATOR_RE = re.compile(r"(?:aten|pyg|torch_scatter|torch_sparse)::[A-Za-z0-9_.]+")
HOME = str(Path.home())


def _redact(value: str) -> str:
    return value.replace(HOME, "<HOME>")


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _hardware() -> dict[str, str | None]:
    info: dict[str, str | None] = {
        "platform": platform.platform(),
        "macos": platform.mac_ver()[0],
        "machine": platform.machine(),
        "chip": None,
        "model": None,
    }
    raw = _command_output(["system_profiler", "SPHardwareDataType", "-json"])
    if raw:
        try:
            hardware = json.loads(raw).get("SPHardwareDataType", [{}])[0]
            info["chip"] = hardware.get("chip_type")
            info["model"] = hardware.get("machine_model")
        except (ValueError, IndexError, TypeError):
            pass
    return info


def _package_versions() -> dict[str, str | None]:
    names = (
        "torch", "torch-geometric", "pyg-lib", "torch-cluster", "torch-scatter",
        "torch-sparse", "numpy", "aiohttp", "xxhash",
    )
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _failure(exc: BaseException) -> dict[str, Any]:
    message = _redact(str(exc))
    trace = _redact(traceback.format_exc())
    operators = sorted(set(OPERATOR_RE.findall(message)))
    return {
        "status": "failed",
        "exception_type": type(exc).__name__,
        "message": message,
        "operators_mentioned_in_message": operators,
        "traceback": trace,
    }


def _child_metadata() -> dict[str, Any]:
    import torch
    import torch_geometric

    return {
        "status": "passed",
        "python": platform.python_version(),
        "package_versions": _package_versions(),
        "torch_version_imported": torch.__version__,
        "pyg_version_imported": torch_geometric.__version__,
        "mps_is_built": torch.backends.mps.is_built(),
        "mps_is_available": torch.backends.mps.is_available(),
        "fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }


def _child_model(model_name: str, device_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model_name,
        "device": device_name,
        "forward": {"status": "not_run"},
        "backward": {"status": "not_run"},
        "overall_status": "not_run",
    }
    try:
        import torch
        from torch_geometric.nn.models import GAT, GCN, GraphSAGE

        if device_name == "mps" and not torch.backends.mps.is_available():
            result["reason"] = "torch.backends.mps.is_available() is False"
            return result

        torch.manual_seed(2026)
        device = torch.device(device_name)
        # Bidirectional ring, a few chords, and deliberately repeated target
        # indices exercise aggregation without external graph data.
        src = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 2, 4, 6, 8, 10]
        dst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 6, 8, 10, 0, 2, 4]
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.int64, device=device)
        x = (torch.arange(36, dtype=torch.float32, device=device).reshape(12, 3) / 10).requires_grad_()
        model = {
            "gcn": lambda: GCN(3, 4, num_layers=2, out_channels=4, dropout=0.0),
            "graphsage": lambda: GraphSAGE(3, 4, num_layers=2, out_channels=4, dropout=0.0),
            "gat": lambda: GAT(3, 4, num_layers=2, out_channels=4, dropout=0.0),
        }[model_name]().to(device)

        try:
            output = model(x, edge_index)
            if device_name == "mps":
                torch.mps.synchronize()
            if output.device.type != device_name:
                raise AssertionError(f"output moved to {output.device}, expected {device}")
            if tuple(output.shape) != (12, 4):
                raise AssertionError(f"unexpected output shape: {tuple(output.shape)}")
            if not bool(torch.isfinite(output).all().item()):
                raise AssertionError("non-finite forward output")
            result["forward"] = {"status": "passed", "output_shape": [12, 4]}
        except Exception as exc:
            result["forward"] = _failure(exc)
            result["backward"]["reason"] = "forward failed"
            result["overall_status"] = "failed"
            return result

        try:
            loss = output.square().mean()
            loss.backward()
            if device_name == "mps":
                torch.mps.synchronize()
            if x.grad is None or x.grad.device.type != device_name or not bool(torch.isfinite(x.grad).all().item()):
                raise AssertionError("missing or non-finite feature gradient")
            if any(
                p.grad is None or p.grad.device.type != device_name or not bool(torch.isfinite(p.grad).all().item())
                for p in model.parameters()
            ):
                raise AssertionError("missing or non-finite parameter gradient")
            result["backward"] = {"status": "passed", "feature_gradient_shape": [12, 3]}
            result["overall_status"] = "passed"
        except Exception as exc:
            result["backward"] = _failure(exc)
            result["overall_status"] = "failed"
    except Exception as exc:
        result["setup_failure"] = _failure(exc)
        result["reason"] = "model construction or data setup failed"
        result["overall_status"] = "failed"
    return result


def _run_child(args: argparse.Namespace) -> None:
    if args.child == "metadata":
        try:
            result = _child_metadata()
        except Exception as exc:
            result = _failure(exc)
    else:
        result = _child_model(args.model, args.device)
    print(RESULT_MARKER + json.dumps(result, sort_keys=True))


def _spawn_child(arguments: list[str], timeout: int) -> tuple[dict[str, Any], str]:
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    command = [sys.executable, str(Path(__file__).resolve()), *arguments]
    try:
        proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        return ({"status": "failed", "exception_type": "TimeoutExpired", "message": str(exc)}, "")
    raw = _redact(proc.stdout + proc.stderr)
    marked = [line[len(RESULT_MARKER) :] for line in proc.stdout.splitlines() if line.startswith(RESULT_MARKER)]
    if len(marked) == 1:
        try:
            result = json.loads(marked[0])
            result["process_exit_code"] = proc.returncode
            return result, raw
        except json.JSONDecodeError:
            pass
    return (
        {"status": "failed", "exception_type": "ChildProcessError", "message": f"child exit {proc.returncode}; no valid result marker"},
        raw,
    )


def _markdown(report: dict[str, Any]) -> str:
    metadata = report["environment"]
    versions = metadata.get("package_versions", {})
    rows = []
    for model in MODELS:
        for device in DEVICES:
            case = report["cases"][f"{model}/{device}"]
            rows.append(
                f"| {model} | {device} | {case['forward']['status']} | "
                f"{case['backward']['status']} | {case['overall_status']} |"
            )
    failures = []
    for key, case in report["cases"].items():
        for stage in ("setup_failure", "forward", "backward"):
            observation = case.get(stage, {})
            if observation.get("status") == "failed":
                failures.append(
                    f"- **{key}, {stage}:** `{observation.get('exception_type')}` — "
                    f"{observation.get('message', '').replace(chr(10), ' ')}"
                )
    if not failures:
        failures = ["- None in these six model/device executions."]
    hardware = report["hardware"]
    return "\n".join(
        [
            "# PyG 2.8.0 MPS operator survey",
            "",
            f"Captured: {report['captured_utc']} (UTC); repository base: `{report['repository_base_commit']}`; "
            f"script SHA-256: `{report['script_sha256']}`.",
            "",
            "## Environment and method",
            "",
            f"- Hardware: {hardware.get('chip') or 'unknown chip'}, {hardware.get('model') or 'unknown model'}; "
            f"macOS {hardware.get('macos') or 'unknown'} ({hardware.get('machine')}).",
            f"- Python {metadata.get('python')}; PyTorch {versions.get('torch')}; PyG {versions.get('torch-geometric')}; "
            f"MPS built/available: {metadata.get('mps_is_built')}/{metadata.get('mps_is_available')}.",
            f"- Optional packages: pyg-lib {versions.get('pyg-lib')}, torch-cluster {versions.get('torch-cluster')}, "
            f"torch-scatter {versions.get('torch-scatter')}, torch-sparse {versions.get('torch-sparse')} "
            "(`None` means not installed).",
            "- `PYTORCH_ENABLE_MPS_FALLBACK=0` was set before every child imported PyTorch. "
            "Each model/device pair ran in a separate process. CPU is a control, not an MPS fallback.",
            "- A fixed 12-node bidirectional synthetic graph (36 directed edges), 3 float32 features, "
            "two-layer PyG model, squared-mean loss, forward and backward. The runner synchronizes "
            "MPS after each stage and checks output and gradients for finiteness.",
            "",
            "## Results",
            "",
            "| Model | Device | Forward | Backward | Overall |",
            "| --- | --- | --- | --- | --- |",
            *rows,
            "",
            "`not_run` means no execution occurred; `failed` means an exception or invalid result. "
            "A `passed` result establishes only this shape, dtype, graph, and package set.",
            "",
            "## Exact failures",
            "",
            *failures,
            "",
            "The adjacent JSON retains full exception messages and stack traces; the `.log` file "
            "retains child stdout/stderr with the home directory redacted. No absence of an MPS operator "
            "is inferred from an unrun case.",
            "",
            "## Reproduce",
            "",
            "Install PyTorch with MPS support and `torch-geometric==2.8.0` in a Python environment, then run:",
            "",
            "```bash",
            "PYTORCH_ENABLE_MPS_FALLBACK=0 python bench/survey_pyg_mps.py --output-prefix docs/pyg-survey/local-pyg28",
            "```",
            "",
            "[GCN](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.nn.models.GCN.html), "
            "[GraphSAGE](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.nn.models.GraphSAGE.html), "
            "and [GAT](https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.nn.models.GAT.html) "
            "are the three PyG 2.8.0 models exercised here. This is a survey probe, not an exhaustive "
            "inventory of PyG models, graph formats, dtypes, aggregation choices, or hardware.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-prefix", type=Path, help="write .json, .md, and .log files")
    parser.add_argument("--expect-pyg-version", default="2.8.0")
    parser.add_argument("--timeout", type=int, default=120, help="seconds per child process")
    parser.add_argument("--child", choices=("metadata", "model"), help=argparse.SUPPRESS)
    parser.add_argument("--model", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--device", choices=DEVICES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        parser.error("set PYTORCH_ENABLE_MPS_FALLBACK=0 before starting Python")
    if args.child:
        if args.child == "model" and (args.model is None or args.device is None):
            parser.error("internal model child requires --model and --device")
        _run_child(args)
        return

    environment, environment_log = _spawn_child(["--child", "metadata"], args.timeout)
    actual_pyg = environment.get("package_versions", {}).get("torch-geometric")
    if environment.get("status") != "passed" or actual_pyg != args.expect_pyg_version:
        raise SystemExit(
            f"PyG preflight failed or version mismatch: expected {args.expect_pyg_version}, "
            f"found {actual_pyg}; details: {environment.get('message', environment_log)}"
        )

    cases: dict[str, Any] = {}
    logs = ["# PyG MPS survey raw child output", "", "## metadata", environment_log]
    for model in MODELS:
        for device in DEVICES:
            key = f"{model}/{device}"
            case, raw = _spawn_child(["--child", "model", "--model", model, "--device", device], args.timeout)
            if "overall_status" not in case:
                case = {
                    "model": model,
                    "device": device,
                    "forward": {"status": "not_run"},
                    "backward": {"status": "not_run"},
                    "overall_status": "failed",
                    "process_failure": case,
                }
            cases[key] = case
            logs.extend([f"## {key}", raw])
            print(f"{key}: {case['forward']['status']} / {case['backward']['status']}")

    script = Path(__file__).resolve()
    report = {
        "schema_version": 1,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "repository_base_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "hardware": _hardware(),
        "environment": environment,
        "survey_input": {
            "node_count": 12,
            "directed_edge_count": 36,
            "feature_dimension": 3,
            "feature_dtype": "float32",
            "edge_index_dtype": "int64",
            "model_layers": 2,
            "loss": "output.square().mean()",
        },
        "cases": cases,
    }
    if args.output_prefix:
        args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        args.output_prefix.with_suffix(".md").write_text(_markdown(report))
        args.output_prefix.with_suffix(".log").write_text("\n".join(logs).rstrip() + "\n")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

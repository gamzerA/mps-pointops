"""Run source-pinned sparse MPS gates on a physical Apple M1 Mac.

This script does not install dependencies, alter global settings, or contact a
remote host. It runs Safe/Fast in separate local processes and writes evidence
only to the caller's chosen output directory outside the checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SUITES = (
    "tests/test_sparse_rulebook.py",
    "tests/test_sparse_conv_cpu.py",
    "tests/test_subm_rulebook_mps.py",
    "tests/test_subm_metal.py",
    "tests/test_sparse_conv_mps.py",
    "tests/test_spconv_compat.py",
)
_SHA40 = re.compile(r"[0-9a-f]{40}\Z")
_PASS = re.compile(r"\b(\d+) passed\b")
_NONPASS = re.compile(r"\b(\d+) (failed|error|errors|skipped|xfailed|xpassed|deselected)\b")


class GateError(RuntimeError):
    """An environment, source-integrity, or test gate failed."""


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=False,
        timeout=30,
    )
    if result.returncode:
        raise GateError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_checkout(root: Path, expected_commit: str) -> None:
    """Reject abbreviated/mismatched commits and any tracked/untracked edits."""
    if not _SHA40.fullmatch(expected_commit):
        raise GateError("--commit must be an exact 40-character lowercase SHA-1")
    actual = _git(root, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise GateError(f"source commit mismatch: expected {expected_commit}, found {actual}")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise GateError("source checkout is not clean; commit or remove local changes first")


def validate_hardware(system: str, machine: str, cpu_brand: str) -> None:
    m1_family = cpu_brand == "Apple M1" or cpu_brand.startswith("Apple M1 ")
    if system != "darwin" or machine != "arm64" or not m1_family:
        raise GateError(
            "this evidence runner requires an Apple M1-family Mac "
            f"(found {system}/{machine}, {cpu_brand!r})"
        )


def validate_torch_version(version: str) -> None:
    match = re.match(r"^(\d+)\.(\d+)(?:\.|\+|$)", version)
    if match is None or tuple(map(int, match.groups())) < (2, 7):
        raise GateError(f"this checkout requires PyTorch 2.7 or later; found {version!r}")


def parse_pytest_summary(output: str, returncode: int) -> int:
    """A zero exit alone is insufficient: even one skip invalidates the gate."""
    if returncode:
        raise GateError(f"pytest exited with status {returncode}")
    counts = [int(match.group(1)) for match in _PASS.finditer(output)]
    if not counts or counts[-1] < 1:
        raise GateError("pytest did not report at least one passing test")
    nonpass = [(int(n), kind) for n, kind in _NONPASS.findall(output) if int(n)]
    if nonpass:
        raise GateError(f"pytest reported non-passing outcomes: {nonpass}")
    return counts[-1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(root: Path) -> dict[str, str]:
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "mps_pointops", "tests", "tools/run_m1_sparse_validation.py"],
        cwd=root, capture_output=True, check=True, timeout=30,
    ).stdout.split(b"\0")
    names = sorted(
        Path(raw.decode("utf-8")) for raw in tracked if raw and
        Path(raw.decode("utf-8")).suffix in {".py", ".metal"}
    )
    return {name.as_posix(): _sha256(root / name) for name in names}


def _run_logged(command: list[str], *, root: Path, env: dict[str, str],
                destination: Path, timeout_s: int) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            command, cwd=root, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, check=False, timeout=timeout_s,
        )
        destination.write_text(result.stdout, encoding="utf-8")
        return result
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout or b""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        destination.write_text(partial + f"\nTIMEOUT after {timeout_s}s\n", encoding="utf-8")
        raise GateError(f"{destination.name} timed out after {timeout_s}s") from exc


def _environment(root: Path, env: dict[str, str], output: Path,
                 timeout_s: int) -> dict:
    probe = (
        "import json,torch; "
        "mps=torch.backends.mps; available=mps.is_available(); "
        "get_name=getattr(mps,'get_name',None); "
        "print(json.dumps({'torch':torch.__version__,"
        "'mps_available':available,"
        "'mps_device':get_name() if available and callable(get_name) else None}))"
    )
    result = _run_logged(
        [sys.executable, "-c", probe], root=root, env=env,
        destination=output, timeout_s=timeout_s,
    )
    if result.returncode:
        raise GateError("PyTorch/MPS environment probe failed")
    try:
        # Some torch builds print a one-line runtime warning before JSON.
        info = json.loads(result.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        raise GateError("PyTorch/MPS environment probe did not emit JSON") from exc
    if not info.get("mps_available"):
        raise GateError("PyTorch MPS backend is unavailable")
    validate_torch_version(str(info.get("torch", "")))
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True, help="exact full HEAD SHA-1")
    parser.add_argument("--output", required=True, type=Path,
                        help="new evidence directory outside this checkout")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--bench-subm", action="store_true",
                        help="after tests, optionally time 1025/10000-row SubM")
    parser.add_argument("--bench-timeout-s", type=int, default=900)
    args = parser.parse_args(argv)
    if args.timeout_s < 1 or args.bench_timeout_s < 1:
        parser.error("timeouts must be positive seconds")

    root = Path(__file__).resolve().parents[1]
    output = args.output.expanduser().resolve()
    manifest: dict = {}
    try:
        verify_checkout(root, args.commit)
        if output == root or root in output.parents:
            raise GateError("--output must be outside the source checkout")
        if output.exists():
            raise GateError("--output must name a new directory to prevent overwriting evidence")
        for suite in SUITES:
            if not (root / suite).is_file():
                raise GateError(f"required test suite is missing: {suite}")
        if not (root / "bench/bench_subm_conv_backends.py").is_file() and args.bench_subm:
            raise GateError("optional SubM benchmark script is missing")

        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        validate_hardware(sys.platform, platform.machine(), brand)
        macos = subprocess.run(
            ["sw_vers", "-productVersion"], capture_output=True,
            text=True, check=True, timeout=10,
        ).stdout.strip()
        output.mkdir(parents=True)
        manifest = {
            "status": "running", "source_commit": args.commit,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": {"cpu_brand": brand, "machine": platform.machine(),
                         "macos": macos},
            "python": platform.python_version(),
            "source_sha256": _source_hashes(root),
            "suites": list(SUITES), "math_modes": {},
            "fallback": "0", "benchmark_requested": args.bench_subm,
        }
        for mode, fast in (("safe", "0"), ("fast", "1")):
            env = os.environ.copy()
            env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
            env["PYTORCH_MPS_FAST_MATH"] = fast
            environment_log = output / f"environment-{mode}.log"
            info = _environment(root, env, environment_log, min(args.timeout_s, 120))
            log = output / f"pytest-{mode}.log"
            result = _run_logged(
                [sys.executable, "-m", "pytest", "-q", *SUITES],
                root=root, env=env, destination=log, timeout_s=args.timeout_s,
            )
            passed = parse_pytest_summary(result.stdout, result.returncode)
            item = {
                "torch": info["torch"], "mps_device": info["mps_device"],
                "mps_fast_math": fast, "tests_passed": passed,
                "pytest_log": log.name, "pytest_log_sha256": _sha256(log),
                "environment_log": environment_log.name,
                "environment_log_sha256": _sha256(environment_log),
            }
            if args.bench_subm:
                bench_json = output / f"subm-benchmark-{mode}.json"
                bench_log = output / f"subm-benchmark-{mode}.log"
                result = _run_logged(
                    [sys.executable, "bench/bench_subm_conv_backends.py", "--rows",
                     "1025", "10000", "--warmup", "1", "--repeats", "3",
                     "--output", str(bench_json)],
                    root=root, env=env, destination=bench_log,
                    timeout_s=args.bench_timeout_s,
                )
                if result.returncode or not bench_json.is_file():
                    raise GateError(f"SubM benchmark {mode} failed")
                data = json.loads(bench_json.read_text(encoding="utf-8"))
                cases = {(case["rows"], case["rulebook_backend"])
                         for case in data.get("cases", ())}
                if cases != {(1025, "cpu"), (1025, "mps"), (10000, "cpu"), (10000, "mps")}:
                    raise GateError(f"SubM benchmark {mode} emitted incomplete cases")
                item["benchmark_json"] = bench_json.name
                item["benchmark_json_sha256"] = _sha256(bench_json)
                item["benchmark_log"] = bench_log.name
                item["benchmark_log_sha256"] = _sha256(bench_log)
            manifest["math_modes"][mode] = item
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        verify_checkout(root, args.commit)
        if _source_hashes(root) != manifest["source_sha256"]:
            raise GateError("source files changed during M1 sparse validation")
        manifest["status"] = "passed"
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts_sha256"] = {
            path.name: _sha256(path) for path in sorted(output.iterdir())
            if path.is_file() and path.name != "manifest.json"
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"M1 sparse validation passed; evidence: {output}")
        return 0
    except (GateError, OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        if output.is_dir() and manifest:
            manifest["status"] = "failed"
            manifest["failure"] = str(exc)
            manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["artifacts_sha256"] = {
                path.name: _sha256(path) for path in sorted(output.iterdir())
                if path.is_file() and path.name != "manifest.json"
            }
            (output / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(f"M1 sparse validation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

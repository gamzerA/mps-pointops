#!/usr/bin/env python3
"""Adjudicate dense feature-kNN CPU/MPS index differences with a float32 oracle.

Reproduce the D=128 case in ``bench_feature_knn.py`` with a separate Safe or
Fast Math process::

    PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0 \
      python bench/probe_feature_knn_tie.py --output /tmp/feature-knn-tie-safe.json

The CPU API uses ``torch.cdist(...).topk(...)``; the Metal feature kernel
selects by direct, dimension-ordered float32 squared distance. This probe
computes that direct distance independently for *every* query/reference pair
without materializing the full pairwise matrix. It reports where each API
differs from the direct oracle. It is a ranking diagnostic, not a benchmark.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
    raise SystemExit("Set PYTORCH_ENABLE_MPS_FALLBACK=0 before importing torch")
if os.environ.get("PYTORCH_MPS_FAST_MATH") not in ("0", "1"):
    raise SystemExit("Set PYTORCH_MPS_FAST_MATH=0 or 1 before importing torch")

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mps_pointops import knn  # noqa: E402

SOURCE_FILES = (
    "bench/probe_feature_knn_tie.py",
    "bench/bench_feature_knn.py",
    "mps_pointops/ops.py",
    "mps_pointops/reference.py",
    "mps_pointops/kernels/feature_knn.metal",
    "tests/test_feature_knn.py",
)


def _command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, cwd=ROOT, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _chip() -> str | None:
    hardware = _command("system_profiler", "SPHardwareDataType")
    if hardware is None:
        return None
    for line in hardware.splitlines():
        if line.strip().startswith("Chip:"):
            return line.partition(":")[2].strip()
    return None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _score_bits(score: np.float32) -> int:
    return int(np.asarray(score, dtype=np.float32).view(np.uint32))


def _score_record(score: np.float32) -> dict:
    """Record the exact float32 number, including a lossless bit encoding."""
    return {
        "decimal": float(score),
        "binary_hex": float(score).hex(),
        "float32_bits": f"0x{_score_bits(score):08x}",
    }


def _ulp_gap(a: np.float32, b: np.float32) -> int | None:
    # The fixture's squared distances are finite and nonnegative. The
    # integer-bit difference is therefore the number of float32 ULP steps.
    if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b < 0:
        return None
    return abs(_score_bits(a) - _score_bits(b))


def _direct_scores(query: np.ndarray, ref_columns: np.ndarray) -> np.ndarray:
    """Match the documented s0, s1, ... float32 rounding sequence.

    Explicit NumPy ufunc outputs keep subtract, multiply, and add separate;
    reference columns are contiguous to avoid a large Q x N allocation.
    """
    n, dim = ref_columns.shape[1], ref_columns.shape[0]
    delta = np.empty(n, dtype=np.float32)
    term = np.empty(n, dtype=np.float32)
    scores = np.empty(n, dtype=np.float32)
    np.subtract(ref_columns[0], query[0], out=delta)
    np.multiply(delta, delta, out=scores)
    for feature in range(1, dim):
        np.subtract(ref_columns[feature], query[feature], out=delta)
        np.multiply(delta, delta, out=term)
        np.add(scores, term, out=scores)
    return scores


def _count_slots(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, default=2048)
    parser.add_argument("--references", type=int, default=2048)
    parser.add_argument("--dimension", type=int, default=128)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1601,
                        help="benchmark base seed; the generator uses seed + dimension")
    args = parser.parse_args()
    if not torch.backends.mps.is_available():
        parser.error("MPS is unavailable")
    if min(args.queries, args.references, args.dimension, args.k) < 1:
        parser.error("queries, references, dimension, and k must be positive")
    if args.dimension == 3:
        parser.error("dimension=3 uses a different kNN kernel")
    if args.k > min(args.references, 256):
        parser.error("k must be <= min(references, 256)")

    # Same generator, order, shapes, and seed as bench_feature_knn.py.
    gen = torch.Generator().manual_seed(args.seed + args.dimension)
    ref_cpu = torch.randn((1, args.references, args.dimension), generator=gen)
    query_cpu = torch.randn((1, args.queries, args.dimension), generator=gen)
    cpu_dist, cpu_idx = knn(query_cpu, ref_cpu, args.k)
    mps_dist, mps_idx = knn(query_cpu.to("mps"), ref_cpu.to("mps"), args.k)
    torch.mps.synchronize()

    cpu_ids = cpu_idx[0].numpy()
    mps_ids = mps_idx[0].cpu().numpy()
    cpu_distance = cpu_dist[0].numpy()
    mps_distance = mps_dist[0].cpu().numpy()
    references = ref_cpu[0].numpy()
    queries = query_cpu[0].numpy()
    ref_columns = np.ascontiguousarray(references.T)
    reference_ids = np.arange(args.references, dtype=np.int64)

    totals = {
        "cpu_vs_mps_slot_mismatches": _count_slots(cpu_ids, mps_ids),
        "cpu_vs_mps_rows_with_mismatch": int(np.count_nonzero(np.any(cpu_ids != mps_ids, axis=1))),
        "cpu_vs_scalar_slot_mismatches": 0,
        "mps_vs_scalar_slot_mismatches": 0,
        "cpu_vs_scalar_rows_with_mismatch": 0,
        "mps_vs_scalar_rows_with_mismatch": 0,
    }
    differing_rows = []
    for row in range(args.queries):
        scores = _direct_scores(queries[row], ref_columns)
        # The finite random fixture has no invalid scores; reject a scaled
        # input that would make NaN ordering an unexamined part of the probe.
        if not np.all(np.isfinite(scores)):
            raise RuntimeError(f"non-finite oracle distance at query {row}")
        ordering = np.lexsort((reference_ids, scores))
        oracle_ids = ordering[:args.k]
        c, m = cpu_ids[row], mps_ids[row]
        cpu_bad = c != oracle_ids
        mps_bad = m != oracle_ids
        totals["cpu_vs_scalar_slot_mismatches"] += int(np.count_nonzero(cpu_bad))
        totals["mps_vs_scalar_slot_mismatches"] += int(np.count_nonzero(mps_bad))
        totals["cpu_vs_scalar_rows_with_mismatch"] += int(np.any(cpu_bad))
        totals["mps_vs_scalar_rows_with_mismatch"] += int(np.any(mps_bad))
        affected_slots = np.flatnonzero((c != m) | cpu_bad | mps_bad)
        if len(affected_slots) == 0:
            continue

        ranks = np.empty(args.references, dtype=np.int64)
        ranks[ordering] = np.arange(args.references, dtype=np.int64)
        slots = []
        for slot in affected_slots:
            ci, mi, oi = int(c[slot]), int(m[slot]), int(oracle_ids[slot])
            entries = {}
            for label, index in (("cpu", ci), ("mps", mi), ("scalar", oi)):
                entries[label] = {
                    "reference_index": index,
                    "scalar_rank_zero_based": int(ranks[index]),
                    "scalar_squared_distance": _score_record(scores[index]),
                }
            slots.append({
                "slot_zero_based": int(slot),
                "candidates": entries,
                "scalar_score_ulp_gaps": {
                    "cpu_vs_mps": _ulp_gap(scores[ci], scores[mi]),
                    "cpu_vs_scalar": _ulp_gap(scores[ci], scores[oi]),
                    "mps_vs_scalar": _ulp_gap(scores[mi], scores[oi]),
                },
                "reported_euclidean_distance": {
                    "cpu": float(cpu_distance[row, slot]),
                    "mps": float(mps_distance[row, slot]),
                },
            })
        differing_rows.append({
            "query_index": row,
            "cpu_indices": c.tolist(),
            "mps_indices": m.tolist(),
            "scalar_indices": oracle_ids.tolist(),
            "cpu_vs_mps_slot_mismatches": _count_slots(c, m),
            "cpu_vs_scalar_slot_mismatches": int(np.count_nonzero(cpu_bad)),
            "mps_vs_scalar_slot_mismatches": int(np.count_nonzero(mps_bad)),
            "differing_slots": slots,
        })

    report = {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "environment": {
            "chip": _chip(),
            "mps_device": torch.backends.mps.get_name(),
            "os": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pytorch_enable_mps_fallback": os.environ["PYTORCH_ENABLE_MPS_FALLBACK"],
            "pytorch_mps_fast_math": os.environ["PYTORCH_MPS_FAST_MATH"],
            "git_commit": _command("git", "rev-parse", "HEAD"),
        },
        "method": {
            "queries": args.queries,
            "references": args.references,
            "dimension": args.dimension,
            "k": args.k,
            "benchmark_base_seed": args.seed,
            "generator_seed": args.seed + args.dimension,
            "input_order": "reference tensor then query tensor from one CPU torch.Generator",
            "cpu_selection": "mps_pointops.knn CPU path: torch.cdist then topk",
            "mps_selection": "mps_pointops.knn Metal feature_knn_dense",
            "oracle": "each float32 subtract, square, then dimension-ordered float32 add; sort by (squared distance, reference index)",
            "oracle_scope": "all query rows and all reference candidates",
            "distance_ulp_scope": "float32 oracle squared distances for competing indices, not cdist or Metal internal squared distances",
            "reference_tensor_sha256": hashlib.sha256(references.tobytes()).hexdigest(),
            "query_tensor_sha256": hashlib.sha256(queries.tobytes()).hexdigest(),
            "source_sha256": {name: _sha(ROOT / name) for name in SOURCE_FILES},
        },
        "counts": totals,
        "differing_rows": differing_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"counts": totals, "differing_query_rows": [r["query_index"] for r in differing_rows]}))


if __name__ == "__main__":
    main()

# M5 Pro spatial branch regression logs

These are raw `pytest -ra tests` outputs from 2026-10-02 on the physical
Apple M5 Pro (48 GiB unified memory, macOS 26.5.2) with PyTorch 2.14.1.
They test the v0.9 development branch after the executable source was fixed
at commit `62fcc0f295e127be47a475fbc122d4de686b4c58`. Subsequent
changes in this worktree only update documentation and tests; the executable
source-file hashes in the 30 clean benchmark JSON records remain identical.

| Log | Isolated environment | Result |
| --- | --- | --- |
| [Safe Math](pytest-safe-m5pro.log) | `PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0` | 515 passed, 41 skipped, 1 xfailed |
| [Fast Math](pytest-fast-m5pro.log) | `PYTHONPATH=. PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=1` | 488 passed, 68 skipped, 1 xfailed |

Both commands used the project's local PyTorch 2.14.1 virtual environment
with `python -m pytest -ra tests` in separate Python processes. The logs
preserve the full pytest collection,
skip reasons, expected failure and summary. The Safe-only spatial BVH tests
are skipped in Fast mode by contract. Optional PyG and real `torch_cluster`
tests are skipped because those upstream packages are absent from this local
environment; they are covered by separate CI jobs. This is an M5 Pro result
and does not substitute for physical M1 verification or a total GPU physical
memory measurement.

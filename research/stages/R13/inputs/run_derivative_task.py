"""Run one exact derivative-task argv with explicit local resource controls.

The storage cap is checked at task boundaries, not enforced as an OS quota.
The workflow owns timeouts, logs, process-tree termination and the single lock.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def storage_bytes(root: Path) -> int:
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(directory) / name
            if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
                raise ValueError("Derivative output contains a symlink or junction")
            if path.is_file():
                total += path.stat().st_size
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-limit-bytes", type=int, default=8 * 1024 ** 3)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    if not argv or args.stage_limit_bytes <= 0:
        parser.error("A positive cap and exact child argv are required")
    controls = {name: "4" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    controls.update(CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                    MPLCONFIGDIR=str(Path.cwd() / "matplotlib-cache"))
    env = dict(os.environ, **controls)
    # Workflow cwd is <attempt>/tasks/<task>; two parents locate this attempt.
    attempt = Path.cwd().parents[1]
    before = storage_bytes(attempt)
    if before > args.stage_limit_bytes:
        raise SystemExit("Stage storage already exceeds its declared cap")
    started = time.monotonic()
    result = subprocess.run(argv, env=env)
    elapsed = time.monotonic() - started
    after = storage_bytes(attempt)
    exceeded = after > args.stage_limit_bytes
    record = {"schema_version": 1, "argv": argv, "exit_code": result.returncode,
              "elapsed_seconds": elapsed, "environment_controls": controls,
              "attempt_storage_before_bytes": before, "attempt_storage_after_bytes": after,
              "stage_storage_cap_bytes": args.stage_limit_bytes, "storage_cap_exceeded": exceeded,
              "storage_enforcement": "At task boundaries; not a live OS quota",
              "model_fits": 0, "network_requests": 0,
              "scientific_validity": "not_assessed"}
    with Path("task-runtime.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    raise SystemExit(result.returncode if result.returncode else 3 if exceeded else 0)

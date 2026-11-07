"""Run the frozen R4 tree protocol. One CPU job, no CatBoost fit."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r4_trees import (  # noqa: E402
    build_confirmation,
    build_discovery,
    freeze_grid,
    load_json,
    sha256_file,
    validate_protocol,
    write_json,
)


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("checks", "discovery", "freeze-grid", "confirmation"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = (args.out or Path.cwd()).resolve()
    protocol_path = args.protocol or (root / "research/stages/R4/inputs/protocol.json")
    if args.command == "checks":
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_r4_trees.py", "-v"],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        write_json(out / "checks.json", {
            "schema_version": 1, "scientific_validity": "not_assessed",
            "scope": "R4 axis unit tests only.",
            "tests_returncode": completed.returncode,
            "stdout_tail": completed.stdout[-6000:],
            "stderr_tail": completed.stderr[-3000:],
        })
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        return completed.returncode
    if args.command == "discovery":
        build_discovery(protocol_path, out)
        return 0
    if args.command == "freeze-grid":
        export_path = args.export.resolve()
        protocol = load_json(protocol_path)
        validate_protocol(protocol)
        write_json(out / "confirmation-grid.json", freeze_grid(load_json(export_path), protocol, sha256_file(export_path)))
        return 0
    build_confirmation(protocol_path, args.grid.resolve(), args.export.resolve(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

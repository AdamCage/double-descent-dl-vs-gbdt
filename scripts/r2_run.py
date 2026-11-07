"""Run the frozen R2 linear/random-feature protocol.

Thread limits are set before NumPy is imported. The process does not load a GPU
runtime and does not download data.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r2_experiment import (  # noqa: E402
    build_confirmation,
    build_discovery_export,
    freeze_confirmation_grid,
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
    protocol_path = args.protocol or (root / "research/stages/R2/inputs/protocol.json")
    if args.command == "checks":
        return run_checks(root, out)
    if args.command == "discovery":
        build_discovery_export(protocol_path, out)
        return 0
    if args.command == "freeze-grid":
        if args.export is None:
            raise SystemExit("freeze-grid requires --export")
        export_path = args.export.resolve()
        protocol = load_json(protocol_path)
        validate_protocol(protocol)
        export = load_json(export_path)
        grid = freeze_confirmation_grid(export, protocol, sha256_file(export_path))
        destination = out / "confirmation-grid.json" if args.grid is None else args.grid.resolve()
        write_json(destination, grid)
        return 0
    if args.grid is None or args.export is None:
        raise SystemExit("confirmation requires --grid and --export")
    build_confirmation(protocol_path, args.grid.resolve(), args.export.resolve(), out)
    return 0


def run_checks(root: Path, out: Path) -> int:
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_r2_experiment.py", "-v"]
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
    payload = {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "scope": "R2 solver, seed, reduction and contrast unit tests. Not a confirmatory fit.",
        "test_command": command,
        "tests_returncode": completed.returncode,
        "stdout_tail": completed.stdout[-8000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    write_json(out / "checks.json", payload)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

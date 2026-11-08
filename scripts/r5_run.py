"""Run the frozen R5 replication. No new dataset and no GPU."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r5_replicate import (  # noqa: E402
    build_confirmation,
    build_discovery,
    canonical,
    load_json,
    sha256_file,
    validate_protocol,
    write_json,
)
import hashlib


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("checks", "discovery", "freeze-grid", "confirmation"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--grid", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    out = (args.out or Path.cwd()).resolve()
    protocol_path = root / "research/stages/R5/inputs/protocol.json"
    if args.command == "checks":
        completed = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_r5_replicate.py", "-v"],
            cwd=root, capture_output=True, text=True, encoding="utf-8", errors="replace")
        write_json(out / "checks.json", {
            "schema_version": 1, "scientific_validity": "not_assessed",
            "tests_returncode": completed.returncode, "stdout_tail": completed.stdout[-4000:],
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
        export = load_json(export_path)
        if export["reduction"].get("kept_full_menu") is not True or export["reduction"].get("used_clean_risk") is not False:
            raise SystemExit("replication grid is not a full time-only menu")
        if hashlib.sha256(canonical(protocol)).hexdigest() != export["protocol_sha256"]:
            raise SystemExit("protocol hash mismatch")
        write_json(out / "confirmation-grid.json", {
            "schema_version": 1, "scientific_validity": "not_assessed",
            "kept_full_menu": True, "used_clean_risk": False,
            "discovery_export_sha256": sha256_file(export_path),
            "projected_confirmation_seconds": export["reduction"]["projected_confirmation_seconds"],
        })
        return 0
    build_confirmation(protocol_path, args.grid.resolve(), args.export.resolve(), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

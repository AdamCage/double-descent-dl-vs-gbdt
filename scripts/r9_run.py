"""Run the frozen R9 CatBoost discovery protocol. One CPU job."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "4"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.r9_catboost import load_json, run_discovery  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    protocol_path = args.protocol.resolve()
    protocol = load_json(protocol_path)
    run_discovery(protocol, Path.cwd(), protocol_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

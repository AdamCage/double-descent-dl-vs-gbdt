"""Execute only the frozen R11 CatBoost schedule-control protocol."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.acceleration_boosting import run_study  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    run_study(json.loads(protocol_path.read_text(encoding="utf-8")), Path.cwd(), protocol_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

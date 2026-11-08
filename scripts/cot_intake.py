"""Run only the separately frozen, activated public CoT dataset intake."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("intake"))
    args = parser.parse_args()
    from src.cot_data import run_intake
    path = args.spec.resolve()
    result = run_intake(json.loads(path.read_text(encoding="utf-8")), args.output, spec_path=path)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

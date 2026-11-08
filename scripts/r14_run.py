"""Execute only the separately reviewed and frozen R14 discovery protocol."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Install caps before importing NumPy/CatBoost through the scientific module.
# Model and inference calls also receive explicit threads; threadpool_limits
# additionally caps already-loaded BLAS pools when run_study is used directly.
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    from src.response_forecast import run_study

    analysis = run_study(json.loads(protocol_path.read_text(encoding="utf-8")), Path.cwd(), protocol_path)
    print(json.dumps({key: analysis[key] for key in ("n_contexts", "n_forecast_rows", "n_directions", "n_fits", "decision")}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

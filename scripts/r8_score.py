"""Score the frozen R8 trajectory catalog. No model is fit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.detector_catalog import score_catalog  # noqa: E402


def pythonize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [pythonize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite score")
        return float(format(value, ".17g"))
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.catalog.read_bytes()
    catalog = json.loads(raw.decode("utf-8"))
    if not isinstance(catalog, dict):
        raise SystemExit("catalog must be an object")
    scored = score_catalog(catalog)
    scored["catalog_sha256"] = hashlib.sha256(raw).hexdigest()
    scored["catalog_path"] = "research/stages/R8/inputs/catalog.json"
    target = Path.cwd() / "scores.json"
    target.write_text(json.dumps(pythonize(scored), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

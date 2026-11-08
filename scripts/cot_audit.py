"""Run only an activated, input-bound CoT acknowledgment audit partition."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "4"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--split", choices=("discovery", "confirmation"))
    parser.add_argument("--metadata-only", action="store_true", help="Read only structural question metadata; never read ACK/outcome columns")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not args.metadata_only and args.split is None:
        parser.error("--split is required unless --metadata-only is selected")
    from src.cot_audit import run_analysis
    result = run_analysis(args.input.resolve(), args.spec.resolve(), args.protocol.resolve(), args.output.resolve(), args.split,
                          metadata_only=args.metadata_only)
    print(json.dumps({"split": result.get("split"), "global": result.get("global"),
                      "metadata_only": result.get("metadata_only", False), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

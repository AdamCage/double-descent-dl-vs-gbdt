"""Record one declared unittest selection as a workflow artifact."""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--model-smoke", action="store_true",
                        help="Opt into tiny fitting checks outside default offline CI")
    args = parser.parse_args()
    if args.model_smoke:
        os.environ["RESEARCH_MODEL_SMOKE"] = "1"
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=args.pattern)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    summary = {"tests_run": result.testsRun, "passed": result.wasSuccessful() and result.testsRun > 0,
               "failures": len(result.failures), "errors": len(result.errors),
               "skipped": len(result.skipped), "log": stream.getvalue()}
    Path("checks.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(stream.getvalue())
    raise SystemExit(0 if summary["passed"] else 1)

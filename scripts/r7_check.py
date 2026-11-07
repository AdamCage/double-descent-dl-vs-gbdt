"""Check the local R7 article against the closed claim ledger. No submission."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.research_claims import validate_claims


REQUIRED = (
    "R2-confirmation-contrasts",
    "R3-confirmation-budget",
    "R4-separate-axes",
    "R5-replication-not-one-mechanism",
    "not_run",
    "опровергнут",
    "не идентифицируют один общий механизм",
    "CatBoost",
    "L05",
    "внешняя подача не",
)


def main() -> int:
    root = ROOT
    article = (root / "research/stages/R7/article.md").read_text(encoding="utf-8")
    claims = json.loads((root / "research/claims.json").read_text(encoding="utf-8"))
    problems = validate_claims(root, claims)
    normalized = " ".join(article.split()).casefold()
    missing = [item for item in REQUIRED if item.casefold() not in normalized]
    lowered = normalized
    if "submitted to" in lowered or "arxiv.org" in lowered:
        problems.append("article contains an external-submission marker")
    payload = {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "claims_valid": not problems,
        "claim_problems": problems,
        "missing_phrases": missing,
        "external_submission": False,
        "passed": not problems and not missing,
    }
    Path.cwd().joinpath("synthesis-check.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

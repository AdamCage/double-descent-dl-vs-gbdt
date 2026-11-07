"""R6 transfer audit. It does not download data and does not fit a model."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    claims = json.loads((root / "research/claims.json").read_text(encoding="utf-8"))
    found = {item["id"] for item in claims["claims"]}
    required = {
        "R2-confirmation-contrasts",
        "R3-confirmation-budget",
        "R4-separate-axes",
        "R5-replication-not-one-mechanism",
    }
    missing = sorted(required - found)
    if missing:
        raise SystemExit(f"Closed claims missing: {missing}")
    disposition = {
        "schema_version": 1,
        "scientific_validity": "not_assessed",
        "disposition": "not_run",
        "models_fit": 0,
        "download_bytes": 0,
        "http_requests": 0,
        "datasets_opened": [],
        "candidate_considered": {
            "name": "sklearn diabetes",
            "opened": False,
            "reason": "Bundled with scikit-learn under the BSD license, 442 rows and 10 columns, no group identifier for a leakage audit.",
        },
        "why_transfer_is_not_justified": [
            "R5 replicated a linear noise contrast and the leaf-versus-forest directions, and recorded that they do not identify one mechanism.",
            "R4 found clean risk rising with the single-tree leaf cap and falling with forest size. Those are not one capacity effect to transfer.",
            "R3 found that 40 epochs had lower clean risk than training to the residual tolerance. That alternative was not re-estimated in R5.",
            "A real table would be a new confirmatory population. No single identified mechanism makes that fit informative.",
            "The historical dataset menu is not downloaded.",
        ],
        "claim_ids": sorted(required),
    }
    out = Path.cwd() / "disposition.json"
    out.write_text(json.dumps(disposition, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

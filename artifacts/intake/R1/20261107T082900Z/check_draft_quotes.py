"""Confirm draft locator quotes occur in the extracted text."""
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
drafts = root / "research/stages/R1/inputs/drafts"
text_root = root / "artifacts/intake/R1/20261107T082900Z/text"
failed = 0
for path in sorted(drafts.glob("L*.json")):
    card = json.loads(path.read_text(encoding="utf-8"))
    extract = (text_root / f"{card['source_id']}.txt").read_text(encoding="utf-8")
    for locator in card["locators"]:
        if locator["quote"] not in extract:
            failed += 1
            print(f"MISSING {card['source_id']} {locator['label']}")
    print(f"{card['source_id']} locators={len(card['locators'])} bytes={len(extract)}")
sys.exit(1 if failed else 0)

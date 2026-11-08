"""Print non-locator fields from literature drafts."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[4]
drafts = root / "research/stages/R1/inputs/drafts"
for name in ("L02", "L03", "L04", "L06"):
    card = json.loads((drafts / f"{name}.json").read_text(encoding="utf-8"))
    print("=" * 20, name, card.get("version"))
    for key in (
        "model_and_data", "loss", "varied_axis", "held_fixed_axes",
        "interpolation_definition", "result_type", "transfer_limits",
    ):
        print(f"\n[{key}]\n{card.get(key)}")
    print("\n[assumptions]")
    for item in card.get("assumptions", []):
        print("-", item)

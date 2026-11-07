"""Assemble agent-adopted literature inputs from verified drafts. Not a stage task."""
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(root))

from src.r1_audit import build_intake_index, load_jsonl
from src.research_workflow import sha256, write_json

INTAKE = "artifacts/intake/R1/20261107T082900Z"
AUTHOR = "ddb85535-1ef1-4069-95c4-46578067887a"
READERS = {
    "L02": "144f9191-3044-4fe3-87d1-0f162ee1fb60",
    "L03": "144f9191-3044-4fe3-87d1-0f162ee1fb60",
    "L04": "144f9191-3044-4fe3-87d1-0f162ee1fb60",
    "L05": "001810aa-bf22-41e9-92fd-4cdc05feb1c1",
    "L06": "001810aa-bf22-41e9-92fd-4cdc05feb1c1",
    "L07": "001810aa-bf22-41e9-92fd-4cdc05feb1c1",
    "L08": "001810aa-bf22-41e9-92fd-4cdc05feb1c1",
}
FIELDS = (
    "locators", "model_and_data", "loss", "varied_axis", "held_fixed_axes",
    "interpolation_definition", "assumptions", "result_type", "transfer_limits",
)


def main() -> None:
    registry = json.loads((root / "docs/literature/sources.json").read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in registry["sources"]}
    records = load_jsonl(root / INTAKE / "acquisition.jsonl")
    retrieved = {}
    for record in records:
        if record.get("status") == "retrieved":
            retrieved[record["source_id"]] = record
    cards = []
    for source_id in ("L02", "L03", "L04", "L05", "L06", "L07", "L08"):
        draft = json.loads((root / "research/stages/R1/inputs/drafts" / f"{source_id}.json").read_text(encoding="utf-8"))
        record = retrieved[source_id]
        registered = by_id[source_id]
        snapshot = f"{INTAKE}/{record['snapshot_path']}"
        text = f"{INTAKE}/text/{source_id}.txt"
        card = {
            "source_id": source_id,
            "title": registered["title"],
            "authors": registered["authors"],
            "version": draft["version"],
            "registry_version": registered["version"],
            "retrieved_utc": record["retrieved_utc"],
            "retrieved_url": record["url"],
            "access_status": "full_text",
            "review_level": "full_text",
            "snapshot_path": snapshot,
            "snapshot_sha256": sha256(root / snapshot),
            "text_path": text,
            "text_sha256": sha256(root / text),
            "text_extraction": "PyMuPDF 1.28.0 text layer; equations may be line-broken and were not repaired.",
            "unavailable_reason": None,
            "draft_reader": READERS[source_id],
        }
        for key in FIELDS:
            card[key] = draft[key]
        if sha256(root / snapshot) != record["sha256"]:
            raise SystemExit(f"snapshot hash disagrees with acquisition log: {source_id}")
        cards.append(card)
    registered = by_id["L01"]
    cards.insert(0, {
        "source_id": "L01",
        "title": registered["title"],
        "authors": registered["authors"],
        "version": "published article; PDF not retrieved",
        "registry_version": registered["version"],
        "retrieved_utc": "2026-11-07T08:31:01.210203+00:00",
        "access_status": "unavailable",
        "review_level": "unavailable",
        "snapshot_path": None,
        "snapshot_sha256": None,
        "text_path": None,
        "text_sha256": None,
        "unavailable_reason": "Both permitted retrievals of the PNAS PDF returned HTTP 403. No third request was made, and the arXiv preprint was not substituted for the published version.",
        "locators": [],
        "model_and_data": "not_read_published_full_text",
        "loss": "not_read_published_full_text",
        "varied_axis": "not_read_published_full_text",
        "held_fixed_axes": "not_read_published_full_text",
        "interpolation_definition": "not_read_published_full_text",
        "assumptions": ["The published PDF was not read after two HTTP 403 responses."],
        "result_type": "not_read_published_full_text",
        "transfer_limits": "No PNAS theorem or figure is transferred into R2 from this audit. The earlier registry status metadata_and_abstract_verified is not promoted.",
    })
    comparison = json.loads((root / "research/stages/R1/inputs/drafts/tree-axis-comparison.json").read_text(encoding="utf-8"))
    document = {
        "schema_version": 1,
        "author_agent": AUTHOR,
        "literature_readers": READERS,
        "intake_path": INTAKE,
        "cards": cards,
        "tree_axis_comparison": comparison,
        "prediction_assessments": [{
            "id": "P2",
            "author_outcome": "consistent",
            "observation_ids": ["L05", "L06", "L07"],
            "reason": "The located tree and boosting papers use different capacity interventions, losses, and interpolation language. Their comparability stays unresolved, which is why the axis comparison is required before any transfer.",
        }],
    }
    notes = root / "research/stages/R1/inputs/source-cards.json"
    write_json(notes, document)
    index = build_intake_index(root, records, INTAKE)
    write_json(root / "research/stages/R1/inputs/intake-index.json", index)
    print(f"cards={len(cards)} http_requests={index['http_requests']} download_bytes={index['download_bytes']}")


if __name__ == "__main__":
    main()

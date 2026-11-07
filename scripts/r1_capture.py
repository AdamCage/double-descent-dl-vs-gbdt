"""Export the R1 audit from frozen agent notes. Not a scientific reader."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_FOR_IMPORT))

from src.r1_audit import (INPUTS, AuditError, audit_legacy, audit_literature, audit_r2,
                          load_jsonl, load_object, render_card_md, render_legacy_md,
                          render_r2_md, repo_file, validation_document)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("legacy", "literature", "r2-design", "validate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "legacy":
            audit = audit_legacy(
                root,
                load_object(repo_file(root, INPUTS["legacy_notes"])),
                load_object(repo_file(root, INPUTS["inventory"])),
            )
            write_json(Path("legacy-audit.json"), audit)
            Path("legacy-audit.md").write_text(render_legacy_md(audit), encoding="utf-8", newline="\n")
        elif args.command == "literature":
            notes = load_object(repo_file(root, INPUTS["source_cards"]))
            records = load_jsonl(repo_file(root, f"{notes['intake_path']}/acquisition.jsonl"))
            audit = audit_literature(
                root, notes, load_object(repo_file(root, INPUTS["registry"])),
                load_object(repo_file(root, INPUTS["intake_index"])), records,
            )
            write_json(Path("literature-audit.json"), audit)
            write_json(Path("source-index.json"), audit["intake"])
            for card in audit["cards"]:
                target = Path("cards") / f"{card['source_id']}.md"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(render_card_md(card), encoding="utf-8", newline="\n")
        elif args.command == "r2-design":
            audit = audit_r2(load_object(repo_file(root, INPUTS["r2_design"])))
            write_json(Path("R2-design-proposal.json"), audit)
            Path("R2-design-proposal.md").write_text(render_r2_md(audit), encoding="utf-8", newline="\n")
        else:
            if args.attempt is None:
                raise AuditError("validate requires --attempt")
            write_json(Path("validation.json"), validation_document(root, args.attempt.resolve()))
    except (AuditError, OSError, KeyError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

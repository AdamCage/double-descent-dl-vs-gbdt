"""Inventory historical artifacts without parsing the corpus or mutating results.

Small evidence is snapshotted; large local files are listed without pretending
they were hashed. Use evidence_bundle.py for explicit complete portable archives.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preserve(root: Path, destination: Path) -> dict:
    root, destination = root.resolve(), destination.resolve()
    if not destination.is_relative_to(root / "research" / "evidence"):
        raise ValueError("Inventory must be inside research/evidence/")
    if destination.exists():
        raise ValueError("Refusing to overwrite a historical inventory")
    candidates = [root / "data/source_manifest.json"]
    for directory in ("result", "reviews"):
        candidates.extend(p for p in (root / directory).rglob("*") if p.is_file())
    rows = []
    small_evidence = {}
    # No SQLite opening here: preserve files and WAL sidecars byte-for-byte only
    # with an explicit database backup; live file copies are not coherent backups.
    for path in sorted(set(candidates)):
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"Unsafe historical path: {path}")
        rel = path.relative_to(root).as_posix()
        size = path.stat().st_size
        row = {"path": rel, "bytes": size, "sha256": None,
               "status": "local_only_not_hashed", "portable": False}
        if size <= 2 * 1024 * 1024:
            row.update(sha256=digest(path), status="hashed_at_inventory")
            if path.suffix in {".json", ".log", ".md", ".csv"}:
                small_evidence[rel] = path.read_bytes()
                row.update(snapshot=f"files/{rel}", portable=True)
        rows.append(row)
    result = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "scientific_validity": "not_assessed",
        "note": "Partial historical preservation; large local artifacts are neither hashed nor portable. Snapshot hashes identify bytes, not scientific validity. SQLite stores are excluded; use SQLite backup or EpistemeOS export for coherent database preservation.",
        "files": rows,
    }
    destination.mkdir(parents=True)
    for rel, content in small_evidence.items():
        target = destination / "files" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (destination / "inventory.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    try:
        result = preserve(root, root / args.destination)
    except ValueError as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps({"files": len(result["files"]),
                      "snapshotted": sum(r["portable"] for r in result["files"])}))


if __name__ == "__main__":
    main()

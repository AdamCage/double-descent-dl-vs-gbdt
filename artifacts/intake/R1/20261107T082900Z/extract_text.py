"""Extract a text layer from retrieved PDFs. Not a scientific reading."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import fitz

INTAKE = Path(__file__).resolve().parent


def main() -> None:
    text_dir = INTAKE / "text"
    text_dir.mkdir(exist_ok=True)
    tool = "PyMuPDF 1.28.0"
    with (INTAKE / "extraction.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        for pdf in sorted((INTAKE / "snapshots").glob("*.pdf")):
            document = fitz.open(pdf)
            parts: list[str] = []
            for index, page in enumerate(document, start=1):
                parts.append(f"\n\n--- page {index} ---\n")
                parts.append(page.get_text("text"))
            target = text_dir / f"{pdf.stem}.txt"
            target.write_bytes("".join(parts).encode("utf-8"))
            record = {
                "extracted_utc": datetime.now(timezone.utc).isoformat(),
                "pages": document.page_count,
                "source_id": pdf.stem,
                "text_bytes": target.stat().st_size,
                "text_path": f"text/{pdf.stem}.txt",
                "tool": tool,
            }
            document.close()
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(f"{pdf.name} pages={record['pages']} text_bytes={record['text_bytes']}")


if __name__ == "__main__":
    main()

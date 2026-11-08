"""Build data/AI_Human.csv from the public Hugging Face mirror.

Kaggle credentials are absent, so kagglehub cannot fetch shanegerami/ai-vs-human-text.
The mirror andythetechnerd03/AI-human-text says it is that Kaggle corpus after
processing and a train/test split. This script concatenates the splits and lets
the study pipeline draw its own stratified split. It prints counts and hashes,
not text.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PARQUET = ROOT / "data" / "hf_parquet"
CSV_PATH = ROOT / "data" / "AI_Human.csv"
MANIFEST = ROOT / "data" / "source_manifest.json"
SOURCE = "https://huggingface.co/datasets/andythetechnerd03/AI-human-text"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = sorted(PARQUET.glob("*.parquet"))
    if not files:
        raise SystemExit(f"no parquet files in {PARQUET}")
    rows = 0
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "generated"])
        for path in files:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=4096, columns=["text", "generated"]):
                texts = batch.column("text").to_pylist()
                labels = batch.column("generated").to_pylist()
                for text, label in zip(texts, labels):
                    writer.writerow([text, int(label)])
                    rows += 1
            print(f"wrote {path.name}", flush=True)
    manifest = {
        "source": SOURCE,
        "note": "Processed public mirror of shanegerami/ai-vs-human-text. Not a byte copy of the Kaggle CSV.",
        "rows_written": rows,
        "csv_sha256": sha256(CSV_PATH),
        "csv_bytes": CSV_PATH.stat().st_size,
        "parquet": [{"name": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in files],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"rows={rows} csv_bytes={manifest['csv_bytes']} csv_sha256={manifest['csv_sha256']}")


if __name__ == "__main__":
    main()

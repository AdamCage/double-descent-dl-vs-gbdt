"""One-shot R1 source retrieval. Not a scientific reader and not a stage task.

Counts each urlopen as one HTTP request. Transport redirects are not counted
separately. Stops at the charter document and total caps. Does not train models.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
INTAKE = Path(__file__).resolve().parent
SNAP = INTAKE / "snapshots"
LOG = INTAKE / "acquisition.jsonl"
MAX_DOC = 25 * 1024 * 1024
MAX_TOTAL = 200 * 1024 * 1024
MAX_REQUESTS = 48
MAX_ATTEMPTS = 2
UA = "double-descent-r1-audit/0.1 (local full-text archive; no training)"

# Registered version first. A second URL is used only if the first attempt
# does not yield a permitted document under the cap.
SOURCES: list[dict[str, object]] = [
    {
        "source_id": "L01",
        "version": "PNAS 116(32):15849-15854",
        "urls": ["https://www.pnas.org/doi/pdf/10.1073/pnas.1903070116"],
    },
    {
        "source_id": "L02",
        "version": "arXiv:1903.08560v5",
        "urls": ["https://arxiv.org/pdf/1903.08560v5"],
    },
    {
        "source_id": "L03",
        "version": "arXiv:1908.05355v5",
        "urls": ["https://arxiv.org/pdf/1908.05355v5"],
    },
    {
        "source_id": "L04",
        "version": "arXiv:1912.02292v1",
        "urls": ["https://arxiv.org/pdf/1912.02292v1"],
    },
    {
        "source_id": "L05",
        "version": "NeurIPS 2023 conference PDF",
        "urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2023/file/aec5e2847c5ae90f939ab786774856cc-Paper-Conference.pdf"
        ],
    },
    {
        "source_id": "L06",
        "version": "arXiv:2509.25216v1",
        "urls": ["https://arxiv.org/pdf/2509.25216v1"],
    },
    {
        "source_id": "L07",
        "version": "arXiv:2608.03111v1",
        "urls": ["https://arxiv.org/pdf/2608.03111v1"],
    },
    {
        "source_id": "L08",
        "version": "arXiv:2206.05608v3",
        "urls": ["https://arxiv.org/pdf/2206.05608v3"],
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(record: dict[str, object]) -> None:
    with LOG.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def looks_like_pdf(data: bytes) -> bool:
    return data.startswith(b"%PDF-")


def main() -> None:
    SNAP.mkdir(parents=True, exist_ok=True)
    total = 0
    requests = 0
    for source in SOURCES:
        source_id = str(source["source_id"])
        urls = list(source["urls"])
        saved = False
        for attempt, url in enumerate(urls[:MAX_ATTEMPTS], start=1):
            if attempt > 1 or source_id != "L01":
                time.sleep(3)
            if requests >= MAX_REQUESTS:
                append({
                    "attempt": attempt, "bytes": 0, "error": "http_request_cap",
                    "retrieved_utc": utc_now(), "source_id": source_id,
                    "status": "not_retrieved", "url": url, "version": source["version"],
                })
                break
            requests += 1
            started = time.monotonic()
            record: dict[str, object] = {
                "attempt": attempt,
                "bytes": 0,
                "content_type": None,
                "http_status": None,
                "retrieved_utc": utc_now(),
                "sha256": None,
                "snapshot_path": None,
                "source_id": source_id,
                "url": url,
                "version": source["version"],
            }
            try:
                request = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(request, timeout=180) as response:
                    record["http_status"] = getattr(response, "status", None)
                    record["content_type"] = response.headers.get("Content-Type")
                    length = response.headers.get("Content-Length")
                    if length is not None and int(length) > MAX_DOC:
                        record["status"] = "rejected_over_document_cap"
                        record["error"] = f"content-length {length}"
                        append(record)
                        continue
                    data = bytearray()
                    too_big = False
                    while True:
                        chunk = response.read(256 * 1024)
                        if not chunk:
                            break
                        if len(data) + len(chunk) > MAX_DOC:
                            too_big = True
                            break
                        data.extend(chunk)
                elapsed = round(time.monotonic() - started, 3)
                record["elapsed_seconds"] = elapsed
                if too_big:
                    record["status"] = "rejected_over_document_cap"
                    record["bytes"] = len(data)
                    append(record)
                    continue
                blob = bytes(data)
                if not looks_like_pdf(blob):
                    record["status"] = "rejected_not_pdf"
                    record["bytes"] = len(blob)
                    record["error"] = "response does not start with %PDF-"
                    append(record)
                    continue
                if total + len(blob) > MAX_TOTAL:
                    record["status"] = "rejected_over_total_cap"
                    record["bytes"] = len(blob)
                    append(record)
                    break
                digest = hashlib.sha256(blob).hexdigest()
                relative = f"snapshots/{source_id}.pdf"
                target = INTAKE / relative
                target.write_bytes(blob)
                total += len(blob)
                record.update(status="retrieved", bytes=len(blob), sha256=digest, snapshot_path=relative)
                append(record)
                saved = True
                break
            except urllib.error.HTTPError as exc:
                record.update(status="http_error", http_status=exc.code, error=str(exc),
                              elapsed_seconds=round(time.monotonic() - started, 3))
                append(record)
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                record.update(status="transport_error", error=f"{type(exc).__name__}: {exc}",
                              elapsed_seconds=round(time.monotonic() - started, 3))
                append(record)
            time.sleep(3)
        if saved:
            time.sleep(3)
        else:
            append({
                "retrieved_utc": utc_now(),
                "source_id": source_id,
                "status": "unavailable",
                "version": source["version"],
                "reason": "No permitted PDF was stored within the retrieval cap.",
            })
    summary = {
        "retrieved_utc": utc_now(),
        "status": "retrieval_finished",
        "http_requests": requests,
        "download_bytes": total,
        "max_http_requests": MAX_REQUESTS,
        "max_document_bytes": MAX_DOC,
        "max_total_bytes": MAX_TOTAL,
    }
    append(summary)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

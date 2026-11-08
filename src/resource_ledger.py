"""Charter accounting for R1–R7. This is a measurement ledger, not a sandbox.

Wall time comes from attempt manifests. A directory without a finite
nonnegative elapsed_seconds is unmeasured: it is charged the charter's
maximum stage hold so it cannot disappear, and that charge is labeled as
not a measurement. Dependent computational activation stays unready while
any attempt is unmeasured or any recorded cap is exceeded.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from src.research_workflow import sha256


class LedgerError(ValueError):
    """An accounting record is missing, inconsistent or outside its root."""


NEW_ARTIFACT_ROOTS = ("artifacts/research", "artifacts/intake", "research/evidence")
EXCLUDED_ROOTS = ("result", ".research", "EpistemeOS-0.2.0")
LEDGER_PATH = "research/resources/ledger.jsonl"


def load_charter(path: Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    digest = sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise LedgerError(f"Charter hash mismatch: {path}")
    try:
        charter = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LedgerError(f"Cannot read charter: {exc}") from exc
    budgets = charter.get("budgets") if isinstance(charter, dict) else None
    r1 = charter.get("r1_budgets") if isinstance(charter, dict) else None
    if not isinstance(budgets, dict) or not isinstance(r1, dict):
        raise LedgerError("Charter is missing budgets")
    for key in ("total_runner_wall_seconds", "max_stage_wall_seconds", "new_storage_bytes",
                "public_dataset_download_bytes"):
        _positive(budgets.get(key), f"budgets.{key}")
    for key in ("document_download_bytes", "max_document_bytes", "new_storage_bytes",
                "dataset_download_bytes", "max_http_requests"):
        if key == "dataset_download_bytes":
            if budgets_value_invalid(r1.get(key), allow_zero=True):
                raise LedgerError(f"Invalid r1_budgets.{key}")
        else:
            _positive(r1.get(key), f"r1_budgets.{key}")
    charter["_sha256"] = digest
    return charter


def budgets_value_invalid(value: Any, *, allow_zero: bool) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return True
    return value < 0 if allow_zero else value <= 0


def _positive(value: Any, label: str) -> float:
    if budgets_value_invalid(value, allow_zero=False):
        raise LedgerError(f"Invalid {label}")
    return float(value)


def _finite_elapsed(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(value)


def iter_attempt_dirs(root: Path) -> list[Path]:
    base = root / "artifacts" / "research"
    if not base.exists():
        return []
    attempts: list[Path] = []
    for stage in sorted(path for path in base.iterdir() if path.is_dir() and not path.is_symlink()):
        for attempt in sorted(path for path in stage.iterdir() if path.is_dir() and not path.is_symlink()):
            attempts.append(attempt)
    return attempts


def attempt_record(attempt: Path, charter: dict[str, Any]) -> dict[str, Any]:
    """Count a failed or interrupted attempt. Never replace unknown time with zero."""
    hold = float(charter["budgets"]["max_stage_wall_seconds"])
    record: dict[str, Any] = {
        "stage_id": attempt.parent.name,
        "attempt_id": attempt.name,
        "measured": False,
        "elapsed_seconds": None,
        "charged_seconds": hold,
        "status": "unmeasured_conservative_hold",
        "charge_is_measurement": False,
    }
    manifest_path = attempt / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return record
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return record
    if not isinstance(manifest, dict):
        return record
    elapsed = _finite_elapsed(manifest.get("elapsed_seconds"))
    if elapsed is None:
        return record
    status = manifest.get("status")
    record.update(
        measured=True,
        elapsed_seconds=elapsed,
        charged_seconds=elapsed,
        status=status if isinstance(status, str) and status else "recorded",
        charge_is_measurement=True,
    )
    return record


def directory_bytes(path: Path) -> int:
    """Sum regular-file sizes. Do not follow links or read file contents."""
    if not path.exists() or path.is_symlink():
        return 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [name for name in dirnames if not (current / name).is_symlink()]
        for name in filenames:
            candidate = current / name
            if candidate.is_symlink() or not candidate.is_file():
                continue
            total += candidate.stat().st_size
    return total


def artifact_storage(root: Path) -> dict[str, Any]:
    roots: dict[str, int] = {}
    for relative in NEW_ARTIFACT_ROOTS:
        roots[relative] = directory_bytes(root / relative)
    return {
        "roots": roots,
        "bytes": sum(roots.values()),
        "excluded_roots": list(EXCLUDED_ROOTS),
        "exclusion_note": "Historical result/, workflow state and the EpistemeOS snapshot are not new-artifact storage.",
    }


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.is_symlink():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LedgerError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise LedgerError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def acquisition_records(root: Path) -> list[dict[str, Any]]:
    base = root / "artifacts" / "intake"
    if not base.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(base.rglob("acquisition.jsonl")):
        if path.is_symlink():
            raise LedgerError(f"Refusing symlink acquisition log: {path}")
        for record in _jsonl(path):
            item = dict(record)
            item["_log"] = path.relative_to(root).as_posix()
            rows.append(item)
    return rows


def network_totals(records: list[dict[str, Any]]) -> dict[str, int]:
    document_bytes = 0
    dataset_bytes = 0
    requests = 0
    for record in records:
        kind = record.get("kind")
        if "url" in record and "attempt" in record:
            requests += 1
        if record.get("status") == "retrieved" and isinstance(record.get("bytes"), int) and not isinstance(record.get("bytes"), bool):
            document_bytes += record["bytes"]
        if kind == "dataset_download":
            amount = record.get("bytes")
            if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0:
                raise LedgerError("Dataset download record needs nonnegative integer bytes")
            dataset_bytes += amount
    return {"document_download_bytes": document_bytes, "dataset_download_bytes": dataset_bytes,
            "http_requests": requests}


def ledger_entries(root: Path) -> list[dict[str, Any]]:
    return _jsonl(root / LEDGER_PATH)


def reconcile_ledger(entries: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> list[str]:
    """Coordinator notes may lag a manifest. They may not contradict one."""
    by_key = {(item["stage_id"], item["attempt_id"]): item for item in attempts}
    problems: list[str] = []
    for entry in entries:
        if entry.get("kind") != "attempt":
            continue
        key = (entry.get("stage_id"), entry.get("attempt_id"))
        measured = by_key.get(key)
        if measured is None:
            problems.append(f"Ledger names an attempt with no manifest directory: {key}")
            continue
        if not measured["measured"]:
            problems.append(f"Ledger elapsed cannot agree with an unmeasured attempt: {key}")
            continue
        elapsed = _finite_elapsed(entry.get("elapsed_seconds"))
        if elapsed is None or elapsed != measured["elapsed_seconds"]:
            problems.append(f"Ledger elapsed disagrees with manifest: {key}")
    return problems


def report(root: Path, *, charter_path: str = "research/decisions/autonomous-research-authorization.json",
           expected_sha256: str | None = None) -> dict[str, Any]:
    charter = load_charter(root / charter_path, expected_sha256=expected_sha256)
    attempts = [attempt_record(path, charter) for path in iter_attempt_dirs(root)]
    charged = sum(item["charged_seconds"] for item in attempts)
    storage = artifact_storage(root)
    network = network_totals(acquisition_records(root))
    problems = reconcile_ledger(ledger_entries(root), attempts)
    budgets = charter["budgets"]
    r1 = charter["r1_budgets"]
    remaining_wall = float(budgets["total_runner_wall_seconds"]) - charged
    remaining_storage = int(budgets["new_storage_bytes"]) - int(storage["bytes"])
    # Whole-study storage uses the 20 GiB cap. R1's 2 GiB cap is checked by the
    # audit against intake plus this stage's new files, not against later stages.
    within_caps = (
        remaining_wall >= 0
        and remaining_storage >= 0
        and network["document_download_bytes"] <= int(r1["document_download_bytes"])
        and network["dataset_download_bytes"] <= int(budgets["public_dataset_download_bytes"])
        and network["dataset_download_bytes"] <= int(r1["dataset_download_bytes"])
    )
    unmeasured = [item for item in attempts if not item["measured"]]
    ready = within_caps and not unmeasured and not problems
    return {
        "schema_version": 1,
        "accounting_implemented": True,
        "charter_sha256": charter["_sha256"],
        "attempts": attempts,
        "charged_wall_seconds": charged,
        "remaining_wall_seconds": remaining_wall,
        "storage": storage,
        "remaining_storage_bytes": remaining_storage,
        "network": network,
        "ledger_path": LEDGER_PATH,
        "ledger_entries": len(ledger_entries(root)),
        "reconciliation_problems": problems,
        "unmeasured_attempts": [f"{item['stage_id']}/{item['attempt_id']}" for item in unmeasured],
        "unknown_elapsed_policy": "Unmeasured attempts charge max_stage_wall_seconds and block computational activation. The hold is not a measured duration.",
        "computational_activation_ready": ready,
        "scientific_validity": "not_assessed",
    }


def append_entry(root: Path, entry: dict[str, Any]) -> None:
    if not isinstance(entry, dict) or not entry.get("kind"):
        raise LedgerError("Ledger entry needs kind")
    path = root / LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")

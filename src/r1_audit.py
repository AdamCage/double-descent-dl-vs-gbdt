"""Validate agent-authored R1 notes and export the stage artifacts.

The scientific text is written by Cursor before freeze. This module checks
structure, hashes, coverage and caps. It does not read a PDF as a reviewer
and it does not assign scientific validity.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.research_workflow import sha256
from src.resource_ledger import report as resource_report

class AuditError(ValueError):
    """A note, snapshot or cap violates the R1 audit contract."""


LARGE_BYTES = 32 * 1024 * 1024
DISPOSITIONS = {"supported", "unsupported", "contradicted", "not_assessed"}
OUTCOMES = {"consistent", "contradicted", "unresolved"}
READINESS = {"ready_for_autonomous_planning", "limited", "blocked"}
BASELINE_IDS = ("L01", "L02", "L03", "L04", "L05", "L06", "L07", "L08")
R2_KEYS = (
    "teacher_and_maximum_feature_space",
    "nested_feature_selection",
    "feature_and_ridge_objective_normalization",
    "separate_data_noise_feature_model_seeds",
    "planned_p_over_n_grid",
    "noise_snr_and_clean_evaluation",
    "discovery_confirmation_exposure",
    "svd_solver_rcond_rank_residual_norm_spectrum",
    "estimand_and_uncertainty",
    "independent_draws_common_evaluation_points",
    "interventions_predictions_falsifiers_inconclusive_rules",
    "timing_pilot_and_cumulative_budget",
    "readiness_and_open_issues",
)
MINIMUM_LEGACY_IDS = {
    "LEG-MLP-ROUNDED-LOG",
    "LEG-MLP-PARTIAL-CURVE",
    "LEG-MLP-NO-COMPLETED-1024",
    "LEG-MLP-NO-CHECKPOINT-FILES",
    "LEG-PCA-CHOICE",
    "LEG-PARTIAL-META-768",
    "LEG-PROBE-4096",
    "LEG-CACHE-KEY-ENDPOINTS",
    "LEG-CACHE-HASH-MISMATCH",
    "LEG-PILOT-NOT-CORPUS",
    "LEG-PILOT-H1-H4",
    "LEG-SCOPE-EMBEDDINGGEMMA",
    "LEG-SOURCE-MANIFEST",
    "LEG-EMB-TRAIN-UNHASHED",
    "LEG-EMB-PARTIAL-UNHASHED",
    "LEG-SPLIT-PARQUET-UNHASHED",
}
INPUTS = {
    "legacy_notes": "research/stages/R1/inputs/legacy-notes.json",
    "source_cards": "research/stages/R1/inputs/source-cards.json",
    "intake_index": "research/stages/R1/inputs/intake-index.json",
    "r2_design": "research/stages/R1/inputs/R2-design.json",
    "preparation": "research/stages/R1/inputs/preparation-log.jsonl",
    "design_review": "research/stages/R1/inputs/design-review.json",
    "dnn_listing": "research/stages/R1/inputs/dnn-directory-listing.json",
    "inventory": "research/evidence/legacy-2026-11-07/inventory.json",
    "registry": "docs/literature/sources.json",
    "contract": "research/stages/R1/TASK-CONTRACT.json",
    "r1_authorization": "research/decisions/R1-authorization.json",
    "charter": "research/decisions/autonomous-research-authorization.json",
    "stages": "research/stages.json",
}


def repo_file(root: Path, relative: str, *, must_exist: bool = True) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or relative.startswith("/"):
        raise AuditError(f"Unsafe relative path: {relative!r}")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise AuditError(f"Unsafe relative path: {relative}")
    path = root.joinpath(*parts)
    if path.is_symlink():
        raise AuditError(f"Refusing symlink: {relative}")
    if not path.resolve().is_relative_to(root.resolve()):
        raise AuditError(f"Path escapes the repository: {relative}")
    if must_exist and not path.is_file():
        raise AuditError(f"Required file is missing: {relative}")
    return path


def load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"Expected object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AuditError(f"Cannot read {path}: {exc}") from exc
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditError(f"Invalid JSONL {path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise AuditError(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def _text(value: Any, label: str, *, minimum: int = 1) -> str:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise AuditError(f"{label} must be substantive text")
    return value


def _read_text_file(path: Path) -> str:
    size = path.stat().st_size
    if size > LARGE_BYTES:
        raise AuditError(f"Refusing to read a large file: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise AuditError(f"Evidence text is not UTF-8: {path}") from exc


def inventory_index(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = inventory.get("files")
    if not isinstance(rows, list):
        raise AuditError("Inventory has no files list")
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise AuditError("Inventory row needs a path")
        if row["path"] in index:
            raise AuditError(f"Duplicate inventory path: {row['path']}")
        index[row["path"]] = row
    return index


def check_observation(root: Path, observation: dict[str, Any], inventory: dict[str, dict[str, Any]]) -> dict[str, Any]:
    for key in ("observation_id", "statement", "evidence_path", "sha256", "locator", "disposition", "scope", "limitations"):
        if key not in observation:
            raise AuditError(f"Observation missing {key}")
    disposition = observation["disposition"]
    if disposition not in DISPOSITIONS:
        raise AuditError(f"Invalid disposition: {disposition}")
    _text(observation["statement"], "statement", minimum=20)
    _text(observation["locator"], "locator", minimum=3)
    _text(observation["scope"], "scope", minimum=10)
    _text(observation["limitations"], "limitations", minimum=10)
    attested = observation.get("content_attestation", True)
    if attested is False:
        if observation["sha256"] is not None:
            raise AuditError(f"{observation['observation_id']} claims a hash without content attestation")
        if disposition != "not_assessed":
            raise AuditError("Unhashed historical content must stay not_assessed")
        entry = inventory.get(observation["evidence_path"])
        if entry is None:
            raise AuditError(f"Unhashed observation is not in the inventory: {observation['evidence_path']}")
        if entry.get("status") != "local_only_not_hashed" or entry.get("sha256") is not None:
            raise AuditError(f"Inventory does not mark {observation['evidence_path']} as unhashed")
        if observation.get("inventory_bytes") != entry.get("bytes"):
            raise AuditError(f"Inventory byte count mismatch: {observation['evidence_path']}")
        live = repo_file(root, observation["evidence_path"], must_exist=False)
        live_bytes = live.stat().st_size if live.is_file() and not live.is_symlink() else None
        checked = dict(observation)
        checked["verified_structural"] = True
        checked["live_bytes"] = live_bytes
        checked["live_bytes_match_inventory"] = live_bytes == entry.get("bytes")
        return checked
    digest = observation["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise AuditError(f"Attested observation needs a SHA-256: {observation['observation_id']}")
    path = repo_file(root, observation["evidence_path"])
    if path.stat().st_size > LARGE_BYTES:
        raise AuditError(f"Refusing to hash a large attested file: {observation['evidence_path']}")
    if sha256(path) != digest:
        raise AuditError(f"SHA-256 mismatch: {observation['evidence_path']}")
    quote = observation.get("quote")
    if not isinstance(quote, str) or len(quote) < 4:
        raise AuditError(f"Attested observation needs a quote: {observation['observation_id']}")
    text = _read_text_file(path)
    if quote not in text:
        raise AuditError(f"Quote not found in {observation['evidence_path']}")
    line_number = observation.get("locator_line")
    if line_number is not None:
        if not isinstance(line_number, int) or isinstance(line_number, bool) or line_number < 1:
            raise AuditError("locator_line must be a positive integer")
        lines = text.splitlines()
        if line_number > len(lines) or quote not in lines[line_number - 1]:
            raise AuditError(f"Locator line does not contain the quote: {observation['observation_id']}")
    checked = dict(observation)
    checked["verified_structural"] = True
    return checked


def audit_legacy(root: Path, notes: dict[str, Any], inventory: dict[str, Any], *,
                 required_ids: set[str] | None = None) -> dict[str, Any]:
    observations = notes.get("observations")
    if not isinstance(observations, list) or not observations:
        raise AuditError("Legacy notes need observations")
    required = MINIMUM_LEGACY_IDS if required_ids is None else required_ids
    index = inventory_index(inventory)
    checked = [check_observation(root, item, index) for item in observations]
    ids = [item["observation_id"] for item in checked]
    if len(ids) != len(set(ids)):
        raise AuditError("Duplicate legacy observation IDs")
    missing = sorted(required - set(ids))
    if missing:
        raise AuditError(f"Missing in-scope legacy observations: {missing}")
    _prediction(notes, "P1")
    return {
        "schema_version": 1,
        "stage_id": "R1",
        "task": "legacy-evidence",
        "author_agent": _text(notes.get("author_agent"), "author_agent"),
        "scientific_validity": "not_assessed",
        "observations": checked,
        "prediction_assessments": notes["prediction_assessments"],
        "negative_and_not_assessed": [item["observation_id"] for item in checked
                                      if item["disposition"] in ("unsupported", "contradicted", "not_assessed")],
    }


def _prediction(document: dict[str, Any], expected: str) -> dict[str, Any]:
    rows = document.get("prediction_assessments")
    if not isinstance(rows, list):
        raise AuditError("Missing prediction_assessments")
    match = [row for row in rows if isinstance(row, dict) and row.get("id") == expected]
    if len(match) != 1:
        raise AuditError(f"Document must assess {expected} once")
    row = match[0]
    if row.get("author_outcome") not in OUTCOMES:
        raise AuditError(f"Invalid author outcome for {expected}")
    _text(row.get("reason"), f"{expected} reason", minimum=20)
    if not isinstance(row.get("observation_ids"), list) or not row["observation_ids"]:
        raise AuditError(f"{expected} needs observation ids")
    return row


def _quotes_in_text(locators: Any, text: str, source_id: str) -> None:
    if not isinstance(locators, list) or not locators:
        raise AuditError(f"{source_id} full text needs locators")
    for locator in locators:
        if not isinstance(locator, dict):
            raise AuditError(f"{source_id} locator must be an object")
        _text(locator.get("label"), f"{source_id} locator label", minimum=2)
        quote = locator.get("quote")
        if not isinstance(quote, str) or len(quote) < 12 or quote not in text:
            raise AuditError(f"{source_id} locator quote is absent from the extracted text")


def audit_card(root: Path, card: dict[str, Any], registry_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_id = card.get("source_id")
    if source_id not in registry_by_id and card.get("additional") is not True:
        raise AuditError(f"Unknown baseline source: {source_id}")
    if card.get("additional") is True:
        _text(card.get("gap"), f"{source_id} gap", minimum=20)
    registered = registry_by_id.get(source_id)
    if registered is not None:
        if card.get("title") != registered.get("title") or card.get("authors") != registered.get("authors"):
            raise AuditError(f"{source_id} title or authors differ from the source registry")
        if card.get("registry_version") != registered.get("version"):
            raise AuditError(f"{source_id} registry_version differs from the source registry")
    _text(card.get("version"), f"{source_id} version", minimum=3)
    _text(card.get("retrieved_utc"), f"{source_id} retrieved_utc", minimum=10)
    access = card.get("access_status")
    review = card.get("review_level")
    if access == "full_text":
        if review != "full_text":
            raise AuditError(f"{source_id} cannot raise review level above the retrieved text")
        if card.get("unavailable_reason") is not None:
            raise AuditError(f"{source_id} full text cannot also be unavailable")
        snapshot = repo_file(root, card.get("snapshot_path"))
        extracted = repo_file(root, card.get("text_path"))
        if snapshot.stat().st_size > 25 * 1024 * 1024:
            raise AuditError(f"{source_id} snapshot exceeds the document cap")
        if sha256(snapshot) != card.get("snapshot_sha256") or sha256(extracted) != card.get("text_sha256"):
            raise AuditError(f"{source_id} snapshot or extract hash mismatch")
        with snapshot.open("rb") as stream:
            if stream.read(5) != b"%PDF-":
                raise AuditError(f"{source_id} snapshot is not a PDF")
        _quotes_in_text(card.get("locators"), _read_text_file(extracted), source_id)
    elif access in ("unavailable", "abstract_only"):
        if review == "full_text" or review != access:
            raise AuditError(f"{source_id} abstract-only or unavailable text was promoted")
        if card.get("snapshot_sha256") is not None or card.get("text_sha256") is not None:
            raise AuditError(f"{source_id} missing text cannot carry a content hash")
        _text(card.get("unavailable_reason"), f"{source_id} unavailable_reason", minimum=10)
    else:
        raise AuditError(f"Invalid access_status: {access}")
    for key in ("model_and_data", "loss", "varied_axis", "held_fixed_axes", "interpolation_definition",
                "result_type", "transfer_limits"):
        _text(card.get(key), f"{source_id} {key}", minimum=8)
    assumptions = card.get("assumptions")
    if not isinstance(assumptions, list) or not assumptions or any(not isinstance(item, str) or len(item.strip()) < 8 for item in assumptions):
        raise AuditError(f"{source_id} needs assumption statements")
    return dict(card)


def retrieval_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if "url" in record and "attempt" in record]


def build_intake_index(root: Path, records: list[dict[str, Any]], intake_relative: str) -> dict[str, Any]:
    requests = retrieval_records(records)
    if len(requests) > 48:
        raise AuditError("HTTP request cap exceeded")
    by_source: dict[str, list[dict[str, Any]]] = {}
    total = 0
    stored: list[dict[str, Any]] = []
    for record in requests:
        source_id = record.get("source_id")
        if not isinstance(source_id, str):
            raise AuditError("Retrieval record missing source_id")
        by_source.setdefault(source_id, []).append(record)
        if record.get("status") == "retrieved":
            amount = record.get("bytes")
            if not isinstance(amount, int) or isinstance(amount, bool) or amount < 0 or amount > 25 * 1024 * 1024:
                raise AuditError(f"Retrieved document outside the per-document cap: {source_id}")
            relative = record.get("snapshot_path")
            if not isinstance(relative, str):
                raise AuditError(f"Retrieved record missing snapshot path: {source_id}")
            path = repo_file(root, f"{intake_relative}/{relative}")
            if path.stat().st_size != amount or sha256(path) != record.get("sha256"):
                raise AuditError(f"Stored snapshot does not match acquisition log: {source_id}")
            total += amount
            stored.append({"source_id": source_id, "bytes": amount, "sha256": record["sha256"],
                           "snapshot_path": f"{intake_relative}/{relative}", "url": record.get("url"),
                           "attempt": record.get("attempt")})
    if total > 200 * 1024 * 1024:
        raise AuditError("Total download cap exceeded")
    for source_id, rows in by_source.items():
        if len(rows) > 2:
            raise AuditError(f"More than two retrieval attempts: {source_id}")
    return {
        "schema_version": 1,
        "intake_path": intake_relative,
        "http_requests": len(requests),
        "download_bytes": total,
        "attempts_by_source": {key: len(value) for key, value in sorted(by_source.items())},
        "stored": stored,
        "dataset_download_bytes": 0,
        "model_training_jobs": 0,
    }


def audit_literature(root: Path, document: dict[str, Any], registry: dict[str, Any],
                     intake_index: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    registered = registry.get("sources")
    if not isinstance(registered, list):
        raise AuditError("Source registry has no sources")
    registry_by_id = {item["id"]: item for item in registered if isinstance(item, dict) and "id" in item}
    if tuple(registry_by_id) != BASELINE_IDS and set(BASELINE_IDS) - set(registry_by_id):
        raise AuditError("Baseline registry does not contain L01-L08")
    cards = document.get("cards")
    if not isinstance(cards, list):
        raise AuditError("Literature notes need cards")
    checked = [audit_card(root, card, registry_by_id) for card in cards]
    ids = [card["source_id"] for card in checked]
    if len(ids) != len(set(ids)):
        raise AuditError("Duplicate source cards")
    missing = [source_id for source_id in BASELINE_IDS if source_id not in ids]
    if missing:
        raise AuditError(f"Missing baseline cards: {missing}")
    additional = [card for card in checked if card.get("additional") is True]
    if len(additional) > 4:
        raise AuditError("More than four additional primary sources")
    intake_relative = document.get("intake_path")
    if not isinstance(intake_relative, str):
        raise AuditError("Literature notes need intake_path")
    rebuilt = build_intake_index(root, records, intake_relative)
    if intake_index != rebuilt:
        raise AuditError("Intake index does not match the acquisition log and stored bytes")
    for card in checked:
        if card["access_status"] != "full_text":
            continue
        stored = [item for item in rebuilt["stored"] if item["source_id"] == card["source_id"]]
        if len(stored) != 1 or stored[0]["sha256"] != card["snapshot_sha256"]:
            raise AuditError(f"Full-text card is not the retrieved snapshot: {card['source_id']}")
        if stored[0]["url"] != card.get("retrieved_url"):
            raise AuditError(f"Retrieved URL disagrees with the card: {card['source_id']}")
    comparison = document.get("tree_axis_comparison")
    if not isinstance(comparison, dict) or comparison.get("sources") != ["L05", "L06", "L07"]:
        raise AuditError("Tree-axis comparison must cover L05-L07")
    if not isinstance(comparison.get("resolved"), bool):
        raise AuditError("Tree-axis comparison needs an explicit resolved flag")
    _text(comparison.get("statement"), "tree comparison", minimum=40)
    if comparison["resolved"] is True:
        quotes = comparison.get("agreement_quotes")
        if not isinstance(quotes, dict) or set(quotes) != {"L05", "L06", "L07"}:
            raise AuditError("A resolved tree comparison needs a quote from each paper")
    _prediction(document, "P2")
    return {
        "schema_version": 1,
        "stage_id": "R1",
        "task": "literature",
        "author_agent": _text(document.get("author_agent"), "author_agent"),
        "scientific_validity": "not_assessed",
        "cards": checked,
        "tree_axis_comparison": comparison,
        "prediction_assessments": document["prediction_assessments"],
        "intake": rebuilt,
        "unavailable": [card["source_id"] for card in checked if card["access_status"] != "full_text"],
    }


def _substantive(value: Any, label: str) -> None:
    if isinstance(value, str):
        _text(value, label, minimum=40)
        return
    if isinstance(value, list) and value:
        return
    if isinstance(value, dict) and value:
        return
    raise AuditError(f"{label} is empty")


def audit_r2(design: dict[str, Any]) -> dict[str, Any]:
    for key in R2_KEYS:
        if key not in design:
            raise AuditError(f"R2 design missing {key}")
        _substantive(design[key], key)
    if design.get("models_executed") is not False:
        raise AuditError("R2 proposal must record that no model was executed")
    readiness = design.get("readiness")
    if readiness not in READINESS:
        raise AuditError("Invalid R2 readiness")
    issues = design.get("open_issues")
    if not isinstance(issues, list) or not issues:
        raise AuditError("R2 design needs open issues")
    interventions = design["interventions_predictions_falsifiers_inconclusive_rules"]
    if not isinstance(interventions, list) or len(interventions) < 3:
        raise AuditError("R2 needs at least three intervention records")
    for item in interventions:
        if not isinstance(item, dict):
            raise AuditError("Intervention must be an object")
        for key in ("intervention", "prediction", "falsifier", "inconclusive"):
            _text(item.get(key), f"intervention {key}", minimum=20)
    if design.get("discovery_data_reused_as_confirmation") is not False:
        raise AuditError("Discovery rows cannot be relabeled as confirmation")
    _prediction(design, "P3")
    return {
        "schema_version": 1,
        "stage_id": "R1",
        "task": "r2-design",
        "author_agent": _text(design.get("author_agent"), "author_agent"),
        "scientific_validity": "not_assessed",
        "models_executed": False,
        "readiness": readiness,
        "open_issues": issues,
        "design": {key: design[key] for key in R2_KEYS},
        "interventions": interventions,
        "prediction_assessments": design["prediction_assessments"],
    }


def audit_preparation(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records or records[-1].get("kind") != "prefreeze_seal":
        raise AuditError("Preparation log must be sealed as its last record")
    kinds = {record.get("kind") for record in records}
    for kind in ("status", "exposure", "retrieval", "implementation", "prefreeze_seal"):
        if kind not in kinds:
            raise AuditError(f"Preparation log missing {kind}")
    for record in records:
        if not isinstance(record.get("ts"), str) or len(record["ts"]) < 20:
            raise AuditError("Preparation record needs a timestamp")
    return {"records": len(records), "sealed": True}


def audit_design_review(review: dict[str, Any]) -> dict[str, Any]:
    author = _text(review.get("author_agent"), "author_agent")
    reviewer = _text(review.get("reviewer_agent"), "reviewer_agent")
    if author.casefold() == reviewer.casefold():
        raise AuditError("Design reviewer must be a different agent from the author")
    if review.get("review_mode") != "subagent_review" or review.get("harness") != "cursor":
        raise AuditError("Design review must be a Cursor subagent review")
    if review.get("verdict") != "accept":
        raise AuditError("Design review verdict is not accept")
    if not isinstance(review.get("blocking_issues"), list):
        raise AuditError("Design review needs a blocking_issues list")
    if review["blocking_issues"]:
        raise AuditError("Accepted design review cannot retain blocking issues")
    _text(review.get("notes"), "design review notes", minimum=40)
    return {"author_agent": author, "reviewer_agent": reviewer, "verdict": "accept"}


def audit_decision_digest(root: Path) -> dict[str, str]:
    stages = load_object(repo_file(root, INPUTS["stages"]))
    entry = next(item for item in stages["stages"] if item.get("id") == "R1")
    expected = {
        INPUTS["charter"]: entry["authorization_decision"]["sha256"],
        INPUTS["r1_authorization"]: entry["stage_design_decision"]["sha256"],
    }
    for relative, digest in expected.items():
        if sha256(repo_file(root, relative)) != digest:
            raise AuditError(f"Decision digest mismatch: {relative}")
    authorization = load_object(repo_file(root, INPUTS["r1_authorization"]))
    contract = INPUTS["contract"]
    if authorization["task_contract"]["sha256"] != sha256(repo_file(root, contract)):
        raise AuditError("Task contract digest mismatch")
    if authorization["authorization_decision"]["sha256"] != sha256(repo_file(root, INPUTS["charter"])):
        raise AuditError("R1 authorization is not bound to the charter bytes")
    task = load_object(repo_file(root, contract))
    if task["authorization_decision"]["sha256"] != sha256(repo_file(root, INPUTS["charter"])):
        raise AuditError("Task contract is not bound to the charter bytes")
    return {"charter_sha256": expected[INPUTS["charter"]], "r1_authorization_sha256": expected[INPUTS["r1_authorization"]]}


def attested_paths(legacy: dict[str, Any], literature: dict[str, Any]) -> list[str]:
    paths = [item["evidence_path"] for item in legacy["observations"] if item.get("content_attestation", True) is not False]
    for card in literature["cards"]:
        if card["access_status"] == "full_text":
            paths.extend((card["snapshot_path"], card["text_path"]))
    return paths


def unhashed_historical_paths(legacy: dict[str, Any]) -> list[str]:
    return [item["evidence_path"] for item in legacy["observations"] if item.get("content_attestation") is False]


def audit_attempt_binding(attempt: Path, attested: list[str], unhashed: list[str]) -> None:
    manifest_path = attempt / "manifest.json"
    if not manifest_path.is_file():
        raise AuditError("Attempt manifest is missing")
    manifest = load_object(manifest_path)
    hashes = manifest.get("input_hashes")
    if not isinstance(hashes, dict):
        raise AuditError("Attempt has no input hash inventory")
    for relative in attested:
        if hashes.get(relative) != sha256(repo_file(attempt / "inputs", relative)):
            raise AuditError(f"Attested path is absent from the frozen input manifest: {relative}")
    for relative in unhashed:
        if relative in hashes:
            raise AuditError(f"Unhashed historical file was copied into the attempt: {relative}")


def check_dnn_listing(root: Path, listing: dict[str, Any]) -> dict[str, Any]:
    if listing.get("path") != "result/05_dnn" or not isinstance(listing.get("entries"), list):
        raise AuditError("DNN listing must name result/05_dnn and its entries")
    directory = root / "result" / "05_dnn"
    if directory.is_symlink() or not directory.is_dir():
        raise AuditError("result/05_dnn is missing; listing cannot be checked")
    live = []
    for child in sorted(directory.iterdir(), key=lambda item: item.name):
        if child.is_symlink():
            raise AuditError("Refusing a symlink inside result/05_dnn")
        if child.is_file():
            live.append({"name": child.name, "bytes": child.stat().st_size})
    if live != listing["entries"]:
        raise AuditError("Frozen DNN directory listing does not match the live directory")
    return {"path": "result/05_dnn", "entries": live, "content_read": False}


def render_legacy_md(audit: dict[str, Any]) -> str:
    lines = ["# R1 legacy audit", "", "Descriptive audit. `scientific_validity=not_assessed`.", ""]
    for item in audit["observations"]:
        lines.extend([
            f"## {item['observation_id']}",
            "",
            item["statement"],
            "",
            f"- Disposition: `{item['disposition']}`",
            f"- Evidence: `{item['evidence_path']}`",
            f"- SHA-256: `{item['sha256']}`",
            f"- Locator: {item['locator']}",
            f"- Scope: {item['scope']}",
            f"- Limitations: {item['limitations']}",
            "",
        ])
    return "\n".join(lines)


def render_card_md(card: dict[str, Any]) -> str:
    locators = "\n".join(f"- {item['label']}: {item['quote']}" for item in card.get("locators") or [])
    assumptions = "\n".join(f"- {item}" for item in card["assumptions"])
    return "\n".join([
        f"# {card['source_id']} — {card['title']}",
        "",
        f"Access: `{card['access_status']}`. Review level: `{card['review_level']}`.",
        f"Version read: {card['version']}. Registry version: {card.get('registry_version')}.",
        f"Retrieved: {card['retrieved_utc']}.",
        "",
        "## Locators",
        "",
        locators or "- No full-text locator.",
        "",
        "## Assumptions and limits",
        "",
        assumptions,
        "",
        card["transfer_limits"],
        "",
    ])


def render_r2_md(audit: dict[str, Any]) -> str:
    lines = [
        "# R2 design proposal",
        "",
        "No model was executed. `scientific_validity=not_assessed`.",
        f"Readiness: `{audit['readiness']}`.",
        "",
        "## Open issues",
        "",
    ]
    lines.extend(f"- {issue}" for issue in audit["open_issues"])
    lines.append("")
    for key, value in audit["design"].items():
        lines.extend([f"## {key}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2), "```", ""])
    return "\n".join(lines)


def validation_document(root: Path, attempt: Path) -> dict[str, Any]:
    legacy_notes = load_object(repo_file(root, INPUTS["legacy_notes"]))
    inventory = load_object(repo_file(root, INPUTS["inventory"]))
    literature_notes = load_object(repo_file(root, INPUTS["source_cards"]))
    registry = load_object(repo_file(root, INPUTS["registry"]))
    intake_index = load_object(repo_file(root, INPUTS["intake_index"]))
    design = load_object(repo_file(root, INPUTS["r2_design"]))
    preparation = load_jsonl(repo_file(root, INPUTS["preparation"]))
    review = load_object(repo_file(root, INPUTS["design_review"]))
    listing = load_object(repo_file(root, INPUTS["dnn_listing"]))
    intake_relative = literature_notes["intake_path"]
    records = load_jsonl(repo_file(root, f"{intake_relative}/acquisition.jsonl"))
    legacy = audit_legacy(root, legacy_notes, inventory)
    literature = audit_literature(root, literature_notes, registry, intake_index, records)
    proposal = audit_r2(design)
    prepared = audit_preparation(preparation)
    reviewed = audit_design_review(review)
    authors = {legacy["author_agent"], literature["author_agent"], proposal["author_agent"], reviewed["author_agent"]}
    if len(authors) != 1:
        raise AuditError("Author agent differs across legacy, literature, R2 and design review")
    digest = audit_decision_digest(root)
    listing_check = check_dnn_listing(root, listing)
    attested = attested_paths(legacy, literature)
    audit_attempt_binding(attempt, attested, unhashed_historical_paths(legacy))
    accounting = resource_report(root, expected_sha256=digest["charter_sha256"])
    if accounting["network"]["dataset_download_bytes"] != 0:
        raise AuditError("R1 recorded a dataset download")
    limitations = [
        "Structural validation does not score scientific truth.",
        "Large historical matrices were not read or hashed.",
        "Extracted PDF text can drop or reorder mathematical notation.",
        "A Cursor subagent review is local team review, not external replication.",
    ]
    return {
        "schema_version": 1,
        "stage_id": "R1",
        "scientific_validity": "not_assessed",
        "structural": {
            "legacy": "pass",
            "literature": "pass",
            "r2": "pass",
            "preparation": "pass",
            "design_review": "pass",
            "decision_digest": "pass",
            "attempt_binding": "pass",
            "resource_accounting": "pass" if accounting["accounting_implemented"] else "fail",
        },
        "criteria_structural_coverage": {
            "C1": "pass", "C2": "pass", "C3": "pass", "C4": "pass", "C5": "pass", "C6": "pass",
        },
        "semantic_review": "required_of_distinct_subagent",
        "author_agent": reviewed["author_agent"],
        "design_reviewer_agent": reviewed["reviewer_agent"],
        "dnn_listing": listing_check,
        "preparation": prepared,
        "decision_digest": digest,
        "resource_accounting": accounting,
        "unavailable_sources": literature["unavailable"],
        "negative_and_not_assessed": legacy["negative_and_not_assessed"],
        "r2_readiness": proposal["readiness"],
        "predictions_author_recorded": [
            next(item for item in legacy["prediction_assessments"] if item["id"] == "P1"),
            next(item for item in literature["prediction_assessments"] if item["id"] == "P2"),
            next(item for item in proposal["prediction_assessments"] if item["id"] == "P3"),
        ],
        "limitations": limitations,
        "computational_activation_ready": accounting["computational_activation_ready"],
    }

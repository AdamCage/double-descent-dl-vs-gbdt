"""Cursor-driven, provider-free planning bridge; never writes historical stores.

``bootstrap`` records a research question, not an experiment preregistration or
scientific approval. ``status`` and ``verify`` open the new store read-only.
The vendored directory is named 0.2.0; its package metadata is version 0.1.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from episteme.commands import CommandService
from episteme.store import Store

ROOT = Path(__file__).resolve().parents[1]
STUDY = "dd-mechanisms-v1"
STUDY_ROOT = ROOT / ".research" / STUDY
PROGRAM = ROOT / "docs" / "research-program.md"
POINTER = "question_pointer.json"


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def question_spec(program: Path = PROGRAM) -> dict:
    """Freeze the actual program bytes; edits require a new explicit revision."""
    return {
        "study_id": STUDY,
        "statement": "При каких условиях двойной спуск возникает, исчезает и меняет положение у нейросетей, линейных моделей, случайных признаков и деревьев/GBDT?",
        "objective": "Различить механизмы интерполяции, спектра, шума, регуляризации и оптимизации контролируемыми проверками, сохраняя отрицательные результаты.",
        "scope": {
            "program_sha256": hashlib.sha256(program.read_bytes()).hexdigest(),
            "model_families": "linear/ridge, random features, MLP, trees/GBDT",
            "inference": "question-only planning; no experiment protocol frozen",
            "legacy_scope": "essay/Qwen results are historical exploratory evidence only",
        },
        "constraints": [
            "Cursor with Grok 4.7 Extra High and distinct reviewer subagents designs, implements, executes, analyzes and writes within the user-authorized research charter; human and Codex provide upper-level critique and critical decisions, not mandatory approval of every stage or claim.",
            "Record protocol, code, environment, data/split hashes, seeds, budgets and stopping rules before each new experiment; a question is not preregistration.",
            "Measure interpolation and optimization separately; a monotone loss drop or late boosting acceleration alone is not double descent.",
            "Keep model-wise, epoch-wise and sample-wise axes distinct; never equate parameter count across model families.",
            "Discovery and confirmation use distinct frozen protocols and exposure records; do not tune a rule on confirmatory test outcomes.",
            "Missing evidence, failed runs and negative findings cannot become support; preserve raw artifacts, logs and competing explanations.",
            "Same-code replay, fresh-seed repetition, independent reanalysis and new-data replication are different modes.",
            "Logical actor labels do not establish reviewer independence or OS isolation; scientific_validity remains not_assessed until actual scientific review.",
            "Do not mutate the original .research/double-descent or registration-scope-mismatch stores; the historical text domain pack is not bound to this question.",
        ],
        "stopping_criteria": [
            "Bootstrap stops after recording one question and its immutable digest; it launches no experiments and no model providers.",
            "Each subsequent stage must have its own finite resource/attempt/time budget and explicit success, negative and inconclusive dispositions.",
            "Critical project forks, substantial unexpected results after bounded diagnosis, unresolved central integrity problems or changes beyond the resource mandate return to Codex/human; routine detailed design and versioned confirmation within the charter remain autonomous.",
        ],
    }


def _new_root(root: Path) -> Path:
    """Imported helpers allow isolated tests, but never historical store names."""
    root = root.resolve()
    if root.name != STUDY:
        raise ValueError(f"bridge only supports the new {STUDY} store")
    return root


def _envelope(spec: dict, digest: str) -> dict:
    return {
        "context": {
            "command_id": f"{STUDY}-question-{digest}",
            "expected_revision": 0,
            "actor": "local-question-recorder",
            "role": "planner",
            "study_id": STUDY,
            "correlation_id": f"{STUDY}-bootstrap",
            "causation_id": None,
        },
        "request": {"version": 1, "action": "planning.question", "payload": spec},
    }


def bootstrap(root: Path = STUDY_ROOT, program: Path = PROGRAM) -> dict:
    root = _new_root(root)
    spec = question_spec(program)
    digest = hashlib.sha256(_canonical(spec)).hexdigest()
    if root.exists() and (root / "state.sqlite3").exists():
        # Read first: a changed declaration must fail without writable opening.
        with Store(root, read_only=True) as existing:
            populated = bool(existing.events())
            existing.receipts()
        if populated:
            current = inspect(root)
            if current["question_spec_sha256"] != digest:
                raise ValueError("research program changed after question registration; use an explicit new planning revision, never overwrite")
            if (root / POINTER).exists():
                return current
    with Store(root) as store:
        question = CommandService(store).execute(_envelope(spec, digest))
        events = store.events()
        event = next(item for item in events if item["id"] == question)
        pointer = {
            "schema_version": 1,
            "study_id": STUDY,
            "question": question,
            "question_hash": event["hash"],
            "question_spec_sha256": digest,
            "registration_kind": "research_question_only",
            "scientific_validity": "not_assessed",
            "review": "not_requested",
            "isolation": "same OS user; no sandbox; actor labels are not independent review",
        }
    # Exclusive creation makes a lost-response retry recover from the receipt
    # without overwriting any already persisted declaration.
    try:
        with (root / POINTER).open("x", encoding="utf-8", newline="\n") as output:
            output.write(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        if json.loads((root / POINTER).read_text(encoding="utf-8")) != pointer:
            raise ValueError("existing question pointer differs from committed question")
    return inspect(root)


def inspect(root: Path = STUDY_ROOT) -> dict:
    """Verify hash chains, receipts and question binding without writes."""
    root = _new_root(root)
    if not (root / "state.sqlite3").is_file():
        return {"study_id": STUDY, "status": "not_initialized",
                "scientific_validity": "not_assessed", "review": "not_requested"}
    with Store(root, read_only=True) as store:
        events, receipts = store.events(), store.receipts()
        questions = [item for item in events if item["kind"] == "research_question"]
        if not questions:
            if not events and not receipts:
                return {"study_id": STUDY, "status": "not_initialized",
                        "scientific_validity": "not_assessed", "review": "not_requested"}
            raise ValueError("new store has no registered question")
        question = questions[0]
        matching = [item for item in receipts if item["result"] == question["id"]
                    and item["request"]["action"] == "planning.question"]
        if len(matching) != 1:
            raise ValueError("question lacks its unique bootstrap command receipt")
        payload = matching[0]["request"]["payload"]
        # CommandService inserts these optional defaults. The bootstrap digest
        # covers the supplied declaration rather than these API defaults.
        spec = {key: value for key, value in payload.items()
                if key not in {"parent", "revision_reason"}}
        digest = hashlib.sha256(_canonical(spec)).hexdigest()
        if matching[0]["context"]["command_id"] != f"{STUDY}-question-{digest}":
            raise ValueError("question declaration differs from bootstrap digest")
        result = {
            "study_id": STUDY,
            "status": "question_registered",
            "registration_kind": "research_question_only",
            "question": question["id"],
            "question_hash": question["hash"],
            "question_spec_sha256": digest,
            "events": len(events),
            "receipts": len(receipts),
            "checkpoint": store.external_checkpoint(),
            "scientific_validity": "not_assessed",
            "review": "not_requested",
        }
    if (root / POINTER).exists():
        pointer = json.loads((root / POINTER).read_text(encoding="utf-8"))
        for field in ("study_id", "question", "question_hash", "question_spec_sha256",
                      "registration_kind", "scientific_validity", "review"):
            if pointer.get(field) != result[field]:
                raise ValueError(f"question pointer mismatch: {field}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "verify", "bootstrap"))
    args = parser.parse_args()
    try:
        result = bootstrap() if args.command == "bootstrap" else inspect()
        if args.command == "verify" and result["status"] == "not_initialized":
            raise ValueError("new store not initialized; bootstrap explicitly when the research question is ready")
    except (ValueError, OSError) as error:
        parser.exit(1, f"episteme bridge: {error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

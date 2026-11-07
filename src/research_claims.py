"""Validate exact claim/evidence/decision bindings for paper-supporting claims.

This is a local integrity gate, not authentication. ``reviewer`` and ``harness``
are logical labels supplied by the caller; this code cannot prove identity,
independence, scientific correctness, or that the named agent actually reviewed.
Every entry in ``claims`` is treated as paper-supporting. Draft proposals belong
outside that accepted-claim registry until their separate decision exists.
Statement digests cover exact UTF-8 bytes, without whitespace normalization.

The optional Cursor lane admits local team-reviewed claims under a byte-bound
user authorization. It requires a distinct review-subagent label, not a second
human/Codex signature for each claim. Admission is neither certification of
scientific truth nor evidence of independent scientific replication.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Any

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _text(value: Any) -> bool:
    return type(value) is str and bool(value.strip())


def _sha256(value: Any) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(root: Path, binding: Any, label: str,
                problems: list[str]) -> Path | None:
    if type(binding) is not dict or set(binding) != {"path", "sha256"}:
        problems.append(f"{label}: expected exactly path and sha256")
        return None
    raw_path, expected = binding["path"], binding["sha256"]
    if not _text(raw_path) or not _sha256(expected):
        problems.append(f"{label}: nonempty relative path and lowercase SHA-256 required")
        return None
    # Reject both platform syntaxes, including C:drive-relative paths and UNC.
    if PurePosixPath(raw_path).is_absolute() or PureWindowsPath(raw_path).drive \
            or PureWindowsPath(raw_path).root:
        problems.append(f"{label}: path must be relative to the repository")
        return None
    try:
        path = (root / raw_path).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            problems.append(f"{label}: path escapes the repository or is not a file")
            return None
        if _digest(path) != expected:
            problems.append(f"{label}: SHA-256 mismatch")
            return None
    except (OSError, ValueError, UnicodeError, RuntimeError) as error:
        problems.append(f"{label}: cannot read bound file ({type(error).__name__})")
        return None
    return path


def _evidence(root: Path, value: Any, label: str,
              problems: list[str]) -> list[dict] | None:
    if type(value) is not list or not value:
        problems.append(f"{label}: nonempty evidence list required")
        return None
    bindings, seen = [], set()
    invalid = False
    for index, binding in enumerate(value):
        if _bound_file(root, binding, f"{label}[{index}]", problems) is None:
            invalid = True
            continue
        path = binding["path"]
        if path in seen:
            problems.append(f"{label}: duplicate evidence path {path!r}")
            invalid = True
        seen.add(path)
        bindings.append(dict(binding))
    return None if invalid else bindings


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate decision JSON key")
        value[key] = item
    return value


def _invalid_number(value: str) -> None:
    raise ValueError(f"non-finite decision JSON number: {value}")


def _read_object(path: Path, label: str, problems: list[str]) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"),
                           object_pairs_hook=_unique_object,
                           parse_constant=_invalid_number)
    except (OSError, ValueError, UnicodeError, RecursionError) as error:
        problems.append(f"{label}: invalid decision JSON ({type(error).__name__})")
        return None
    if type(value) is not dict:
        problems.append(f"{label}: decision must be an object")
        return None
    return value


def _cursor_authorization(root: Path, claims_doc: dict,
                          problems: list[str]) -> dict | None:
    """Validate the opt-in authority artifact; labels are not authenticated."""
    if "autonomy_authorization" not in claims_doc:
        return None
    binding = claims_doc["autonomy_authorization"]
    path = _bound_file(root, binding, "claims autonomy_authorization", problems)
    if path is None:
        return None
    authorization = _read_object(path, "claims autonomy_authorization", problems)
    if authorization is None:
        return None
    required = {"authority": "user", "status": "accepted"}
    if (type(authorization.get("schema_version")) is not int
            or authorization["schema_version"] != 1
            or not _text(authorization.get("decision_id"))
            or any(authorization.get(key) != value for key, value in required.items())
            or authorization.get("cursor_claim_review") is not True
            or authorization.get("require_distinct_review_subagent") is not True):
        problems.append("claims autonomy_authorization: accepted user authorization "
                        "must explicitly enable Cursor claims and distinct review subagents")
        return None
    return dict(binding)


def _cursor_review(claim: dict, decision: dict, authorization: dict | None, label: str,
                   problems: list[str]) -> None:
    if claim.get("assessment") != "evidence_supported" or claim.get("review_status") != "cursor_team_reviewed":
        problems.append(f"{label}: Cursor claims require assessment=evidence_supported "
                        "and review_status=cursor_team_reviewed")
    if any(value.get("scientific_validity", "not_assessed") != "not_assessed"
           for value in (claim, decision)):
        problems.append(f"{label}: Cursor team review cannot certify scientific_validity")
    if authorization is None:
        problems.append(f"{label}: Cursor scientific review requires a valid autonomy_authorization")
    elif decision.get("authorization_decision") != authorization:
        problems.append(f"{label}: Cursor review authorization_decision must match exact charter path/hash binding")
    if decision.get("review_mode") != "subagent_review":
        problems.append(f"{label}: Cursor scientific review requires review_mode=subagent_review")
    author, reviewer = decision.get("author_agent"), decision.get("reviewer_agent")
    if not _text(author) or not _text(reviewer):
        problems.append(f"{label}: Cursor review requires author_agent and reviewer_agent")
    elif author.strip().casefold() == reviewer.strip().casefold():
        problems.append(f"{label}: Cursor review requires a distinct review subagent; sole self-review is insufficient")


def validate_claims(root: Path, claims_doc: Any) -> list[str]:
    """Return diagnostics for malformed or unbound claims; never approve science.

    The function performs only file reads, accepts already parsed JSON, and
    returns malformed-input diagnostics rather than raising validation errors.
    Bound decision JSON requires ``claim_id``, ``statement_sha256``, ``evidence``,
    ``reviewer``, ``harness`` (human/codex or explicitly authorized cursor), and
    ``verdict`` (accepted). The Cursor lane also binds the user authorization
    and distinct author/review-subagent labels. It is local project admission.
    """
    problems: list[str] = []
    try:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            return ["claims: repository root is not a directory"]
    except (TypeError, OSError, ValueError, UnicodeError, RuntimeError):
        return ["claims: invalid repository root"]
    if type(claims_doc) is not dict:
        return ["claims: document must be an object"]
    if type(claims_doc.get("schema_version")) is not int or claims_doc["schema_version"] != 1:
        problems.append("claims: schema_version must be 1")
    claims = claims_doc.get("claims")
    if type(claims) is not list:
        return problems + ["claims: claims must be a list"]
    cursor_authorization = _cursor_authorization(root, claims_doc, problems)
    seen_ids = set()
    for index, claim in enumerate(claims):
        label = f"claim[{index}]"
        if type(claim) is not dict:
            problems.append(f"{label}: claim must be an object")
            continue
        claim_id = claim.get("id")
        if not _text(claim_id):
            problems.append(f"{label}: nonempty id required")
        else:
            label = f"claim {claim_id!r}"
            if claim_id in seen_ids:
                problems.append(f"{label}: duplicate claim id")
            seen_ids.add(claim_id)
        statement = claim.get("statement")
        if not _text(statement):
            problems.append(f"{label}: nonempty statement required")
        scope = claim.get("scope")
        if not (_text(scope) or (type(scope) is dict and bool(scope)
                                and all(_text(key) for key in scope))):
            problems.append(f"{label}: nonempty scope string or object required")
        limitations = claim.get("limitations")
        if type(limitations) is not list or not limitations \
                or not all(_text(item) for item in limitations):
            problems.append(f"{label}: nonempty list of limitations required")
        evidence = _evidence(root, claim.get("evidence"), f"{label} evidence", problems)
        decision_path = _bound_file(root, claim.get("scientific_review"),
                                    f"{label} scientific_review", problems)
        if decision_path is None:
            continue
        decision = _read_object(decision_path, label, problems)
        if decision is None:
            continue
        if not _text(claim_id) or decision.get("claim_id") != claim_id:
            problems.append(f"{label}: decision claim_id mismatch")
        try:
            statement_digest = hashlib.sha256(statement.encode("utf-8")).hexdigest() \
                if _text(statement) else None
        except UnicodeError:
            statement_digest = None
        if statement_digest is None or decision.get("statement_sha256") != statement_digest:
            problems.append(f"{label}: decision statement_sha256 mismatch")
        if decision.get("verdict") != "accepted":
            problems.append(f"{label}: scientific decision verdict must be accepted")
        if decision.get("harness") == "cursor":
            _cursor_review(claim, decision, cursor_authorization, label, problems)
        elif decision.get("harness") not in ("human", "codex"):
            problems.append(f"{label}: scientific decision harness must be human, codex or authorized cursor")
        if not _text(decision.get("reviewer")):
            problems.append(f"{label}: scientific decision reviewer required")
        decision_evidence = _evidence(root, decision.get("evidence"),
                                     f"{label} decision evidence", problems)
        if evidence is not None and decision_evidence is not None:
            expected = {item["path"]: item["sha256"] for item in evidence}
            actual = {item["path"]: item["sha256"] for item in decision_evidence}
            if actual != expected:
                problems.append(f"{label}: decision evidence must match exact claim path/hash bindings")
    return problems

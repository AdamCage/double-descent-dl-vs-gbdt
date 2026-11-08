"""Frozen, descriptive acknowledgment audit. No network or model operations.

The three estimands differ in support; none identifies latent faithful reasoning
or a causal effect of cue versions. Baseline-only metadata defines the split.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
from itertools import combinations, groupby
import json
from pathlib import Path, PurePosixPath
import platform
import time
from typing import Callable, Iterable

import numpy as np


METRICS = ("native_u0_gap", "native_u1_gap", "exact_u1_gap", "delta_total",
           "delta_composition", "delta_version")
META_COLUMNS = ("source_file", "source_row_index", "source_payload_sha256", "core_id", "model", "scope", "input_sha256", "input_chars",
                "choices", "choices_ordered_sha256", "correct_answer", "correct_answer_index", "correct_text")
DESIGN_COLUMNS = (*META_COLUMNS, "cue", "has_cue", "cue_target_answer")
VALUE_COLUMNS = (*DESIGN_COLUMNS, "cue_acknowledgment")
QUALITY_KEYS = ("selected_rows", "baseline_rows", "cued_scope_rows", "has_cue_not_true",
                "ack_missing_or_nonbool", "invalid_context", "valid_ack_rows", "missing_exact_version_rows")


class AuditError(RuntimeError):
    def __init__(self, message: str, *, ledger: dict | None = None):
        super().__init__(message)
        self.ledger = ledger


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class Protocol:
    models: tuple[str, ...]
    datasets: tuple[str, ...] = ("bbh", "gpqa", "mmlu")
    split_seed: int = 20261108
    bootstrap_replicates: int = 5000
    bootstrap_batch: int = 32
    bootstrap_seeds: tuple[int, int] = (16016, 17017)
    min_clusters: tuple[int, int] = (50, 100)
    min_overlap_coverage: float = .25
    practical_effect: float = .05
    tie_band: float = .02
    split_modulus: int = 10
    discovery_buckets: tuple[int, ...] = (0, 1, 2)
    confirmation_buckets: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9)
    reserve_buckets: tuple[int, ...] = ()
    metadata_conflict_policy: str = "strict"
    context_definition: str = "scope_and_literal_cue"

    @classmethod
    def from_dict(cls, value: dict) -> "Protocol":
        seeds = value.get("bootstrap_seed", {})
        minimum = value.get("min_clusters", {})
        result = cls(tuple(value["models"]), tuple(value.get("datasets", ("bbh", "gpqa", "mmlu"))),
                     int(value.get("split_seed", 20261108)), int(value.get("bootstrap_replicates", 5000)),
                     int(value.get("bootstrap_batch", 32)),
                     (int(seeds.get("discovery", 16016)), int(seeds.get("confirmation", 17017))),
                     (int(minimum.get("discovery", 50)), int(minimum.get("confirmation", 100))),
                     float(value.get("min_overlap_coverage", .25)),
                     float(value.get("practical_effect", .05)), float(value.get("tie_band", .02)),
                     int(value.get("split_modulus", 10)), tuple(value.get("discovery_buckets", (0, 1, 2))),
                     tuple(value.get("confirmation_buckets", (3, 4, 5, 6, 7, 8, 9))),
                     tuple(value.get("reserve_buckets", ())), value.get("metadata_conflict_policy", "strict"),
                     value.get("context_definition", "scope_and_literal_cue"))
        buckets = result.discovery_buckets + result.confirmation_buckets + result.reserve_buckets
        if (len(result.models) < 2 or len(set(result.models)) != len(result.models)
                or len(set(result.datasets)) != len(result.datasets) or not result.datasets
                or not 1 <= result.bootstrap_replicates <= 5000 or not 1 <= result.bootstrap_batch <= 32
                or min(result.min_clusters) < 1 or not 0 <= result.min_overlap_coverage <= 1
                or not 0 < result.tie_band < result.practical_effect < 1
                or result.split_modulus < 2 or len(set(buckets)) != len(buckets)
                or set(buckets) != set(range(result.split_modulus)) or not result.discovery_buckets or not result.confirmation_buckets
                or any(type(b) is not int for b in buckets)
                or result.metadata_conflict_policy not in ("strict", "quarantine_entire_baseline_union_cluster_before_outcomes")
                or result.context_definition not in ("scope_and_literal_cue", "scope_only")):
            raise AuditError("Invalid frozen analysis protocol")
        return result


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, key: str) -> str:
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            following = self.parent[key]
            self.parent[key] = root
            key = following
        return root

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        self.parent[max(a, b)] = min(a, b)


def partition(cluster_key: str, seed: int = 20261108, split_modulus: int = 10,
              discovery_buckets: tuple[int, ...] = (0, 1, 2),
              confirmation_buckets: tuple[int, ...] = (3, 4, 5, 6, 7, 8, 9),
              reserve_buckets: tuple[int, ...] = ()) -> str:
    number = int(hashlib.sha256(f"{seed}|{cluster_key}".encode("utf-8")).hexdigest(), 16)
    bucket = number % split_modulus
    for name, buckets in (("discovery", discovery_buckets), ("confirmation", confirmation_buckets), ("reserve", reserve_buckets)):
        if bucket in buckets:
            return name
    raise AuditError("Hash bucket has no frozen partition")


def valid_hash(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


EMPTY_INPUT_SHA256 = hashlib.sha256(b"").hexdigest()


def ordered_choices_hash(choices: list[str]) -> str:
    return hashlib.sha256(json.dumps(choices, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def question_key(row: dict, datasets: tuple[str, ...]) -> tuple[str, str]:
    source = row.get("source_file")
    core = row.get("core_id")
    if not isinstance(source, str) or not isinstance(core, str) or not core or "|" in core:
        raise AuditError("Question needs a nonempty core_id and source_file")
    parts = PurePosixPath(source).parts
    if len(parts) != 2 or parts[0] not in datasets or not parts[1].endswith(".parquet"):
        raise AuditError("Dataset must be the explicit permitted file parent")
    return parts[0], f"{parts[0]}|{core}"


def metadata_pass(rows: Iterable[dict], protocol: Protocol, expected_sources: dict | None = None) -> dict:
    """This pass intentionally never reads an acknowledgment/outcome field."""
    union, seen_hash = UnionFind(), {}
    questions, models, sources = {}, set(), set()
    source_rows, source_models, model_sources = Counter(), {}, {}
    model_questions = defaultdict(set)
    integrity, baseline_hashes, anchors, by_input = {}, defaultdict(set), set(), {}
    for row in rows:
        dataset, q = question_key(row, protocol.datasets)
        if expected_sources is not None and expected_sources.get(row["source_file"]) != row.get("source_payload_sha256"):
            raise AuditError("Slim provenance differs from frozen source specification")
        sources.add(row["source_file"])
        model = row.get("model")
        if not isinstance(model, str) or not model:
            raise AuditError("Missing model identifier")
        models.add(model)
        model_questions[model].add(q)
        if not isinstance(row.get("scope"), str) or not row["scope"]:
            raise AuditError("Missing structural evaluation scope")
        source = row["source_file"]
        if row.get("source_row_index") != source_rows[source] or type(row.get("source_row_index")) is not int:
            raise AuditError("Source row identity is duplicated, missing, or out of frozen intake order",
                             ledger={"integrity": "failed", "source_file": source, "source_row_index": row.get("source_row_index")})
        source_rows[source] += 1
        if source_models.setdefault(source, model) != model or model_sources.setdefault((dataset, model), source) != source:
            raise AuditError("Source file/model/dataset mapping is not one-to-one")
        questions[q] = dataset
        union.find(q)
        choices = row.get("choices")
        choice_hash = row.get("choices_ordered_sha256")
        if choices is not None:
            if not isinstance(choices, list) or not choices or not all(isinstance(c, str) for c in choices):
                raise AuditError("Invalid ordered choices metadata")
            if ordered_choices_hash(choices) != choice_hash:
                raise AuditError("Ordered choices SHA256 does not match retained choices")
        elif choice_hash is not None:
            raise AuditError("Choices hash has no retained choices to verify")
        info = integrity.setdefault(q, {"unordered": set(), "correct_text": set(), "orders": {}})
        if choices is not None:
            unordered = tuple(sorted(choices))
            info["unordered"].add(unordered)
        text = row.get("correct_text")
        if text is not None:
            if not isinstance(text, str):
                raise AuditError("Invalid correct_text metadata")
            info["correct_text"].add(text)
        if choice_hash is not None:
            reference = info["orders"].setdefault(choice_hash, {})
            for name in ("correct_answer", "correct_answer_index"):
                value = row.get(name)
                if value is not None:
                    reference.setdefault(name, set()).add(value)
        input_hash = row.get("input_sha256")
        if valid_hash(input_hash) and input_hash != EMPTY_INPUT_SHA256:
            reference = by_input.setdefault(input_hash, {"questions": set(), "values": {}})
            reference["questions"].add(q)
            for name in ("choices_ordered_sha256", "correct_text", "correct_answer", "correct_answer_index"):
                value = row.get(name)
                if value is not None:
                    reference["values"].setdefault(name, {}).setdefault(value, set()).add(q)
        if row.get("scope") == "baseline" and row.get("input_sha256") is not None:
            signature = row["input_sha256"]
            if signature in ("", EMPTY_INPUT_SHA256):
                continue
            if not valid_hash(signature):
                raise AuditError("Malformed baseline input SHA256")
            if type(row.get("input_chars")) is not int or row["input_chars"] <= 0:
                continue
            anchors.add((q, model))
            baseline_hashes[q].add(signature)
            if signature in seen_hash:
                union.union(q, seen_hash[signature])
            else:
                seen_hash[signature] = q
    if models != set(protocol.models):
        raise AuditError(f"Frozen models differ from structural input metadata: {sorted(models)}")
    if expected_sources is not None and sources != set(expected_sources):
        raise AuditError("Slim does not contain the complete frozen source-file matrix")
    clusters = {q: union.find(q) for q in questions}
    assignments = {q: partition(clusters[q], protocol.split_seed, protocol.split_modulus, protocol.discovery_buckets,
                                protocol.confirmation_buckets, protocol.reserve_buckets) for q in questions}
    within_question = []
    for q, info in sorted(integrity.items()):
        for name in ("unordered", "correct_text"):
            if len(info[name]) > 1:
                within_question.append({"question_key": q, "field": name, "values": sorted(info[name])})
        for order, reference in sorted(info["orders"].items()):
            for name, values in sorted(reference.items()):
                if len(values) > 1:
                    within_question.append({"question_key": q, "field": name, "ordered_choices_sha256": order, "values": sorted(values)})
    conflicts, input_within_question = [], []
    for input_hash, reference in sorted(by_input.items()):
        fields = {}
        for name, values in sorted(reference["values"].items()):
            if len(values) > 1:
                fields[name] = [{"value": value, "question_keys": sorted(qs)} for value, qs in sorted(values.items())]
                memberships = Counter(q for qs in values.values() for q in qs)
                input_within_question.extend({"question_key": q, "input_sha256": input_hash, "field": name}
                                             for q, count in memberships.items() if count > 1)
        if fields:
            conflicts.append({"input_sha256": input_hash, "fields": fields,
                              "question_keys": sorted(reference["questions"]),
                              "cluster_keys": sorted({clusters[q] for q in reference["questions"]})})
    excluded_clusters = {cluster for conflict in conflicts for cluster in conflict["cluster_keys"]}
    excluded_questions = {q for q in questions if clusters[q] in excluded_clusters}
    exclusion_rows = [{"question_key": q, "dataset": questions[q], "original_cluster_key": clusters[q],
                       "original_partition": assignments[q]} for q in sorted(excluded_questions)]
    conflict_ledger = {"policy": protocol.metadata_conflict_policy, "conflicting_input_hashes": len(conflicts),
                       "conflicts": conflicts, "within_question_conflicts": within_question,
                       "same_input_within_question_conflicts": input_within_question,
                       "excluded_question_count": len(excluded_questions), "excluded_cluster_count": len(excluded_clusters),
                       "excluded_cluster_keys": sorted(excluded_clusters), "excluded_questions": exclusion_rows,
                       "exclusions_by_partition_dataset": {
                           f"{part}|{dataset}": {"questions": sum(r["dataset"] == dataset and r["original_partition"] == part for r in exclusion_rows),
                                                  "clusters": len({r["original_cluster_key"] for r in exclusion_rows if r["dataset"] == dataset and r["original_partition"] == part})}
                           for part in ("discovery", "confirmation", "reserve") for dataset in protocol.datasets},
                       "original_cluster_keys_and_bucket_assignments_preserved": True,
                       "acknowledgment_or_generated_answer_values_used": False}
    if within_question or input_within_question:
        fields = sorted({entry["field"] for entry in within_question + input_within_question})
        raise AuditError(f"Contradictory metadata within one question: {fields}", ledger=conflict_ledger)
    if conflicts and protocol.metadata_conflict_policy == "strict":
        raise AuditError("Identical released input has contradictory question metadata", ledger=conflict_ledger)
    if protocol.metadata_conflict_policy == "strict":
        excluded_questions, excluded_clusters = set(), set()
    return {"questions": questions, "clusters": clusters,
            "partitions": assignments,
            "anchors": anchors,
            "excluded_questions": excluded_questions,
            "metadata_integrity": {"status": "passed_after_explicit_quarantine" if excluded_questions else "passed", "source_rows": dict(source_rows),
                                   "conflict_ledger": conflict_ledger,
                                   "question_ids": len(questions), "questions_with_baseline_anchor": len(baseline_hashes),
                                   "partition_metadata": {name: {"questions": sum(part == name for part in assignments.values()),
                                                                  "clusters": len({clusters[q] for q in questions if assignments[q] == name})}
                                                          for name in ("discovery", "confirmation", "reserve")},
                                   "retained_partition_metadata": {name: {"questions": sum(assignments[q] == name and q not in excluded_questions for q in questions),
                                                                           "clusters": len({clusters[q] for q in questions if assignments[q] == name and q not in excluded_questions})}
                                                                   for name in ("discovery", "confirmation", "reserve")},
                                   "questions_with_multiple_baseline_inputs": sum(len(v) > 1 for v in baseline_hashes.values()),
                                   "baseline_input_versions_per_question": {q: len(baseline_hashes[q]) for q in sorted(questions)},
                                   "baseline_anchor_questions_per_model": {m: sum(mm == m for _q, mm in anchors) for m in protocol.models},
                                   "baseline_coverage_per_model": {m: {"questions": len(model_questions[m]),
                                                                        "anchored_questions": sum((q, m) in anchors for q in model_questions[m]),
                                                                        "missing_anchor_questions": sum((q, m) not in anchors for q in model_questions[m])}
                                                                   for m in protocol.models}}}


def _intern(mapping: dict, value) -> int:
    if value not in mapping:
        mapping[value] = len(mapping)
    return mapping[value]


def _gap_values(native0: float, native1: float, exact: float) -> np.ndarray:
    return np.array((native0, native1, exact, exact - native0,
                     native1 - native0, exact - native1), dtype=np.float64)


def exact_signature(row: dict, q: str, structural: dict) -> tuple:
    """Validate cued-record design metadata without accessing an ACK value."""
    model = row["model"]
    ledger = {"integrity": "failed", "question_key": q, "model": model,
              "source_file": row.get("source_file"), "source_row_index": row.get("source_row_index"),
              "metadata_integrity": structural["metadata_integrity"]}
    if (q, model) not in structural["anchors"]:
        raise AuditError("Cued record has no nonempty baseline input anchor", ledger=ledger)
    signature = (row.get("input_sha256"), row.get("choices_ordered_sha256"), row.get("cue_target_answer"))
    valid = (all(valid_hash(s) for s in signature[:2]) and signature[0] != EMPTY_INPUT_SHA256
             and type(row.get("input_chars")) is int and row["input_chars"] > 0
             and isinstance(signature[2], str) and len(signature[2]) == 1
             and isinstance(row.get("choices"), list)
             and signature[2] in [chr(65 + i) for i in range(len(row["choices"]))])
    if not valid:
        raise AuditError("Cued record has missing or invalid exact-version signature", ledger=ledger)
    if row.get("cue") is not None and not isinstance(row["cue"], str):
        raise AuditError("Cue context must preserve string/null values", ledger=ledger)
    return signature


def design_pass(rows: Iterable[dict], protocol: Protocol, structural: dict) -> dict:
    """Uniform, all-partition preflight for retained has_cue=true records; no ACK column."""
    counts = Counter()
    excluded = 0
    for row in rows:
        dataset, q = question_key(row, protocol.datasets)
        if q in structural["excluded_questions"]:
            excluded += 1
            continue
        if row.get("has_cue") is True:
            exact_signature(row, q, structural)
            counts[f"{structural['partitions'][q]}|{dataset}"] += 1
    return {"retained_cued_rows_checked": sum(counts.values()), "excluded_rows_skipped": excluded,
            "rows_by_partition_dataset": {f"{part}|{dataset}": counts[f"{part}|{dataset}"]
                                          for part in ("discovery", "confirmation", "reserve") for dataset in protocol.datasets},
            "all_retained_partitions_checked": True, "ack_column_read": False}


def audit_records(rows_factory: Callable[[tuple[str, ...]], Iterable[dict]], protocol: Protocol,
                  split: str, *, question_sink: Callable[[dict], None] | None = None,
                  expected_sources: dict | None = None) -> dict:
    """Pure offline computation; fixtures can supply the row factory and sink.

    The callback receives only selected-partition per-question contributions.
    Repeated generations are averaged inside a model/question/context/version.
    """
    if split not in ("discovery", "confirmation"):
        raise AuditError("Unknown partition")
    structural = metadata_pass(rows_factory(META_COLUMNS), protocol, expected_sources)
    selected = sorted(q for q, part in structural["partitions"].items() if part == split and q not in structural["excluded_questions"])
    q_ids = {q: i for i, q in enumerate(selected)}
    model_ids = {name: i for i, name in enumerate(protocol.models)}
    # Complete partition-specific validity preflight before any bool values are aggregated.
    for row in rows_factory(VALUE_COLUMNS):
        _dataset, q = question_key(row, protocol.datasets)
        if q not in q_ids or row.get("scope") == "baseline":
            continue
        if row.get("has_cue") is True and type(row.get("cue_acknowledgment")) is bool:
            exact_signature(row, q, structural)
    context_ids, version_ids, version_valid = {}, {}, []
    aggregates: dict[tuple[int, int, int, int], list[int]] = {}
    quality = {(dataset, model): Counter(dict.fromkeys(QUALITY_KEYS, 0))
               for dataset in protocol.datasets for model in protocol.models}
    for row in rows_factory(VALUE_COLUMNS):
        dataset, q = question_key(row, protocol.datasets)
        # Critical: no outcome fields are read before this split decision.
        if q not in q_ids:
            continue
        model = row["model"]
        if model not in model_ids:
            raise AuditError("Unexpected model after metadata pass")
        count = quality[(dataset, model)]
        count["selected_rows"] += 1
        if row.get("scope") == "baseline":
            count["baseline_rows"] += 1
            continue
        count["cued_scope_rows"] += 1
        if row.get("has_cue") is not True:
            count["has_cue_not_true"] += 1
            continue
        acknowledgment = row.get("cue_acknowledgment")
        if type(acknowledgment) is not bool:
            count["ack_missing_or_nonbool"] += 1
            continue
        if not isinstance(row.get("scope"), str) or not row["scope"]:
            count["invalid_context"] += 1
            continue
        context = (row["scope"], row.get("cue") if protocol.context_definition == "scope_and_literal_cue" else None)
        signature = exact_signature(row, q, structural)
        count["valid_ack_rows"] += 1
        c, v = _intern(context_ids, context), _intern(version_ids, signature)
        if v == len(version_valid):
            version_valid.append(True)
        key = (q_ids[q], model_ids[model], c, v)
        if key not in aggregates:
            aggregates[key] = [0, 0]
        aggregates[key][0] += int(acknowledgment)
        aggregates[key][1] += 1

    pairs = list(combinations(range(len(protocol.models)), 2))
    pair_count = len(pairs)
    cells, cluster_data = [], {}
    active_questions = {selected[key[0]] for key in aggregates}
    for dataset in protocol.datasets:
        labels = sorted({structural["clusters"][q] for q in active_questions if structural["questions"][q] == dataset})
        cluster_data[dataset] = {"labels": labels, "indices": {c: i for i, c in enumerate(labels)},
                                 "sums": np.zeros((len(labels), pair_count, 3)),
                                 "denoms": np.zeros((len(labels), pair_count, 3))}
        for a, b in pairs:
            cells.append({"dataset": dataset, "model_a": protocol.models[a], "model_b": protocol.models[b],
                          "cell_id": f"{dataset}|{protocol.models[a]}|{protocol.models[b]}",
                          "counts": Counter({name: 0 for name in ("u0_questions", "u1_questions", "u0_contexts", "u1_contexts",
                                                                  "a_observed_versions", "b_observed_versions", "common_exact_versions")})})
    dataset_index = {d: i for i, d in enumerate(protocol.datasets)}
    for qi, group in groupby(sorted(aggregates), key=lambda key: key[0]):
        q, by_model = selected[qi], defaultdict(lambda: defaultdict(dict))
        dataset = structural["questions"][q]
        for key in group:
            _, model, context, version = key
            acknowledged, generations = aggregates[key]
            by_model[model][context][version] = acknowledged / generations
        for pi, (a, b) in enumerate(pairs):
            contexts0 = sorted(set(by_model[a]) & set(by_model[b]))
            if not contexts0:
                continue
            native0, native1, exact = [], [], []
            common_versions, aversions, bversions = 0, 0, 0
            for context in contexts0:
                va, vb = by_model[a][context], by_model[b][context]
                native = float(np.mean(list(va.values())) - np.mean(list(vb.values())))
                native0.append(native)
                aversions += len(va)
                bversions += len(vb)
                common = sorted(v for v in set(va) & set(vb) if version_valid[v])
                if common:
                    common_versions += len(common)
                    native1.append(native)
                    exact.append(float(np.mean([va[v] - vb[v] for v in common])))
            values = np.array((np.mean(native0), np.mean(native1) if native1 else np.nan,
                               np.mean(exact) if exact else np.nan))
            cd = cluster_data[dataset]
            ci = cd["indices"][structural["clusters"][q]]
            present = np.isfinite(values)
            cd["sums"][ci, pi, present] += values[present]
            cd["denoms"][ci, pi, present] += 1
            cell = cells[dataset_index[dataset] * pair_count + pi]
            counts = cell["counts"]
            counts.update(u0_questions=1, u1_questions=int(bool(native1)),
                          u0_contexts=len(contexts0), u1_contexts=len(native1),
                          a_observed_versions=aversions, b_observed_versions=bversions,
                          common_exact_versions=common_versions)
            if question_sink:
                question_sink({"dataset": dataset, "question_key": q,
                               "cluster_key": structural["clusters"][q], "split": split,
                               "model_a": protocol.models[a], "model_b": protocol.models[b],
                               "u0_contexts": len(contexts0), "u1_contexts": len(native1),
                               "a_observed_versions": aversions, "b_observed_versions": bversions,
                               "common_exact_versions": common_versions,
                               "native_u0_gap": float(values[0]),
                               "native_u1_gap": float(values[1]) if native1 else None,
                               "exact_u1_gap": float(values[2]) if exact else None})
    # Release high-cardinality row structures before bootstrap allocation.
    del aggregates, context_ids, version_ids, version_valid
    part_index = 0 if split == "discovery" else 1
    for di, dataset in enumerate(protocol.datasets):
        cd = cluster_data[dataset]
        for pi, _pair in enumerate(pairs):
            cell = cells[di * pair_count + pi]
            numerator, denominator = cd["sums"][:, pi].sum(axis=0), cd["denoms"][:, pi].sum(axis=0)
            base = np.divide(numerator, denominator, out=np.full(3, np.nan), where=denominator > 0)
            values = _gap_values(*base)
            cell["estimates"] = {m: float(v) if np.isfinite(v) else None for m, v in zip(METRICS, values)}
            counts = cell["counts"]
            counts["dataset_selected_questions"] = sum(structural["questions"][q] == dataset for q in selected)
            counts["dataset_selected_clusters"] = len({structural["clusters"][q] for q in selected if structural["questions"][q] == dataset})
            counts["dataset_bootstrap_clusters"] = len(cd["labels"])
            counts["u0_clusters"] = int(np.count_nonzero(cd["denoms"][:, pi, 0]))
            counts["u1_clusters"] = int(np.count_nonzero(cd["denoms"][:, pi, 1]))
            counts["overlap_question_coverage"] = counts["u1_questions"] / counts["u0_questions"] if counts["u0_questions"] else 0.
            for side, model in (("a", cell["model_a"]), ("b", cell["model_b"])):
                for key in ("selected_rows", "cued_scope_rows", "valid_ack_rows", "has_cue_not_true",
                            "ack_missing_or_nonbool", "missing_exact_version_rows"):
                    counts[f"{side}_{key}"] = quality[(dataset, model)][key]
            reasons = []
            if counts["u1_clusters"] < protocol.min_clusters[part_index]:
                reasons.append("insufficient_overlap_clusters")
            if counts["overlap_question_coverage"] < protocol.min_overlap_coverage:
                reasons.append("insufficient_question_overlap")
            if not np.all(np.isfinite(values)):
                reasons.append("undefined_estimand")
            cell["support_eligible"] = not reasons
            cell["ineligible_reasons"] = reasons
            cell["counts"] = dict(counts)
    boots = cluster_bootstrap(cluster_data, protocol, part_index)
    apply_intervals(cells, boots, protocol)
    eligible = [c for c in cells if c["eligible"]]
    effects = [c for c in eligible if c["material_effect"]]
    support_cells = [c for c in cells if c["support_eligible"]]
    analyzable = len(support_cells) >= 4 and len({c["dataset"] for c in support_cells}) >= 2
    robust = analyzable and bool(eligible) and len(eligible) == len(support_cells) and all(
        c["intervals"]["delta_total"]["low"] >= -protocol.practical_effect
        and c["intervals"]["delta_total"]["high"] <= protocol.practical_effect for c in eligible)
    sensitivity_rule = len(effects) >= 2 and len({c["dataset"] for c in effects}) >= 2
    global_result = {"hypothesis": ("supported" if sensitivity_rule else "not_supported") if split == "confirmation" else "not_evaluated_on_discovery",
                     "sensitivity_rule_pass": sensitivity_rule,
                     "robustness": "supported_within_eligible_scope" if robust else "inconclusive",
                     "eligible_cells": len(eligible), "support_eligible_cells": len(support_cells),
                     "material_effect_cells": len(effects),
                     "discovery_go": analyzable}
    return {"cells": cells, "global": global_result, "bootstrap": boots,
            "dataquality": {"split": split, "context_definition": protocol.context_definition, "selected_questions": len(selected),
                            "selected_clusters": len({structural["clusters"][q] for q in selected}),
                            "bootstrap_clusters": len({structural["clusters"][q] for q in active_questions}),
                            "metadata_integrity": structural["metadata_integrity"],
                            "models": {f"{d}|{m}": dict(v) for (d, m), v in sorted(quality.items())},
                            "unselected_outcomes_used": False,
                            "parquet_io_may_decode_unselected_batch_rows": True,
                            "ack_false_may_include_source_parse_failure": True}}


def cluster_bootstrap(cluster_data: dict, protocol: Protocol, part_index: int) -> np.ndarray:
    """Global Efron cluster draws shared by every pair and every dataset.

    A baseline-prompt cluster can span datasets. Conditional ratio means retain
    dataset-specific estimands while using the same multiplicity for aliases.
    This is an empirical cluster-bootstrap approximation, not a finite-sample guarantee.
    """
    pair_count = len(protocol.models) * (len(protocol.models) - 1) // 2
    boot = np.full((protocol.bootstrap_replicates, len(protocol.datasets) * pair_count, len(METRICS)), np.nan)
    rng = np.random.default_rng(protocol.bootstrap_seeds[part_index])
    global_labels = sorted(set().union(*(set(cd["labels"]) for cd in cluster_data.values())))
    global_indices = {label: i for i, label in enumerate(global_labels)}
    if not global_labels:
        return boot
    for start in range(0, protocol.bootstrap_replicates, protocol.bootstrap_batch):
        size = min(protocol.bootstrap_batch, protocol.bootstrap_replicates - start)
        global_weights = rng.multinomial(len(global_labels), np.full(len(global_labels), 1 / len(global_labels)), size=size)
        for di, dataset in enumerate(protocol.datasets):
            cd = cluster_data[dataset]
            clusters = len(cd["labels"])
            if not clusters:
                continue
            weights = global_weights[:, [global_indices[label] for label in cd["labels"]]]
            numerator = (weights @ cd["sums"].reshape(clusters, -1)).reshape(size, pair_count, 3)
            denominator = (weights @ cd["denoms"].reshape(clusters, -1)).reshape(size, pair_count, 3)
            base = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
            block = boot[start:start + size, di * pair_count:(di + 1) * pair_count]
            block[:, :, :3] = base
            block[:, :, 3] = base[:, :, 2] - base[:, :, 0]
            block[:, :, 4] = base[:, :, 1] - base[:, :, 0]
            block[:, :, 5] = base[:, :, 2] - base[:, :, 1]
    return boot


def apply_intervals(cells: list[dict], boots: np.ndarray, protocol: Protocol) -> None:
    """Separate simultaneous max-|t| families for each declared delta endpoint."""
    for cell in cells:
        cell["intervals"] = {}
    for metric_index, metric in enumerate(METRICS[3:], 3):
        indices, standard_errors = [], {}
        tested = [i for i, cell in enumerate(cells) if cell["support_eligible"]]
        joint_mask = (np.all(np.isfinite(boots[:, tested, metric_index]), axis=1)
                      if tested else np.ones(boots.shape[0], dtype=bool))
        joint_fraction = float(np.mean(joint_mask))
        family_valid = joint_fraction >= .99 and np.count_nonzero(joint_mask) > 1
        for i, cell in enumerate(cells):
            samples = boots[joint_mask, i, metric_index]
            se = float(np.std(samples, ddof=1)) if family_valid and np.all(np.isfinite(samples)) and len(samples) > 1 else None
            valid = cell["support_eligible"] and se is not None and se > 1e-12
            reason = (None if valid else "insufficient_support" if not cell["support_eligible"] else
                      "joint_finite_replicates_below_99pct" if not family_valid else "degenerate_or_nonfinite_bootstrap")
            cell["intervals"][metric] = {"low": None, "high": None, "se": se, "critical": None,
                                          "simultaneous_family": metric, "family_size": 0,
                                          "joint_finite_fraction": joint_fraction,
                                          "replicates_used": int(np.count_nonzero(joint_mask)),
                                          "status": "pending" if valid else reason}
            if valid:
                indices.append(i)
                standard_errors[i] = se
        if indices:
            centers = np.array([cells[i]["estimates"][metric] for i in indices])
            scales = np.array([standard_errors[i] for i in indices])
            studentized = np.abs((boots[joint_mask, :, metric_index][:, indices] - centers) / scales)
            critical = float(np.quantile(np.max(studentized, axis=1), .95, method="higher"))
            for i in indices:
                center, se = cells[i]["estimates"][metric], standard_errors[i]
                cells[i]["intervals"][metric].update(low=center - critical * se, high=center + critical * se,
                                                       critical=critical, family_size=len(indices), status="simultaneous_95_max_t")
    for cell in cells:
        interval = cell["intervals"]["delta_total"]
        cell["eligible"] = cell["support_eligible"] and interval["low"] is not None
        if cell["support_eligible"] and not cell["eligible"]:
            cell["ineligible_reasons"].append("degenerate_or_nonfinite_primary_bootstrap")
        cell["material_effect"] = bool(cell["eligible"] and abs(cell["estimates"]["delta_total"]) >= protocol.practical_effect
                                       and (interval["low"] > protocol.tie_band or interval["high"] < -protocol.tie_band))
        estimates = cell["estimates"]
        cell["decomposition_closure_error"] = (abs(estimates["delta_total"] - estimates["delta_composition"] - estimates["delta_version"])
                                                 if estimates["delta_total"] is not None else None)


def run_analysis(input_path: Path, spec_path: Path, protocol_path: Path, output: Path, split: str | None,
                 *, metadata_only: bool = False) -> dict:
    """Bound file inputs before metadata access; leave every failed attempt intact."""
    import csv
    import pyarrow as pa
    import pyarrow.parquet as pq

    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    started = time.monotonic()
    frozen = json.loads(protocol_path.read_text(encoding="utf-8"))
    if frozen.get("execution_authorized") is not True:
        raise AuditError("Analysis execution is not activated in frozen protocol")
    if input_path.stat().st_size > 256 * 1024 ** 2:
        raise AuditError("Slim input exceeds frozen 256MiB cap")
    binding = {"slim_sha256": sha256_file(input_path), "spec_sha256": sha256_file(spec_path),
               "protocol_sha256": sha256_file(protocol_path), "slim_bytes": input_path.stat().st_size,
               "source_sha256": sha256_file(Path(__file__))}
    if binding["slim_sha256"] != frozen.get("input_sha256") or binding["spec_sha256"] != frozen.get("spec_sha256"):
        raise AuditError("Input hashes differ from frozen analysis protocol")
    protocol = Protocol.from_dict(frozen)
    if len(protocol.models) != 8 or set(protocol.datasets) != {"bbh", "gpqa", "mmlu"}:
        raise AuditError("Production audit requires all eight frozen models and all three datasets")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    expected_sources = {f["path"]: f["sha256"] for f in spec["files"]}
    if len(expected_sources) != 24:
        raise AuditError("Complete pinned 24-file source matrix required")
    output.mkdir(parents=True, exist_ok=False)
    writer, buffered = None, []
    question_schema = pa.schema([(name, pa.string()) for name in ("dataset", "question_key", "cluster_key", "split", "model_a", "model_b")]
                                + [(name, pa.int64()) for name in ("u0_contexts", "u1_contexts", "a_observed_versions", "b_observed_versions", "common_exact_versions")]
                                + [(name, pa.float64()) for name in METRICS[:3]])
    def flush():
        nonlocal writer
        if not buffered:
            return
        table = pa.Table.from_pylist(buffered, schema=question_schema)
        if writer is None:
            writer = pq.ParquetWriter(output / "per_question.parquet", question_schema, compression="zstd")
        writer.write_table(table)
        buffered.clear()
    def sink(row):
        buffered.append(row)
        if len(buffered) >= 1024:
            flush()
    def rows_factory(columns):
        parquet = pq.ParquetFile(input_path)
        if not set(columns).issubset(parquet.schema_arrow.names):
            raise AuditError("Slim is missing required analysis columns")
        for batch in parquet.iter_batches(batch_size=2048, columns=list(columns), use_threads=False):
            yield from batch.to_pylist()
    try:
        if metadata_only:
            structural = metadata_pass(rows_factory(META_COLUMNS), protocol, expected_sources)
            result = {"schema_version": 1, "metadata_only": True, "outcome_columns_read": False,
                      "status": "metadata_validated_design_pending", "input_binding": binding, "protocol": frozen,
                      "metadata_integrity": structural["metadata_integrity"],
                      "timing": {"wall_seconds": time.monotonic() - started},
                      "environment": {"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__,
                                      "arrow_cpu_count": pa.cpu_count(), "arrow_io_thread_count": pa.io_thread_count()}}
            write_json(output / "metadata_audit.json", result)
            try:
                result["design_preflight"] = design_pass(rows_factory(DESIGN_COLUMNS), protocol, structural)
            except AuditError as exc:
                result.update(status="design_preflight_failed", design_error={"message": str(exc), "ledger": exc.ledger})
                write_json(output / "metadata_audit.json", result)
                raise
            result.update(status="complete", timing={"wall_seconds": time.monotonic() - started})
            write_json(output / "metadata_audit.json", result)
            write_json(output / "protocol.json", frozen)
            return result
        result = audit_records(rows_factory, protocol, split, question_sink=sink, expected_sources=expected_sources)
        flush()
        if writer is None:
            pq.write_table(pa.Table.from_pylist([], schema=question_schema), output / "per_question.parquet")
        else:
            writer.close()
            writer = None
        boot = result.pop("bootstrap")
        np.savez_compressed(output / "bootstrap.npz", samples=boot, metric_names=np.array(METRICS),
                            cell_ids=np.array([c["cell_id"] for c in result["cells"]]))
        quality = result.pop("dataquality")
        write_json(output / "dataquality.json", quality)
        result.update(schema_version=1, split=split, estimand="question-weighted released acknowledgment proxy",
                      context_definition=protocol.context_definition,
                      models=list(protocol.models), datasets=list(protocol.datasets), settings=frozen,
                      input_binding=binding, environment={"python": platform.python_version(), "numpy": np.__version__, "pyarrow": pa.__version__,
                                                         "arrow_cpu_count": pa.cpu_count(), "arrow_io_thread_count": pa.io_thread_count()},
                      timing={"wall_seconds": time.monotonic() - started},
                      bootstrap_method={"name": "global empirical question-cluster bootstrap",
                                        "interval": "centered max absolute standardized bootstrap deviation; fixed bootstrap SE",
                                        "replicates": protocol.bootstrap_replicates,
                                        "seed": protocol.bootstrap_seeds[0 if split == "discovery" else 1],
                                        "finite_sample_guarantee": False},
                      limitations=["descriptive support-dependent contrasts; no causal cue-version attribution",
                                   "acknowledgment is a source judge annotation, not latent faithful reasoning",
                                   "question-weighted released proxy is not exact source subdataset-macro reproduction"])
        write_json(output / "analysis.json", result)
        write_json(output / "protocol.json", frozen)
        flat = []
        for cell in result["cells"]:
            row = {k: cell[k] for k in ("cell_id", "dataset", "model_a", "model_b", "eligible", "material_effect")}
            row.update(cell["counts"])
            row.update(cell["estimates"])
            row.update(delta_total_low=cell["intervals"]["delta_total"]["low"], delta_total_high=cell["intervals"]["delta_total"]["high"],
                       ineligible_reasons=";".join(cell["ineligible_reasons"]))
            flat.append(row)
        with (output / "pairs.csv").open("w", encoding="utf-8", newline="") as stream:
            fieldnames = sorted(set().union(*(r.keys() for r in flat)))
            csv_writer = csv.DictWriter(stream, fieldnames=fieldnames)
            csv_writer.writeheader()
            csv_writer.writerows(flat)
        return result
    except BaseException as exc:
        if writer is not None:
            writer.close()
        write_json(output / "failure.json", {"error": type(exc).__name__, "message": str(exc), "input_binding": binding,
                                              "wall_seconds": time.monotonic() - started})
        if isinstance(exc, AuditError) and exc.ledger is not None:
            write_json(output / "dataquality.json", exc.ledger)
        raise

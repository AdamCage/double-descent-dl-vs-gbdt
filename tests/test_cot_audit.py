"""Handwritten fixtures only: no public data, network, models, or fitting."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.cot_audit import Protocol, METRICS, META_COLUMNS, VALUE_COLUMNS, AuditError, audit_records, metadata_pass, partition, cluster_bootstrap, apply_intervals, run_analysis, sha256_file, ordered_choices_hash


def h(value):
    return hashlib.sha256(value.encode()).hexdigest()


def row(q, model, ack, *, target="B", version="shared", scope="cue", cue="constant"):
    choices = ["choice A", "choice B"]
    return {"source_file": f"bbh/{model}.parquet", "source_payload_sha256": h(model), "core_id": q,
            "model": model, "scope": scope, "input_sha256": h(f"{q}:{version}"),
            "input_chars": len(q + version), "choices": choices, "correct_answer": "A", "correct_answer_index": 0,
            "correct_text": "choice A", "choices_ordered_sha256": ordered_choices_hash(choices), "cue_target_answer": target,
            "cue": cue, "has_cue": scope != "baseline", "cue_acknowledgment": ack}


def factory(rows):
    def read(columns):
        existing = {(r["source_file"], r["core_id"], r["model"]) for r in rows if r["scope"] == "baseline"}
        expanded = list(rows)
        for value in rows:
            key = value["source_file"], value["core_id"], value["model"]
            if key not in existing:
                baseline = dict(value, scope="baseline", has_cue=False, cue_acknowledgment=False,
                                input_sha256=h(f"baseline:{key[0].split('/')[0]}:{key[1]}:{value.get('choices_ordered_sha256')}"), input_chars=10)
                expanded.append(baseline)
                existing.add(key)
        offsets = {}
        for value in expanded:
            source = value["source_file"]
            offset = offsets.get(source, 0)
            offsets[source] = offset + 1
            present = dict(value, source_row_index=value.get("source_row_index", offset))
            yield {key: present.get(key) for key in columns}
    return read


def keys_for(split, count):
    return [f"q{i}" for i in range(10000) if partition(f"bbh|q{i}") == split][:count]


def protocol(models=("A", "B")):
    return Protocol(tuple(models), ("bbh",), bootstrap_replicates=101, bootstrap_batch=7,
                    min_clusters=(1, 1), min_overlap_coverage=0.)


class CotAuditTests(unittest.TestCase):
    def test_decomposition_composition_and_same_support(self):
        q1, q2 = keys_for("discovery", 2)
        rows = [row(q1, "A", True), row(q1, "B", False),
                row(q2, "A", False, version="a"), row(q2, "B", True, version="b")]
        result = audit_records(factory(rows), protocol(), "discovery")
        e = result["cells"][0]["estimates"]
        self.assertEqual(e["native_u0_gap"], 0.)
        self.assertEqual(e["native_u1_gap"], 1.)
        self.assertEqual(e["exact_u1_gap"], 1.)
        self.assertEqual(e["delta_total"], 1.)
        self.assertEqual(e["delta_composition"], 1.)
        self.assertEqual(e["delta_version"], 0.)
        self.assertEqual(result["cells"][0]["decomposition_closure_error"], 0.)

    def test_generations_versions_and_contexts_have_distinct_weights(self):
        q = keys_for("discovery", 1)[0]
        # Three repeated successes in v1 and one failure v2 => version-balanced A=.5, not .75.
        rows = [row(q, "A", True, version="v1") for _ in range(3)]
        rows += [row(q, "A", False, version="v2"), row(q, "B", False, version="v1")]
        result = audit_records(factory(rows), protocol(), "discovery")
        e = result["cells"][0]["estimates"]
        self.assertEqual(e["native_u0_gap"], .5)
        self.assertEqual(e["native_u1_gap"], .5)
        self.assertEqual(e["exact_u1_gap"], 1.)
        self.assertEqual(e["delta_composition"], 0.)
        self.assertEqual(e["delta_version"], .5)

    def test_context_intersection_precedes_question_average(self):
        q = keys_for("discovery", 1)[0]
        rows = [row(q, "A", True), row(q, "B", False), row(q, "B", True, cue="onlyB")]
        result = audit_records(factory(rows), protocol(), "discovery")
        self.assertEqual(result["cells"][0]["estimates"]["native_u0_gap"], 1.)

    def test_baseline_hash_union_prevents_duplicate_split_leakage(self):
        q1 = keys_for("discovery", 1)[0]
        q2 = keys_for("confirmation", 1)[0]
        rows = [row(q1, "A", False, scope="baseline"), row(q2, "B", False, scope="baseline")]
        for r in rows:
            r["input_sha256"] = h("identical baseline prompt")
        meta = metadata_pass(factory(rows)(META_COLUMNS), protocol())
        self.assertEqual(meta["clusters"][f"bbh|{q1}"], meta["clusters"][f"bbh|{q2}"])
        self.assertEqual(meta["partitions"][f"bbh|{q1}"], meta["partitions"][f"bbh|{q2}"])

    def test_unselected_outcomes_do_not_affect_discovery(self):
        q1, q2 = keys_for("discovery", 1)[0], keys_for("confirmation", 1)[0]
        rows = [row(q, m, m == "A") for q in (q1, q2) for m in ("A", "B")]
        first = audit_records(factory(rows), protocol(), "discovery")
        for r in rows:
            if r["core_id"] == q2:
                r["cue_acknowledgment"] = {"unreadable": object()}
        second = audit_records(factory(rows), protocol(), "discovery")
        self.assertEqual(first["cells"], second["cells"])
        np.testing.assert_equal(first["bootstrap"], second["bootstrap"])

    def test_missing_ack_excluded_baseline_not_false_observation(self):
        q = keys_for("discovery", 1)[0]
        rows = [row(q, "A", True), row(q, "B", False), row(q, "A", None), row(q, "A", False, scope="baseline")]
        result = audit_records(factory(rows), protocol(), "discovery")
        self.assertEqual(result["cells"][0]["estimates"]["native_u0_gap"], 1.)
        self.assertEqual(result["dataquality"]["models"]["bbh|A"]["ack_missing_or_nonbool"], 1)

    def test_common_cluster_draws_and_degenerate_intervals(self):
        qs = keys_for("discovery", 12)
        rows = [row(q, m, (i % 2 == 0) if m == "A" else False,
                    version="shared" if i % 3 else m)
                for i, q in enumerate(qs) for m in ("A", "B", "C")]
        result = audit_records(factory(rows), protocol(("A", "B", "C")), "discovery")
        # B and C outcomes/support are identical, so AB and AC must share bootstrap values exactly.
        np.testing.assert_equal(result["bootstrap"][:, 0], result["bootstrap"][:, 1])
        self.assertTrue(all(c["intervals"]["delta_version"]["low"] is None for c in result["cells"]))
        bc = result["cells"][2]
        self.assertFalse(bc["eligible"])
        self.assertEqual(result["global"]["robustness"], "inconclusive")

    def test_cross_dataset_aliases_share_global_bootstrap_multiplicity(self):
        p = Protocol(("A", "B"), ("bbh", "mmlu"), bootstrap_replicates=101, bootstrap_batch=7)
        values = np.array([[[1., .2, .3]], [[0., .7, .8]]])
        cd = {d: {"labels": ["same-cluster-1", "same-cluster-2"], "sums": values.copy(),
                  "denoms": np.ones_like(values)} for d in p.datasets}
        bootstrap = cluster_bootstrap(cd, p, 0)
        np.testing.assert_equal(bootstrap[:, 0], bootstrap[:, 1])

    def test_max_t_family_and_structural_zero_not_automatic_robustness(self):
        p = protocol()
        cells = [{"support_eligible": True, "ineligible_reasons": [],
                  "estimates": dict.fromkeys(METRICS, 0.)} for _ in range(2)]
        samples = np.zeros((5, 2, 6))
        samples[:, 0, 3] = [-.1, 0, .1, 0, 0]
        samples[:, 1, 3] = [0, -.2, 0, .2, 0]
        apply_intervals(cells, samples, p)
        self.assertAlmostEqual(cells[0]["intervals"]["delta_total"]["high"], .1)
        self.assertAlmostEqual(cells[1]["intervals"]["delta_total"]["high"], .2)
        self.assertEqual(cells[0]["intervals"]["delta_total"]["critical"], cells[1]["intervals"]["delta_total"]["critical"])
        self.assertIsNone(cells[0]["intervals"]["delta_version"]["low"])

    def test_bound_output_contract_with_handwritten_eight_model_fixture(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            models = [f"M{i}" for i in range(8)]
            datasets, rows, sources = ["bbh", "gpqa", "mmlu"], [], []
            qs = keys_for("discovery", 8)
            for dataset in datasets:
                for model in models:
                    source = f"{dataset}/{model}.parquet"
                    sources.append({"path": source, "sha256": h(source)})
                    for i, q in enumerate(qs):
                        for scope in ("baseline", "cue"):
                            value = row(q, model, i % 2 == 0, scope=scope,
                                        version="shared" if i % 3 else model)
                            value.update(source_file=source, source_payload_sha256=h(source))
                            rows.append(value)
            fields = [(name, pa.bool_() if name in ("has_cue", "cue_acknowledgment") else
                       pa.int64() if name in ("source_row_index", "input_chars", "correct_answer_index") else
                       pa.list_(pa.string()) if name == "choices" else pa.string()) for name in VALUE_COLUMNS]
            input_path, spec_path, proto_path = root / "slim.parquet", root / "spec.json", root / "protocol.json"
            pq.write_table(pa.Table.from_pylist(list(factory(rows)(VALUE_COLUMNS)), schema=pa.schema(fields)), input_path)
            spec_path.write_text(json.dumps({"files": sources}), encoding="utf-8")
            frozen = {"models": models, "datasets": datasets, "bootstrap_replicates": 51,
                      "bootstrap_batch": 7, "min_clusters": {"discovery": 1, "confirmation": 1},
                      "execution_authorized": True, "input_sha256": "bad", "spec_sha256": sha256_file(spec_path)}
            proto_path.write_text(json.dumps(frozen), encoding="utf-8")
            with self.assertRaises(AuditError):
                run_analysis(input_path, spec_path, proto_path, root / "rejected", "discovery")
            self.assertFalse((root / "rejected").exists())
            frozen["input_sha256"] = sha256_file(input_path)
            proto_path.write_text(json.dumps(frozen), encoding="utf-8")
            result = run_analysis(input_path, spec_path, proto_path, root / "output", "discovery")
            self.assertEqual(len(result["cells"]), 84)
            self.assertEqual(set(p.name for p in (root / "output").iterdir()),
                             {"analysis.json", "dataquality.json", "pairs.csv", "per_question.parquet", "bootstrap.npz", "protocol.json"})
            self.assertEqual(json.loads((root / "output" / "analysis.json").read_text())["input_binding"]["slim_sha256"], sha256_file(input_path))
            self.assertEqual(np.load(root / "output" / "bootstrap.npz")["samples"].shape, (51, 84, 6))
            original_parquet = pq.ParquetFile
            class MetadataProbe:
                def __init__(self, *args, **kwargs):
                    self.inner = original_parquet(*args, **kwargs)
                @property
                def schema_arrow(self):
                    return self.inner.schema_arrow
                def iter_batches(self, *args, **kwargs):
                    forbidden = {"cue_acknowledgment", "model_answer", "reasoning", "output", "is_correct"}
                    self_columns = set(kwargs.get("columns", []))
                    if self_columns & forbidden:
                        raise AssertionError("Metadata-only audit requested outcome columns")
                    return self.inner.iter_batches(*args, **kwargs)
            with patch("pyarrow.parquet.ParquetFile", MetadataProbe):
                metadata = run_analysis(input_path, spec_path, proto_path, root / "metadata-only", None, metadata_only=True)
            self.assertTrue(metadata["metadata_only"])
            self.assertFalse(metadata["outcome_columns_read"])
            self.assertEqual(set(p.name for p in (root / "metadata-only").iterdir()), {"metadata_audit.json", "protocol.json"})

    def test_invalid_version_missing_baseline_and_metadata_contradictions_fail(self):
        q = keys_for("discovery", 1)[0]
        rows = [row(q, "A", True), row(q, "B", False)]
        malformed = deepcopy(rows)
        malformed[0]["input_sha256"] = None
        with self.assertRaisesRegex(AuditError, "exact-version"):
            audit_records(factory(malformed), protocol(), "discovery")
        absent_anchor = deepcopy(rows)
        absent_anchor.append(dict(row(q, "A", False, scope="baseline"), input_sha256=h(""), input_chars=0))
        with self.assertRaisesRegex(AuditError, "baseline input anchor"):
            audit_records(factory(absent_anchor), protocol(), "discovery")
        wrong_choices = deepcopy(rows)
        wrong_choices[0]["choices_ordered_sha256"] = h("wrong digest")
        with self.assertRaisesRegex(AuditError, "choices SHA256"):
            audit_records(factory(wrong_choices), protocol(), "discovery")
        contradictory = deepcopy(rows)
        contradictory[0]["correct_text"] = "different answer"
        with self.assertRaisesRegex(AuditError, "correct_text"):
            audit_records(factory(contradictory), protocol(), "discovery")
        duplicate = deepcopy(rows)
        duplicate.append(dict(rows[0], source_row_index=0))
        with self.assertRaisesRegex(AuditError, "Source row identity"):
            audit_records(factory(duplicate), protocol(), "discovery")

    def test_reordered_choices_allow_corresponding_correct_letter_change(self):
        q = keys_for("discovery", 1)[0]
        rows = [row(q, "A", True), row(q, "B", False)]
        rows[1].update(choices=["choice B", "choice A"], correct_answer="B", correct_answer_index=1,
                       input_sha256=h("different rendered order"))
        rows[1]["choices_ordered_sha256"] = ordered_choices_hash(rows[1]["choices"])
        result = audit_records(factory(rows), protocol(), "discovery")
        self.assertEqual(result["cells"][0]["estimates"]["native_u0_gap"], 1.)
        self.assertEqual(result["cells"][0]["counts"]["u1_questions"], 0)

    def test_support_only_discovery_gate_does_not_select_on_zero_outcome_variance(self):
        models, datasets = ("A", "B", "C"), ("bbh", "gpqa")
        rows = []
        for dataset in datasets:
            qs = [f"q{i}" for i in range(1000) if partition(f"{dataset}|q{i}") == "discovery"][:5]
            for q in qs:
                for model in models:
                    value = row(q, model, False)
                    value["source_file"] = f"{dataset}/{model}.parquet"
                    rows.append(value)
        p = Protocol(models, datasets, bootstrap_replicates=101, bootstrap_batch=7, min_clusters=(1, 1))
        result = audit_records(factory(rows), p, "discovery")
        self.assertTrue(result["global"]["discovery_go"])
        self.assertEqual(result["global"]["eligible_cells"], 0)
        self.assertEqual(result["global"]["robustness"], "inconclusive")

    def test_joint_finite_mask_rejects_family_when_individual_masks_disagree(self):
        cells = [{"support_eligible": True, "ineligible_reasons": [],
                  "estimates": dict.fromkeys(METRICS, 0.)} for _ in range(2)]
        samples = np.zeros((100, 2, 6))
        samples[:, :, 3] = np.linspace(-.1, .1, 100)[:, None]
        samples[0, 0, 3] = np.nan
        samples[1, 1, 3] = np.nan
        apply_intervals(cells, samples, protocol())
        for cell in cells:
            self.assertEqual(cell["intervals"]["delta_total"]["joint_finite_fraction"], .98)
            self.assertEqual(cell["intervals"]["delta_total"]["status"], "joint_finite_replicates_below_99pct")
            self.assertFalse(cell["eligible"])

    def test_null_and_string_cue_contexts_remain_distinct_without_sort_errors(self):
        q = keys_for("discovery", 1)[0]
        rows = [row(q, "A", True, cue=None), row(q, "B", False, cue=None),
                row(q, "A", False, cue="text"), row(q, "B", True, cue="text")]
        result = audit_records(factory(rows), protocol(), "discovery")
        self.assertEqual(result["cells"][0]["counts"]["u0_contexts"], 2)
        self.assertEqual(result["cells"][0]["estimates"]["native_u0_gap"], 0.)

    def test_three_way_partition_reserve_outcomes_are_never_used(self):
        frozen = {"models": ["A", "B"], "datasets": ["bbh"], "bootstrap_replicates": 51,
                  "bootstrap_batch": 7, "min_clusters": {"discovery": 1, "confirmation": 1},
                  "split_modulus": 20, "discovery_buckets": list(range(6)),
                  "confirmation_buckets": list(range(6, 13)), "reserve_buckets": list(range(13, 20))}
        p = Protocol.from_dict(frozen)
        args = (p.split_seed, p.split_modulus, p.discovery_buckets, p.confirmation_buckets, p.reserve_buckets)
        groups = {name: [f"q{i}" for i in range(1000) if partition(f"bbh|q{i}", *args) == name][:3]
                  for name in ("discovery", "confirmation", "reserve")}
        rows = [row(q, model, model == "A") for qs in groups.values() for q in qs for model in p.models]
        initial = audit_records(factory(rows), p, "discovery")
        for value in rows:
            if value["core_id"] in groups["reserve"] + groups["confirmation"]:
                value["cue_acknowledgment"] = {"must_not_aggregate": object()}
        repeated = audit_records(factory(rows), p, "discovery")
        self.assertEqual(initial["cells"], repeated["cells"])
        np.testing.assert_equal(initial["bootstrap"], repeated["bootstrap"])
        self.assertEqual(repeated["dataquality"]["metadata_integrity"]["partition_metadata"]["reserve"]["questions"], 3)

    def test_cross_question_conflict_quarantines_whole_original_clusters_across_splits(self):
        p = Protocol(("A", "B"), ("bbh", "gpqa", "mmlu"), bootstrap_replicates=51, bootstrap_batch=7,
                     min_clusters=(1, 1), split_modulus=20, discovery_buckets=tuple(range(6)),
                     confirmation_buckets=tuple(range(6, 13)), reserve_buckets=tuple(range(13, 20)),
                     metadata_conflict_policy="quarantine_entire_baseline_union_cluster_before_outcomes")
        args = p.split_seed, p.split_modulus, p.discovery_buckets, p.confirmation_buckets, p.reserve_buckets
        def pick(dataset, split, prefix):
            return next(f"{prefix}{i}" for i in range(1000) if partition(f"{dataset}|{prefix}{i}", *args) == split)
        qa, qb, qc = pick("bbh", "discovery", "a"), pick("mmlu", "reserve", "b"), "alias"
        keep = pick("bbh", "discovery", "keep")
        rows = []
        for dataset, q, label, base in (("bbh", qa, "A", "base-a"), ("mmlu", qb, "B", "base-b"), ("gpqa", qc, "A", "base-a")):
            for model in p.models:
                for scope in ("baseline", "cue"):
                    value = row(q, model, False, scope=scope)
                    value.update(source_file=f"{dataset}/{model}.parquet", correct_answer=label,
                                 correct_answer_index=ord(label)-65, correct_text=f"choice {label}",
                                 input_sha256=h(base if scope == "baseline" else "identical conflicting cued input"))
                    if scope == "cue":
                        value["cue_acknowledgment"] = {"quarantined_outcome_must_never_be_used": object()}
                    rows.append(value)
        rows.extend(row(keep, model, model == "A") for model in p.models)
        with self.assertRaisesRegex(AuditError, "Identical released input"):
            metadata_pass(factory(rows)(META_COLUMNS), replace(p, metadata_conflict_policy="strict"))
        structural = metadata_pass(factory(rows)(META_COLUMNS), p)
        ledger = structural["metadata_integrity"]["conflict_ledger"]
        self.assertEqual(ledger["excluded_question_count"], 3)
        self.assertEqual(ledger["excluded_cluster_count"], 2)
        self.assertEqual(structural["clusters"][f"gpqa|{qc}"], f"bbh|{qa}")
        self.assertEqual(structural["partitions"][f"gpqa|{qc}"], "discovery")
        self.assertEqual(structural["partitions"][f"mmlu|{qb}"], "reserve")
        result = audit_records(factory(rows), p, "discovery")
        self.assertEqual(result["dataquality"]["selected_questions"], 1)
        self.assertEqual(result["cells"][0]["estimates"]["native_u0_gap"], 1.)
        contradiction = deepcopy(rows)
        contradiction[-1]["correct_text"] = "contradictory within retained question"
        with self.assertRaisesRegex(AuditError, "within one question"):
            metadata_pass(factory(contradiction)(META_COLUMNS), p)

    def test_scope_only_context_retains_different_literal_targets_in_native_population(self):
        q1, q2 = keys_for("discovery", 2)
        rows = [row(q1, "A", True, cue="choose B"), row(q1, "B", False, cue="choose B"),
                row(q2, "A", False, cue="choose A", target="A", version="target-A"),
                row(q2, "B", True, cue="choose B", target="B", version="target-B")]
        original = audit_records(factory(rows), protocol(), "discovery")
        revised = audit_records(factory(rows), replace(protocol(), context_definition="scope_only"), "discovery")
        first, second = original["cells"][0], revised["cells"][0]
        self.assertEqual(first["counts"]["u0_questions"], 1)
        self.assertEqual(first["estimates"]["native_u0_gap"], 1.)
        self.assertEqual(first["estimates"]["delta_total"], 0.)
        self.assertEqual(second["counts"]["u0_questions"], 2)
        self.assertEqual(second["estimates"]["native_u0_gap"], 0.)
        self.assertEqual(second["estimates"]["exact_u1_gap"], 1.)
        self.assertEqual(second["estimates"]["delta_total"], 1.)
        self.assertEqual(second["estimates"]["delta_version"], 0.)
        self.assertEqual(revised["dataquality"]["context_definition"], "scope_only")


if __name__ == "__main__":
    unittest.main()

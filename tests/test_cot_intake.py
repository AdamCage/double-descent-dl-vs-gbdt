"""Handwritten Parquet and mocked HTTP only; no public payloads or models."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
import urllib.request

import pyarrow as pa
import pyarrow.parquet as pq

from src.cot_data import (
    AuditLog, IntakeError, TransferBudget, _BoundedRedirect, download_one,
    file_sha256, run_intake, safe_relative, slim_parquets, validate_spec,
)


STRINGS = ["core_id", "model", "id", "scope", "input", "output", "reasoning",
           "model_answer", "state_transition", "correct_answer", "correct_text",
           "baseline_model_answer", "cue", "cue_target_answer"]
BOOLS = ["is_correct", "cue_acknowledgment", "switched_to_cue", "has_cue"]
INTS = ["factor_utilization_used", "factor_utilization_total", "correct_answer_index"]
OMITTED = ["input", "output", "reasoning"]
FEATURES = {**{name: {"dtype": "string", "_type": "Value"} for name in STRINGS},
            **{name: {"dtype": "bool", "_type": "Value"} for name in BOOLS},
            **{name: {"dtype": "int64", "_type": "Value"} for name in INTS},
            "factor_utilization_rate": {"dtype": "float64", "_type": "Value"},
            "choices": [{"dtype": "string", "_type": "Value"}],
            "causal_factors": [{"dtype": "string", "_type": "Value"}]}


def payload(*, bad_schema=False) -> bytes:
    fields = [pa.field(name, pa.string()) for name in STRINGS]
    fields += [pa.field(name, pa.bool_()) for name in BOOLS]
    fields += [pa.field(name, pa.int64()) for name in INTS]
    fields += [pa.field("factor_utilization_rate", pa.float64()),
               pa.field("choices", pa.list_(pa.string())), pa.field("causal_factors", pa.list_(pa.string()))]
    values = {name: ["A", "B", None] for name in STRINGS}
    values.update({name: [True, False, None] for name in BOOLS})
    values.update({name: [1, 0, None] for name in INTS})
    values.update(factor_utilization_rate=[.5, float("nan"), None],
                  choices=[["A", "B"], [], None], causal_factors=[["cue"], [], None])
    values["input"] = ["α exact\ninput", "", None]
    if bad_schema:
        fields = [f for f in fields if f.name != "core_id"]
        values.pop("core_id")
    stream = io.BytesIO()
    pq.write_table(pa.Table.from_pydict(values, schema=pa.schema(fields)), stream)
    return stream.getvalue()


def spec_for(body: bytes, *, count=1) -> dict:
    repo, revision = "fixture/data", "a" * 40
    files = []
    for i in range(count):
        name = f"fixture/model-{i}.parquet"
        files.append({"path": name,
                      "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}",
                      "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    return {"schema_version": 1, "download_authorized": True,
            "dataset": {"repo_id": repo, "revision": revision, "private": False, "gated": False,
                        "license": "fixture", "license_url": "https://example.org/fixture-only"},
            "files": files, "file_count": count, "total_payload_bytes": len(body) * count,
            "limits": {"max_download_bytes": max(1024, len(body) * count * 3), "max_http_requests": 12,
                       "max_attempts_per_file": 2, "max_workers": 2, "timeout_seconds": 5,
                       "max_slim_bytes": 1_000_000, "batch_rows": 1},
            "analysis_columns": [name for name in FEATURES if name not in OMITTED],
            "hashed_text_columns": OMITTED[:], "observed_hf_features": deepcopy(FEATURES)}


class Response:
    def __init__(self, body: bytes, *, break_after: int | None = None, content_length=True):
        self.stream = io.BytesIO(body)
        self.break_after = break_after
        self.headers = {"Content-Length": str(len(body))} if content_length else {}

    def getcode(self):
        return 200

    def read(self, amount):
        if self.break_after is not None:
            if self.stream.tell() >= self.break_after:
                raise OSError("fixture partial read")
            amount = min(amount, self.break_after - self.stream.tell())
        return self.stream.read(amount)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stream.close()


class Opener:
    def __init__(self, response):
        self.response = response

    def open(self, request, timeout):
        return self.response


class CoTIntakeTests(unittest.TestCase):
    def test_pinned_spec_and_path_guards_before_any_request(self):
        spec = spec_for(b"abc")
        validate_spec(spec)
        for path in ("../escape.parquet", "./file.parquet", "a//b.parquet", "C:/file.parquet", "NUL.txt", ".env/key.parquet"):
            with self.subTest(path=path), self.assertRaises(IntakeError):
                safe_relative(path)
        for mutation in (lambda s: s["dataset"].update(gated=True),
                         lambda s: s["files"][0].update(url=s["files"][0]["url"].replace("a" * 40, "main")),
                         lambda s: s["files"][0].update(sha256="not-a-sha"),
                         lambda s: s["limits"].update(max_workers=5)):
            candidate = deepcopy(spec); mutation(candidate)
            with self.assertRaises(IntakeError):
                validate_spec(candidate)
        spec["download_authorized"] = False
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(IntakeError):
            run_intake(spec, Path(directory) / "out", opener_factory=lambda *args: self.fail("network must not be reached"))

    def test_failed_partial_retry_and_actual_transfer_accounting(self):
        body = b"abcdef"
        spec = spec_for(body)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = AuditLog(root / "acquisition.jsonl")
            budget = TransferBudget(spec["limits"], log)
            factory = lambda budget, source, attempt: Opener(Response(body, break_after=3 if attempt == 1 else None))
            result = download_one(0, spec["files"][0], root, budget, factory)
            self.assertEqual(result["status"], "verified")
            self.assertEqual(budget.received, 9)
            self.assertEqual(budget.requests, 2)
            self.assertEqual((root / "download_attempts/000/attempt-1.partial").read_bytes(), b"abc")
            self.assertEqual((root / "data/fixture/model-0.parquet").read_bytes(), body)
            rows = [json.loads(line) for line in log.path.read_text().splitlines()]
            self.assertEqual(sum(r.get("bytes", 0) for r in rows if r["kind"] == "dataset_download"), 9)
            self.assertEqual(sum("url" in r and "attempt" in r for r in rows), 2)
            self.assertFalse(any(r.get("status") == "retrieved" for r in rows))

    def test_byte_cap_and_hash_failures_preserve_attempts(self):
        spec = spec_for(b"a" * 10)
        spec["limits"]["max_download_bytes"] = 15
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); log = AuditLog(root / "acquisition.jsonl")
            budget = TransferBudget(spec["limits"], log)
            result = download_one(0, spec["files"][0], root, budget,
                                  lambda *args: Opener(Response(b"b" * 20, content_length=False)))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(budget.received, 15)
            self.assertEqual(sum(p.stat().st_size for p in (root / "download_attempts").rglob("*.partial")), 15)
            self.assertFalse((root / "data/fixture/model-0.parquet").exists())
        spec = spec_for(b"good")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); budget = TransferBudget(spec["limits"], AuditLog(root / "acquisition.jsonl"))
            result = download_one(0, spec["files"][0], root, budget, lambda *args: Opener(Response(b"evil")))
            self.assertEqual(result["status"], "failed")
            self.assertEqual(len(list((root / "download_attempts").rglob("*.partial"))), 2)
            self.assertEqual(budget.received, 8)

    def test_redirects_have_separate_physical_request_budget(self):
        spec = spec_for(b"abc")
        spec["limits"]["max_http_requests"] = 2
        source = spec["files"][0]
        with tempfile.TemporaryDirectory() as directory:
            budget = TransferBudget(spec["limits"], AuditLog(Path(directory) / "acquisition.jsonl"))
            budget.request(source, 1)
            handler = _BoundedRedirect(budget, source, 1)
            req = urllib.request.Request(source["url"])
            handler.redirect_request(req, None, 302, "redirect", {}, "https://cdn-lfs.huggingface.co/public?signature=not-logged")
            self.assertEqual(budget.requests, 2)
            with self.assertRaises(IntakeError):
                handler.redirect_request(req, None, 302, "redirect", {}, "https://cdn-lfs.huggingface.co/another")
            with self.assertRaises(IntakeError):
                handler.redirect_request(req, None, 302, "redirect", {}, "http://evil.example/secret")
            self.assertNotIn("signature", budget.log.path.read_text())

    def test_full_mock_pipeline_preserves_every_row_column_and_exact_text_hash(self):
        body = payload()
        spec = spec_for(body, count=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "out"
            result = run_intake(spec, root, opener_factory=lambda *args: Opener(Response(body)))
            self.assertEqual(result["status"], "complete")
            self.assertEqual((result["rows"], result["raw_files"]), (6, 2))
            self.assertEqual(result["network"]["transferred_bytes"], 2 * len(body))
            rows = pq.read_table(root / "slim.parquet").to_pylist()
            self.assertEqual(len(rows), 6)
            for name in spec["analysis_columns"]:
                self.assertIn(name, rows[0])
            self.assertNotIn("input", rows[0])
            self.assertEqual(rows[0]["input_sha256"], hashlib.sha256("α exact\ninput".encode()).hexdigest())
            self.assertEqual(rows[1]["input_sha256"], hashlib.sha256(b"").hexdigest())
            self.assertIsNone(rows[2]["input_sha256"])
            self.assertEqual(rows[0]["input_chars"], len("α exact\ninput"))
            self.assertEqual(rows[0]["input_utf8_bytes"], len("α exact\ninput".encode()))
            self.assertEqual(rows[0]["choices"], ["A", "B"])
            self.assertEqual(rows[0]["choices_ordered_sha256"], hashlib.sha256(b'["A","B"]').hexdigest())
            self.assertEqual(rows[1]["choices_ordered_sha256"], hashlib.sha256(b'[]').hexdigest())
            self.assertIsNone(rows[2]["choices_ordered_sha256"])
            self.assertIsNone(rows[2]["cue_acknowledgment"])
            self.assertEqual([r["source_row_index"] for r in rows], [0, 1, 2, 0, 1, 2])
            self.assertEqual(len({r["row_content_sha256"] for r in rows}), 3)
            profile = json.loads((root / "schema_profile.json").read_text())
            self.assertEqual(profile["nulls"]["input"], 2)
            self.assertFalse(profile["statistical_endpoints_computed"])
            with self.assertRaises(IntakeError):
                run_intake(spec, root, opener_factory=lambda *args: self.fail("no overwrite request"))

    def test_schema_failure_has_failed_report_and_keeps_verified_payload(self):
        body = payload(bad_schema=True)
        spec = spec_for(body)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "out"
            with self.assertRaises(IntakeError):
                run_intake(spec, root, opener_factory=lambda *args: Opener(Response(body)))
            self.assertEqual(file_sha256(root / "data/fixture/model-0.parquet"), spec["files"][0]["sha256"])
            self.assertEqual(json.loads((root / "intake_report.json").read_text())["status"], "failed")
            self.assertFalse((root / "slim.parquet").exists())

    def test_slim_cap_preserves_partial_without_silently_dropping_rows(self):
        body = payload(); spec = spec_for(body)
        spec["limits"]["max_slim_bytes"] = 100
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "out"
            with self.assertRaises(IntakeError):
                run_intake(spec, root, opener_factory=lambda *args: Opener(Response(body)))
            self.assertTrue((root / "slim.parquet.partial").exists())
            self.assertFalse((root / "slim.parquet").exists())
            self.assertEqual(json.loads((root / "download_manifest.json").read_text())["status"], "failed")

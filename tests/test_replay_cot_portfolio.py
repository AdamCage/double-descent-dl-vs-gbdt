"""Tiny handwritten comparator fixtures only; never replay the study corpus."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import socket
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.replay_cot_portfolio import (
    Comparison, ReplayError, compare_csv, compare_json, compare_npz,
    compare_parquet, csv_schema, offline_guard, replay,
)


class ReplayComparatorTests(unittest.TestCase):
    def test_json_root_exclusions_numeric_tolerance_and_exact_count_types(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "reference.json", root / "actual.json"
            reference = {"timing": {"seconds": 1.}, "environment": {"host": "old"},
                         "cells": [{"count": 3, "eligible": False, "gap": .1}], "settings": {"timing": 2}}
            actual = {**reference, "timing": {"seconds": 9.}, "environment": {"host": "new"},
                      "cells": [{"count": 3, "eligible": False, "gap": .1 + 1e-13}]}
            left.write_text(json.dumps(reference))
            right.write_text(json.dumps(actual))
            self.assertEqual(compare_json(left, right, analysis=True)["mismatches"], 0)
            actual["settings"] = {"timing": 3}
            right.write_text(json.dumps(actual))
            self.assertGreater(compare_json(left, right, analysis=True)["mismatches"], 0)
        comparison = Comparison("strict types")
        comparison.compare({"count": 1, "flag": False}, {"count": 1., "flag": 0})
        self.assertEqual(comparison.mismatches, 2)

    def test_entire_bootstrap_and_identifier_order_nan_masks_and_dtype(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "reference.npz", root / "actual.npz"
            samples = np.array([[[.1, np.nan]], [[.2, .3]]])
            np.savez(left, samples=samples, metric_names=np.array(["a", "b"]), cell_ids=np.array(["cell"]))
            near = samples.copy()
            near[1, 0, 1] += 1e-13
            np.savez(right, samples=near, metric_names=np.array(["a", "b"]), cell_ids=np.array(["cell"]))
            self.assertEqual(compare_npz(left, right)["mismatches"], 0)
            np.savez(right, samples=near, metric_names=np.array(["b", "a"]), cell_ids=np.array(["cell"]))
            self.assertGreater(compare_npz(left, right)["mismatches"], 0)
            near[0, 0, 1] = 0.
            np.savez(right, samples=near, metric_names=np.array(["a", "b"]), cell_ids=np.array(["cell"]))
            self.assertGreater(compare_npz(left, right)["mismatches"], 0)
            np.savez(right, samples=samples.astype("float32"), metric_names=np.array(["a", "b"]), cell_ids=np.array(["cell"]))
            self.assertGreater(compare_npz(left, right)["mismatches"], 0)

    def test_parquet_all_rows_null_nan_metadata_and_different_row_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "reference.parquet", root / "actual.parquet"
            schema = pa.schema([("question", pa.string()), ("value", pa.float64()), ("count", pa.int64())],
                               metadata={b"unit": b"question"})
            rows = [{"question": "q1", "value": None, "count": 1}, {"question": "q2", "value": .2, "count": 2},
                    {"question": "q3", "value": float("nan"), "count": 3}]
            table = pa.Table.from_pylist(rows, schema=schema)
            pq.write_table(table, left, row_group_size=1)
            pq.write_table(table, right, row_group_size=3)
            self.assertEqual(compare_parquet(left, right)["mismatches"], 0)
            pq.write_table(pa.Table.from_pylist(rows[:-1], schema=schema), right)
            self.assertGreater(compare_parquet(left, right)["mismatches"], 0)
            changed = [dict(row) for row in rows]
            changed[0]["value"] = float("nan")
            pq.write_table(pa.Table.from_pylist(changed, schema=schema), right)
            self.assertGreater(compare_parquet(left, right)["mismatches"], 0)
            pq.write_table(table.replace_schema_metadata({b"unit": b"wrong"}), right)
            self.assertGreater(compare_parquet(left, right)["mismatches"], 0)

    def test_csv_preserves_integer_boolean_float_and_row_semantics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left, right = root / "reference.csv", root / "actual.csv"
            schema = csv_schema({"cells": [{"counts": {"n": 3, "coverage": .5}, "estimates": {"gap": .1}}]})
            row = {"cell_id": "pair", "dataset": "fixture", "model_a": "A", "model_b": "B", "ineligible_reasons": "",
                   "eligible": False, "material_effect": False, "delta_total_low": None, "delta_total_high": .2,
                   "n": 3, "coverage": .5, "gap": .1}
            def write(path, values):
                with path.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(schema))
                    writer.writeheader()
                    writer.writerows(values)
            write(left, [row])
            write(right, [{**row, "gap": .1 + 1e-13}])
            self.assertEqual(compare_csv(left, right, schema)["mismatches"], 0)
            write(right, [{**row, "n": "3.0", "eligible": "0"}])
            self.assertGreaterEqual(compare_csv(left, right, schema)["mismatches"], 2)
            write(right, [row, row])
            self.assertGreater(compare_csv(left, right, schema)["mismatches"], 0)

    def test_network_is_blocked_and_existing_evidence_or_output_not_modified(self):
        with offline_guard(), self.assertRaises(ReplayError):
            socket.create_connection(("127.0.0.1", 1))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence, existing = root / "evidence", root / "existing"
            evidence.mkdir()
            existing.mkdir()
            marker = existing / "keep.txt"
            marker.write_text("preserve")
            with self.assertRaises(ReplayError):
                replay(evidence, existing)
            self.assertEqual(marker.read_text(), "preserve")
            with self.assertRaises(ReplayError):
                replay(evidence, evidence / "nested-output")
            self.assertFalse((evidence / "nested-output").exists())


if __name__ == "__main__":
    unittest.main()

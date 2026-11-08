"""Corrupt delivery inventories must never pass release verification."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.verify_science_release import digest, verify_gallery, check_replayed_values


class ReleaseValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="dd-release-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "figures").mkdir()
        (self.root / "tables").mkdir()
        (self.root / "index.html").write_text("<html>gallery</html>")
        figures = []
        for i in range(2):
            paths = {kind: f"figures/static{i}.{kind}" for kind in ("png", "svg")}
            Image.new("RGB", (600, 300), "white").save(self.root / paths["png"])
            (self.root / paths["svg"]).write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
            figures.append({"id": f"static{i}", "paths": paths,
                            "sha256": {key: digest(self.root / value) for key, value in paths.items()}})
        (self.root / "figures/interactive.html").write_text("<html>interactive</html>")
        figures.append({"id": "interactive", "paths": {"html": "figures/interactive.html"},
                        "sha256": {"html": digest(self.root / "figures/interactive.html")}})
        table = self.root / "tables/toy.csv"
        table.write_text("value\n1\n2\n")
        self.manifest = {"figure_count": 3, "figures": figures,
                         "tables": [{"path": "tables/toy.csv", "rows": 2, "sha256": digest(table)}]}

    def check(self, document=None):
        (self.root / "figure_manifest.json").write_text(json.dumps(document or self.manifest))
        return verify_gallery(self.root, expected_static=2, expected_interactive=1, expected_tables={"toy": 2})

    def test_complete_inventory(self):
        self.assertEqual(self.check()["source_tables"], 1)

    def test_reused_paths_and_mixed_entry_kinds_are_rejected(self):
        bad = copy.deepcopy(self.manifest)
        bad["figures"][1]["paths"] = bad["figures"][0]["paths"]
        bad["figures"][1]["sha256"] = bad["figures"][0]["sha256"]
        with self.assertRaises(ValueError):
            self.check(bad)
        bad = copy.deepcopy(self.manifest)
        bad["figures"][0]["paths"]["html"] = "figures/interactive.html"
        with self.assertRaises(ValueError):
            self.check(bad)

    def test_empty_duplicate_escaping_and_truncated_tables_are_rejected(self):
        for entries in ([], self.manifest["tables"] * 2,
                        [{"path": "../outside.csv", "rows": 2, "sha256": "0" * 64}]):
            bad = copy.deepcopy(self.manifest)
            bad["tables"] = entries
            with self.assertRaises(ValueError):
                self.check(bad)
        bad = copy.deepcopy(self.manifest)
        table = self.root / "tables/toy.csv"
        table.write_text("value\n1\n")
        bad["tables"][0].update(rows=1, sha256=digest(table))
        with self.assertRaises(ValueError):
            self.check(bad)

    def test_extra_and_missing_files_are_rejected(self):
        (self.root / "tables/unregistered.csv").write_text("x\n1\n")
        with self.assertRaises(ValueError):
            self.check()
        (self.root / "tables/unregistered.csv").unlink()
        (self.root / "figures/static1.svg").unlink()
        with self.assertRaises(ValueError):
            self.check()

    def test_equal_but_wrong_or_nonfinite_replay_is_rejected(self):
        with self.assertRaises(ValueError):
            check_replayed_values(np.ones((2, 2)), np.ones((2, 2)))
        values = np.zeros((7188, 256))
        values[0, 0] = np.nan
        with self.assertRaises(ValueError):
            check_replayed_values(values, values)

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from scripts.deduplicated_evidence_bundle import INDEX, TOOL, create, restore, verify


class DeduplicatedEvidenceTests(unittest.TestCase):
    def test_exact_roundtrip_deduplication_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            (source / "attempt1").mkdir(parents=True)
            (source / "attempt2").mkdir()
            for name in ("attempt1/data.npz", "attempt2/data.npz"):
                (source / name).write_bytes(bytes(range(256)) * 200)
            (source / "attempt2/log.txt").write_bytes(b"failure preserved\n")
            archive = root / "evidence.zip"
            result = create(source, archive)
            self.assertEqual((result["files"], result["unique_blobs"]), (3, 2))
            self.assertLess(result["unique_bytes"], result["restored_bytes"])
            restore(archive, root / "restored")
            for path in source.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (root / "restored" / path.relative_to(source)).read_bytes())
            with self.assertRaises(FileExistsError):
                create(source, archive)
            with self.assertRaises(ValueError):
                restore(archive, root / "restored")

    def _write_bad(self, path, rows, blobs):
        with zipfile.ZipFile(path, "w") as archive:
            tool = b"# Fixture only; never executed\n"
            archive.writestr(INDEX, json.dumps({"schema_version": 1, "format": "sha256-blobs-v1", "files": rows,
                                               "restore_tool_sha256": hashlib.sha256(tool).hexdigest()}))
            archive.writestr(TOOL, tool)
            for name, payload in blobs.items():
                archive.writestr(name, payload)

    def test_corruption_extra_blob_path_escape_and_aliases_rejected(self):
        payload = b"bound raw metrics"
        sha = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [
                ([{"path": "../escape", "sha256": sha, "bytes": len(payload)}], {f"blobs/{sha}": payload}),
                ([{"path": "data", "sha256": sha, "bytes": len(payload)}], {f"blobs/{sha}": b"x" * len(payload)}),
                ([{"path": "data", "sha256": sha, "bytes": len(payload)}], {f"blobs/{sha}": payload, "extra": b"x"}),
                ([{"path": p, "sha256": sha, "bytes": len(payload)} for p in ("Data", "data")], {f"blobs/{sha}": payload}),
                ([{"path": p, "sha256": sha, "bytes": len(payload)} for p in ("data", "data/child")], {f"blobs/{sha}": payload}),
            ]
            for i, (rows, blobs) in enumerate(cases):
                path = root / f"bad{i}.zip"
                self._write_bad(path, rows, blobs)
                with self.assertRaises(ValueError):
                    verify(path)

    def test_environment_and_external_source_link_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / ".env").write_text("dummy-only")
            with self.assertRaises(ValueError):
                create(source, root / "bad.zip")

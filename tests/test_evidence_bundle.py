from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.evidence_bundle import create_bundle, verify_bundle


class EvidenceBundleTest(unittest.TestCase):
    def test_bundle_is_portable_and_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            source.mkdir()
            (source / "metrics.json").write_text('{"loss":0.5}', encoding="utf-8")
            output = root / "evidence.tar.gz"
            result = create_bundle(source, output)
            (source / "metrics.json").unlink()
            source.rmdir()
            self.assertEqual(verify_bundle(output), result)
            source.mkdir()
            (source / "x").write_text("x", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_bundle(source, output)

    def test_corrupted_bytes_fail_detached_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.tar.gz"
            manifest = {"schema_version": 1, "files": [
                {"path": "metrics.json", "bytes": 3, "sha256": "0" * 64}]}
            with tarfile.open(path, "w:gz") as archive:
                for name, payload in (("BUNDLE-MANIFEST.json", json.dumps(manifest).encode()),
                                      ("metrics.json", b"bad")):
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "Evidence mismatch"):
                verify_bundle(path)

    def test_refuses_secret_environment(self):
        for name in (".env", ".env.local", ".Env.production", ".env.local/nested.txt"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                source = root / "source"
                source.mkdir()
                target = source / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("test sentinel", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Secret"):
                    create_bundle(source, root / "evidence.tar.gz")


class LegacyEntryGuardTest(unittest.TestCase):
    def test_default_and_original_output_cannot_start_training(self):
        root = Path(__file__).resolve().parents[1]
        for argv in ([], ["--stage", "s05"],
                     ["--legacy", "--stage", "s05", "--result-dir", "result"]):
            result = subprocess.run([sys.executable, str(root / "main.py"), *argv],
                                    cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Pipeline started", result.stdout)


if __name__ == "__main__":
    unittest.main()

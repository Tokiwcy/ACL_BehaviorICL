"""Offline failure-boundary tests for the new portable artifact tools."""
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import handoff_artifact_utils as artifacts
import materialize_handoff_artifacts as materialize


class HandoffTests(unittest.TestCase):
    def fixture(self, root):
        original = b'{"frozen": "exact original bytes"}\r\n'
        data = gzip.compress(original, mtime=0)
        (root / "archive.json.gz").write_bytes(data)
        entry = {"path": "archive.json.gz", "materialize_path": "analysis/lock.json",
                 "kind": "gzip_exact", "bytes": len(data), "sha256": artifacts.sha256(data),
                 "source_bytes": len(original), "source_sha256": artifacts.sha256(original)}
        (root / "handoff").mkdir()
        (root / "handoff/publish_manifest.json").write_text(json.dumps({"files": [entry]}))
        return original, entry

    def test_exact_restore_and_existing_conflict_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original, entry = self.fixture(root)
            with patch.object(artifacts, "ROOT", root):
                materialize.main()
                target = root / entry["materialize_path"]
                self.assertEqual(target.read_bytes(), original)
                materialize.main()
                target.write_bytes(b"conflicting evidence")
                with self.assertRaisesRegex(ValueError, "Conflicting existing"):
                    materialize.main()
                self.assertEqual(target.read_bytes(), b"conflicting evidence")

    def test_wrong_compressed_source_sha_and_size_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, entry = self.fixture(root)
            with patch.object(artifacts, "ROOT", root):
                for wrong in ({"sha256": "0" * 64}, {"source_sha256": "0" * 64}, {"source_bytes": 1}):
                    with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                        artifacts.verified_payload({**entry, **wrong})

    def test_path_escape_and_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as external:
            root = Path(temp)
            with patch.object(artifacts, "ROOT", root):
                for path in ("../outside", "/absolute", "C:/machine", "folder\\path"):
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        artifacts.repository_path(path)
                try:
                    (root / "linked").symlink_to(external, target_is_directory=True)
                except OSError:
                    return  # Windows may require symlink privileges; Unix CI exercises this boundary.
                with self.assertRaises(ValueError):
                    artifacts.repository_path("linked/evidence.json")


if __name__ == "__main__":
    unittest.main()

"""Offline checks for preserving publication history while extending the README."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import verify_local_publication_v1 as verifier


def digest(data):
    return hashlib.sha256(data).hexdigest()


class SnapshotTests(unittest.TestCase):
    def fixture(self, root):
        (root / 'handoff').mkdir()
        audit_path = root / 'analysis/internvl35_joint_v2/combined_results_audit.json'
        audit_path.parent.mkdir(parents=True)
        rows = {str(i): {'state': 'complete' if i < 10 else 'partial'} for i in range(24)}
        audit_path.write_text(json.dumps({'state': 'incomplete', 'datasets': {
            'fixture': {'methods': rows}}, 'incomplete': list(range(14))}), encoding='utf-8')
        original = root / 'handoff/publish_manifest.json'
        original.write_bytes(b'{}')
        document = root / 'README.md'
        document.write_bytes(b'collaborator handoff\n')
        paths = [document, audit_path]
        manifest = {'version': 'old_fixture', 'dynamic_full_test_audited_cells': 10,
                    'original_manifest_sha256': digest(original.read_bytes()),
                    'files': [{'path': p.relative_to(root).as_posix(),
                               'bytes': len(p.read_bytes()), 'sha256': digest(p.read_bytes())}
                              for p in paths]}
        previous = root / 'handoff/previous.json'
        previous.write_text(json.dumps(manifest), encoding='utf-8')
        latest = root / 'handoff/latest.json'
        latest.write_text(json.dumps({**manifest, 'version': 'new_fixture',
                                      'previous_manifest_sha256': digest(previous.read_bytes())}),
                          encoding='utf-8')
        return previous, latest, document

    def verify(self, root, previous, latest, arguments=()):
        output = io.StringIO()
        with patch.object(verifier, 'ROOT', root), patch.object(verifier, 'MANIFEST', previous), \
                patch.object(verifier, 'COLLABORATOR_MANIFEST', latest), \
                patch.object(sys, 'argv', ['verify', *arguments]), contextlib.redirect_stdout(output):
            verifier.main()
        return json.loads(output.getvalue())

    def test_latest_default_and_explicit_historical_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous, latest, _ = self.fixture(root)
            self.assertEqual(self.verify(root, previous, latest)['publication_version'], 'new_fixture')
            result = self.verify(root, previous, latest, ['--manifest', str(previous)])
            self.assertEqual(result['publication_version'], 'old_fixture')

    def test_changed_previous_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous, latest, _ = self.fixture(root)
            previous.write_bytes(previous.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'Previous publication manifest changed'):
                self.verify(root, previous, latest)

    def test_changed_published_document_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            previous, latest, document = self.fixture(root)
            document.write_bytes(b'unbound change\n')
            with self.assertRaisesRegex(ValueError, 'Published bytes differ'):
                self.verify(root, previous, latest)


if __name__ == '__main__':
    unittest.main()

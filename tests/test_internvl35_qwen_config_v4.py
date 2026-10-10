"""CPU integrity checks for duplicate supervision and recorded five-image budgets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_internvl35_qwen_pipeline_v4 as pipeline
import audit_internvl35_qwen_config_v4 as audit
from internvl35_qwen_config_v4 import PROTOCOL_PATH, digest
from multidataset_protocol import Sample


class IntegrityTests(unittest.TestCase):
    def test_second_supervisor_cannot_acquire_active_lock(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline, "OUTPUT", Path(temp)):
            with pipeline.lock_gpu():
                # Windows can reject reading the already locked byte before LK_NBLCK.
                with self.assertRaises(RuntimeError):
                    pipeline.lock_gpu()
            with pipeline.lock_gpu():
                pass

    def context(self, folder, tokens=1280):
        bank = [Sample(f"b{i}", f"/bank/{i}.jpg", "class", "bank") for i in range(4)]
        query = [Sample("q", "/query/q.jpg", "class", "query")]
        budget = {"protocol_sha256": digest(PROTOCOL_PATH),
                  "images": {s.sample_id: {"visual_tokens": 256} for s in bank + query}}
        (folder / "image_budget.json").write_text(json.dumps(budget))
        (folder / "telemetry").mkdir()
        row = {"images": [[s.path for s in bank + query]], "visual_tokens": [tokens],
               "pixel_values_shape": [5, 3, 448, 448], "input_tokens": [1400]}
        path = folder / "telemetry/icl_raw.jsonl"
        path.write_text(json.dumps(row) + "\n")
        return bank, query, {"q": [0, 1, 2, 3]}, path, row

    def test_valid_five_images_are_audited(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            bank, query, selection, _, _ = self.context(folder)
            result = audit.audit_vision(folder, bank, query, selection, "raw")
            self.assertEqual(result["visual_tokens"]["min"], 1280)

    def test_changed_token_budget_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            bank, query, selection, _, _ = self.context(folder, tokens=256)
            with self.assertRaises(ValueError):
                audit.audit_vision(folder, bank, query, selection, "raw")

    def test_changed_image_order_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            bank, query, selection, path, row = self.context(folder)
            row["images"][0][0], row["images"][0][1] = row["images"][0][1], row["images"][0][0]
            path.write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                audit.audit_vision(folder, bank, query, selection, "raw")


if __name__ == "__main__":
    unittest.main()

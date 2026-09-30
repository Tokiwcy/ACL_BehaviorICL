import argparse
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_multidataset_main as runner
from multidataset_protocol import Sample


class MultidatasetRunnerTests(unittest.TestCase):
    def test_vision_budget_uses_separate_run_tree(self):
        root = Path("results/cdr_main")
        self.assertEqual(runner.vision_run_root(root, None), root)
        self.assertEqual(runner.vision_run_root(root, 50176), root / "vision_50176")

    def test_identity_prevents_cross_run_checkpoint_reuse(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            identity = {"dataset": "dtd", "model_slug": "qwen3vl4b", "model_id": "q", "seed": 73}
            runner.validate_or_write_identity(root, identity, resume=False)
            runner.validate_or_write_identity(root, identity, resume=True)
            changed = {**identity, "dataset": "cub"}
            with self.assertRaises(RuntimeError):
                runner.validate_or_write_identity(root, changed, resume=True)
            with self.assertRaises(RuntimeError):
                runner.validate_or_write_identity(root, {**identity, "vision_pixels": 50176}, resume=True)

    def test_nearest_demo_order_is_ascending_within_top_k(self):
        embeddings = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [1.0, 0.0]])
        query = [argparse.Namespace(sample_id="q")]
        selected = runner.nearest_from_embeddings(embeddings, 3, query, shots=2)["q"]
        self.assertEqual(selected, [1, 0])

    def test_generation_resume_validates_split_and_selections_without_cached_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bank = [Sample("b", "local.jpg", "label", "train")]
            query = [Sample("q", "query.jpg", "label", "test")]
            runner.write_json(root / "manifest.json", {
                "bank": [asdict(bank[0])], "query": [asdict(query[0])],
            })
            runner.write_json(root / "selections.json", {
                "rices": {"q": [asdict(bank[0])]},
            })
            runner.validate_generation_resume(root, bank, query, ["rices"], 1)
            changed = [Sample("b", "local.jpg", "other", "train")]
            with self.assertRaises(RuntimeError):
                runner.validate_generation_resume(root, changed, query, ["rices"], 1)
            with self.assertRaises(RuntimeError):
                runner.validate_generation_resume(root, bank, query, ["gpt_mm"], 1)



if __name__ == "__main__":
    unittest.main()

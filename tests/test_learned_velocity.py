import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_dtd_learned_velocity as learned
import analyze_learned_velocity as analysis


class LearnedVelocityTests(unittest.TestCase):
    def test_split_is_balanced_and_disjoint(self):
        labels = ["a"] * 8 + ["b"] * 8
        train, val = learned.stratified_development_split(labels, 2, 7)
        self.assertEqual(len(train), 12)
        self.assertEqual(len(val), 4)
        self.assertFalse(set(train) & set(val))
        self.assertEqual([labels[i] for i in val].count("a"), 2)
        self.assertEqual([labels[i] for i in val].count("b"), 2)

    def test_velocity_shape_and_normalization(self):
        model = learned.VelocityRetriever(8, 3, 4)
        encoded = model(torch.randn(5, 4, 8))
        self.assertEqual(tuple(encoded.shape), (5, 3, 4))
        torch.testing.assert_close(encoded.norm(dim=-1), torch.ones(5, 3), atol=1e-5, rtol=1e-5)

    def test_contrastive_loss_prefers_same_class(self):
        labels = torch.tensor([0, 0, 1, 1])
        good = torch.tensor(
            [[1.0, 0.9, 0.0, 0.0], [0.9, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.9], [0.0, 0.0, 0.9, 1.0]]
        )
        bad = -good
        self.assertLess(
            float(learned.supervised_contrastive_loss(good, labels, 0.1)),
            float(learned.supervised_contrastive_loss(bad, labels, 0.1)),
        )

    def test_canonicalization_is_formatting_only(self):
        classes = ["dotted", "polka-dotted", "lacelike"]
        self.assertEqual(
            analysis.canonical_prediction("lace-like", None, classes), "lacelike"
        )
        self.assertEqual(
            analysis.canonical_prediction("polka-dotted", "polka-dotted", classes),
            "polka-dotted",
        )
        self.assertIsNone(analysis.canonical_prediction("charred", None, classes))


if __name__ == "__main__":
    unittest.main()

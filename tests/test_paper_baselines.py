import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_dtd_paper_baselines as paper


class PaperBaselineTests(unittest.TestCase):
    def test_layer_cadence_includes_endpoints(self):
        self.assertEqual(paper.detriever_layers(36), [0, 4, 9, 14, 19, 24, 29, 34, 35])

    def test_proxy_candidates_obey_labels(self):
        labels = ["a"] * 3 + ["b"] * 3
        positive, negative = paper.proxy_candidate_indices(labels, 4, 5, 7)
        for anchor in range(len(labels)):
            self.assertTrue(all(labels[i] == labels[anchor] for i in positive[anchor]))
            self.assertTrue(all(labels[i] != labels[anchor] for i in negative[anchor]))
            self.assertNotIn(anchor, positive[anchor])

    def test_detriever_output_is_normalized(self):
        model = paper.DeTriever(input_size=8, layer_count=3)
        output = model(torch.randn(5, 3, 8))
        self.assertEqual(tuple(output.shape), (5, 512))
        np.testing.assert_allclose(output.norm(dim=-1).detach().numpy(), 1.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()

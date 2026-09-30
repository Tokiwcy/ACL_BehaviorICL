import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from detriever_proxy import output_proxy_candidate_indices


class OutputProxyTests(unittest.TestCase):
    def test_similarity_ranking_is_not_label_identity(self):
        vectors = np.asarray([
            [1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [-1.0, 0.0],
        ], dtype=np.float32)
        positives, negatives = output_proxy_candidate_indices(vectors, 1, 1)
        self.assertEqual(int(positives[0, 0]), 1)
        self.assertEqual(int(negatives[0, 0]), 3)
        for anchor in range(len(vectors)):
            self.assertNotIn(anchor, positives[anchor])
            self.assertNotIn(anchor, negatives[anchor])
            self.assertFalse(set(positives[anchor]) & set(negatives[anchor]))

    def test_rejects_too_small_or_invalid_bank(self):
        with self.assertRaises(ValueError):
            output_proxy_candidate_indices(np.eye(2), 1, 1)
        with self.assertRaises(ValueError):
            output_proxy_candidate_indices(np.asarray([[1., 0.], [0., np.nan], [0., 1.]]), 1, 1)


if __name__ == "__main__":
    unittest.main()

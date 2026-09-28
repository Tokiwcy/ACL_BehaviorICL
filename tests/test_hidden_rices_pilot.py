import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_dtd_hidden_rices_pilot.py"
SPEC = importlib.util.spec_from_file_location("pilot", SCRIPT)
pilot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


class SimilarityTests(unittest.TestCase):
    def test_cosine_rows(self):
        query = np.array([1.0, 0.0])
        bank = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(pilot.cosine_rows(query, bank), [1.0, 0.0, -1.0])

    def test_matrix_similarity_is_row_aligned(self):
        query = np.eye(2)
        bank = np.stack([np.eye(2), np.flip(np.eye(2), axis=0)])
        scores = pilot.matrix_similarity(query, bank)
        np.testing.assert_allclose(scores, [1.0, 0.0])

    def test_parse_prediction(self):
        labels = ["braided", "bubbly", "cracked"]
        self.assertEqual(pilot.parse_prediction("cracked", labels), "cracked")
        self.assertEqual(pilot.parse_prediction("Label: bubbly.", labels), "bubbly")
        self.assertIsNone(pilot.parse_prediction("wooden", labels))

    def test_direct_matrix_retrievers_rank_full_bank(self):
        bank = [
            pilot.Sample(f"b{i}", f"b{i}.jpg", label, "bank")
            for i, label in enumerate(["a", "b", "c"])
        ]
        query = [pilot.Sample("q0", "q0.jpg", "a", "query")]
        clip = np.array([[1, 0], [0, 1], [-1, 0], [1, 0]], dtype=float)
        final = clip.copy()
        matrix = np.array(
            [
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]],
                [[-1, 0], [-1, 0]],
                [[1, 0], [1, 0]],
            ],
            dtype=float,
        )
        delta = np.array(
            [
                [[1, 0], [1, 0]],
                [[0, 1], [0, 1]],
                [[0, -1], [0, -1]],
                [[0, 1], [0, 1]],
            ],
            dtype=float,
        )
        selections = pilot.rank_methods(
            bank,
            query,
            clip,
            final,
            matrix,
            delta,
            final,
            matrix,
            delta,
            shots=1,
            pool_size=3,
            seed=73,
            class_order=["a", "b", "c"],
        )
        self.assertEqual(selections["direct_selected_matrix"]["q0"], [0])
        self.assertEqual(selections["direct_delta_matrix"]["q0"], [1])


if __name__ == "__main__":
    unittest.main()

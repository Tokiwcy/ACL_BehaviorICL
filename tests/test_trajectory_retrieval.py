import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_dtd_trajectory_retrieval.py"
SPEC = importlib.util.spec_from_file_location("trajectory", SCRIPT)
trajectory = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(trajectory)


class TrajectoryTests(unittest.TestCase):
    def test_activity_weights_are_normalized_and_activity_sensitive(self):
        speed = torch.tensor([[1.0, 2.0, 1.0]])
        curvature = torch.tensor([[0.0, 1.0, 0.0]])
        weights = trajectory.activity_weights(speed, curvature)
        torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1))
        self.assertGreater(float(weights[0, 1]), float(weights[0, 0]))

    def test_dtw_prefers_matching_trajectory(self):
        matching = np.eye(4, dtype=np.float32)
        mismatching = -np.eye(4, dtype=np.float32)
        scores = trajectory.dtw_scores(np.stack([matching, mismatching]), band=1)
        self.assertGreater(float(scores[0]), float(scores[1]))


if __name__ == "__main__":
    unittest.main()

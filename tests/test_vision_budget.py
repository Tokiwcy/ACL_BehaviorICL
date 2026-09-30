import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_vision_budget import development_selection
from multidataset_protocol import Sample


class VisionBudgetTests(unittest.TestCase):
    def test_pilot_holds_out_development_queries_from_all_demonstrations(self):
        bank = [Sample(f"{label}_{index}", "unused.jpg", label, "bank")
                for label in ("a", "b", "c") for index in range(4)]
        selected = development_selection(bank, per_class=1, shots=2, seed=73)
        self.assertEqual(len(selected), 3)
        self.assertEqual({sample.label for sample, _ in selected}, {"a", "b", "c"})
        held_out = {sample.sample_id for sample, _ in selected}
        for _, demos in selected:
            self.assertEqual(len(demos), 2)
            self.assertEqual(len({demo.label for demo in demos}), 2)
            self.assertTrue(all(demo.sample_id not in held_out for demo in demos))


if __name__ == "__main__":
    unittest.main()

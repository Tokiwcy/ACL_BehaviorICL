import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import multidataset_protocol as protocol


class MultidatasetProtocolTests(unittest.TestCase):
    def test_every_official_split_has_matching_label_spaces(self):
        expected = {
            "dtd": (3760, 1880, 47),
            "aircraft": (6667, 3333, 100),
            "cub": (5994, 5794, 200),
            "dogs": (12000, 8580, 120),
            "pets": (3680, 3669, 37),
        }
        for name, counts in expected.items():
            bank, query = protocol.load_dataset(name)
            self.assertEqual((len(bank), len(query), len({x.label for x in bank})), counts)
            self.assertEqual({x.label for x in bank}, {x.label for x in query})

    def test_run_directory_scopes_checkpoints(self):
        root = Path("results")
        left = protocol.run_directory(root, "dtd", "qwen3vl4b", 73)
        right = protocol.run_directory(root, "cub", "qwen3vl4b", 73)
        other_model = protocol.run_directory(root, "dtd", "glm4v9b", 73)
        other_seed = protocol.run_directory(root, "dtd", "qwen3vl4b", 74)
        self.assertEqual(len({left, right, other_model, other_seed}), 4)


if __name__ == "__main__":
    unittest.main()

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from legal_label_decoding import LegalLabelTrie, exact_generated_label
from multimodal_model_adapter import task_instruction


class LegalLabelDecodingTests(unittest.TestCase):
    def test_trie_only_allows_legal_continuations(self):
        trie = LegalLabelTrie([[1, 2], [1, 3], [4]], eos_token_id=9)
        self.assertEqual(trie.allowed([]), [1, 4])
        self.assertEqual(trie.allowed([1]), [2, 3])
        self.assertEqual(trie.allowed([1, 2]), [9])
        self.assertEqual(trie.allowed([4]), [9])

    def test_prefix_callback_removes_prompt(self):
        trie = LegalLabelTrie([[5, 6]], eos_token_id=9)
        callback = trie.prefix_allowed_tokens_fn(prompt_length=3)
        self.assertEqual(callback(0, torch.tensor([20, 21, 22])), [5])
        self.assertEqual(callback(0, torch.tensor([20, 21, 22, 5])), [6])

    def test_prefix_callback_tolerates_finished_rows_in_a_batch(self):
        trie = LegalLabelTrie([[5], [6, 7]], eos_token_id=9)
        callback = trie.prefix_allowed_tokens_fn(prompt_length=2)
        self.assertEqual(callback(0, torch.tensor([20, 21, 5, 9])), [9])
        # Transformers can append its pad token while a longer batch row runs.
        self.assertEqual(callback(0, torch.tensor([20, 21, 5, 9, 99, 99])), [9])

    def test_prefix_callback_rejects_early_eos(self):
        trie = LegalLabelTrie([[5, 6]], eos_token_id=9)
        callback = trie.prefix_allowed_tokens_fn(prompt_length=1)
        with self.assertRaises(RuntimeError):
            callback(0, torch.tensor([20, 5, 9]))

    def test_illegal_prefix_fails_loudly(self):
        trie = LegalLabelTrie([[1, 2]], eos_token_id=9)
        with self.assertRaises(RuntimeError):
            trie.allowed([7])

    def test_exact_output_does_not_merge_distinct_dtd_labels(self):
        labels = ["dotted", "polka-dotted", "lacelike"]
        self.assertEqual(exact_generated_label(" polka-dotted ", labels), "polka-dotted")
        with self.assertRaises(RuntimeError):
            exact_generated_label("lace-like", labels)

    def test_every_dataset_has_task_specific_wording(self):
        labels = ["a", "b"]
        self.assertIn("texture category", task_instruction("dtd", labels))
        self.assertIn("aircraft variant", task_instruction("aircraft", labels))
        self.assertIn("bird species", task_instruction("cub", labels))
        self.assertIn("dog breed", task_instruction("dogs", labels))


if __name__ == "__main__":
    unittest.main()

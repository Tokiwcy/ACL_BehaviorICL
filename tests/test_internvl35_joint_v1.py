"""CPU-only cache branch and interruption tests. Never load pretrained weights."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import internvl35_joint_source_v1 as joint


class JointTests(unittest.TestCase):
    def test_answer_index_preserves_eos_and_cap_behavior(self):
        self.assertEqual(joint.answer_state_index([2], 2, 1), (1, 0))
        self.assertEqual(joint.answer_state_index([7, 8, 2], 2, 3), (2, 2))
        self.assertEqual(joint.answer_state_index([7] * 12, 2, 12), (12, 11))

    def test_suffix_includes_eos_and_trailing_newline(self):
        suffix = joint.exact_suffix(torch.tensor([[1, 4]]), torch.tensor([[1, 4, 7, 2, 9]]))
        self.assertEqual(suffix.tolist(), [[7, 2, 9]])
        with self.assertRaises(ValueError):
            joint.exact_suffix(torch.tensor([[1, 4]]), torch.tensor([[1, 5, 7]]))

    def test_resume_truncates_only_uncommitted_answer_tail(self):
        samples = [SimpleNamespace(sample_id="a"), SimpleNamespace(sample_id="b")]
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp:
            path = Path(temp) / "answers.jsonl"
            path.write_bytes(b'{"sample_id":"a"}\n{"sample_id":"b"}\n{"partial":')
            joint.recover_answers(path, samples, 1)
            self.assertEqual(path.read_bytes(), b'{"sample_id":"a"}\n')
            path.write_bytes(b'{"sample_id":"wrong"}\n')
            with self.assertRaises(ValueError):
                joint.recover_answers(path, samples, 1)

    def test_real_small_cpu_qwen_cache_branch_matches_full_forward(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(73)
        torch.set_num_threads(2)
        config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=36, num_attention_heads=2,
                             num_key_value_heads=1, head_dim=8, eos_token_id=2,
                             bos_token_id=1, pad_token_id=0)
        config._attn_implementation = "eager"
        inner = Qwen3ForCausalLM(config).eval()
        calls = []
        class Model:
            def set_attn_implementation(self, name):
                pass
            def generate(self, **kwargs):
                calls.append("image_prefill")
                kwargs.pop("pixel_values")
                return inner.generate(**kwargs)
            def __call__(self, **kwargs):
                calls.append("gold_suffix")
                self.last_suffix = kwargs["input_ids"].clone()
                return inner(**kwargs)
        class Processor:
            tokenizer = SimpleNamespace(eos_token_id=2)
            def apply_chat_template(self, messages, **kwargs):
                self.gold = messages[0][-1]["content"]
                return {"input_ids": torch.tensor([[1, 4, 5, 7, 2, 9]]),
                        "attention_mask": torch.ones(1, 6, dtype=torch.long),
                        "pixel_values": torch.zeros(1, 3, 2, 2)}
            def batch_decode(self, ids, **kwargs):
                return ["answer"]
        model, processor = Model(), Processor()
        inputs = {"input_ids": torch.tensor([[1, 4, 5]]), "attention_mask": torch.ones(1, 3, dtype=torch.long),
                  "pixel_values": torch.zeros(1, 3, 2, 2)}
        with patch("multimodal_model_adapter.prepare_batch", return_value=inputs), patch("multimodal_model_adapter.zero_shot_messages", return_value=[]), torch.inference_mode():
            sample = SimpleNamespace(label="bank label")
            result = joint.extract_one(model, processor, sample, "cub", ["bank label"], is_bank=True)
            full = inner(input_ids=torch.tensor([[1, 4, 5, 7, 2, 9]]), output_hidden_states=True, use_cache=False)
            np.testing.assert_allclose(result["gold"], full.hidden_states[-1][0,-1].numpy(), atol=1e-6, rtol=1e-5)
            prefix = inner(input_ids=inputs["input_ids"], output_hidden_states=True, use_cache=False)
            np.testing.assert_allclose(result["anchor"], torch.stack([h[0,-1] for h in prefix.hidden_states[1:]]).numpy(), atol=1e-6)
            self.assertEqual(calls, ["image_prefill", "gold_suffix"])
            self.assertEqual(model.last_suffix.tolist(), [[7, 2, 9]])
            calls.clear()
            # Object has no label: query gold cannot accidentally be read.
            query = joint.extract_one(model, processor, object(), "cub", ["bank label"], is_bank=False)
            self.assertIsNone(query["gold"])
            self.assertEqual(calls, ["image_prefill"])

    def test_array_commit_recovers_after_progress_publish_interruption(self):
        samples = [SimpleNamespace(sample_id="bank"), SimpleNamespace(sample_id="query")]
        binding = {"manifest_sha256": "fixed"}
        values = {"anchor": np.ones((36,2560)), "gpt": np.ones(2560), "gold": np.ones(2560),
                  "tail_ids": [7,2], "raw_zero_shot_answer": "x", "answer_token_count": 1}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temp, patch.object(joint, "extract_one", return_value=values):
            folder = Path(temp)
            original = joint.write_json
            def fail_progress(path, value):
                if Path(path).name == "anchor_states_progress.json" and value["completed"] == 1:
                    raise OSError("simulated interruption after authoritative commit")
                original(path, value)
            with patch.object(joint, "write_json", side_effect=fail_progress):
                with self.assertRaises(OSError):
                    joint.extract_dataset(None, None, samples[:1], samples[1:], "cub", ["x"], folder, binding)
            self.assertEqual(json.loads((folder / "joint_progress.json").read_text())["completed"], 1)
            joint.extract_dataset(None, None, samples[:1], samples[1:], "cub", ["x"], folder, binding)
            rows = [json.loads(line) for line in (folder / "gpt_mm_zero_shot_outputs.jsonl").read_text().splitlines()]
            self.assertEqual([r["sample_id"] for r in rows], ["bank", "query"])
            self.assertEqual(json.loads((folder / "anchor_states_progress.json").read_text())["completed"], 2)
            with self.assertRaises(ValueError):
                joint.extract_dataset(None, None, samples[:1], samples[1:], "cub", ["x"], folder, {"manifest_sha256":"changed"})

if __name__ == "__main__":
    unittest.main()

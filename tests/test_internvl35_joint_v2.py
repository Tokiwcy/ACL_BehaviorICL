"""CPU assertions for exact gold-forward fallback; no pretrained model/CUDA."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import internvl35_joint_source_v2 as joint

class ExactProxyTests(unittest.TestCase):
    def test_bank_uses_original_full_gold_kernel_and_keeps_suffix(self):
        batches = [{"input_ids": torch.tensor([[1,4]]), "pixel_values": torch.zeros(1,3,2,2)},
                   {"input_ids": torch.tensor([[1,4,7,2,9]]), "pixel_values": torch.zeros(1,3,2,2)}]
        processor = SimpleNamespace(apply_chat_template=lambda *a, **k: batches.pop(0))
        def original(model, proc, *a, **k):
            self.assertFalse(k['is_bank'])
            proc.apply_chat_template(None)
            return {"gold": None, "gold_suffix_ids": None}
        def proxy(model, proc, *a):
            proc.apply_chat_template(None)
            return torch.ones(2560)
        with patch.object(joint, 'original_extract_one', side_effect=original), patch('run_detriever_output_proxy.gold_answer_eos_state', side_effect=proxy) as full:
            value=joint.extract_one(None,processor,object(),'cub',['bank label'],is_bank=True)
        self.assertEqual(value['gold_suffix_ids'],[7,2,9])
        np.testing.assert_array_equal(value['gold'],np.ones(2560))
        full.assert_called_once()

    def test_query_never_calls_gold_kernel(self):
        with patch.object(joint,'original_extract_one',return_value={'gold':None}) as prefill, patch('run_detriever_output_proxy.gold_answer_eos_state') as full:
            value=joint.extract_one(None,None,object(),'cub',['bank label'],is_bank=False)
        full.assert_not_called()
        self.assertFalse(prefill.call_args.kwargs['is_bank'])
        self.assertIsNone(value['gold'])

if __name__=='__main__':
    unittest.main()

"""Exact shared generation prefill; original separate bank gold proxy forward."""
from __future__ import annotations
import numpy as np
import internvl35_joint_source_v1 as core
from internvl35_joint_config_v2 import write_json

original_extract_one = core.extract_one

class CaptureProcessor:
    def __init__(self, processor):
        self.processor = processor
        self.batches = []
    def __getattr__(self, key):
        return getattr(self.processor, key)
    def apply_chat_template(self, *args, **kwargs):
        batch = self.processor.apply_chat_template(*args, **kwargs)
        self.batches.append(batch)
        return batch

def extract_one(model, processor, sample, dataset, labels, *, is_bank):
    from run_detriever_output_proxy import gold_answer_eos_state
    import torch
    capture = CaptureProcessor(processor)
    value = original_extract_one(model, capture, sample, dataset, labels, is_bank=False)
    if is_bank:
        # The cached suffix optimization failed the original threshold. Use the
        # frozen full-forward kernel; do not lower the threshold or alter labels.
        value["gold"] = gold_answer_eos_state(model, capture, sample, dataset, labels, None).numpy()
        prefix, full = capture.batches
        suffix = core.exact_suffix(prefix["input_ids"], full["input_ids"])
        if not torch.equal(prefix["pixel_values"], full["pixel_values"]):
            raise ValueError("Bank prefix/full gold pixels changed")
        value["gold_suffix_ids"] = suffix[0].tolist()
        if not np.isfinite(value["gold"]).all():
            raise ValueError("Nonfinite original gold proxy")
    return value

def extract_dataset(*args, **kwargs):
    # Reuse the tested authoritative cursor implementation without changing its
    # frozen bytes. This process-local dispatch swaps only the per-row extractor.
    core.extract_one = extract_one
    core.write_json = write_json
    return core.extract_dataset(*args, **kwargs)

answer_state_index = core.answer_state_index

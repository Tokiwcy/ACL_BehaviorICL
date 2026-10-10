"""Operational per-query cache cleanup; call the frozen decoder unchanged."""
import gc
import hashlib
import json
from pathlib import Path
import torch
import multimodal_model_adapter as adapter

original_generate = adapter.generate_legal_labels

def guarded_generate(*args, **kwargs):
    value = original_generate(*args, **kwargs)
    torch.cuda.synchronize()
    before = torch.cuda.memory_reserved()
    gc.collect()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved()
    guarded_generate.last_memory = {"reserved_before":before,"reserved_after":after,
                                   "allocated_after":torch.cuda.memory_allocated()}
    return value

def install():
    adapter.generate_legal_labels = guarded_generate

def main():
    import sys
    import internvl35_joint_config_v2 as config
    config.require_gate()
    dataset = sys.argv[sys.argv.index("--dataset")+1]
    method = sys.argv[sys.argv.index("--method")+1]
    receipt = config.OUTPUT/dataset/"predictions"/f"{method}.cache_guard_identity.json"
    value = {"version":"cache_guard_v1","protocol_sha256":config.digest(config.PROTOCOL_PATH),
             "guard_code_sha256":config.digest(Path(__file__)),
             "decoder_code_sha256":config.digest(config.ROOT/"scripts/multimodal_model_adapter.py"),
             "policy":"call original legal decoder; after each returned prediction synchronize, collect unreachable objects and release unallocated CUDA cache; no prompt/model/vision/selection/precision change"}
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) != value:
        raise RuntimeError("Operational memory policy changed")
    config.write_json(receipt,value)
    install()
    import run_internvl35_joint_icl_v2 as runner
    runner.main()

if __name__=="__main__":
    main()

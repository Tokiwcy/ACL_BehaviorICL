"""Versioned extraction optimization; v4 scientific kernels remain frozen."""
from __future__ import annotations
import json
import os
from pathlib import Path
import time
import internvl35_qwen_config_v4 as legacy

ROOT = legacy.ROOT
DATASETS = ("cub", "dogs", "pets")
METHODS = legacy.METHODS
MODEL, REVISION = legacy.MODEL, legacy.REVISION
OUTPUT = ROOT / "results/internvl35_joint_v2"
ANALYSIS = ROOT / "analysis/internvl35_joint_v2"
PROTOCOL_PATH = ANALYSIS / "protocol.json"
digest, now, load_model = legacy.digest, legacy.now, legacy.load_model


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.2 * (attempt + 1))


def ensure_protocol():
    legacy.ensure_protocol()
    names = ("internvl35_joint_config_v2.py", "internvl35_joint_source_v2.py", "internvl35_joint_source_v1.py",
             "run_internvl35_joint_sources_v2.py", "run_internvl35_joint_methods_v2.py",
             "run_internvl35_joint_icl_v2.py", "audit_internvl35_joint_v2.py",
             "run_internvl35_joint_pipeline_v2.py", "check_internvl35_joint_v2.py")
    value = {"version": "joint_v2", "base_protocol_sha256": digest(legacy.PROTOCOL_PATH),
             "model": MODEL, "revision": REVISION, "datasets": list(DATASETS),
             "methods": list(METHODS), "seed": 73, "shots": 4,
             "policy": "one HF greedy generate prefill; 36 input anchors from step0; historical GPT-MM answer-state indexing; bank-only original full teacher-forced forward for gold proxy",
             "gold_backend": "original full cuDNN prefill; cached gold branch rejected by joint_v1 parity pilot, cosine0.9996466 below0.9999",
             "joint_commit": "one durable row cursor after all arrays and answer rows flush; dependent progress published after authoritative cursor",
             "gate": {"samples_per_dataset": "8 bank + 4 query, SHA256(dataset/sample_id) order",
                      "cosine_min": .9999, "generated_token_ids_exact": True,
                      "ordered_top4_exact": "Raw and GPT-MM over the fixed 8-bank pilot; bank proxy cosine top4 excluding self",
                      "not_full_retrieval_equivalence_proof": True},
             "unchanged": "native images, prompts, batch1 BF16, training, candidates, seed, 4shot and final legal decode from frozen v4",
             "code_sha256": {n: digest(ROOT / "scripts" / n) for n in names}}
    if PROTOCOL_PATH.exists():
        if json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")) != value:
            raise ValueError("Joint extraction binding changed; create a new version")
    else:
        write_json(PROTOCOL_PATH, value)
    return value


def require_gate():
    ensure_protocol()
    gate = json.loads((ANALYSIS / "compatibility.json").read_text(encoding="utf-8"))
    if gate.get("state") != "complete" or gate.get("protocol_sha256") != digest(PROTOCOL_PATH):
        raise RuntimeError("Joint GPU compatibility gate has not passed for this code binding")
    return gate

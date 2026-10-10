"""Pinned dynamic-image InternVL protocol matching Qwen's scientific settings."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = "OpenGVLab/InternVL3_5-4B-HF"
REVISION = "6bd4487402110ef9889ba50eb7aefeb302526fed"
DATASETS = ("aircraft", "cub", "dogs", "pets")
METHODS = ("raw", "proj", "learn", "gpt_mm", "detriever", "rices")
OUTPUT = ROOT / "results/internvl35_qwen_config_v4"
ANALYSIS = ROOT / "analysis/internvl35_qwen_config_v4"
PROTOCOL_PATH = ANALYSIS / "protocol.json"
SNAPSHOT = ROOT / ".hf_cache/models--OpenGVLab--InternVL3_5-4B-HF/snapshots" / REVISION


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def ensure_protocol():
    names = (
        "internvl35_qwen_config_v4.py", "run_internvl35_qwen_sources_v4.py",
        "run_internvl35_qwen_methods_v4.py", "run_internvl35_qwen_icl_v4.py",
        "run_internvl35_qwen_pipeline_v4.py", "audit_internvl35_qwen_config_v4.py", "internvl35_sdpa_runtime_v1.py",
        "multimodal_model_adapter.py", "legal_label_decoding.py", "multidataset_protocol.py",
        "run_multidataset_main.py", "run_dtd_decision_states.py",
        "run_behavior_ablation.py", "run_dtd_learned_velocity.py",
        "run_dtd_paper_baselines.py", "run_detriever_output_proxy.py",
    )
    receipt_path = ANALYSIS / "model_files.json"
    if not receipt_path.exists():
        raise RuntimeError("Run CPU prepare to bind local model file bytes first")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    # Stat identity provides a fast change detector; full SHA is checked during prepare.
    for name, item in receipt.items():
        st = (SNAPSHOT / name).stat()
        if st.st_size != item["size"] or st.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Pinned model file changed: {name}")
    import torch
    import transformers
    import numpy as np
    value = {
        "version": 4, "model": MODEL, "revision": REVISION,
        "datasets": list(DATASETS), "methods": list(METHODS), "seed": 73, "shots": 4,
        "official_full_bank_and_test": True, "layers": 36, "width": 2560,
        "vision": {"crop_to_patches": True, "max_patches": 12,
                   "patch_size": 448, "thumbnail": True, "tokens_per_patch": 256},
        "image_policy": "native dynamic processor, no global token cap; same policy as Qwen main vision_pixels=None",
        "source": "36 decoder states excluding embedding at final input position before answer under full task+image",
        "raw": "per-layer L2 cosine, uniform mean of all 36 layers",
        "proj": "shared bias-free 256 projection with uniform layers",
        "learn": "same shared 256 projection plus learned softmax layer weights",
        "behavior_training": {"bank_only": True, "dev_per_class": 10, "steps": 1500,
                              "lr": 0.0003, "weight_decay": 0.01, "temperature": 0.07,
                              "classes_per_batch": 16, "samples_per_class": 4, "eval_every": 100,
                              "checkpoint": "earliest best bank-dev top4_same_label_fraction"},
        "gpt_mm": "local GPT-MM-style greedy generated-answer state, max_new_tokens=12; not author code",
        "detriever": {"bank_gold_answer_eos_proxy": True, "steps": 10000, "batch": 64,
                      "positive_count": 40, "negative_count": 100, "memory_refresh": 100,
                      "lr": 0.0001, "weight_decay": 0.01, "betas": [0.9, 0.98],
                      "temperature": 0.07, "layer_grid": "zero-based [0,5,10,...,last]",
                      "implementation": "task adaptation, not author code"},
        "rices": "same CLIP ViT-B/32 cosine, rerun InternVL final ICL",
        "icl": "shared multimodal_model_adapter.icl_messages, lower similarity first, greedy legal-label trie",
        "test_labels": "evaluation only, never retrieval or training",
        "attention_backend": "cuDNN SDPA prefill (query length>1), math SDPA single-token decode; native images unchanged",
        "inference_batch": 1, "precision": "BF16, no quantization", "torch_threads": 4,
        "training_recovery": "incomplete retriever training restarts at seed73; completed metadata+checkpoint are reused only within this protocol",
        "environment": {"torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__},
        "model_files_sha256": digest(receipt_path),
        "code_sha256": {name: digest(ROOT / "scripts" / name) for name in names},
        "comparison_limit": "DTD excluded from rerun; cannot mix old single-patch DTD into a uniform dynamic five-dataset table",
    }
    if PROTOCOL_PATH.exists() and json.loads(PROTOCOL_PATH.read_text(encoding="utf-8")) != value:
        raise ValueError("Frozen v2 protocol/code/environment changed; create a new version")
    if not PROTOCOL_PATH.exists():
        write_json(PROTOCOL_PATH, value)
    return value


class RecordingProcessor:
    """Delegate processor behavior unchanged, recording actual image/input tokens."""
    def __init__(self, processor, telemetry):
        self._processor = processor
        self._telemetry = telemetry

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def apply_chat_template(self, messages, *args, **kwargs):
        batch = self._processor.apply_chat_template(messages, *args, **kwargs)
        if isinstance(batch, Mapping) and "input_ids" in batch and self._telemetry is not None:
            ids = batch["input_ids"]
            image_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
            paths = []
            for chat in messages:
                image_paths = []
                for turn in chat:
                    content = turn.get("content", [])
                    if isinstance(content, list):
                        image_paths.extend(part.get("image", part.get("url")) for part in content if part.get("type") == "image")
                paths.append(image_paths)
            row = {"utc": now(), "pid": os.getpid(), "images": paths,
                   "input_tokens": list(batch["attention_mask"].sum(-1).tolist()),
                   "visual_tokens": list((ids == image_id).sum(-1).tolist()),
                   "pixel_values_shape": list(batch["pixel_values"].shape),
                   "input_ids_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest()}
            if sum(row["visual_tokens"]) != row["pixel_values_shape"][0] * 256:
                raise ValueError("Image token count differs from actual patch tensor")
            self._telemetry.parent.mkdir(parents=True, exist_ok=True)
            with self._telemetry.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        return batch


def load_model(telemetry=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    torch.set_num_threads(4)
    torch.manual_seed(73)
    from internvl35_sdpa_runtime_v1 import register_backend, NAME
    register_backend()
    processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
    if processor.image_processor.max_patches != 12 or processor.image_seq_length != 256:
        raise ValueError("Unexpected native InternVL image defaults")
    # Processor's native images_kwargs defaults enable dynamic patches.
    from transformers.models.internvl.processing_internvl import InternVLProcessorKwargs
    if not InternVLProcessorKwargs._defaults["images_kwargs"]["crop_to_patches"]:
        raise ValueError("Dynamic patch processing is disabled")
    model = AutoModelForImageTextToText.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation=NAME,
    ).eval().to("cuda")
    if (model.config.text_config.num_hidden_layers, model.config.text_config.hidden_size) != (36, 2560):
        raise ValueError("Unexpected text model dimensions")
    return model, RecordingProcessor(processor, telemetry)

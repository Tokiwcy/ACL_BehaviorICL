"""Minimal pinned LFM GPU check under the Qwen main experiment prompts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from multidataset_protocol import load_dataset
from multimodal_model_adapter import (
    anchor_hidden_states, generate_legal_labels, icl_messages,
    model_dimensions, prepare_batch, zero_shot_messages,
)

REVISION = "35a118d938ce6d123ac2d371649f24a8efb69058"
SNAPSHOT = ROOT / ".hf_cache/models--LiquidAI--LFM2.5-VL-3B/snapshots" / REVISION
OUTPUT = ROOT / "analysis/lfm25_vl3b_full_v1/gpu_smoke.json"


def save(value):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    value["checked_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = OUTPUT.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, OUTPUT)


def main() -> None:
    torch.set_num_threads(4)
    result = {"model": "LiquidAI/LFM2.5-VL-3B", "revision": REVISION,
              "pid": os.getpid(), "state": "loading", "cases": {}}
    save(result)
    if not SNAPSHOT.is_dir() or not torch.cuda.is_available():
        raise RuntimeError("Pinned local snapshot and CUDA are required")
    bank, query = load_dataset("aircraft", ROOT / "datasets")
    labels = sorted({row.label for row in bank})
    processor = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True)
    image_processor = processor.image_processor
    image_config = {
        key: getattr(image_processor, key, None)
        for key in ("max_tiles", "max_image_tokens", "do_image_splitting", "tile_size")
    }
    print(f"image_config={image_config}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval().to("cuda")
    layers, width = model_dimensions(model)
    if (layers, width) != (30, 2048):
        raise RuntimeError(f"Unexpected LFM text dimensions: {(layers, width)}")
    result.update(dtype="bfloat16", image_config=image_config,
                  layers=layers, width=width)
    save(result)
    for name, messages in (
        ("one_image_source", zero_shot_messages(bank[0], "aircraft", labels, None)),
        ("five_image_icl", icl_messages(bank[:4], query[0], "aircraft", labels, None)),
    ):
        result.update(state="running", stage=name)
        save(result)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inputs = prepare_batch(processor, [messages])
        entry = {"input_tokens": int(inputs["input_ids"].shape[1]),
                 "visual_tokens": int((inputs["input_ids"] == model.config.image_token_id).sum()),
                 "pixel_values_shape": list(inputs["pixel_values"].shape),
                 "peak_reserved_after_prepare_gb": torch.cuda.max_memory_reserved() / 2**30}
        if name == "one_image_source":
            states = anchor_hidden_states(model, inputs)
            entry["states_shape"] = list(states.shape)
            entry["states_finite"] = bool(torch.isfinite(states).all())
            del states
        else:
            del inputs
            prediction, raw = generate_legal_labels(model, processor, [messages], labels)[0]
            entry["prediction"] = prediction
            entry["raw_output"] = raw
            entry["prediction_legal"] = prediction in labels
        entry["peak_reserved_gb"] = torch.cuda.max_memory_reserved() / 2**30
        print(f"{name}: {entry}", flush=True)
        result["cases"][name] = entry
        if name == "one_image_source":
            del inputs
        save(result)
    result["state"] = "complete"
    save(result)
    print(f"wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()

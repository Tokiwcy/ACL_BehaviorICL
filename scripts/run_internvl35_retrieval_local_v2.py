"""Pinned InternVL3.5-4B HF replay of the fixed five-dataset prompt study.

The base bank/query rows and prompt strings are reused byte-for-byte from the
Qwen study. This runner stores only pre-answer decoder states; it does not
perform ICL generation or train a retriever.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from query_tam_local_plan_v1 import digest

MODEL_ID = "OpenGVLab/InternVL3_5-4B-HF"
REVISION = "6bd4487402110ef9889ba50eb7aefeb302526fed"
DATASETS = ("dtd", "aircraft", "cub", "dogs", "pets")
ARMS = ("generic", "task_short", "task_full", "label_list_only")
PLAN_PATH = ROOT / "analysis/internvl35_retrieval_local_v2/protocol.json"


def build_plan() -> dict:
    base_path = ROOT / "analysis/query_retrieval_local_v1/protocol.json"
    follow_path = ROOT / "analysis/query_retrieval_local_v2/protocol.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    follow = json.loads(follow_path.read_text(encoding="utf-8"))
    if follow["base_protocol_SHA256"] != digest(base_path):
        raise ValueError("Qwen prompt protocol chain changed")
    datasets = {}
    for name in DATASETS:
        first, second = base["datasets"][name], follow["datasets"][name]
        if first["bank"] != second["bank"] or first["query"] != second["query"]:
            raise ValueError(f"Frozen bank/query mismatch for {name}")
        datasets[name] = {
            "bank": first["bank"], "query": first["query"],
            "labels": first["labels"],
            "prompts": {"generic": "Describe the image briefly.",
                        "task_short": first["prompts"]["task_short"] if "prompts" in first else
                        f"Classify this image by its {dict(dtd='texture category', aircraft='aircraft variant', cub='bird species', dogs='dog breed', pets='cat or dog breed')[name]}. Give the most specific label.",
                        **second["prompts"]},
        }
    return {
        "version": 2, "model": MODEL_ID, "revision": REVISION,
        "image_max_patches": 1, "image_patch_size": 448,
        "base_protocol_SHA256": digest(base_path),
        "follow_protocol_SHA256": digest(follow_path),
        "datasets": datasets, "arms": list(ARMS),
        "features": "36 pre-answer last-input decoder states, float16 storage, no generation",
        "metrics": ["equal-layer cosine Raw top1", "Raw top4 any same label", "same-label AP"],
        "scope": "Two training images per class and 32 test images per dataset; retrieval diagnostic, not full-bank ICL accuracy.",
    }


def ensure_plan() -> dict:
    plan = build_plan()
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PLAN_PATH.exists() and json.loads(PLAN_PATH.read_text(encoding="utf-8")) != plan:
        raise ValueError("InternVL experiment plan changed after freezing")
    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    plan = ensure_plan()
    if args.plan_only:
        print(f"Frozen {len(plan['datasets'])} datasets and {len(ARMS)} arms at {PLAN_PATH}")
        return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_model_adapter import anchor_hidden_states, model_dimensions, prepare_batch

    snapshot = ROOT / ".hf_cache/models--OpenGVLab--InternVL3_5-4B-HF/snapshots" / REVISION
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    processor.image_processor.max_patches = 1
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    ).eval().to("cuda")
    layers, width = model_dimensions(model)
    if (layers, width) != (36, 2560):
        raise RuntimeError(f"Unexpected InternVL state shape {(layers, width)}")

    for dataset in ([args.dataset] if args.dataset else DATASETS):
        table = plan["datasets"][dataset]
        rows = table["bank"] + table["query"]
        output = ROOT / "results/internvl35_retrieval_local_v2" / dataset
        output.mkdir(parents=True, exist_ok=True)
        for arm in (args.arms or ARMS):
            path = output / f"{arm}_states.npy"
            progress = output / f"{arm}_progress.json"
            shape = (len(rows), layers, width)
            identity = {"shape": list(shape), "protocol_SHA256": digest(PLAN_PATH)}
            if path.exists() and progress.exists():
                state = json.loads(progress.read_text(encoding="utf-8"))
                if any(state.get(key) != value for key, value in identity.items()):
                    raise ValueError(f"Resume identity differs for {dataset}/{arm}")
                values = np.lib.format.open_memmap(path, mode="r+", dtype=np.float16, shape=shape)
                start = int(state["completed"])
            else:
                values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
                start = 0
            for offset in range(start, len(rows), args.batch_size):
                batch = rows[offset:offset + args.batch_size]
                messages = []
                for row in batch:
                    image = ROOT / row["path"]
                    if digest(image) != row["image_SHA256"]:
                        raise ValueError(f"Image changed: {image}")
                    messages.append([{"role": "user", "content": [
                        {"type": "image", "image": str(image)},
                        {"type": "text", "text": table["prompts"][arm]},
                    ]}])
                inputs = prepare_batch(processor, messages)
                with torch.inference_mode():
                    anchor = anchor_hidden_states(model, inputs)
                values[offset:offset + len(batch)] = anchor.float().cpu().numpy().astype(np.float16)
                values.flush()
                completed = offset + len(batch)
                progress.write_text(json.dumps({**identity, "completed": completed}), encoding="utf-8")
                del inputs, anchor
                if completed % 20 == 0 or completed == len(rows):
                    torch.cuda.empty_cache()
                    print(f"{dataset}/{arm} {completed}/{len(rows)}", flush=True)
            del values


if __name__ == "__main__":
    main()


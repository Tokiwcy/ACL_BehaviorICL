"""Extract pinned Qwen pre-answer states for a small, frozen retrieval ablation."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from query_tam_local_plan_v1 import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dtd", "aircraft", "cub", "dogs", "pets"], required=True)
    parser.add_argument("--protocol", type=Path, default=Path("analysis/query_retrieval_local_v1/protocol.json"))
    parser.add_argument("--arms", nargs="+")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_model_adapter import prepare_batch, anchor_hidden_states
    plan_path = ROOT / args.protocol
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    table = plan["datasets"][args.dataset]
    rows = table["bank"] + table["query"]
    snapshot = ROOT / ".hf_cache/models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / plan["revision"]
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(snapshot, dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True).eval().to("cuda")
    output = ROOT / "results/query_retrieval_local_v1" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    available_arms = plan.get("arms", plan.get("new_arms"))
    arms = args.arms or available_arms
    if any(arm not in available_arms for arm in arms):
        raise ValueError("Requested arm is not in frozen protocol")
    for arm in arms:
        path = output / f"{arm}_states.npy"
        progress = output / f"{arm}_progress.json"
        shape = (len(rows), 36, 2560)
        if path.exists() and progress.exists():
            state = json.loads(progress.read_text(encoding="utf-8"))
            if state["shape"] != list(shape) or state["protocol_SHA256"] != digest(plan_path):
                raise ValueError("Resume identity differs")
            values = np.lib.format.open_memmap(path, mode="r+", dtype=np.float16, shape=shape)
            start = state["completed"]
        else:
            values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
            start = 0
        for i in range(start, len(rows)):
            row = rows[i]
            image = ROOT / row["path"]
            if digest(image) != row["image_SHA256"]:
                raise ValueError(f"Image changed: {image}")
            query = (table["prompts"][arm] if "prompts" in table else
                "Describe the image briefly." if arm == "generic" else
                f"Classify this image by its {dict(dtd='texture category', aircraft='aircraft variant', cub='bird species', dogs='dog breed', pets='cat or dog breed')[args.dataset]}. Give the most specific label.")
            messages = [[{"role": "user", "content": [
                {"type": "image", "image": str(image), "resized_height": 224, "resized_width": 224},
                {"type": "text", "text": query}]}]]
            inputs = prepare_batch(processor, messages)
            with torch.inference_mode():
                anchor = anchor_hidden_states(model, inputs)
            values[i] = anchor[0].float().cpu().numpy().astype(np.float16)
            values.flush()
            progress.write_text(json.dumps({"shape": list(shape), "completed": i + 1,
                "protocol_SHA256": digest(plan_path)}), encoding="utf-8")
            del inputs, anchor
            if (i + 1) % 20 == 0 or i + 1 == len(rows):
                torch.cuda.empty_cache()
                print(f"{args.dataset}/{arm} {i + 1}/{len(rows)}", flush=True)
        del values


if __name__ == "__main__":
    main()


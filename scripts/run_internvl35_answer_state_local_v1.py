"""Small-bank InternVL GPT-MM-style generated-answer-state comparator.

Uses the same fixed samples, full-task prompt, and one-patch image budget as
the pre-answer study. It follows the repository's local GPT-MM-style proxy:
greedy zero-shot generation followed by a final-layer answer-position state.
This is not the GPT-MM authors' official implementation.
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

REVISION = "6bd4487402110ef9889ba50eb7aefeb302526fed"
DATASETS = ("dtd", "aircraft", "cub", "dogs", "pets")
PLAN_PATH = ROOT / "analysis/internvl35_answer_state_local_v1/protocol.json"


def ensure_plan() -> dict:
    base_path = ROOT / "analysis/internvl35_retrieval_local_v2/protocol.json"
    base = json.loads(base_path.read_text(encoding="utf-8"))
    plan = {"version": 1, "base_protocol_SHA256": digest(base_path),
            "model": base["model"], "revision": REVISION,
            "image_max_patches": 1, "image_patch_size": 448,
            "prompt_arm": "task_full", "max_new_tokens": 12,
            "representation": "local GPT-MM-style proxy: final decoder layer at generated answer position (or 12-token cap)",
            "scope": "Same two-per-class bank and 32 test queries per dataset; generated-answer-state retrieval control",
            "datasets": {name: {"bank": block["bank"], "query": block["query"],
                                 "prompt": block["prompts"]["task_full"]}
                         for name, block in base["datasets"].items()}}
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PLAN_PATH.exists() and json.loads(PLAN_PATH.read_text(encoding="utf-8")) != plan:
        raise ValueError("Answer-state protocol changed")
    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and (args.dataset is None or args.limit < 1):
        raise ValueError("positive --limit requires --dataset")
    plan = ensure_plan()
    if args.plan_only:
        print(f"Frozen answer-state control at {PLAN_PATH}")
        return

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_model_adapter import prepare_batch

    snapshot = ROOT / ".hf_cache/models--OpenGVLab--InternVL3_5-4B-HF/snapshots" / REVISION
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    processor.image_processor.max_patches = 1
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    ).eval().to("cuda")
    for dataset in ([args.dataset] if args.dataset else DATASETS):
        block = plan["datasets"][dataset]
        rows = block["bank"] + block["query"]
        output = ROOT / "results/internvl35_answer_state_local_v1" / dataset
        output.mkdir(parents=True, exist_ok=True)
        state_path = output / "answer_states.npy"
        progress_path = output / "progress.json"
        answers_path = output / "answers.jsonl"
        identity = {"shape": [len(rows), 2560], "protocol_SHA256": digest(PLAN_PATH)}
        if state_path.exists() and progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            if any(progress.get(k) != v for k, v in identity.items()):
                raise ValueError(f"Resume identity mismatch: {dataset}")
            states = np.lib.format.open_memmap(state_path, mode="r+", dtype=np.float16, shape=tuple(identity["shape"]))
            start = int(progress["completed"])
        else:
            states = np.lib.format.open_memmap(state_path, mode="w+", dtype=np.float16, shape=tuple(identity["shape"]))
            start = 0
        stop = min(len(rows), args.limit) if args.limit is not None else len(rows)
        with answers_path.open("a", encoding="utf-8") as handle:
            for i in range(start, stop):
                row = rows[i]
                image = ROOT / row["path"]
                if digest(image) != row["image_SHA256"]:
                    raise ValueError(f"Image changed: {image}")
                messages = [[{"role": "user", "content": [
                    {"type": "image", "image": str(image)},
                    {"type": "text", "text": block["prompt"]},
                ]}]]
                inputs = prepare_batch(processor, messages)
                prompt_len = int(inputs["input_ids"].shape[1])
                with torch.inference_mode():
                    generation = model.generate(
                        **inputs, max_new_tokens=plan["max_new_tokens"], do_sample=False,
                        use_cache=True, return_dict_in_generate=True, output_hidden_states=True,
                    )
                tail = generation.sequences[0, prompt_len:]
                eos = processor.tokenizer.eos_token_id
                positions = (tail == eos).nonzero(as_tuple=False).flatten()
                length = max(1, int(positions[0]) if len(positions) else len(tail))
                hidden_steps = generation.hidden_states
                if hidden_steps is None:
                    raise RuntimeError("Generated hidden states unavailable")
                vector = hidden_steps[min(length, len(hidden_steps) - 1)][-1][0, -1]
                states[i] = vector.float().cpu().numpy().astype(np.float16)
                states.flush()
                raw = processor.batch_decode(tail[None], skip_special_tokens=True)[0].strip()
                handle.write(json.dumps({"sample_id": row["sample_id"], "raw_answer": raw,
                                         "answer_token_count": length}, ensure_ascii=False) + "\n")
                handle.flush()
                progress_path.write_text(json.dumps({**identity, "completed": i + 1}), encoding="utf-8")
                del inputs, generation, hidden_steps, vector, tail
                if (i + 1) % 20 == 0 or i + 1 == stop:
                    torch.cuda.empty_cache()
                    print(f"{dataset}/answer_state {i + 1}/{len(rows)}", flush=True)
        del states


if __name__ == "__main__":
    main()

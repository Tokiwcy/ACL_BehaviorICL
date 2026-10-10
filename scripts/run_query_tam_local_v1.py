"""Resumable single-GPU local Qwen/TAM study; first answer token only."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from query_tam_local_plan_v1 import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["dtd", "aircraft", "cub", "dogs", "pets"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--protocol", type=Path, default=Path("analysis/query_tam_local_v1/protocol.json"))
    parser.add_argument("--arms", nargs="+", choices=["generic", "task_short", "task_full"], default=["generic", "task_short"])
    parser.add_argument("--mode", choices=["fixed", "natural"], default="fixed")
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from scipy.optimize import minimize_scalar
    from transformers import AutoModelForImageTextToText, AutoProcessor
    import query_tam_adapter_v1 as adapter

    if not torch.cuda.is_available():
        raise RuntimeError("Local CUDA GPU unavailable")
    plan_path = ROOT / args.protocol
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    examples = [x for x in plan["examples"] if args.dataset is None or x["dataset"] == args.dataset]
    if args.limit is not None:
        if args.dataset is None:
            raise ValueError("--limit requires --dataset")
        examples = examples[:args.limit]
    snapshot = ROOT / ".hf_cache/models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / plan["revision"]
    if not snapshot.is_dir() or digest(ROOT / ".tools/tam_official_v1/tam.py") != plan["TAM_SHA256"]:
        raise ValueError("Pinned local model/TAM source missing or changed")
    out = ROOT / "results/query_tam_local_v1"
    out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True
    ).eval().to("cuda")
    if (model.config.text_config.num_hidden_layers, model.config.text_config.hidden_size) != (36, 2560):
        raise ValueError("Unexpected Qwen dimensions")
    tam, receipt = adapter.load(ROOT / ".tools/tam_official_v1/tam.py", torch, np, minimize_scalar)
    (out / "adapter_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    img_id = int(model.config.image_token_id)
    vision_end = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    footer = processor.tokenizer.encode("<|im_end|>\n<|im_start|>assistant", add_special_tokens=False)
    answer_start = processor.tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    for example in examples:
        path = ROOT / example["image_path"]
        if digest(path) != example["image_SHA256"]:
            raise ValueError(f"Image changed: {path}")
        image = Image.open(path).convert("RGB")
        fixed = processor.tokenizer.encode(example["gold_label"], add_special_tokens=False)[0]
        previous = None
        for arm in args.arms:
            key = f"{example['dataset']}__{example['sample_id']}__{arm}__{args.mode}"
            folder = out / key
            if (folder / "record.json").exists():
                print(f"skip {key}", flush=True)
                continue
            query = example["queries"][arm]
            message = [{"role": "user", "content": [
                {"type": "image", "image": str(path), "resized_height": 224, "resized_width": 224},
                {"type": "text", "text": query},
            ]}]
            padding = processor.tokenizer.padding_side
            try:
                processor.tokenizer.padding_side = "left"
                inputs = processor.apply_chat_template([message], tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt", processor_kwargs={"padding": True})
            finally:
                processor.tokenizer.padding_side = padding
            grid = inputs["image_grid_thw"][0].tolist()
            merge = int(model.config.vision_config.spatial_merge_size)
            shape = (grid[1] // merge, grid[2] // merge)
            if grid[0] != 1 or grid[1] % merge or grid[2] % merge or int((inputs["input_ids"] == img_id).sum()) != shape[0] * shape[1]:
                raise ValueError("Image grid and tokens disagree")
            pixels_sha = hashlib.sha256(inputs["pixel_values"].contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()
            if previous is not None and previous != (pixels_sha, grid):
                raise ValueError("Query changed image pixels/grid")
            previous = (pixels_sha, grid)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            L = inputs["input_ids"].shape[1]
            base = inputs["input_ids"][0].tolist()
            if base[-len(answer_start):] != answer_start:
                raise ValueError("Assistant prefix differs")
            special = {"img_id": [img_id], "prompt_id": [vision_end, footer], "answer_id": [answer_start, -1]}
            generation = {"max_new_tokens": 1}
            if args.mode == "fixed":
                generation["prefix_allowed_tokens_fn"] = lambda _batch, _ids: [fixed]
            print(f"start {key} L={L} grid={shape}", flush=True)
            start = time.time()
            with torch.inference_mode():
                output = model.generate(**inputs, do_sample=False, use_cache=True, output_hidden_states=True,
                    return_dict_in_generate=True, **generation)
                tokens = output.sequences[0].tolist()
                if args.mode == "fixed" and tokens[L] != fixed:
                    raise ValueError("Forced first gold token differs")
                states = output.hidden_states[0]
                if len(states) != 37 or states[-1].shape[1] != L:
                    raise ValueError("Unexpected prefill states")
                anchor = torch.stack([x[0, -1].float().cpu() for x in states[1:]]).numpy()
                logits = [model.lm_head(states[-1]).detach().cpu()]
                del output, states
                rows = adapter.maps(tam, tokens, shape, logits, special, image, processor, 1)
            folder.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(folder / "maps.npz", maps=np.stack([r["map"] for r in rows]),
                raw_CAM=np.stack([r["raw_CAM"] for r in rows]),
                post_ECI=np.stack([r["post_ECI_before_filter"] for r in rows]), anchor_states=anchor)
            record = {"key": key, "dataset": example["dataset"], "sample_id": example["sample_id"],
                "arm": arm, "mode": args.mode, "query": query, "target_token_id": tokens[L],
                "target_token": processor.tokenizer.convert_ids_to_tokens(tokens[L]),
                "gold_label": example["gold_label"], "image_SHA256": example["image_SHA256"],
                "processed_pixels_SHA256": pixels_sha, "grid": grid, "map_shape": shape,
                "input_length": L, "seconds": time.time() - start,
                "maps_SHA256": digest(folder / "maps.npz"),
                "fixed_is_teacher_forced": args.mode == "fixed", "first_answer_token_only": True}
            (folder / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"done {key} {record['seconds']:.1f}s", flush=True)
            del inputs, logits, rows, anchor
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()


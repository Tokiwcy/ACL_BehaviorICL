"""Four-shot ICL using the generated-answer-state retrieval baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_query_retrieval_local_v1 import normalize
from multidataset_protocol import Sample
from query_tam_local_plan_v1 import digest

REVISION = "6bd4487402110ef9889ba50eb7aefeb302526fed"


def sample(row: dict, split: str) -> Sample:
    return Sample(row["sample_id"], str(ROOT / row["path"]), row["label"], split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dtd", "aircraft", "cub", "dogs", "pets"), required=True)
    parser.add_argument("--max-queries", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.max_queries <= 32:
        raise ValueError("max-queries must be 1..32")
    answer_path = ROOT / "analysis/internvl35_answer_state_local_v1/protocol.json"
    answer_plan = json.loads(answer_path.read_text(encoding="utf-8"))
    base_path = ROOT / "analysis/internvl35_retrieval_local_v2/protocol.json"
    if answer_plan["base_protocol_SHA256"] != digest(base_path):
        raise ValueError("Answer-state base protocol changed")
    table = json.loads(base_path.read_text(encoding="utf-8"))["datasets"][args.dataset]
    bank, query = ([sample(row, "bank") for row in table["bank"]],
                   [sample(row, "query") for row in table["query"]])
    labels = table["labels"]
    folder = ROOT / "results/internvl35_answer_state_local_v1" / args.dataset
    progress = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
    if progress["protocol_SHA256"] != digest(answer_path) or progress["completed"] != len(bank) + len(query):
        raise ValueError("Incomplete generated-answer states")
    states = normalize(np.asarray(np.load(folder / "answer_states.npy", mmap_mode="r"), dtype=np.float32))
    scores = states[len(bank):] @ states[:len(bank)].T
    demo_indices = np.argsort(scores, axis=1, kind="stable")[:, -4:]

    outdir = ROOT / "results/internvl35_answer_state_icl_local_v1" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    identity_path = outdir / "identity.json"
    identity = {"model": answer_plan["model"], "revision": REVISION,
                "answer_state_protocol_SHA256": digest(answer_path),
                "base_protocol_SHA256": digest(base_path),
                "ICL_prompt": "repository multimodal_model_adapter.icl_messages",
                "shots": 4, "image_max_patches": 1, "bank_n": len(bank), "query_n": len(query)}
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("ICL resume identity mismatch")
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    predictions_path = outdir / "predictions.jsonl"
    completed = {}
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["query_id"] in completed:
                raise ValueError("Duplicate ICL query")
            completed[row["query_id"]] = row

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_model_adapter import generate_legal_labels, icl_messages

    snapshot = ROOT / ".hf_cache/models--OpenGVLab--InternVL3_5-4B-HF/snapshots" / REVISION
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    processor.image_processor.max_patches = 1
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    ).eval().to("cuda")
    with predictions_path.open("a", encoding="utf-8") as handle:
        for i, target in enumerate(query[:args.max_queries]):
            if target.sample_id in completed:
                continue
            demos = [bank[j] for j in demo_indices[i]]
            prediction, raw = generate_legal_labels(
                model, processor, [icl_messages(demos, target, args.dataset, labels, None)], labels
            )[0]
            row = {"query_id": target.sample_id, "target": target.label,
                   "prediction": prediction, "correct": prediction == target.label,
                   "raw_output": raw, "demo_ids": [demo.sample_id for demo in demos],
                   "demo_labels": [demo.label for demo in demos]}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            completed[target.sample_id] = row
            if (i + 1) % 8 == 0 or i + 1 == args.max_queries:
                torch.cuda.empty_cache()
                print(f"{args.dataset}/answer_state_icl {i + 1}/{args.max_queries}", flush=True)
    if args.max_queries == len(query):
        count = sum(completed[target.sample_id]["correct"] for target in query)
        (outdir / "summary.json").write_text(json.dumps({"correct": count, "total": len(query),
                                                           "accuracy": count / len(query)}, indent=2), encoding="utf-8")
        print(f"{args.dataset}: {count}/{len(query)}")


if __name__ == "__main__":
    main()

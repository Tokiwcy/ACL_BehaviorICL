"""Matched four-shot ICL for generic/full InternVL retrieval on the small bank.

The ICL prompt is held fixed. Only the four demonstration images selected from
generic versus task-full retrieval states differ. This is a 32-query diagnostic
per dataset and is not the historical full-bank main experiment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_query_retrieval_original_raw_v1 import raw_scores
from multidataset_protocol import Sample
from query_tam_local_plan_v1 import digest

REVISION = "6bd4487402110ef9889ba50eb7aefeb302526fed"
ARMS = ("generic", "task_full")


def selections(table: dict, dataset: str, plan_sha: str) -> dict:
    source = ROOT / "results/internvl35_retrieval_local_v1" / dataset
    bank_n = len(table["bank"])
    output = {}
    for arm in ARMS:
        progress = json.loads((source / f"{arm}_progress.json").read_text(encoding="utf-8"))
        if progress["protocol_SHA256"] != plan_sha or progress["completed"] != bank_n + len(table["query"]):
            raise ValueError(f"Incomplete retrieval states for {dataset}/{arm}")
        states = np.load(source / f"{arm}_states.npy", mmap_mode="r")
        scores = raw_scores(states, bank_n)
        # Same four best matches and the main runner's ascending demo order.
        ranked = np.argsort(scores, axis=1, kind="stable")[:, -4:]
        output[arm] = ranked.tolist()
    return output


def sample(row: dict, split: str) -> Sample:
    return Sample(row["sample_id"], str(ROOT / row["path"]), row["label"], split)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dtd", "aircraft", "cub", "dogs", "pets"), required=True)
    parser.add_argument("--max-queries", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.max_queries <= 32:
        raise ValueError("max-queries must be 1..32")
    plan_path = ROOT / "analysis/internvl35_retrieval_local_v1/protocol.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    table = plan["datasets"][args.dataset]
    indices = selections(table, args.dataset, digest(plan_path))
    bank = [sample(row, "bank") for row in table["bank"]]
    query = [sample(row, "query") for row in table["query"]]
    labels = table["labels"]
    if sorted({item.label for item in bank}) != sorted(labels):
        raise ValueError("Class vocabulary mismatch")

    outdir = ROOT / "results/internvl35_icl_local_v1" / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "predictions.jsonl"
    identity_path = outdir / "identity.json"
    identity = {"model": plan["model"], "revision": REVISION,
                "retrieval_protocol_SHA256": digest(plan_path),
                "ICL_prompt": "repository multimodal_model_adapter.icl_messages",
                "shots": 4, "bank_n": len(bank), "query_n": len(query)}
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("ICL resume identity mismatch")
    identity_path.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    completed = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            key = (row["arm"], row["query_id"])
            if key in completed:
                raise ValueError(f"Duplicate ICL result {key}")
            completed[key] = row

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from multimodal_model_adapter import generate_legal_labels, icl_messages

    snapshot = ROOT / ".hf_cache/models--OpenGVLab--InternVL3_5-4B-HF/snapshots" / REVISION
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    ).eval().to("cuda")
    with path.open("a", encoding="utf-8") as handle:
        for arm in ARMS:
            for i, target in enumerate(query[:args.max_queries]):
                key = (arm, target.sample_id)
                if key in completed:
                    continue
                demo_ids = indices[arm][i]
                demos = [bank[j] for j in demo_ids]
                messages = [icl_messages(demos, target, args.dataset, labels, None)]
                prediction, raw = generate_legal_labels(model, processor, messages, labels)[0]
                row = {"arm": arm, "query_id": target.sample_id, "target": target.label,
                       "prediction": prediction, "correct": prediction == target.label,
                       "raw_output": raw, "demo_ids": [demo.sample_id for demo in demos],
                       "demo_labels": [demo.label for demo in demos]}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed[key] = row
                if (i + 1) % 8 == 0 or i + 1 == args.max_queries:
                    torch.cuda.empty_cache()
                    print(f"{args.dataset}/{arm} {i + 1}/{args.max_queries}", flush=True)

    if args.max_queries == len(query):
        summary = {}
        for arm in ARMS:
            rows = [completed[(arm, target.sample_id)] for target in query]
            summary[arm] = {"correct": sum(row["correct"] for row in rows),
                            "total": len(rows),
                            "accuracy": sum(row["correct"] for row in rows) / len(rows)}
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

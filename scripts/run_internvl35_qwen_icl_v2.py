"""Resumable full official-test 4-shot InternVL3.5-4B generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_internvl35_qwen_methods_v2 import (
    DATASETS, METHODS, PROTOCOL_PATH, ensure_protocol, load_context,
)
from internvl35_qwen_config_v2 import digest, load_model, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size != 1:
        raise ValueError("Pinned v2 ICL runs one query per batch")
    if (args.limit is not None and args.limit < 1) or args.batch_size < 1:
        raise ValueError("--limit and --batch-size must be positive")
    ensure_protocol()
    folder, bank, query, identity = load_context(args.dataset)
    labels = sorted({x.label for x in bank})
    selection_path = folder / "selections" / f"{args.method}.json"
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    if list(selections) != [q.sample_id for q in query]:
        raise ValueError("Selections do not cover official test in order")
    for indices in selections.values():
        if len(indices) != 4 or len(set(indices)) != 4 or min(indices) < 0 or max(indices) >= len(bank):
            raise ValueError("Invalid demonstration indices")
    outdir = folder / "predictions"
    outdir.mkdir(exist_ok=True)
    result_path = outdir / f"{args.method}.jsonl"
    identity_path = outdir / f"{args.method}.identity.json"
    prediction_identity = {
        "version": 2, "dataset": args.dataset, "method": args.method,
        "full_protocol_sha256": digest(PROTOCOL_PATH),
        "source_manifest_sha256": identity["manifest_sha256"],
        "selections_sha256": digest(selection_path),
        "shots": 4, "image_max_patches": 12, "crop_to_patches": True, "generation_batch_size": args.batch_size,
        "icl_prompt": "multimodal_model_adapter.icl_messages",
        "decoder": "multimodal_model_adapter.generate_legal_labels",
        "query_n": len(query),
    }
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != prediction_identity:
        raise ValueError("Prediction resume identity changed")
    write_json(identity_path, prediction_identity)
    completed = []
    if result_path.exists():
        with result_path.open("rb") as handle:
            for line in handle:
                if not line.endswith(b"\n"):
                    raise ValueError("Truncated final prediction row")
                completed.append(json.loads(line))
        if len(completed) > len(query):
            raise ValueError("Prediction file exceeds official test")
        for i, row in enumerate(completed):
            if (row["query_id"], row["target"], row["method"]) != (
                query[i].sample_id, query[i].label, args.method
            ):
                raise ValueError("Prediction prefix/order differs")
            demos = [bank[j] for j in selections[query[i].sample_id]]
            if (row.get("demo_ids") != [s.sample_id for s in demos]
                    or row.get("demo_labels") != [s.label for s in demos]
                    or row.get("correct") != (row.get("prediction") == query[i].label)):
                raise ValueError("Prediction resume demos/correctness differs")
    stop = min(len(query), args.limit) if args.limit is not None else len(query)
    if len(completed) < stop:
        import torch
        from multimodal_model_adapter import generate_legal_labels, icl_messages

        torch.manual_seed(73)
        model, processor = load_model(folder / "telemetry" / f"icl_{args.method}.jsonl")
        with result_path.open("a", encoding="utf-8") as handle:
            for start in range(len(completed), stop, args.batch_size):
                targets = query[start:min(start + args.batch_size, stop)]
                demo_batches = [[bank[j] for j in selections[target.sample_id]] for target in targets]
                outputs = generate_legal_labels(
                    model, processor,
                    [icl_messages(demos, target, args.dataset, labels, None)
                     for demos, target in zip(demo_batches, targets)], labels,
                )
                for target, demos, (prediction, raw) in zip(targets, demo_batches, outputs):
                    row = {
                        "method": args.method, "query_id": target.sample_id,
                        "target": target.label, "prediction": prediction,
                        "correct": prediction == target.label, "raw_output": raw,
                        "demo_ids": [demo.sample_id for demo in demos],
                        "demo_labels": [demo.label for demo in demos],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed.append(row)
                done = len(completed)
                if done % 20 == 0 or done == stop:
                    torch.cuda.empty_cache()
                if done % 100 == 0 or done == stop:
                    print(f"{args.dataset}/{args.method} predictions {done}/{len(query)}", flush=True)
        del model, processor
    if len(completed) == len(query):
        correct = sum(row["correct"] for row in completed)
        write_json(outdir / f"{args.method}.summary.json", {
            **prediction_identity, "correct": correct, "total": len(query),
            "accuracy": correct / len(query), "predictions_sha256": digest(result_path),
        })
        print(f"COMPLETE {args.dataset}/{args.method} {correct}/{len(query)}", flush=True)


if __name__ == "__main__":
    main()

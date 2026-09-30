#!/usr/bin/env python
"""Compare Qwen image budgets on a held-out slice of the training bank.

No official test/query image or test label is used. Class-balanced random
demonstrations come from the remaining bank and are fixed across budgets.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from multidataset_protocol import Sample
from multimodal_model_adapter import (
    MODEL_SPECS,
    configure_vision_pixels,
    generate_legal_labels,
    icl_messages,
    load_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="aircraft")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--per-class", type=int, default=2)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--budgets", type=int, nargs="+", default=[0, 50176, 100352, 200704])
    return parser.parse_args()


def development_selection(
    bank: list[Sample], per_class: int, shots: int, seed: int
) -> list[tuple[Sample, list[Sample]]]:
    by_label: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(bank):
        by_label[sample.label].append(index)
    rng = random.Random(seed)
    held_out = sorted(index for label in sorted(by_label)
                      for index in rng.sample(by_label[label], per_class))
    held_out_set = set(held_out)
    candidate_by_label = {
        label: [index for index in by_label[label] if index not in held_out_set]
        for label in sorted(by_label)
    }
    candidate_labels = [label for label, indices in candidate_by_label.items() if indices]
    if len(candidate_labels) < shots:
        raise ValueError("Not enough demonstration classes")
    result = []
    for index in held_out:
        demo_labels = rng.sample(candidate_labels, shots)
        demo_indices = [rng.choice(candidate_by_label[label]) for label in demo_labels]
        result.append((bank[index], [bank[int(value)] for value in demo_indices]))
    return result


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the throughput benchmark")
    if args.per_class < 1 or args.batch_size < 1 or args.shots < 1:
        raise ValueError("per-class, batch-size, and shots must be positive")
    if len(set(args.budgets)) != len(args.budgets):
        raise ValueError("Duplicate budgets")
    for budget in args.budgets:
        if budget != 0 and (budget < 1024 or budget % 1024):
            raise ValueError("Every nonzero budget must be a positive multiple of 1024")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bank = [Sample(**row) for row in manifest["bank"]]
    selected = development_selection(bank, args.per_class, args.shots, args.seed)
    labels = sorted({sample.label for sample in bank})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "dataset": args.dataset,
        "model": MODEL_SPECS["qwen3vl4b"].model_id,
        "seed": args.seed,
        "per_class": args.per_class,
        "shots": args.shots,
        "batch_size": args.batch_size,
        "budgets": args.budgets,
        "selected": [{"query": sample.sample_id, "demos": [d.sample_id for d in demos]}
                     for sample, demos in selected],
    }
    identity_path = args.output_dir / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Refusing to resume a different vision-budget pilot")
    else:
        write_json(identity_path, identity)

    predictions_path = args.output_dir / "predictions.jsonl"
    completed: dict[tuple[int, str], dict] = {}
    if predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[(int(row["vision_pixels"]), row["query_id"])] = row

    model, processor = load_model(MODEL_SPECS["qwen3vl4b"], args.cache_dir)
    original_size = copy.deepcopy(processor.image_processor.size)
    with predictions_path.open("a", encoding="utf-8") as output:
        for budget in args.budgets:
            processor.image_processor.size = copy.deepcopy(original_size)
            if budget:
                configure_vision_pixels(processor, budget)
            pending = [(sample, demos) for sample, demos in selected
                       if (budget, sample.sample_id) not in completed]
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                messages = [icl_messages(demos, sample, args.dataset, labels,
                                         224 if budget == 0 else None)
                            for sample, demos in batch]
                torch.cuda.synchronize()
                begun = time.perf_counter()
                predictions = generate_legal_labels(model, processor, messages, labels)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - begun
                for (sample, demos), (prediction, raw) in zip(batch, predictions):
                    row = {
                        "vision_pixels": budget,
                        "query_id": sample.sample_id,
                        "target": sample.label,
                        "prediction": prediction,
                        "correct": prediction == sample.label,
                        "raw_output": raw,
                        "demo_ids": [demo.sample_id for demo in demos],
                        "seconds_per_query": elapsed / len(batch),
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                    completed[(budget, sample.sample_id)] = row
                done = len(selected) - len(pending) + min(start + len(batch), len(pending))
                if done % 20 == 0 or done == len(selected):
                    torch.cuda.empty_cache()
                    print(f"budget {budget}: {done}/{len(selected)}", flush=True)
            rows = [completed[(budget, sample.sample_id)] for sample, _ in selected]
            summary = {
                "vision_pixels": budget,
                "correct": sum(row["correct"] for row in rows),
                "total": len(rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows),
                "seconds_per_query": sum(row["seconds_per_query"] for row in rows) / len(rows),
            }
            write_json(args.output_dir / f"summary_{budget}.json", summary)
            print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

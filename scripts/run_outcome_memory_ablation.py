#!/usr/bin/env python
"""Compare correct-only, all-trajectory, and error-aware trajectory memories."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from run_dtd_hidden_rices_pilot import (
    Sample,
    build_icl_messages,
    classification_instruction,
    generate_prediction,
    image_part,
    load_qwen,
    matrix_similarity,
    minmax,
)


NEW_METHODS = ("visual_matrix_correct_only", "visual_matrix_error_aware")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--rices-pool", type=int, default=50)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--corrective-shots", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ranked(indices: list[int] | np.ndarray, scores: np.ndarray, count: int) -> list[int]:
    values = np.asarray(list(indices), dtype=int)
    if len(values) == 0:
        return []
    chosen = values[np.argsort(scores[values])[-min(count, len(values)) :]]
    return chosen.tolist()  # ascending: the strongest demonstration is closest to the query.


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    config = json.loads((args.run_dir / "config.json").read_text(encoding="utf-8"))
    classes = config["classes"]
    bank = [Sample(**sample) for sample in manifest["bank"]]
    query = [Sample(**sample) for sample in manifest["query"]]
    all_samples = bank + query
    features = np.load(args.run_dir / "features.npz")
    clip = features["clip"]
    visual_matrix = features["visual_matrix"]

    model, processor = load_qwen(args.qwen_model, args.cache_dir)
    zero_path = args.run_dir / "zero_shot_predictions.jsonl"
    zero_rows = read_jsonl(zero_path) if args.resume else []
    zero_by_id = {row["sample_id"]: row for row in zero_rows}
    with zero_path.open("a", encoding="utf-8") as handle:
        for index, sample in enumerate(all_samples, start=1):
            if sample.sample_id in zero_by_id:
                continue
            messages = [
                {
                    "role": "user",
                    "content": [
                        image_part(sample.path, args.image_size),
                        {"type": "text", "text": classification_instruction(classes)},
                    ],
                }
            ]
            prediction, raw = generate_prediction(
                model, processor, messages, classes, args.max_new_tokens
            )
            row = {
                "sample_id": sample.sample_id,
                "split": sample.split,
                "target": sample.label,
                "prediction": prediction,
                "correct": prediction == sample.label,
                "raw_output": raw,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            zero_by_id[sample.sample_id] = row
            print(
                f"zero-shot {index:03d}/{len(all_samples):03d}: "
                f"target={sample.label} pred={prediction}",
                flush=True,
            )

    bank_n = len(bank)
    selections: dict[str, dict[str, list[int]]] = {method: {} for method in NEW_METHODS}
    selection_metadata: dict[str, dict] = {}
    for q_index, query_sample in enumerate(query):
        clip_scores = clip[:bank_n] @ clip[bank_n + q_index]
        pool = np.argsort(clip_scores)[-min(args.rices_pool, bank_n) :]
        hidden_scores = matrix_similarity(visual_matrix[bank_n + q_index], visual_matrix[:bank_n])
        combined = np.full(bank_n, -np.inf, dtype=float)
        combined[pool] = 0.5 * minmax(clip_scores[pool]) + 0.5 * minmax(hidden_scores[pool])

        correct_pool = [i for i in pool if zero_by_id[bank[i].sample_id]["correct"]]
        correct_only = ranked(correct_pool, combined, args.shots)
        if len(correct_only) < args.shots:
            for index in reversed(ranked(pool, combined, len(pool))):
                if index not in correct_only:
                    correct_only.insert(0, index)
                if len(correct_only) == args.shots:
                    break
        selections["visual_matrix_correct_only"][query_sample.sample_id] = correct_only

        query_prediction = zero_by_id[query_sample.sample_id]["prediction"]
        error_pool = [
            i
            for i in pool
            if not zero_by_id[bank[i].sample_id]["correct"]
            and zero_by_id[bank[i].sample_id]["prediction"] == query_prediction
        ]
        corrections = ranked(error_pool, combined, args.corrective_shots)
        supports = ranked(correct_pool, combined, args.shots - len(corrections))
        mixed = supports + corrections
        if len(mixed) < args.shots:
            for index in reversed(ranked(pool, combined, len(pool))):
                if index not in mixed:
                    mixed.insert(0, index)
                if len(mixed) == args.shots:
                    break
        mixed.sort(key=lambda index: combined[index])
        selections["visual_matrix_error_aware"][query_sample.sample_id] = mixed
        selection_metadata[query_sample.sample_id] = {
            "query_zero_shot_prediction": query_prediction,
            "corrective_candidates_available": len(error_pool),
            "corrective_demo_ids": [bank[i].sample_id for i in corrections],
        }

    detailed = {
        method: {
            query_id: [
                {
                    **asdict(bank[index]),
                    "zero_shot_prediction": zero_by_id[bank[index].sample_id]["prediction"],
                    "zero_shot_correct": zero_by_id[bank[index].sample_id]["correct"],
                }
                for index in indices
            ]
            for query_id, indices in rows.items()
        }
        for method, rows in selections.items()
    }
    (args.run_dir / "outcome_selections.json").write_text(
        json.dumps({"selections": detailed, "metadata": selection_metadata}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictions_path = args.run_dir / "outcome_predictions.jsonl"
    previous = read_jsonl(predictions_path) if args.resume else []
    completed = {(row["method"], row["query_id"]): row for row in previous}
    with predictions_path.open("a", encoding="utf-8") as handle:
        for method in NEW_METHODS:
            for q_index, query_sample in enumerate(query, start=1):
                key = (method, query_sample.sample_id)
                if key in completed:
                    continue
                indices = selections[method][query_sample.sample_id]
                demos = [bank[index] for index in indices]
                prediction, raw = generate_prediction(
                    model,
                    processor,
                    build_icl_messages(demos, query_sample, classes, args.image_size),
                    classes,
                    args.max_new_tokens,
                )
                row = {
                    "method": method,
                    "query_id": query_sample.sample_id,
                    "target": query_sample.label,
                    "prediction": prediction,
                    "correct": prediction == query_sample.label,
                    "raw_output": raw,
                    "demo_ids": [sample.sample_id for sample in demos],
                    "demo_labels": [sample.label for sample in demos],
                    "demo_zero_shot_correct": [
                        zero_by_id[sample.sample_id]["correct"] for sample in demos
                    ],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed[key] = row
                print(
                    f"{method} {q_index:02d}/{len(query):02d}: "
                    f"target={query_sample.label} pred={prediction}",
                    flush=True,
                )

    original = read_jsonl(args.run_dir / "predictions.jsonl")
    all_rows = [row for row in original if row["method"] == "rices_visual_selected_matrix"]
    summary = {
        "zero_shot": {
            "bank_correct": sum(zero_by_id[sample.sample_id]["correct"] for sample in bank),
            "bank_total": len(bank),
            "query_correct": sum(zero_by_id[sample.sample_id]["correct"] for sample in query),
            "query_total": len(query),
        },
        "visual_matrix_all_trajectories": {
            "correct": sum(row["correct"] for row in all_rows),
            "total": len(all_rows),
        },
    }
    for method in NEW_METHODS:
        method_rows = [row for row in completed.values() if row["method"] == method]
        summary[method] = {
            "correct": sum(row["correct"] for row in method_rows),
            "total": len(method_rows),
        }
    (args.run_dir / "outcome_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

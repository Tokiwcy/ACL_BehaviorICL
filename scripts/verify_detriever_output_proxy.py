#!/usr/bin/env python
"""Validate one completed bank-only gold-output DeTriever run and compare it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prediction_map(path: Path, query: dict[str, dict], labels: set[str],
                   method: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["method"] != method:
            continue
        key = row["query_id"]
        if key in rows or key not in query:
            raise RuntimeError(f"Duplicate or unknown {method} query: {key}")
        if row["target"] != query[key]["label"] or row["prediction"] not in labels:
            raise RuntimeError(f"Target or predicted label mismatch for {key}")
        if row["correct"] != (row["prediction"] == row["target"]):
            raise RuntimeError(f"Incorrect correctness flag for {key}")
        rows[key] = row
    if set(rows) != set(query):
        raise RuntimeError(f"{method} has {len(rows)}/{len(query)} queries")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("new_run", type=Path)
    parser.add_argument("original_run", type=Path, nargs="?",
                        help="Optional legacy tuple for paired comparison")
    args = parser.parse_args()
    new = json.loads((args.new_run / "manifest.json").read_text(encoding="utf-8"))
    if args.original_run is not None:
        old = json.loads((args.original_run / "manifest.json").read_text(encoding="utf-8"))
        for split in ("bank", "query"):
            a = [(r["sample_id"], r["label"], r["split"]) for r in new[split]]
            b = [(r["sample_id"], r["label"], r["split"]) for r in old[split]]
            if a != b:
                raise RuntimeError(f"{split} differs between new and original protocols")
    identity = json.loads((args.new_run / "run_identity.json").read_text(encoding="utf-8"))
    if identity.get("baseline") != "detriever_gold_input_answer_eos_v1":
        raise RuntimeError("Not a gold-output-proxy DeTriever run")
    metadata = json.loads((args.new_run / "detriever_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("proxy") != "gold_input_answer_eos" or not metadata.get("proxy_sha256"):
        raise RuntimeError("Training metadata does not identify the gold-output proxy")
    if not (args.new_run / "detriever_checkpoint.pt").is_file():
        raise RuntimeError("Local retriever checkpoint is missing")
    progress = json.loads((args.new_run / "gold_input_answer_eos_progress.json").read_text(encoding="utf-8"))
    if progress.get("completed") != len(new["bank"]):
        raise RuntimeError("Bank-only gold-output feature extraction is incomplete")
    selections = json.loads((args.new_run / "selections.json").read_text(encoding="utf-8"))
    query = {row["sample_id"]: row for row in new["query"]}
    bank = {row["sample_id"]: row for row in new["bank"]}
    labels = {row["label"] for row in new["bank"]}
    if set(selections["detriever"]) != set(query):
        raise RuntimeError("New DeTriever selections are incomplete")
    for query_id, demos in selections["detriever"].items():
        if len(demos) != 4 or len({row["sample_id"] for row in demos}) != 4:
            raise RuntimeError(f"Invalid four-shot selection for {query_id}")
        if any(row["sample_id"] not in bank or row["label"] != bank[row["sample_id"]]["label"] for row in demos):
            raise RuntimeError(f"Non-bank or mislabeled demonstration for {query_id}")
    fresh = prediction_map(args.new_run / "predictions.jsonl", query, labels, "detriever")
    fresh_correct = sum(row["correct"] for row in fresh.values())
    result = {
        "total": len(query), "new_accuracy": fresh_correct / len(query),
        "new_correct": fresh_correct,
    }
    if args.original_run is not None:
        legacy = prediction_map(args.original_run / "predictions.jsonl", query, labels, "detriever")
        legacy_correct = sum(row["correct"] for row in legacy.values())
        wins = sum(fresh[key]["correct"] and not legacy[key]["correct"] for key in query)
        losses = sum(not fresh[key]["correct"] and legacy[key]["correct"] for key in query)
        result.update({
            "legacy_correct": legacy_correct,
            "legacy_accuracy": legacy_correct / len(query),
            "new_minus_legacy": (fresh_correct - legacy_correct) / len(query),
            "paired_wins": wins, "paired_losses": losses,
        })
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

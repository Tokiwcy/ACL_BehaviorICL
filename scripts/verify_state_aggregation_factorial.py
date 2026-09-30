#!/usr/bin/env python
"""Validate the complete raw/delta x pooled/layerwise DTD ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


METHODS = ("raw_pooled", "raw_layerwise", "delta_pooled", "delta_layerwise")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(run_dir: Path, source_cache: Path | None = None) -> dict:
    identity = read_json(run_dir / "run_identity.json")
    if identity.get("ablation") != "raw_delta_x_pooled_layerwise_v1":
        raise ValueError("Wrong ablation identity")
    if (identity.get("shots"), identity.get("projection_dim"), identity.get("train_steps")) != (4, 256, 1500):
        raise ValueError("Shots, projection dimension, or training steps differ")
    if source_cache is not None and sha256_file(source_cache) != identity["source_cache_sha256"]:
        raise ValueError("Source cache SHA256 differs")
    manifest = read_json(run_dir / "manifest.json")
    bank = {row["sample_id"]: row for row in manifest["bank"]}
    query = {row["sample_id"]: row for row in manifest["query"]}
    if len(bank) != len(manifest["bank"]) or len(query) != len(manifest["query"]):
        raise ValueError("Duplicate bank or query sample ID")
    if set(bank) & set(query):
        raise ValueError("Bank/query sample ID overlap")
    labels = {row["label"] for row in bank.values()}
    split = read_json(run_dir / "development_split.json")
    train_ids, val_ids = split["train_ids"], split["validation_ids"]
    if len(set(train_ids)) != len(train_ids) or len(set(val_ids)) != len(val_ids):
        raise ValueError("Duplicate development sample ID")
    if set(train_ids) & set(val_ids) or set(train_ids) | set(val_ids) != set(bank):
        raise ValueError("Development train/validation are not a bank partition")
    if set(Counter(bank[sample_id]["label"] for sample_id in val_ids).values()) != {10}:
        raise ValueError("Validation does not contain ten bank examples per class")

    selections = read_json(run_dir / "selections.json")
    if set(selections) != set(METHODS):
        raise ValueError("Selection method set differs")
    for method, rows in selections.items():
        if set(rows) != set(query):
            raise ValueError(f"{method} does not cover each query exactly once")
        for query_id, demos in rows.items():
            if len(demos) != 4 or len({demo["sample_id"] for demo in demos}) != 4:
                raise ValueError(f"Invalid four-shot selection for {method}/{query_id}")
            for demo in demos:
                if bank.get(demo["sample_id"]) != demo:
                    raise ValueError(f"Demo differs from bank for {method}/{query_id}")
        metadata = read_json(run_dir / f"{method}_metadata.json")
        layers = 36 if method.startswith("raw_") else 35
        if (metadata.get("method"), metadata.get("train_steps"),
            metadata.get("projection_dim")) != (method, 1500, 256):
            raise ValueError(f"Training metadata differs for {method}")
        if metadata.get("parameter_count") != 256 * 2560 + layers:
            raise ValueError(f"Parameter count differs for {method}")
        if len(metadata.get("learned_layer_weights", [])) != layers:
            raise ValueError(f"Layer-weight count differs for {method}")
        if not 0 <= metadata.get("best_step", -1) <= 1500:
            raise ValueError(f"Best step differs for {method}")
        if [row["step"] for row in metadata.get("history", [])] != list(range(0, 1501, 100)):
            raise ValueError(f"Training history is incomplete for {method}")
        if not (run_dir / f"{method}_best.pt").is_file():
            raise ValueError(f"Missing best checkpoint for {method}")

    predictions = run_dir / "predictions.jsonl"
    seen = set()
    counts = Counter()
    correct = Counter()
    for line in predictions.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        method, query_id = row["method"], row["query_id"]
        key = method, query_id
        if method not in METHODS or query_id not in query or key in seen:
            raise ValueError(f"Invalid or duplicate prediction key: {key}")
        seen.add(key)
        if row["target"] != query[query_id]["label"] or row["prediction"] not in labels:
            raise ValueError(f"Target or prediction label differs for {key}")
        if bool(row["correct"]) != (row["target"] == row["prediction"]):
            raise ValueError(f"Correctness flag differs for {key}")
        demos = selections[method][query_id]
        if row["demo_ids"] != [demo["sample_id"] for demo in demos]:
            raise ValueError(f"Demo IDs differ for {key}")
        if row["demo_labels"] != [demo["label"] for demo in demos]:
            raise ValueError(f"Demo labels differ for {key}")
        counts[method] += 1
        correct[method] += bool(row["correct"])
    if counts != Counter({method: len(query) for method in METHODS}):
        raise ValueError(f"Incomplete predictions: {dict(counts)}")
    summary = read_json(run_dir / "summary.json")["summary"]
    if set(summary) != set(METHODS):
        raise ValueError("Summary method set differs")
    for method in METHODS:
        row = summary[method]
        if row["correct"] != correct[method] or row["total"] != len(query):
            raise ValueError(f"Summary counts differ for {method}")
        if abs(row["accuracy"] - correct[method] / len(query)) > 1e-12:
            raise ValueError(f"Summary accuracy differs for {method}")
    return {"passed": True, "dataset": identity["dataset"], "bank": len(bank),
            "query": len(query), "source_cache_sha256": identity["source_cache_sha256"],
            "methods": {method: {"correct": correct[method], "total": len(query),
                                 "accuracy": correct[method] / len(query)} for method in METHODS}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-cache", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run_dir, args.source_cache), indent=2))


if __name__ == "__main__":
    main()

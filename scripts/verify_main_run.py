#!/usr/bin/env python
"""Read-only completion gate for one multidataset CDR run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


METHODS = {"rices", "gpt_mm", "detriever", "cdr_zero", "cdr_learn"}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run_dir: Path) -> dict:
    problems: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    identity = read_json(run_dir / "run_identity.json")
    config = read_json(run_dir / "config.json")
    protocol = read_json(run_dir / "protocol.json")
    manifest = read_json(run_dir / "manifest.json")
    bank, query = manifest["bank"], manifest["query"]
    bank_by_id = {sample["sample_id"]: sample for sample in bank}
    query_by_id = {sample["sample_id"]: sample for sample in query}
    labels = {sample["label"] for sample in bank}
    total = len(bank) + len(query)
    shots = int(config["shots"])

    check(len(bank_by_id) == len(bank), "duplicate bank IDs")
    check(len(query_by_id) == len(query), "duplicate query IDs")
    check(not (bank_by_id.keys() & query_by_id.keys()), "bank/query IDs overlap")
    check({sample["label"] for sample in query} <= labels, "query label absent from bank")
    check(identity["dataset"] == config["dataset"] == protocol["dataset"], "dataset identity mismatch")
    check(identity["model_slug"] == config["model"], "model identity mismatch")
    check(identity["model_id"] == protocol["model"], "model ID mismatch")
    check(identity["seed"] == config["seed"] == protocol["seed"], "seed identity mismatch")
    check(protocol["bank_count"] == len(bank), "bank count mismatch")
    check(protocol["query_count"] == len(query), "query count mismatch")
    check(set(config["methods"]) == METHODS, "configured method set mismatch")

    for name, progress_name in (
        ("anchor_states.npy", "anchor_states_progress.json"),
        ("clip_features.npy", "clip_features.progress.json"),
        ("gpt_mm_embeddings.npy", "gpt_mm_progress.json"),
    ):
        path, progress_path = run_dir / name, run_dir / progress_name
        if not path.exists() or not progress_path.exists():
            problems.append(f"missing cache or progress: {name}")
            continue
        check(int(read_json(progress_path)["completed"]) == total, f"incomplete cache: {name}")
        check(np.load(path, mmap_mode="r").shape[0] == total, f"cache shape mismatch: {name}")

    for checkpoint, metadata in (
        ("projected_velocity_best.pt", "projected_velocity_metadata.json"),
        ("detriever_checkpoint.pt", "detriever_metadata.json"),
    ):
        check((run_dir / checkpoint).is_file(), f"missing local checkpoint: {checkpoint}")
        check((run_dir / metadata).is_file(), f"missing local training metadata: {metadata}")

    selections = read_json(run_dir / "selections.json")
    check(set(selections) == METHODS, "selection method set mismatch")
    for method in METHODS & selections.keys():
        selected = selections[method]
        check(set(selected) == query_by_id.keys(), f"{method}: selection query IDs mismatch")
        for query_id, demos in selected.items():
            check(len(demos) == shots, f"{method}/{query_id}: demo count mismatch")
            demo_ids = [demo["sample_id"] for demo in demos]
            check(len(set(demo_ids)) == len(demo_ids), f"{method}/{query_id}: duplicate demos")
            for demo in demos:
                sample = bank_by_id.get(demo["sample_id"])
                check(sample is not None, f"{method}/{query_id}: demo outside bank")
                if sample is not None:
                    check(demo == sample, f"{method}/{query_id}: demo metadata mismatch")

    rows = [json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    counts: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    correct: Counter[str] = Counter()
    for row in rows:
        method, query_id = row["method"], row["query_id"]
        key = (method, query_id)
        check(key not in seen, f"duplicate prediction: {method}/{query_id}")
        seen.add(key)
        counts[method] += 1
        target = query_by_id.get(query_id)
        check(target is not None, f"prediction outside query: {method}/{query_id}")
        if target is None:
            continue
        check(row["target"] == target["label"], f"target mismatch: {method}/{query_id}")
        check(row["prediction"] in labels, f"illegal prediction: {method}/{query_id}")
        check(row["correct"] == (row["prediction"] == target["label"]), f"correct flag mismatch: {method}/{query_id}")
        correct[method] += bool(row["correct"])
        demos = selections.get(method, {}).get(query_id)
        if demos is None:
            problems.append(f"prediction has no selection: {method}/{query_id}")
            continue
        check(row["demo_ids"] == [demo["sample_id"] for demo in demos], f"demo IDs mismatch: {method}/{query_id}")
        check(row["demo_labels"] == [demo["label"] for demo in demos], f"demo labels mismatch: {method}/{query_id}")
    check(set(counts) == METHODS, "prediction method set mismatch")
    for method in METHODS:
        check(counts[method] == len(query), f"{method}: prediction count mismatch")
        check({qid for m, qid in seen if m == method} == query_by_id.keys(), f"{method}: prediction query IDs mismatch")

    summary = read_json(run_dir / "summary.json")
    summary = summary.get("summary", summary)
    for method in METHODS & summary.keys():
        check(summary[method]["correct"] == correct[method], f"{method}: summary correct mismatch")
        check(summary[method]["total"] == len(query), f"{method}: summary total mismatch")
    check(set(summary) == METHODS, "summary method set mismatch")

    result = {
        "run_dir": str(run_dir.resolve()),
        "dataset": identity["dataset"],
        "model": identity["model_slug"],
        "seed": identity["seed"],
        "bank": len(bank),
        "query": len(query),
        "prediction_counts": dict(counts),
        "correct_counts": dict(correct),
        "problems": problems,
        "passed": not problems,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = verify(args.run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

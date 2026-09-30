#!/usr/bin/env python
"""Read-only completion gate for 36-state Behavior retrieval runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


METHODS = {"decision_zero", "decision_learn"}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run_dir: Path, source_run: Path, check_source_cache: bool = True) -> dict:
    problems = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            problems.append(message)

    identity = read_json(run_dir / "run_identity.json")
    config = read_json(run_dir / "config.json")
    protocol = read_json(run_dir / "protocol.json")
    manifest = read_json(run_dir / "manifest.json")
    bank, query = manifest["bank"], manifest["query"]
    bank_by_id = {row["sample_id"]: row for row in bank}
    query_by_id = {row["sample_id"]: row for row in query}
    labels = {row["label"] for row in bank}
    shots = int(config["shots"])
    check(identity.get("ablation") == "36_raw_layer_states_v1", "ablation identity mismatch")
    check(identity.get("dataset") == config.get("dataset") == protocol.get("dataset")
          and identity.get("dataset") in {"dtd", "aircraft", "cub", "dogs", "pets"},
          "dataset mismatch")
    check(identity.get("model_slug") == config.get("model") == "qwen3vl4b", "model mismatch")
    check(identity.get("seed") == config.get("seed") == protocol.get("seed"), "seed mismatch")
    check(set(config.get("methods", [])) == METHODS, "configured method set mismatch")
    check(config.get("train_steps") == 1500, "training step protocol mismatch")
    check(config.get("state_shape") == [len(bank) + len(query), 36, 2560], "state shape mismatch")
    check(len(bank_by_id) == len(bank) and len(query_by_id) == len(query), "duplicate sample IDs")
    check(not (bank_by_id.keys() & query_by_id.keys()), "bank/query overlap")
    check({row["label"] for row in query} <= labels, "query label absent from bank")
    check(protocol.get("bank_count") == len(bank) and protocol.get("query_count") == len(query),
          "protocol split count mismatch")

    source_identity = read_json(source_run / "run_identity.json")
    source_manifest = read_json(source_run / "manifest.json")
    check(all(source_identity.get(k) == identity.get(k) for k in
              ("dataset", "model_slug", "model_id", "seed")), "source identity mismatch")
    for split in ("bank", "query"):
        source_rows = [(r["sample_id"], r["label"], r["split"]) for r in source_manifest[split]]
        result_rows = [(r["sample_id"], r["label"], r["split"]) for r in manifest[split]]
        check(source_rows == result_rows, f"source {split} order/labels mismatch")
    if check_source_cache:
        progress = read_json(source_run / "anchor_states_progress.json")
        check(progress.get("completed") == len(bank) + len(query), "source cache incomplete")
        check(progress.get("shape") == [len(bank) + len(query), 36, 2560],
              "source cache progress shape mismatch")
        check(list(np.load(source_run / "anchor_states.npy", mmap_mode="r").shape)
              == [len(bank) + len(query), 36, 2560], "source cache array shape mismatch")

    checkpoint = run_dir / "projected_state_best.pt"
    metadata_path = run_dir / "projected_state_metadata.json"
    check(checkpoint.is_file(), "missing Decision-Learn checkpoint")
    check(metadata_path.is_file(), "missing Decision-Learn training metadata")
    if metadata_path.is_file():
        metadata = read_json(metadata_path)
        history = metadata.get("history", [])
        check(metadata.get("variant") == "projected_state", "training variant mismatch")
        check(bool(history) and history[0].get("step") == 0
              and history[-1].get("step") == 1500, "training history incomplete")
        check(len(metadata.get("learned_layer_weights", [])) == 36,
              "learned layer weight count mismatch")

    selections = read_json(run_dir / "selections.json")
    check(set(selections) == METHODS, "selection method set mismatch")
    for method in METHODS & selections.keys():
        check(set(selections[method]) == query_by_id.keys(), f"{method} selection query mismatch")
        for query_id, demos in selections[method].items():
            check(len(demos) == shots, f"{method}/{query_id} demo count mismatch")
            ids = [demo["sample_id"] for demo in demos]
            check(len(set(ids)) == len(ids), f"{method}/{query_id} duplicate demo")
            for demo in demos:
                source = bank_by_id.get(demo["sample_id"])
                check(source is not None and demo == source,
                      f"{method}/{query_id} demo not in bank")

    counts, correct = Counter(), Counter()
    seen = set()
    for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        method, query_id = row["method"], row["query_id"]
        key = (method, query_id)
        check(key not in seen, f"duplicate prediction {key}")
        seen.add(key)
        counts[method] += 1
        sample = query_by_id.get(query_id)
        check(sample is not None, f"prediction outside query {key}")
        if sample is None:
            continue
        check(row["target"] == sample["label"], f"target mismatch {key}")
        check(row["prediction"] in labels, f"illegal label {key}")
        check(row["correct"] == (row["prediction"] == sample["label"]),
              f"correct flag mismatch {key}")
        correct[method] += bool(row["correct"])
        demos = selections.get(method, {}).get(query_id)
        check(demos is not None, f"missing selection {key}")
        if demos is not None:
            check(row["demo_ids"] == [demo["sample_id"] for demo in demos],
                  f"demo IDs mismatch {key}")
            check(row["demo_labels"] == [demo["label"] for demo in demos],
                  f"demo labels mismatch {key}")
    check(set(counts) == METHODS, "prediction method set mismatch")
    for method in METHODS:
        check(counts[method] == len(query), f"{method} prediction count mismatch")
        check({qid for m, qid in seen if m == method} == query_by_id.keys(),
              f"{method} prediction query coverage mismatch")
    summary = read_json(run_dir / "summary.json").get("summary", {})
    check(set(summary) == METHODS, "summary method set mismatch")
    for method in METHODS & summary.keys():
        check(summary[method]["correct"] == correct[method], f"{method} summary correct mismatch")
        check(summary[method]["total"] == len(query), f"{method} summary total mismatch")
    return {"dataset": identity.get("dataset"), "bank": len(bank), "query": len(query),
            "counts": dict(counts), "correct": dict(correct),
            "source_cache_checked": check_source_cache,
            "problems": problems, "passed": not problems}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--source-run", type=Path)
    parser.add_argument("--skip-source-cache", action="store_true")
    args = parser.parse_args()
    identity = read_json(args.run_dir / "run_identity.json")
    source_run = args.source_run or (Path("results/cdr_main") / identity["dataset"]
                                     / identity["model_slug"] / f"seed_{identity['seed']}")
    result = verify(args.run_dir, source_run, not args.skip_source_cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

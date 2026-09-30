#!/usr/bin/env python
"""Completion and split-leakage gate for forward-trace ablations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


METHODS = {"forward_zero", "forward_learn"}
SHAPE_SUFFIX = [36, 3, 2560]


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(run_dir: Path, check_cache: bool = True) -> dict:
    errors = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    identity = read(run_dir / "run_identity.json")
    config = read(run_dir / "config.json")
    protocol = read(run_dir / "protocol.json")
    manifest = read(run_dir / "manifest.json")
    bank, query = manifest["bank"], manifest["query"]
    bank_by_id = {row["sample_id"]: row for row in bank}
    query_by_id = {row["sample_id"]: row for row in query}
    labels = {row["label"] for row in bank}
    count = len(bank) + len(query)
    check(identity.get("ablation") == "forward_trace_anchor_visual_update_v1",
          "ablation identity mismatch")
    check(identity.get("dataset") == config.get("dataset") == protocol.get("dataset")
          and identity.get("dataset") in {"dtd", "aircraft"},
          "dataset mismatch")
    check(identity.get("model_slug") == config.get("model") == "qwen3vl4b", "model mismatch")
    check(identity.get("seed") == config.get("seed") == protocol.get("seed"), "seed mismatch")
    check(set(config.get("methods", [])) == METHODS, "method set mismatch")
    check(config.get("train_steps") == 1500, "training-step protocol mismatch")
    check(config.get("shots") == 4, "shot count mismatch")
    check(config.get("vision_pixels") is None, "visual protocol changed")
    check(len(bank_by_id) == len(bank) and len(query_by_id) == len(query),
          "duplicate bank/query sample IDs")
    check(not (bank_by_id.keys() & query_by_id.keys()), "bank/query overlap")
    check({row["label"] for row in query} <= labels, "query label missing from bank")
    check(protocol.get("bank_count") == len(bank) and protocol.get("query_count") == len(query),
          "protocol split count mismatch")
    check(protocol.get("retrieval_feature_prompt") ==
          "zero-shot image plus task; one prefill; no generated answer",
          "retrieval feature prompt mismatch")

    progress = read(run_dir / "forward_trace_progress.json")
    expected_shape = [count] + SHAPE_SUFFIX
    check(progress.get("shape") == expected_shape, "feature progress shape mismatch")
    check(progress.get("completed") == count, "feature cache incomplete")
    check(progress.get("streams") == ["answer_anchor", "visual_token_mean",
                                      "answer_layer_update"], "feature stream mismatch")
    if check_cache:
        cache = np.load(run_dir / "forward_trace.npy", mmap_mode="r")
        check(list(cache.shape) == expected_shape, "feature array shape mismatch")
        check(cache.dtype == np.float16, "feature array dtype mismatch")

    checkpoint = run_dir / "projected_state_best.pt"
    metadata_path = run_dir / "projected_state_metadata.json"
    check(checkpoint.is_file(), "missing Learn checkpoint")
    check(metadata_path.is_file(), "missing Learn training metadata")
    if metadata_path.is_file():
        metadata = read(metadata_path)
        history = metadata.get("history", [])
        check(metadata.get("variant") == "projected_state", "Learn variant mismatch")
        check(bool(history) and history[0].get("step") == 0 and
              history[-1].get("step") == 1500, "Learn training incomplete")
        check(len(metadata.get("learned_layer_weights", [])) == 108,
              "Learn weight count mismatch")
        check(metadata.get("parameter_count") == 256 * 2560 + 108,
              "Learn parameter count mismatch")
    learn_info = read(run_dir / "forward_learn_metadata.json")
    check(learn_info.get("bank_train_count") + learn_info.get("bank_validation_count")
          == len(bank), "Learn split not bank-only")

    selections = read(run_dir / "selections.json")
    check(set(selections) == METHODS, "selection method set mismatch")
    for method in METHODS & selections.keys():
        check(set(selections[method]) == query_by_id.keys(),
              f"{method} selection query coverage mismatch")
        for query_id, demos in selections[method].items():
            check(len(demos) == 4, f"{method}/{query_id} demo count mismatch")
            ids = [demo["sample_id"] for demo in demos]
            check(len(set(ids)) == len(ids), f"{method}/{query_id} repeated demo")
            for demo in demos:
                check(bank_by_id.get(demo["sample_id"]) == demo,
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
              f"correctness mismatch {key}")
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
              f"{method} prediction coverage mismatch")
    summary = read(run_dir / "summary.json").get("summary", {})
    check(set(summary) == METHODS, "summary method set mismatch")
    for method in METHODS & summary.keys():
        check(summary[method]["correct"] == correct[method],
              f"{method} summary count mismatch")
        check(summary[method]["total"] == len(query), f"{method} summary total mismatch")
    return {"bank": len(bank), "query": len(query), "counts": dict(counts),
            "correct": dict(correct), "cache_checked": check_cache,
            "errors": errors, "passed": not errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--skip-cache", action="store_true")
    args = parser.parse_args()
    result = verify(args.run_dir, not args.skip_cache)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

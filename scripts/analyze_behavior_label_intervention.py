"""CPU-only complete five-dataset fixed-context label intervention analysis.

Never publish partial arms or replace the historical P0 analysis. The new
constrained zero reference defines rescue/harm; old free zero is sensitivity.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_behavior_priority_analysis as a
from analyze_behavior_repair_all_queries import failure_features, paired_bootstrap
from collect_behavior_priority_zero import digest
from run_behavior_label_intervention import (ARMS, DATASETS, job_identity, read, schedule,
    validate_inputs, validate_prefix, verify_images)


def safe_pair(first, second):
    if not first:
        return {"queries": 0, "difference_pp": None, "paired_bootstrap_low_pp": None,
                "paired_bootstrap_high_pp": None, "bootstrap_repetitions": 10000, "seed": 73}
    value = paired_bootstrap(first, second)
    for old, new in (("first_rescued", "first_positive"), ("second_rescued", "second_positive"),
                     ("first_only_rescued", "first_only_positive"), ("second_only_rescued", "second_only_positive")):
        value[new] = value.pop(old)
    return value


def make_tables(rows, records, free_zero):
    expected = {(r["query_id"], arm) for r in rows for arm in ARMS}
    data = {(r["query_id"], r["arm"]): r for r in records}
    if len(records) != len(expected) or set(data) != expected:
        raise ValueError("Missing/duplicate intervention query-arm rows")
    outcomes, summary, paired = [], [], []
    for row in rows:
        qid = row["query_id"]; z = data[(qid, ARMS[0])]
        if z["status"] != "generated":
            raise ValueError("Constrained zero must cover every prespecified query")
        old = free_zero[qid]
        if old["target"] != row["target_for_evaluation_only"]:
            raise ValueError("Free-zero query identity differs from original plan")
        features = failure_features(old["target"], old["prediction"], row["gold_labels"], row["demo_own_zero_labels"])
        for arm in ARMS:
            value = data[(qid, arm)]; generated = value["status"] == "generated"
            outcomes.append({"dataset": row["dataset"], "query_id": qid, "arm": arm,
                    "status": value["status"], "correct": value["correct"],
                    "constrained_zero_correct": z["correct"],
                    "rescued": (not z["correct"] and value["correct"]) if generated else None,
                    "harmed": (z["correct"] and not value["correct"]) if generated else None,
                    "free_zero_correct": old["correct"], "free_zero_valid": old["prediction"] is not None,
                    "free_zero_rescued": (not old["correct"] and value["correct"]) if generated else None,
                    "free_zero_harmed": (old["correct"] and not value["correct"]) if generated else None,
                    "own_labels_changed": row["own_legal_labels_changed"],
                    "own_arm_missing": row["own_arm_status"] != "ready",
                    "changed_demo_labels": 0 if arm in ARMS[:2] else row["own_legal_labels_changed"] if arm == ARMS[2] else 4,
                    "input_tokens": value["input_tokens"], "output_tokens": value["output_tokens"],
                    "elapsed_seconds": value["elapsed_seconds"],
                    "demo_label_token_counts": value.get("demo_label_token_counts"),
                    "old_free_zero_strict_status": features["strict_status"],
                    "old_free_zero_strict_count": features["strict_observed_count"]})
    for arm in ARMS:
        all_rows = [r for r in outcomes if r["arm"] == arm]
        valid = [r for r in all_rows if r["status"] == "generated"]
        errors = [r for r in valid if not r["constrained_zero_correct"]]
        correct = [r for r in valid if r["constrained_zero_correct"]]
        n = len(valid); count = sum(r["correct"] for r in valid)
        accuracy_low, accuracy_high = a.wilson(count, n)
        summary.append({"dataset": rows[0]["dataset"], "arm": arm, "planned_queries": len(rows),
                "generated_queries": n, "explicit_missing": len(all_rows) - n,
                "correct": count, "accuracy": a.rate(count, n),
                "accuracy_wilson_low": accuracy_low, "accuracy_wilson_high": accuracy_high,
                "constrained_zero_errors_in_same_arm_cohort": len(errors),
                "rescued": sum(r["rescued"] for r in errors),
                "rescue_rate": a.rate(sum(r["rescued"] for r in errors), len(errors)),
                "constrained_zero_correct_in_same_arm_cohort": len(correct),
                "harmed": sum(r["harmed"] for r in correct),
                "harm_rate": a.rate(sum(r["harmed"] for r in correct), len(correct)),
                "input_tokens_sum": sum(r["input_tokens"] for r in valid),
                "output_tokens_sum": sum(r["output_tokens"] for r in valid),
                "actual_generation_seconds_sum": sum(r["elapsed_seconds"] for r in valid),
                "timing_scope": "input prepare + serial constrained generate/decode synchronization; model loading separate"})
    for second in ARMS[2:]:
        valid = [row for row in rows if data[(row["query_id"], second)]["status"] == "generated"]
        for scope, subset in (("all_available_pairs", valid),
                              ("labels_actually_changed", [row for row in valid if second == ARMS[3] or row["own_legal_labels_changed"] > 0])):
            for metric, selected in (("accuracy", subset),
                                     ("rescue_given_constrained_zero_error", [row for row in subset if not data[(row["query_id"], ARMS[0])]["correct"]]),
                                     ("harm_given_constrained_zero_correct", [row for row in subset if data[(row["query_id"], ARMS[0])]["correct"]])):
                first = [data[(row["query_id"], ARMS[1])]["correct"] for row in selected]
                last = [data[(row["query_id"], second)]["correct"] for row in selected]
                if metric.startswith("harm"):
                    first, last = [not v for v in first], [not v for v in last]
                paired.append({"dataset": rows[0]["dataset"], "first_arm": ARMS[1], "second_arm": second,
                               "scope": scope, "metric": metric, **safe_pair(first, last)})
    zero_ids = [row["query_id"] for row in rows]
    paired.append({"dataset": rows[0]["dataset"], "first_arm": ARMS[1], "second_arm": ARMS[0],
                   "scope": "all_prespecified_queries", "metric": "accuracy",
                   **safe_pair([data[(i, ARMS[1])]["correct"] for i in zero_ids],
                               [data[(i, ARMS[0])]["correct"] for i in zero_ids])})
    unchanged = [r for r in rows if r["own_arm_status"] == "ready" and r["own_legal_labels_changed"] == 0]
    consistency = {"dataset": rows[0]["dataset"], "unchanged_label_queries": len(unchanged),
            "identical_label_prediction_disagreements": sum(data[(r["query_id"], ARMS[1])]["prediction"] != data[(r["query_id"], ARMS[2])]["prediction"] for r in unchanged)}
    return outcomes, summary, paired, consistency


def load_free_zero(csv_path, rows):
    wanted = {(r["dataset"], r["query_id"]): r for r in rows}
    found = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = (r["dataset"], r["query_id"])
            if r["method"] != "BehaviorICL-Proj" or key not in wanted:
                continue
            row = wanted[key]
            if key in found or json.loads(r["demo_ids"]) != row["demo_ids_in_actual_prompt_order"] or json.loads(r["demo_labels"]) != row["gold_labels"]:
                raise ValueError("P0 source query/demo identity mismatch")
            legal = {v.casefold(): v for v in row["legal_classes"]}
            pred = legal.get(r["raw_zero_shot_answer"].strip().casefold())
            stored = r["zero_shot_pred"] or None
            if pred != stored or r["true_label"] != row["target_for_evaluation_only"]:
                raise ValueError("P0 zero answer changed")
            correct = pred == r["true_label"]
            if r["zero_shot_correct"] != str(correct):
                raise ValueError("P0 zero correctness changed")
            found[key] = {"target": r["true_label"], "prediction": pred, "correct": correct}
    if set(found) != set(wanted):
        raise ValueError("Missing locked queries in complete P0 source")
    return found


def audit_generated_tokens(records, rows, tokenizer):
    from legal_label_decoding import LegalLabelTrie, label_token_sequences
    classes = rows[0]["legal_classes"]
    sequences, maximum = label_token_sequences(tokenizer, classes)
    trie = LegalLabelTrie(sequences, tokenizer.eos_token_id)
    for r in records:
        if r["status"] != "generated":
            continue
        tokens = r["generated_token_ids"]
        prefix = []
        for t in tokens:
            if t not in trie.allowed(prefix):
                raise ValueError("Generated token left the exact legal-label trie")
            if t == tokenizer.eos_token_id:
                break
            prefix.append(t)
        if tokens[-1] != tokenizer.eos_token_id or tokens.count(tokenizer.eos_token_id) != 1 or len(tokens) > maximum:
            raise ValueError("Missing/extra EOS in constrained output")
        if tokenizer.decode(tokens, skip_special_tokens=True).strip() != r["raw_answer"]:
            raise ValueError("Stored literal answer differs from actual generated tokens")
        counts = [len(tokenizer.encode(" " + v, add_special_tokens=False)) for v in r["demo_labels"]]
        if counts != r["demo_label_token_counts"]:
            raise ValueError("Incorrect intervention label-token accounting")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--inputs-lock", type=Path, default=ROOT / "analysis/paired_label_inputs_v1/attached_inputs_lock.json")
    p.add_argument("--base-plan", type=Path, default=ROOT / "analysis/paired_label_plan_v1/query_arm_lock.json")
    p.add_argument("--cache-dir", type=Path, default=ROOT / ".hf_cache")
    p.add_argument("--p0-query-csv", type=Path, default=ROOT / "analysis/paper_repair_five_dataset_v1/query_level_results.csv")
    args = p.parse_args()
    out = args.output_root.resolve(); source = args.result_root.resolve()
    if out.exists() or not out.is_relative_to(ROOT / "analysis") or not source.is_relative_to(ROOT / "results"):
        raise ValueError("Fresh derived analysis root and local result snapshot required")
    inputs, base = read(args.inputs_lock), read(args.base_plan)
    validate_inputs(inputs, base)
    if digest(args.base_plan) != inputs["base_plan_file_sha256"]:
        raise ValueError("Original plan bytes changed")
    # Guard all registered evidence before any new scientific result is written.
    provenance = dict(inputs["source_sha256"])
    for rel, expected in provenance.items():
        if digest(ROOT / rel) != expected:
            raise ValueError("Frozen P0/plan source changed: " + rel)
    free = load_free_zero(args.p0_query_csv.resolve(), inputs["all_queries"])
    from transformers import AutoTokenizer
    snapshot = args.cache_dir.resolve() / "models--Qwen--Qwen3-VL-4B-Instruct/snapshots" / inputs["revision"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    all_outcomes, all_summary, all_pairs, consistency = [], [], [], []
    for dataset in DATASETS:
        rows = [r for r in inputs["all_queries"] if r["dataset"] == dataset]
        verify_images(rows)
        job = job_identity(inputs, dataset, args.cache_dir.resolve())
        folder = source / dataset
        if read(folder / "identity.json") != job:
            raise ValueError("Capture job differs from audited source/model/code")
        verification = read(folder / "verification.json")
        if (verification.get("status") != "all_complete_verified" or verification.get("records") != 1024
                or verification.get("queries") != 256 or verification.get("job_hash") != job["job_hash"]
                or verification.get("outputs_sha256") != digest(folder / "predictions.jsonl")):
            raise ValueError("Only complete verified dataset arms may be analyzed")
        records = validate_prefix(folder / "predictions.jsonl", schedule(rows), job["job_hash"])
        if len(records) != 1024:
            raise ValueError("Incomplete intervention prefix")
        if (not verification.get("runtime_sha256")
                or verification.get("generated") != sum(r["status"] == "generated" for r in records)
                or verification.get("explicit_missing") != sum(r["status"] != "generated" for r in records)):
            raise ValueError("Missing runtime or changed verified generation denominators")
        for name, expected in verification["runtime_sha256"].items():
            path = folder / name
            if path.parent != folder or digest(path) != expected:
                raise ValueError("Runtime receipt changed")
            r = read(path)
            if r.get("job_hash") != job["job_hash"] or not str(r.get("torch", "")).startswith("2.8.0") or r.get("transformers") != "5.13.0":
                raise ValueError("Wrong pinned runtime")
        audit_generated_tokens(records, rows, tokenizer)
        values = make_tables(rows, records, {r["query_id"]: free[(dataset, r["query_id"])] for r in rows})
        all_outcomes.extend(values[0]); all_summary.extend(values[1]); all_pairs.extend(values[2]); consistency.append(values[3])
        for path in (folder / "identity.json", folder / "verification.json", folder / "predictions.jsonl", *folder.glob("runtime_*.json")):
            provenance[path.relative_to(ROOT).as_posix()] = digest(path)
    for path in (args.inputs_lock, args.base_plan, args.p0_query_csv, Path(__file__)):
        provenance[path.resolve().relative_to(ROOT).as_posix()] = digest(path)
    for rel, expected in provenance.items():
        if digest(ROOT / rel) != expected:
            raise ValueError("Source changed during analysis")
    out.mkdir(parents=True)
    a.csv_out(out / "query_arm_outcomes.csv", all_outcomes)
    a.csv_out(out / "arm_summary.csv", all_summary)
    a.csv_out(out / "paired_label_effects.csv", all_pairs)
    a.csv_out(out / "identical_label_consistency.csv", consistency)
    from collect_behavior_priority_zero import atomic
    atomic(out / "audit.json", {"status": "all_five_complete_verified_CPU_analysis", "source_sha256": provenance,
            "queries": 1280, "records": len(all_outcomes), "attachment_hash": inputs["attachment_hash"],
            "primary_zero_reference": "new same-environment constrained zero",
            "free_zero": "original P0 sensitivity only, not mixed denominator",
            "pairs": "same query/images/order with gold vs own-legal or deterministic wrong demo labels; missingness explicit",
            "CI": "10000 query paired bootstrap seed73, descriptive no multiplicity adjustment",
            "causal_scope": "fixed-context label content; not unique causal value of same-failure retrieval",
            "identical_labels": consistency, "GPU_execution_in_analysis": False})
    print(json.dumps({"status": "all_five_complete_verified_CPU_analysis", "queries": 1280,
                      "records": len(all_outcomes), "output_root": str(out)}))


if __name__ == "__main__":
    main()

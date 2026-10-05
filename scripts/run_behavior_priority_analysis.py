"""CPU-only P0/P1/P3 reuse audit. Does not launch models or infer mechanisms.

Sources are explicit, full official test groups. Output files are derived only;
source artifacts, scientific identities and historical RESULTS are never edited.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("dtd", "aircraft", "cub", "pets", "dogs")
METHODS = ("RICES", "GPT-MM", "DeTriever", "BehaviorICL-Raw",
           "BehaviorICL-Proj", "BehaviorICL-Learn")
MODEL = "Qwen/Qwen3-VL-4B-Instruct"
COHORT = "existing_fixed256_fullbank"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(manifest):
    return {split: [(s["sample_id"], s["label"], s["split"]) for s in manifest[split]]
            for split in ("bank", "query")}


def rate(n, d):
    return n / d if d else None


def wilson(n, d):
    if not d:
        return None, None
    z = 1.959963984540054
    p = n / d
    den = 1 + z * z / d
    mid = (p + z * z / (2 * d)) / den
    half = z * math.sqrt(p * (1 - p) / d + z * z / (4 * d * d)) / den
    return max(0., mid - half), min(1., mid + half)


def csv_out(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    if not fields:
        raise ValueError("Explicit columns required for an empty table")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict))
                             else v for k, v in r.items()})


def source_catalog():
    cloud = ROOT / "results/cloud_trial"
    out = {}
    for d in DATASETS:
        base = (ROOT / f"results/cdr_main/{d}/qwen3vl4b/seed_73" if d in ("dtd", "aircraft")
                else cloud / f"jiirguh96xew6j-four-methods/{d}/seed_73")
        behavior = (cloud / "jiirguh96xew6j-decision/dtd/seed_73" if d == "dtd"
                    else cloud / "nwbxy265c6vqvw-aircraft/decision/aircraft/qwen3vl4b/seed_73"
                    if d == "aircraft" else cloud / f"pbzipnh9i4d180-decision/{d}/seed_73")
        proj = (cloud / "kt3efjpk5c3qmz-behavior-ablation/dtd/seed_73" if d == "dtd"
                else cloud / "3gddhpel2b38g4-behavior-resume/aircraft/seed_73" if d == "aircraft"
                else cloud / f"kp5b5tkcqbi547-behavior-complete/snapshot/results/behavior_ablation/{d}/qwen3vl4b/seed_73")
        out[d] = {"base": base, "behavior": behavior, "proj": proj,
                  "detriever": cloud / f"jiirguh96xew6j-detriever/{d}/seed_73"}
    return out


def load_group(folder, method, manifest, provenance):
    other = read(folder / "manifest.json")
    if identity(other) != identity(manifest):
        raise ValueError(f"Official bank/query ID/label/split/order mismatch: {folder}")
    for split in ("bank", "query"):
        ids = [s["sample_id"] for s in other[split]]
        if len(ids) != len(set(ids)) or any(s["split"] != split for s in other[split]):
            raise ValueError("Duplicate sample or split mismatch")
    bank = {s["sample_id"]: s for s in manifest["bank"]}
    query = {s["sample_id"]: s for s in manifest["query"]}
    selections = read(folder / "selections.json")[method]
    rows = [r for r in jsonl(folder / "predictions.jsonl") if r["method"] == method]
    if len(rows) != len(query) or set(r["query_id"] for r in rows) != set(query):
        raise ValueError(f"Missing/duplicate full-test rows: {folder} {method}")
    if set(selections) != set(query):
        raise ValueError("Incomplete selections")
    legal = {s["label"] for s in manifest["bank"]}
    for r in rows:
        q = query[r["query_id"]]
        chosen = selections[r["query_id"]]
        if r["target"] != q["label"] or r["correct"] != (r["prediction"] == q["label"]):
            raise ValueError("Target/correctness mismatch")
        if r["prediction"] not in legal:
            raise ValueError("Illegal ICL label")
        if len(r["demo_ids"]) != 4 or len(set(r["demo_ids"])) != 4:
            raise ValueError("Not four unique demonstrations")
        if r["demo_ids"] != [s["sample_id"] for s in chosen]:
            raise ValueError("Demonstration order mismatch")
        if r["demo_labels"] != [bank[i]["label"] for i in r["demo_ids"]]:
            raise ValueError("Bank label mismatch")
        if any((s["label"], s["split"]) != (bank[s["sample_id"]]["label"], "bank")
               for s in chosen):
            raise ValueError("Selection outside bank")
    for f in ("manifest.json", "selections.json", "predictions.jsonl", "protocol.json", "run_identity.json"):
        if (folder / f).exists():
            provenance[str((folder / f).relative_to(ROOT))] = sha(folder / f)
    return {r["query_id"]: r for r in rows}


def load_zero(folder, manifest, provenance):
    path = folder / "gpt_mm_zero_shot_outputs.jsonl"
    if not path.exists():
        return None
    if identity(read(folder / "manifest.json")) != identity(manifest):
        raise ValueError("Zero-shot source manifest mismatch")
    samples = manifest["bank"] + manifest["query"]
    raw = jsonl(path)
    if len(raw) != len(samples) or [r["sample_id"] for r in raw] != [s["sample_id"] for s in samples]:
        raise ValueError("Zero-shot cache missing/duplicate/changed order")
    labels = {s["label"].casefold(): s["label"] for s in manifest["bank"]}
    provenance[str(path.relative_to(ROOT))] = sha(path)
    return {s["sample_id"]: {"sample_id": s["sample_id"], "true_label": s["label"],
            "split": s["split"], "raw_zero_shot_answer": r["raw_zero_shot_answer"],
            "zero_shot_pred": labels.get(r["raw_zero_shot_answer"].strip().casefold()),
            "zero_shot_valid": r["raw_zero_shot_answer"].strip().casefold() in labels,
            "zero_shot_correct": labels.get(r["raw_zero_shot_answer"].strip().casefold()) == s["label"],
            "decoding": "unconstrained_greedy_max_new_tokens12"}
            for s, r in zip(samples, raw)}


def pair_features(target, pred, demo_labels, demo_preds):
    wrong = pred is not None and pred != target
    strict = sum(wrong and label == target and p is not None and p == pred
                 for label, p in zip(demo_labels, demo_preds))
    weak = sum(wrong and p is not None and p == pred for p in demo_preds)
    return strict, weak


def majority(labels, scores=None):
    votes = Counter(labels)
    tied = [c for c, n in votes.items() if n == max(votes.values())]
    if len(tied) == 1:
        return tied[0]
    if scores is None:
        return None
    sums = {c: sum(s for label, s in zip(labels, scores) if label == c) for c in tied}
    tied = [c for c in tied if sums[c] == max(sums.values())]
    # Original demo order is lowest->highest similarity. Highest rank wins exact ties.
    return next(label for label in reversed(labels) if label in tied)


def weighted(labels, scores, temperature=.07):
    top = max(scores)
    sums = defaultdict(float)
    for label, s in zip(labels, scores):
        sums[label] += math.exp((s - top) / temperature)
    tied = {c for c, s in sums.items() if s == max(sums.values())}
    return next(label for label in reversed(labels) if label in tied)


def expected_sameclass(bank_n, bank_matches, retrieved_n):
    if not 0 <= bank_matches <= bank_n or retrieved_n > bank_n:
        raise ValueError("Invalid bank support/count")
    any_match = (1 - math.comb(bank_n - bank_matches, retrieved_n) / math.comb(bank_n, retrieved_n)
                 if retrieved_n else 0.)
    return retrieved_n * bank_matches / bank_n if bank_n else 0., any_match


def summarize(rows):
    n = len(rows)
    matched = [r for r in rows if r["num_strict_failure_matches"] > 0]
    unmatched = [r for r in rows if r["num_strict_failure_matches"] == 0]
    ci = wilson(len(matched), n)
    return {"zero_wrong_queries": n, "retrieved_demos": 4 * n,
            "strict_failure_match_demos": sum(r["num_strict_failure_matches"] for r in rows),
            "strict_failure_match_rate": rate(sum(r["num_strict_failure_matches"] for r in rows), 4*n),
            "failure_match_at4_count": len(matched), "failure_match_at4": rate(len(matched), n),
            "failure_match_at4_wilson_low": ci[0], "failure_match_at4_wilson_high": ci[1],
            "same_wrong_prediction_rate": rate(sum(r["num_same_wrong_prediction"] for r in rows), 4*n),
            "same_true_label_demo_rate": rate(sum(r["num_same_true_label"] for r in rows), 4*n),
            "matched_rescued": sum(r["rescued"] for r in matched), "matched_queries": len(matched),
            "rescue_with_match": rate(sum(r["rescued"] for r in matched), len(matched)),
            "unmatched_rescued": sum(r["rescued"] for r in unmatched), "unmatched_queries": len(unmatched),
            "rescue_without_match": rate(sum(r["rescued"] for r in unmatched), len(unmatched)),
            "rescue_rate": rate(sum(r["rescued"] for r in rows), n),
            "observed_strict_matches_per_query": rate(sum(r["num_strict_failure_matches"] for r in rows), n),
            "expected_strict_matches_at_sameclass_count": rate(sum(r["expected_strict_matches"] for r in rows), n),
            "expected_failure_at4_at_sameclass_count": rate(sum(r["expected_failure_at4"] for r in rows), n)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "analysis/priority_experiments")
    args = parser.parse_args()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT / "analysis"):
        raise ValueError("Derived outputs must stay under workspace analysis/")
    if out.exists() and (out / "audit.json").exists():
        raise ValueError("Use a fresh output root; do not overwrite completed analysis")
    out.mkdir(parents=True, exist_ok=True)
    provenance, accuracy, coverage, all_query_rows, summaries, counts, confusion = {}, [], [], [], [], [], []
    knn_summary, knn_rows, controls, bank_support = [], [], [], []
    for d, folders in source_catalog().items():
        manifest = read(folders["base"] / "manifest.json")
        specs = {"RICES": ("base", "rices"), "GPT-MM": ("base", "gpt_mm"),
                 "DeTriever": ("detriever", "detriever"), "BehaviorICL-Raw": ("behavior", "decision_zero"),
                 "BehaviorICL-Learn": ("behavior", "decision_learn"), "BehaviorICL-Proj": ("proj", "projection_only")}
        groups = {name: load_group(folders[key], internal, manifest, provenance)
                  for name, (key, internal) in specs.items()}
        for name in METHODS:
            rows = list(groups[name].values())
            accuracy.append({"dataset": d, "model": MODEL, "cohort": COHORT, "method": name,
                             "correct": sum(r["correct"] for r in rows), "total": len(rows),
                             "accuracy": rate(sum(r["correct"] for r in rows), len(rows)), "seed": 73,
                             "dimension": 256 if name in ("BehaviorICL-Proj", "BehaviorICL-Learn") else None,
                             "status": "complete_existing_predictions", "retrieval_seconds": None,
                             "icl_seconds": None, "timing_status": "not_separately_available",
                             "source_folder": str(folders[specs[name][0]].relative_to(ROOT))})
        for representation in ("SupCon-CLIP", "SupCon-QwenVision", "Raw QwenVision"):
            controls.append({"dataset": d, "model": MODEL, "method": representation,
                             "correct": None, "total": len(manifest["query"]), "accuracy": None,
                             "status": "missing_representation_matched_protocol_evaluation"})
        for name in ("BehaviorICL-Raw", "BehaviorICL-Proj", "BehaviorICL-Learn"):
            local = []
            for r in groups[name].values():
                vote = majority(r["demo_labels"])
                local.append({"dataset": d, "model": MODEL, "method": name, "cohort": COHORT,
                              "query_id": r["query_id"], "true_label": r["target"],
                              "retrieved_ids": r["demo_ids"], "retrieved_true_labels": r["demo_labels"],
                              "knn_pred": vote, "knn_correct": None if vote is None else vote == r["target"],
                              "icl_pred": r["prediction"], "icl_correct": r["correct"],
                              "num_unique_demo_labels": len(set(r["demo_labels"])),
                              "similarity_scores": None, "weighted_knn_pred": None,
                              "score_status": "not_saved_in_original_selections"})
            knn_rows.extend(local)
            for group in ("all", "all_four_same_label", "multiple_labels"):
                sub = [r for r in local if group == "all" or
                       (r["num_unique_demo_labels"] == 1 if group == "all_four_same_label"
                        else r["num_unique_demo_labels"] > 1)]
                known = [r for r in sub if r["knn_pred"] is not None]
                wrong = [r for r in known if not r["knn_correct"]]
                unresolved = len(sub)-len(known)
                knn_summary.append({"dataset": d, "model": MODEL, "retriever": name, "group": group,
                    "queries": len(sub), "resolved_majority_queries": len(known), "unresolved_vote_ties": unresolved,
                    "knn_correct_on_resolved": sum(r["knn_correct"] for r in known),
                    "knn_accuracy_on_resolved": rate(sum(r["knn_correct"] for r in known), len(known)),
                    "knn_full_accuracy": None if unresolved else rate(sum(r["knn_correct"] for r in known), len(sub)),
                    "weighted_knn_full_accuracy": None,
                    "icl_correct": sum(r["icl_correct"] for r in sub), "icl_accuracy": rate(sum(r["icl_correct"] for r in sub),len(sub)),
                    "knn_wrong_icl_correct": sum(r["icl_correct"] for r in wrong), "knn_wrong_queries": len(wrong),
                    "icl_correct_given_knn_wrong": rate(sum(r["icl_correct"] for r in wrong),len(wrong)),
                    "knn_correct_icl_wrong": sum(r["knn_correct"] and not r["icl_correct"] for r in known),
                    "both_correct": sum(r["knn_correct"] and r["icl_correct"] for r in known),
                    "both_wrong": sum(not r["knn_correct"] and not r["icl_correct"] for r in known),
                    "status": "complete_majority_group" if not unresolved else "partial_unambiguous_majority_only",
                    "weighted_status": "missing_original_similarity_scores"})
        zero = load_zero(folders["base"], manifest, provenance)
        samples = manifest["bank"] + manifest["query"]
        if zero is None:
            coverage.append({"dataset": d, "bank_total": len(manifest["bank"]), "query_total": len(manifest["query"]),
                             "zero_rows": 0, "status": "missing_full_bank_and_test_zero_cache"})
            for name in METHODS:
                summaries.append({"dataset": d, "model": MODEL, "cohort": COHORT, "method": name,
                                  "status": "missing_full_bank_and_test_zero_cache"})
            continue
        csv_out(out / f"zero_shot/{d}.csv", list(zero.values()))
        legal_wrong = {s["sample_id"] for s in manifest["query"]
                       if zero[s["sample_id"]]["zero_shot_valid"] and not zero[s["sample_id"]]["zero_shot_correct"]}
        common = {qid for qid in legal_wrong if all(all(zero[i]["zero_shot_valid"] for i in groups[name][qid]["demo_ids"])
                                                   for name in METHODS)}
        coverage.append({"dataset": d, "bank_total":len(manifest["bank"]), "query_total":len(manifest["query"]),
                         "zero_rows":len(zero), "invalid_bank_zero":sum(not zero[s["sample_id"]]["zero_shot_valid"] for s in manifest["bank"]),
                         "invalid_query_zero":sum(not zero[s["sample_id"]]["zero_shot_valid"] for s in manifest["query"]),
                         "valid_wrong_queries":len(legal_wrong), "common_complete_case_wrong_queries":len(common),
                         "excluded_valid_wrong_queries":len(legal_wrong)-len(common),
                         "status":"preliminary_unconstrained_zero_complete_case_analysis"})
        valid_by_class = Counter(zero[s["sample_id"]]["true_label"] for s in manifest["bank"] if zero[s["sample_id"]]["zero_shot_valid"])
        match_support = Counter((zero[s["sample_id"]]["true_label"],zero[s["sample_id"]]["zero_shot_pred"])
                                for s in manifest["bank"] if zero[s["sample_id"]]["zero_shot_valid"])
        for (y,pred), support in match_support.items():
            bank_support.append({"dataset":d,"true_label":y,"zero_shot_pred":pred,"valid_bank_in_class":valid_by_class[y],
                                 "bank_confusion_pair_support":support})
        for name in METHODS:
            local = []
            for s in manifest["query"]:
                qid = s["sample_id"]; z = zero[qid]; r = groups[name][qid]
                demo_preds = [zero[i]["zero_shot_pred"] for i in r["demo_ids"]]
                strict, weak = pair_features(s["label"], z["zero_shot_pred"], r["demo_labels"], demo_preds)
                same = sum(label == s["label"] for label in r["demo_labels"])
                exp, exp_any = expected_sameclass(valid_by_class[s["label"]],
                                                  match_support[(s["label"],z["zero_shot_pred"])], same)
                local.append({"dataset":d,"model":MODEL,"cohort":COHORT,"query_id":qid,"true_label":s["label"],
                    "zero_shot_pred":z["zero_shot_pred"],"zero_shot_correct":z["zero_shot_correct"],
                    "zero_shot_valid":z["zero_shot_valid"],"raw_zero_shot_answer":z["raw_zero_shot_answer"],
                    "method":name,"retrieved_ids":r["demo_ids"],"retrieved_true_labels":r["demo_labels"],
                    "retrieved_zero_shot_preds":demo_preds,"num_strict_failure_matches":strict,
                    "num_same_wrong_prediction":weak,"num_same_true_label":same,
                    "num_invalid_demo_zero":sum(p is None for p in demo_preds),
                    "icl_pred":r["prediction"],"icl_correct":r["correct"],
                    "rescued":not z["zero_shot_correct"] and r["correct"],
                    "harmed":z["zero_shot_correct"] and not r["correct"],
                    "eligible_common_complete_case_wrong":qid in common,
                    "expected_strict_matches":exp,"expected_failure_at4":exp_any,
                    "bank_valid_sameclass":valid_by_class[s["label"]],
                    "bank_confusion_pair_support":match_support[(s["label"],z["zero_shot_pred"])],
                    "similarity_scores":None,"zero_decoding":z["decoding"]})
            all_query_rows.extend(local)
            wrong = [r for r in local if r["eligible_common_complete_case_wrong"]]
            summaries.append({"dataset":d,"model":MODEL,"cohort":COHORT,"method":name,**summarize(wrong),
                              "zero_query_correct":sum(zero[s["sample_id"]]["zero_shot_correct"] for s in manifest["query"]),
                              "query_total":len(manifest["query"]),
                              "all_zero_errors_rescued":sum(r["rescued"] for r in local),
                              "all_zero_correct_harmed":sum(r["harmed"] for r in local),
                              "status":"preliminary_common_legal_wrong_queries_unconstrained_zero"})
            for k in range(5):
                sub = [r for r in wrong if r["num_strict_failure_matches"] == k]
                lo,hi = wilson(sum(r["rescued"] for r in sub),len(sub))
                counts.append({"dataset":d,"method":name,"num_strict_failure_matches":k,"queries":len(sub),
                               "rescued":sum(r["rescued"] for r in sub),"rescue_rate":rate(sum(r["rescued"] for r in sub),len(sub)),
                               "wilson_low":lo,"wilson_high":hi})
            pairs = defaultdict(list)
            for r in wrong:
                pairs[(r["true_label"],r["zero_shot_pred"])].append(r)
            for (y,pred), sub in sorted(pairs.items()):
                if len(sub) >= 10:
                    confusion.append({"dataset":d,"method":name,"true_label":y,"zero_shot_pred":pred,
                                      **summarize(sub)})
    csv_out(out/"behavior_raw_results.csv",accuracy)
    csv_out(out/"metric_learning_control_results.csv",controls)
    csv_out(out/"knn_vs_icl_results.csv",knn_summary)
    csv_out(out/"knn_query_level_results.csv",knn_rows)
    csv_out(out/"failure_matching_summary.csv",summaries)
    csv_out(out/"failure_matching/query_level_results.csv",all_query_rows)
    csv_out(out/"failure_matching/method_summary.csv",summaries)
    csv_out(out/"failure_matching/confusion_pair_summary.csv",confusion, list(confusion[0]) if confusion else ["dataset","method","true_label","zero_shot_pred","zero_wrong_queries"])
    csv_out(out/"failure_matching/rescue_by_num_failure_matches.csv",counts)
    csv_out(out/"failure_matching/bank_confusion_support.csv",bank_support)
    csv_out(out/"failure_matching/coverage.csv",coverage)
    strata = defaultdict(list)
    for r in all_query_rows:
        if r["eligible_common_complete_case_wrong"]:
            strata[(r["dataset"],r["method"],r["num_same_true_label"],r["num_strict_failure_matches"])].append(r)
    csv_out(out/"failure_matching/rescue_by_sameclass_and_failure_count.csv",[
        {"dataset":d,"method":m,"num_same_true_label":same,"num_strict_failure_matches":fail,"queries":len(rs),
         "rescued":sum(r["rescued"] for r in rs),"rescue_rate":rate(sum(r["rescued"] for r in rs),len(rs))}
        for (d,m,same,fail),rs in sorted(strata.items())])
    comparison = []
    for d in DATASETS:
        row = {"dataset":d,"model":MODEL,"cohort":COHORT}
        for name in METHODS:
            row[name] = next(r["accuracy"] for r in accuracy if r["dataset"]==d and r["method"]==name)
        row.update({"SupCon-QwenVision":None,"SupCon-CLIP":None,"Behavior-kNN":None})
        comparison.append(row)
    csv_out(out/"summary_table.csv",comparison)
    audit = {"experiment":"behavior_priority_existing_evidence_v1","source_sha256":provenance,
             "cohort":COHORT,"datasets":list(DATASETS),"methods":list(METHODS),"seed":73,
             "zero_cache_scope":["dtd","aircraft"],"source_cache_byte_equality_across_runs":"not_asserted",
             "zero_prompt_source":"multimodal_model_adapter.zero_shot_messages; same helper used for anchor extraction",
             "zero_decoding":"existing unconstrained greedy max_new_tokens12; NOT constrained ICL decoding",
             "invalid_zero_policy":"exact trim/casefold only; no semantic repair; preserve raw answers",
             "P0_primary_denominator":"common valid-wrong queries with four legal demo zero predictions for ALL six methods",
             "P0_full_requirement":"not complete: invalid class outputs and three missing full zero caches",
             "sameclass_reference":"uniform without-replacement draw among legally predicted bank items of true class, conditional on sameclass retrieved count",
             "P3_missing_scores":"Do not invent summed-similarity ties or weighted voting; report unambiguous majority subsets only",
             "P2":"not run; no matched external/vision controls available",
             "second_model":"no complete second-model representations found in chosen verified sources",
             "runtime":"CPU only; no GPU/model generation, training, resource mutation, or test tuning",
             "interpretation":"numeric tables only; observational association is not a corrective-label intervention"}
    (out/"audit.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output_root":str(out),"complete_accuracy_groups":len(accuracy),
                      "P0_datasets":len(coverage)-sum(r["zero_rows"]==0 for r in coverage),
                      "query_level_rows":len(all_query_rows),"knn_rows":len(knn_rows),
                      "coverage":coverage},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()

"""CPU-only repair audit on ALL official test queries, with missingness explicit.

This is a new derived observational analysis, not a change to retrieval, zero
generation, or the historical common-complete-case table. No model is executed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_behavior_priority_analysis as a
from analyze_behavior_paper_p0 import load_complete_new

PROXIES = ("strict", "weak", "same_prediction_wrong_demo", "sameclass_other_error", "sameclass")
SPEC = {"RICES": ("base", "rices"), "GPT-MM": ("base", "gpt_mm"),
        "DeTriever": ("detriever", "detriever"), "BehaviorICL-Raw": ("behavior", "decision_zero"),
        "BehaviorICL-Proj": ("proj", "projection_only"), "BehaviorICL-Learn": ("behavior", "decision_learn")}


def failure_features(target, query_pred, demo_labels, demo_preds):
    """Observed positive, known absence, or unknown; never match None to None.

    Absence for behavior-dependent proxies is conservatively known only when
    query and all four demo predictions are legal. Presence can be established
    by one legal demo despite another illegal answer. True-class presence uses
    the actual known bank labels, so it remains observable for illegal queries.
    """
    if len(demo_labels) != 4 or len(demo_preds) != 4:
        raise ValueError("Four actual demonstrations required")
    wrong = query_pred is not None and query_pred != target
    counts = {
        "strict": sum(wrong and p is not None and y == target and p == query_pred
                      for y, p in zip(demo_labels, demo_preds)),
        "weak": sum(wrong and p is not None and p == query_pred for p in demo_preds),
        "same_prediction_wrong_demo": sum(wrong and p is not None and p == query_pred and p != y
                                          for y, p in zip(demo_labels, demo_preds)),
        "sameclass_other_error": sum(wrong and p is not None and y == target and p != y and p != query_pred
                                     for y, p in zip(demo_labels, demo_preds)),
        "sameclass": sum(y == target for y in demo_labels),
    }
    full_legal = wrong and all(p is not None for p in demo_preds)
    result = {"num_invalid_demo_zero": sum(p is None for p in demo_preds)}
    for proxy, count in counts.items():
        status = "present" if count else "absent" if (full_legal or proxy == "sameclass") else "unknown"
        if query_pred == target and proxy != "sameclass":
            status = "not_initial_error"
        result[proxy + "_observed_count"] = count
        result[proxy + "_status"] = status
    return result


def build_rows(dataset, manifest, zero, groups):
    queries = manifest["query"]
    qids = {q["sample_id"] for q in queries}
    if set(groups) != set(a.METHODS) or any(set(g) != qids for g in groups.values()):
        raise ValueError("All six methods must cover the identical full official test")
    common = {q["sample_id"] for q in queries if zero[q["sample_id"]]["zero_shot_valid"]
              and not zero[q["sample_id"]]["zero_shot_correct"]
              and all(all(zero[i]["zero_shot_valid"] for i in groups[m][q["sample_id"]]["demo_ids"])
                      for m in a.METHODS)}
    bank_valid = Counter(s["label"] for s in manifest["bank"] if zero[s["sample_id"]]["zero_shot_valid"])
    bank_pairs = Counter((s["label"], zero[s["sample_id"]]["zero_shot_pred"])
                         for s in manifest["bank"] if zero[s["sample_id"]]["zero_shot_valid"])
    rows = []
    for m in a.METHODS:
        for q in queries:
            qid = q["sample_id"]; z = zero[qid]; r = groups[m][qid]
            preds = [zero[i]["zero_shot_pred"] for i in r["demo_ids"]]
            f = failure_features(q["label"], z["zero_shot_pred"], r["demo_labels"], preds)
            expected, expected_any = a.expected_sameclass(
                bank_valid[q["label"]], bank_pairs[(q["label"], z["zero_shot_pred"])], f["sameclass_observed_count"])
            rows.append({"dataset": dataset, "method": m, "query_id": qid, "true_label": q["label"],
                         "zero_shot_pred": z["zero_shot_pred"], "raw_zero_shot_answer": z["raw_zero_shot_answer"],
                         "zero_shot_valid": z["zero_shot_valid"], "zero_shot_correct": z["zero_shot_correct"],
                         "initial_error": not z["zero_shot_correct"],
                         "demo_ids": r["demo_ids"], "demo_labels": r["demo_labels"], "demo_zero_preds": preds,
                         "icl_pred": r["prediction"], "icl_correct": r["correct"],
                         "rescued": not z["zero_shot_correct"] and r["correct"],
                         "harmed": z["zero_shot_correct"] and not r["correct"],
                         "eligible_common_complete_case_wrong": qid in common,
                         "num_strict_failure_matches": f["strict_observed_count"],
                         "num_same_wrong_prediction": f["weak_observed_count"],
                         "num_same_true_label": f["sameclass_observed_count"],
                         "expected_strict_matches": expected, "expected_failure_at4": expected_any,
                         "zero_decoding": z["decoding"], **f})
    return rows


def outcome_summary(rows):
    n = len(rows); rescued = sum(r["rescued"] for r in rows)
    lo, hi = a.wilson(rescued, n)
    return {"queries": n, "rescued": rescued, "rescue_rate": a.rate(rescued, n),
            "wilson_low": lo, "wilson_high": hi}


def method_summary(rows):
    errors = [r for r in rows if r["initial_error"]]
    correct = [r for r in rows if not r["initial_error"]]
    legal = [r for r in errors if r["zero_shot_valid"]]
    invalid = [r for r in errors if not r["zero_shot_valid"]]
    repaired = sum(r["rescued"] for r in errors); harmed = sum(r["harmed"] for r in correct)
    final = sum(r["icl_correct"] for r in rows)
    if len(correct) - harmed + repaired != final:
        raise ValueError("Repair/harm decomposition does not equal full-test ICL correctness")
    result = {"test_queries": len(rows), "zero_correct": len(correct), "zero_errors": len(errors),
              "legal_zero_errors": len(legal), "illegal_zero_errors": len(invalid),
              "rescued": repaired, "rescue_rate": a.rate(repaired, len(errors)),
              "legal_error_rescued": sum(r["rescued"] for r in legal),
              "illegal_error_rescued": sum(r["rescued"] for r in invalid),
              "harmed": harmed, "harm_rate": a.rate(harmed, len(correct)),
              "icl_correct": final, "icl_accuracy": a.rate(final, len(rows))}
    result["rescue_wilson_low"], result["rescue_wilson_high"] = a.wilson(repaired, len(errors))
    result["harm_wilson_low"], result["harm_wilson_high"] = a.wilson(harmed, len(correct))
    for status in ("present", "absent", "unknown"):
        sub = [r for r in errors if r["strict_status"] == status]
        result["strict_" + status + "_queries"] = len(sub)
        result["strict_" + status + "_rescued"] = sum(r["rescued"] for r in sub)
    if sum(result["strict_" + s + "_queries"] for s in ("present", "absent", "unknown")) != len(errors):
        raise ValueError("Matching classification dropped an initial error")
    return result


def association_tables(rows):
    errors = [r for r in rows if r["initial_error"]]
    summaries, strata = [], []
    for scope, subset in (("all_initial_errors", errors),
                          ("known_no_strict_match", [r for r in errors if r["strict_status"] == "absent"]),
                          ("common_complete_case_wrong", [r for r in errors if r["eligible_common_complete_case_wrong"]])):
        for proxy in PROXIES:
            for status in ("present", "absent", "unknown"):
                sub = [r for r in subset if r[proxy + "_status"] == status]
                summaries.append({"scope": scope, "proxy": proxy, "status": status, **outcome_summary(sub)})
                for k in range(5):
                    fixed = [r for r in sub if r["num_same_true_label"] == k]
                    strata.append({"scope": scope, "proxy": proxy, "status": status,
                                   "num_same_true_label": k, **outcome_summary(fixed)})
    return summaries, strata


def paired_bootstrap(first, second, repetitions=10000, seed=73):
    """Multinomial resampling of paired win/loss/tie outcomes = query bootstrap."""
    import numpy as np
    if len(first) != len(second) or not first:
        raise ValueError("Nonempty same-query paired outcomes required")
    wins = sum(bool(x) and not bool(y) for x, y in zip(first, second))
    losses = sum(not bool(x) and bool(y) for x, y in zip(first, second))
    n = len(first); ties = n - wins - losses
    samples = np.random.default_rng(seed).multinomial(n, [wins/n, losses/n, ties/n], size=repetitions)
    lo, hi = np.quantile((samples[:, 0] - samples[:, 1]) * 100 / n, [.025, .975])
    return {"queries": n, "first_rescued": sum(first), "second_rescued": sum(second),
            "first_only_rescued": wins, "second_only_rescued": losses,
            "difference_pp": (wins-losses)*100/n, "paired_bootstrap_low_pp": float(lo),
            "paired_bootstrap_high_pp": float(hi), "bootstrap_repetitions": repetitions, "seed": seed}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zero-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--datasets", nargs="+", choices=a.DATASETS, default=list(a.DATASETS))
    args = p.parse_args(); out = args.output_root.resolve(); zero_root = args.zero_root.resolve()
    if not out.is_relative_to(ROOT / "analysis") or not zero_root.is_relative_to(ROOT / "results"):
        raise ValueError("Inputs/results and derived analysis must stay in the workspace")
    if out.exists():
        raise ValueError("Fresh output directory required; no overwrite")
    if len(args.datasets) != len(set(args.datasets)):
        raise ValueError("Duplicate dataset requested")
    sources = a.source_catalog(); provenance = {}; data = {}; specs = {}
    old_hashes = a.read(ROOT / "analysis/priority_experiments/audit.json")["source_sha256"]
    # Pre-audit every requested COMPLETE dataset before creating any output.
    for d in args.datasets:
        folders = sources[d]; manifest = a.read(folders["base"] / "manifest.json")
        groups = {m: a.load_group(folders[key], internal, manifest, provenance)
                  for m, (key, internal) in SPEC.items()}
        if d in ("dtd", "aircraft"):
            zero = a.load_zero(folders["base"], manifest, provenance)
            if zero is None:
                raise ValueError("Missing full original zero cache")
        else:
            from collect_behavior_priority_zero import prepare, digest
            samples, _, _, expected = prepare(d, ROOT / "datasets", ROOT / ".hf_cache")
            images = {s.sample_id: digest(s.path) for s in samples}
            zero = load_complete_new(zero_root / d, manifest, expected, images, provenance)
            specs[d] = {"rows": len(samples), "job_hash": expected["job_hash"],
                        "all_official_image_sha_checked": True}
        data[d] = (manifest, zero, groups)
        print("Complete official test + six ICL groups + full zero cache audited: " + d, flush=True)
    original_checked = 0
    for path, current in provenance.items():
        if path in old_hashes:
            if old_hashes[path] != current:
                raise ValueError("Original evidence SHA changed: " + path)
            original_checked += 1
        elif not path.startswith(str(zero_root.relative_to(ROOT)).replace("\\", "/")) and "behavior_priority_zero_v1" not in path:
            raise ValueError("Original evidence not registered in initial audit: " + path)
    full_rows, coverage, summaries, common_summaries, associations, strata, paired = [], [], [], [], [], [], []
    for d, (manifest, zero, groups) in data.items():
        rows = build_rows(d, manifest, zero, groups); full_rows.extend(rows)
        qzero = [zero[q["sample_id"]] for q in manifest["query"]]
        common_n = sum(r["eligible_common_complete_case_wrong"] for r in rows if r["method"] == a.METHODS[0])
        coverage.append({"dataset": d, "bank_total": len(manifest["bank"]), "query_total": len(manifest["query"]),
                         "zero_rows": len(zero), "invalid_bank_zero": sum(not zero[s["sample_id"]]["zero_shot_valid"] for s in manifest["bank"]),
                         "invalid_query_zero": sum(not z["zero_shot_valid"] for z in qzero),
                         "all_initial_errors": sum(not z["zero_shot_correct"] for z in qzero),
                         "valid_wrong_queries": sum(z["zero_shot_valid"] and not z["zero_shot_correct"] for z in qzero),
                         "common_complete_case_wrong_queries": common_n,
                         "excluded_valid_wrong_queries": sum(z["zero_shot_valid"] and not z["zero_shot_correct"] for z in qzero)-common_n})
        error_ids = None; by_method = {}
        for m in a.METHODS:
            sub = [r for r in rows if r["method"] == m]; by_method[m] = sub
            ids = [r["query_id"] for r in sub if r["initial_error"]]
            if error_ids is not None and ids != error_ids:
                raise ValueError("Method-specific error cohort detected")
            error_ids = ids
            summaries.append({"dataset": d, "method": m, **method_summary(sub)})
            common_summaries.append({"dataset": d, "method": m,
                                     **a.summarize([r for r in sub if r["eligible_common_complete_case_wrong"]])})
            assoc, fixed = association_tables(sub)
            associations.extend({"dataset": d, "method": m, **r} for r in assoc)
            strata.extend({"dataset": d, "method": m, **r} for r in fixed)
        for m in ("BehaviorICL-Raw", "BehaviorICL-Proj", "BehaviorICL-Learn"):
            for baseline in ("RICES", "GPT-MM", "DeTriever"):
                first = [r["rescued"] for r in by_method[m] if r["initial_error"]]
                second = [r["rescued"] for r in by_method[baseline] if r["initial_error"]]
                paired.append({"dataset": d, "first_method": m, "second_method": baseline,
                               **paired_bootstrap(first, second)})
    a.csv_out(out / "query_level_results.csv", full_rows)
    a.csv_out(out / "coverage.csv", coverage)
    a.csv_out(out / "all_error_repair_summary.csv", summaries)
    a.csv_out(out / "common_case_method_summary.csv", common_summaries)
    a.csv_out(out / "match_associations.csv", associations)
    a.csv_out(out / "fixed_sameclass_count_associations.csv", strata)
    a.csv_out(out / "paired_all_error_differences.csv", paired)
    audit = {"experiment": "behavior_all_official_query_repair_v1", "datasets": args.datasets,
             "full_five_dataset_P0": set(args.datasets) == set(a.DATASETS), "methods": list(a.METHODS),
             "source_sha256": provenance, "original_source_sha_rechecked": original_checked,
             "new_complete_zero_cache_specs": specs, "analysis_script_sha256": a.sha(Path(__file__)),
             "imported_analysis_sha256": a.sha(ROOT / "scripts/run_behavior_priority_analysis.py"),
             "imported_complete_validator_sha256": a.sha(ROOT / "scripts/analyze_behavior_paper_p0.py"),
             "primary_repair_denominator": "all official zero-shot errors including illegal outputs; identical ordered IDs for six methods",
             "all_official_queries_retained": True, "matching_policy": "observed present / known absent / unknown; illegal query behavior remains unknown, never None==None",
             "unknown_demo_policy": "one legal match proves presence; absence conservatively requires all four legal demo answers; no query removed from repair",
             "sensitivity_analysis": "original common legal wrong query + all four demo outputs legal for all six methods; definition unchanged",
             "strict_definition": "same query/demo true class and same legal wrong prediction",
             "weak_definition": "same legal prediction; demo need not itself be wrong",
             "same_prediction_wrong_demo_definition": "same legal prediction AND demo itself wrong; not necessarily query true class",
             "sameclass_other_error_definition": "same true class AND demo itself wrong with a different legal predicted class; does NOT establish semantic similarity of errors",
             "sameclass_definition": "at least one demo gold class equals query gold class; query labels used for analysis only",
             "paired_CI": "10000 paired-query percentile bootstrap, seed73; categorical multinomial is equivalent to resampling paired win/loss/tie outcomes; descriptive, no multiplicity adjustment",
             "decoding_confound": "free greedy max12 zero vs constrained ICL; correct-label intervention and constrained zero still required",
             "runtime_equivalence_across_old_and_new_caches": "not asserted",
             "interpretation": "retrospective observation; association not causal mediation or a Behavior-unique corrective mechanism",
             "runtime": "CPU only; no generation, training, source mutation, query selection, or test tuning"}
    (out / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_root": str(out), "datasets": args.datasets, "query_method_rows": len(full_rows),
                      "complete_method_groups": len(summaries), "coverage": coverage}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

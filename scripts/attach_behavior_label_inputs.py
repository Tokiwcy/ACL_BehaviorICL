"""CPU-only, immutable attachment of audited demo behavior to the locked plan.

No new query selection, retrieval, label repair, model execution, or source edit.
The original 1280-query lock stays byte-identical. Missing own-label arms stay
missing; all prespecified gold/hash-wrong/zero arms remain in the schedule.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_behavior_priority_analysis as a
from analyze_behavior_paper_p0 import load_complete_new
from collect_behavior_priority_zero import atomic, canonical, digest, REVISION
from prepare_behavior_label_intervention import pick_queries, wrong_label

EXPECTED_PLAN_HASH = "e2e2d2b7a8b830e1cc254c06c1b5d06ba85228a42e1be7042b238752d4e451ad"
ARMS = ("same_constrained_zero", "gold_demo_labels", "demo_own_legal_zero_labels", "hashed_wrong_demo_labels")


def validate_plan(plan, expected_hash=EXPECTED_PLAN_HASH):
    body = {k: v for k, v in plan.items() if k != "plan_hash"}
    if canonical(body) != plan.get("plan_hash") or plan["plan_hash"] != expected_hash:
        raise ValueError("Original query/arm lock identity changed")
    if (plan.get("model") != a.MODEL or plan.get("revision") != REVISION
            or plan.get("paired_arms") != list(ARMS)):
        raise ValueError("Wrong model/revision/arms in original plan")
    seen = set()
    for row in plan["all_queries"]:
        key = (row["dataset"], row["query_id"])
        if key in seen:
            raise ValueError("Duplicate locked query")
        seen.add(key)
        if (len(row["demo_ids_in_actual_prompt_order"]) != 4
                or len(set(row["demo_ids_in_actual_prompt_order"])) != 4
                or len(row["gold_labels"]) != 4 or len(row["hashed_wrong_labels"]) != 4):
            raise ValueError("Need exactly four unique ordered demos")
        if row["demo_own_zero_labels"] is not None:
            raise ValueError("Original lock must not be mutated to attach behavior")


def attach_row(row, zero):
    """Pure attachment. Illegal literal answers are retained and never relabeled."""
    demos = row["demo_ids_in_actual_prompt_order"]
    gold = row["gold_labels"]
    labels = row["legal_classes"]
    preds, evidence = [], []
    for demo_id, label in zip(demos, gold):
        z = zero[demo_id]
        if (z["sample_id"] != demo_id or z["true_label"] != label or z["split"] != "bank"):
            raise ValueError("Own demo behavior belongs to another sample/label/split")
        legal = {v.casefold(): v for v in labels}
        pred = legal.get(z["raw_zero_shot_answer"].strip().casefold())
        if z["zero_shot_pred"] != pred or z["zero_shot_valid"] != (pred is not None):
            raise ValueError("Own labels differ from literal zero answer")
        preds.append(pred)
        evidence.append({"demo_id": demo_id, "gold_label": label,
                         "raw_zero_shot_answer": z["raw_zero_shot_answer"],
                         "predicted_label": pred, "valid": pred is not None})
    valid = all(p is not None for p in preds)
    return {**row, "demo_own_zero_labels": preds,
            "own_arm_status": "ready" if valid else "missing_illegal_demo_zero",
            "own_zero_evidence": evidence,
            "own_legal_labels_changed": sum(p is not None and p != y for p, y in zip(preds, gold)),
            "invalid_demo_zero_count": sum(p is None for p in preds),
            "hashed_wrong_labels_changed": sum(p != y for p, y in zip(row["hashed_wrong_labels"], gold))}


def image_record(sample):
    path = Path(sample.path).resolve()
    if not path.is_relative_to(ROOT / "datasets"):
        raise ValueError("Official selected image outside workspace datasets")
    return {"sample_id": sample.sample_id, "relative_path": path.relative_to(ROOT).as_posix(),
            "sha256": digest(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", type=Path, default=ROOT / "analysis/paired_label_plan_v1/query_arm_lock.json")
    p.add_argument("--zero-root", type=Path, required=True)
    p.add_argument("--p0-audit", type=Path, default=ROOT / "analysis/paper_p0_full_v1/audit.json")
    p.add_argument("--output-root", type=Path, default=ROOT / "analysis/paired_label_inputs_v1")
    args = p.parse_args()
    args.plan = args.plan.resolve()
    args.zero_root = args.zero_root.resolve()
    args.p0_audit = args.p0_audit.resolve()
    out = args.output_root.resolve()
    if not out.is_relative_to(ROOT / "analysis") or out.exists():
        raise ValueError("Use a fresh derived analysis directory; never overwrite locks")
    for path in (args.plan, args.zero_root, args.p0_audit):
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError("Inputs must stay within this workspace")
    plan_before = digest(args.plan)
    plan = a.read(args.plan)
    validate_plan(plan)
    p0 = a.read(args.p0_audit)
    if set(p0.get("zero_cache_scope", [])) != set(a.DATASETS):
        raise ValueError("Full five-dataset P0 audit required")
    provenance = {}
    for rel, expected in {**p0["source_sha256"], **plan["source_sha256"]}.items():
        path = (ROOT / rel.replace("\\", "/")).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise ValueError("Audited source changed: " + rel)
        provenance[path.relative_to(ROOT).as_posix()] = expected
    provenance[args.plan.resolve().relative_to(ROOT).as_posix()] = plan_before
    provenance[args.p0_audit.resolve().relative_to(ROOT).as_posix()] = digest(args.p0_audit)
    from multidataset_protocol import load_dataset
    sources = a.source_catalog()
    attached, coverage = [], []
    for dataset in a.DATASETS:
        src = sources[dataset]
        manifest = a.read(src["base"] / "manifest.json")
        group = a.load_group(src["proj"], "projection_only", manifest, provenance)
        bank, query = load_dataset(dataset, ROOT / "datasets")
        current = {"bank": [s.__dict__ for s in bank], "query": [s.__dict__ for s in query]}
        if a.identity(current) != a.identity(manifest):
            raise ValueError("Official sample ID/label/split/order changed")
        locked = [r for r in plan["all_queries"] if r["dataset"] == dataset]
        expected_ids = [s["sample_id"] for s in pick_queries(dataset, manifest["query"])]
        if len(locked) != 256 or [r["query_id"] for r in locked] != expected_ids:
            raise ValueError("Preselected query cohort changed")
        labels = sorted({s.label for s in bank})
        if dataset in ("dtd", "aircraft"):
            zero = a.load_zero(src["base"], manifest, provenance)
            if zero is None:
                raise ValueError("Missing complete old zero cache")
        else:
            from collect_behavior_priority_zero import prepare
            samples, _, _, expected = prepare(dataset, ROOT / "datasets", ROOT / ".hf_cache")
            images = {s.sample_id: digest(s.path) for s in samples}
            zero = load_complete_new(args.zero_root / dataset, manifest, expected, images, provenance)
        images = {s.sample_id: s for s in bank + query}
        image_cache = {}
        def record(sample_id):
            if sample_id not in image_cache:
                image_cache[sample_id] = image_record(images[sample_id])
            return image_cache[sample_id]
        local = []
        for row in locked:
            actual = group[row["query_id"]]
            if (row["legal_classes"] != labels or actual["target"] != row["target_for_evaluation_only"]
                    or actual["demo_ids"] != row["demo_ids_in_actual_prompt_order"]
                    or actual["demo_labels"] != row["gold_labels"]):
                raise ValueError("Locked actual top4/labels/target/classes changed")
            expected_wrong = [wrong_label(dataset, i, y, labels)
                              for i, y in zip(actual["demo_ids"], actual["demo_labels"])]
            if row["hashed_wrong_labels"] != expected_wrong:
                raise ValueError("Original deterministic wrong-label arm changed")
            value = attach_row(row, zero)
            value["query_image"] = record(row["query_id"])
            value["demo_images_in_actual_prompt_order"] = [record(i) for i in actual["demo_ids"]]
            local.append(value)
        attached.extend(local)
        coverage.append({"dataset": dataset, "locked_queries": len(local),
                         "own_arm_ready": sum(r["own_arm_status"] == "ready" for r in local),
                         "own_arm_missing": sum(r["own_arm_status"] != "ready" for r in local),
                         "own_arm_ready_with_changed_labels": sum(r["own_arm_status"] == "ready" and r["own_legal_labels_changed"] > 0 for r in local),
                         "gold_zero_hashwrong_arms": "all 256 preserved, no query dropped",
                         "unique_selected_images": len(image_cache)})
        print("Full audited zero + original locked context attached: " + dataset, flush=True)
    if Counter(r["dataset"] for r in attached) != Counter({d: 256 for d in a.DATASETS}):
        raise ValueError("Incomplete five-dataset locked cohort")
    if digest(args.plan) != plan_before:
        raise ValueError("Original query plan changed during attachment")
    # Recheck all frozen evidence after processing, not only before attachment.
    provenance = {(ROOT / k.replace("\\", "/")).relative_to(ROOT).as_posix(): v
                  for k, v in provenance.items()}
    for rel, expected in provenance.items():
        if digest(ROOT / rel) != expected:
            raise ValueError("Source changed during attachment: " + rel)
    spec = {"experiment": "behavior_corrective_label_inputs_v1", "model": a.MODEL,
            "revision": REVISION, "base_plan_hash": plan["plan_hash"],
            "base_plan_file_sha256": plan_before, "paired_arms": list(ARMS),
            "all_queries": attached, "source_sha256": provenance,
            "attachment_script_sha256": digest(Path(__file__)),
            "coverage": coverage, "zero_behavior_decoding": "original free greedy max12, illegal literal answers retained",
            "generation_protocol": "existing model adapter original vision/prompt; new identical legal decoding in all four arms; same-environment gold rerun",
            "query_truth_usage": "evaluation only; no label/correctness/rescue-based query selection or retrieval",
            "own_arm_invalid_policy": "explicit missing row if any demo free zero answer illegal; no repair and no query reselection",
            "image_scope": "selected official image bytes audited now; historical byte-level environment equivalence not asserted",
            "causal_scope": "fixed-context label intervention, not same-failure retrieval's unique causal effect"}
    spec["attachment_hash"] = canonical(spec)
    out.mkdir(parents=True)
    atomic(out / "attached_inputs_lock.json", spec)
    a.csv_out(out / "planned_arm_coverage.csv", coverage)
    atomic(out / "status.json", {"stage": "CPU_inputs_attached_verified_no_GPU_execution",
                               "utc": datetime.now(timezone.utc).isoformat(),
                               "attachment_hash": spec["attachment_hash"],
                               "original_plan_sha256_unchanged": plan_before,
                               "queries": len(attached), "coverage": coverage,
                               "source_files_rehashed": len(provenance),
                               "model_execution": False})
    print(json.dumps({"attachment_hash": spec["attachment_hash"], "queries": len(attached),
                      "coverage": coverage, "GPU": "not executed"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

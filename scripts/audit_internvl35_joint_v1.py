"""CPU audit of full official-test InternVL3.5-4B four-shot predictions."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_behavior_repair_all_queries import paired_bootstrap
from run_internvl35_joint_methods_v1 import DATASETS, METHODS, PROTOCOL_PATH, ensure_protocol, load_context
from internvl35_joint_config_v1 import digest, write_json


def audit_source(folder, bank, query, identity):
    path = folder / "joint_progress.json"
    if not path.exists():
        return {"state": "not_started", "completed": 0}
    import numpy as np
    from internvl35_joint_config_v1 import ANALYSIS
    progress = json.loads(path.read_text(encoding="utf-8"))
    binding = json.loads((folder / "joint_binding.json").read_text(encoding="utf-8"))
    if (progress["binding"] != binding or binding["manifest_sha256"] != identity["manifest_sha256"]
            or binding["protocol_sha256"] != digest(PROTOCOL_PATH)
            or binding["compatibility_sha256"] != digest(ANALYSIS / "compatibility.json")):
        raise ValueError("Joint source identity/compatibility changed")
    total, done = len(bank) + len(query), progress["completed"]
    if progress["shape"] != [total,36,2560] or progress["bank_n"] != len(bank) or not 0 <= done <= total:
        raise ValueError("Joint source commit shape/count invalid")
    files = [(folder / "anchor_states.npy", (total,36,2560)),
             (folder / "gpt_mm_embeddings.npy", (total,2560)),
             (folder / "trained/detriever/gold_input_answer_eos.npy", (len(bank),2560))]
    for file, shape in files:
        value = np.load(file, mmap_mode="r")
        if value.shape != shape or value.dtype != np.float16:
            raise ValueError("Joint array shape or dtype invalid")
    answers = folder / "gpt_mm_zero_shot_outputs.jsonl"
    with answers.open("rb") as handle:
        for index in range(done):
            line = handle.readline()
            if not line.endswith(b"\n"):
                raise ValueError("Committed answer row truncated")
            row = json.loads(line)
            if row["sample_id"] != (bank + query)[index].sample_id:
                raise ValueError("Joint answer order differs")
            if (index < len(bank)) != (row["gold_suffix_ids"] is not None):
                raise ValueError("Gold proxy branch violates bank-only contract")
            if not 1 <= len(row["tail_ids"]) <= 12:
                raise ValueError("GPT-MM generated token count outside protocol")
        if done == total and handle.read():
            raise ValueError("Full joint answer file contains uncommitted rows")
    report = {"state": "complete" if done == total else "partial", "completed": done,
              "total": total, "binding": binding}
    if done == total:
        report["array_sha256"] = {file.name: digest(file) for file, _ in files}
        report["answers_sha256"] = digest(answers)
    return report


def wilson(successes: int, total: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [centre - radius, centre + radius]


def read_rows(path: Path) -> list[dict]:
    with path.open("rb") as handle:
        data = handle.read()
    if data and not data.endswith(b"\n"):
        raise ValueError(f"Truncated final prediction row: {path}")
    return [json.loads(line) for line in data.splitlines()]


def audit_vision(folder, bank, query, selections, method):
    budgets = json.loads((folder / "image_budget.json").read_text(encoding="utf-8"))
    if budgets["protocol_sha256"] != digest(PROTOCOL_PATH):
        raise ValueError("Image budget belongs to a different protocol")
    records = read_rows(folder / "telemetry" / f"icl_{method}.jsonl")
    by_images = {}
    for record in records:
        if len(record["images"]) != 1 or len(record["images"][0]) != 5:
            raise ValueError("Expected exactly five images in each ICL context")
        by_images[tuple(record["images"][0])] = record
    visual, total = [], []
    for target in query:
        context = [bank[j] for j in selections[target.sample_id]] + [target]
        key = tuple(s.path for s in context)
        if key not in by_images:
            raise ValueError(f"Missing actual vision telemetry for {target.sample_id}")
        record = by_images[key]
        expected = sum(budgets["images"][s.sample_id]["visual_tokens"] for s in context)
        if (record["visual_tokens"] != [expected]
                or record["pixel_values_shape"][0] * 256 != expected
                or record["input_tokens"][0] <= expected):
            raise ValueError("Actual ICL image/input token counts differ from frozen images")
        visual.append(expected)
        total.append(record["input_tokens"][0])
    def stats(values):
        values = sorted(values)
        return {"min": values[0], "median": values[len(values) // 2],
                "p95": values[min(len(values) - 1, int(len(values) * .95))],
                "max": values[-1], "mean": sum(values) / len(values)}
    return {"visual_tokens": stats(visual), "total_input_tokens": stats(total),
            "telemetry_sha256": digest(folder / "telemetry" / f"icl_{method}.jsonl")}


def audit_method(folder: Path, bank: list, query: list, identity: dict, method: str) -> dict:
    selection_path = folder / "selections" / f"{method}.json"
    if not selection_path.is_file():
        return {"state": "missing_selection", "completed": 0, "total": len(query)}
    selections = json.loads(selection_path.read_text(encoding="utf-8"))
    if list(selections) != [q.sample_id for q in query]:
        raise ValueError(f"{folder.name}/{method}: selection coverage/order differs")
    for indices in selections.values():
        if len(indices) != 4 or len(set(indices)) != 4 or min(indices) < 0 or max(indices) >= len(bank):
            raise ValueError(f"{folder.name}/{method}: invalid four-shot selection")
    same_label_slots = sum(
        bank[index].label == target.label
        for target in query for index in selections[target.sample_id]
    )
    any_same = sum(
        any(bank[index].label == target.label for index in selections[target.sample_id])
        for target in query
    )
    selection_metrics = {
        "top4_same_label_fraction": same_label_slots / (4 * len(query)),
        "top4_any_same_label_fraction": any_same / len(query),
    }
    result_path = folder / "predictions" / f"{method}.jsonl"
    if not result_path.is_file():
        return {"state": "selected", "completed": 0, "total": len(query),
                "selection_metrics": selection_metrics}
    rows = read_rows(result_path)
    if len(rows) > len(query):
        raise ValueError(f"{folder.name}/{method}: excess prediction rows")
    labels = {sample.label for sample in bank}
    for index, row in enumerate(rows):
        target = query[index]
        demos = [bank[j] for j in selections[target.sample_id]]
        if (row.get("query_id") != target.sample_id or row.get("target") != target.label
                or row.get("method") != method or row.get("demo_ids") != [x.sample_id for x in demos]
                or row.get("demo_labels") != [x.label for x in demos]
                or row.get("correct") != (row.get("prediction") == target.label)):
            raise ValueError(f"{folder.name}/{method}: row {index} differs from frozen selection/target")
    result = {"state": "partial", "completed": len(rows), "total": len(query),
              "selection_metrics": selection_metrics,
              "correct": sum(row["correct"] for row in rows),
              "illegal_or_unknown": sum(row.get("prediction") not in labels for row in rows)}
    if len(rows) != len(query):
        return result
    summary_path = folder / "predictions" / f"{method}.summary.json"
    if not summary_path.is_file():
        raise ValueError(f"{folder.name}/{method}: full rows without summary")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (summary.get("predictions_sha256") != digest(result_path)
            or summary.get("selections_sha256") != digest(selection_path)
            or summary.get("source_manifest_sha256") != identity["manifest_sha256"]
            or summary.get("full_protocol_sha256") != digest(PROTOCOL_PATH)
            or summary.get("correct") != result["correct"]
            or summary.get("total") != len(query)
            or summary.get("shots") != 4 or summary.get("image_max_patches") != 12
            or summary.get("crop_to_patches") is not True):
        raise ValueError(f"{folder.name}/{method}: summary/identity mismatch")
    per_class = {}
    for row in rows:
        item = per_class.setdefault(row["target"], {"correct": 0, "total": 0})
        item["correct"] += int(row["correct"])
        item["total"] += 1
    result.update(state="complete", accuracy=result["correct"] / len(query),
                  accuracy_wilson_95=wilson(result["correct"], len(query)),
                  macro_class_accuracy=sum(v["correct"] / v["total"] for v in per_class.values()) / len(per_class),
                  per_class=per_class,
                  predictions_sha256=digest(result_path),
                  vision=audit_vision(folder, bank, query, selections, method),
                  outcomes=[bool(row["correct"]) for row in rows])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    ensure_protocol()
    report = {"version": "joint_v1", "model": "OpenGVLab/InternVL3_5-4B-HF",
              "shots": 4, "seed": 73, "official_full_test": True,
              "paired_ci": "10000 paired-query multinomial bootstrap, seed73; descriptive, no multiplicity correction",
              "datasets": {}, "pairs": []}
    incomplete = []
    for dataset in DATASETS:
        folder, bank, query, identity = load_context(dataset)
        source = audit_source(folder, bank, query, identity)
        method_results = {}
        for method in METHODS:
            result = audit_method(folder, bank, query, identity, method)
            outcomes = result.pop("outcomes", None)
            if outcomes is not None:
                result["outcomes_internal"] = outcomes
            if result["state"] != "complete":
                incomplete.append(f"{dataset}/{method}: {result['state']} {result['completed']}/{result['total']}")
            method_results[method] = result
        for baseline in ("gpt_mm", "detriever"):
            for candidate in ("raw", "proj", "learn"):
                a, b = method_results[candidate], method_results[baseline]
                if a["state"] == b["state"] == "complete":
                    pair = paired_bootstrap(a["outcomes_internal"], b["outcomes_internal"])
                    first_selection = json.loads((folder / "selections" / f"{candidate}.json").read_text(encoding="utf-8"))
                    second_selection = json.loads((folder / "selections" / f"{baseline}.json").read_text(encoding="utf-8"))
                    overlap = sum(
                        len(set(first_selection[target.sample_id]) & set(second_selection[target.sample_id]))
                        for target in query
                    ) / len(query)
                    report["pairs"].append({"dataset": dataset, "first": candidate,
                                             "second": baseline, "mean_shared_demonstrations_of_four": overlap,
                                             **pair})
        for result in method_results.values():
            result.pop("outcomes_internal", None)
        report["datasets"][dataset] = {"bank_n": len(bank), "query_n": len(query),
                                        "manifest_sha256": identity["manifest_sha256"],
                                        "joint_source": source,
                                        "methods": method_results}
    report["state"] = "incomplete" if incomplete else "complete"
    report["incomplete"] = incomplete
    output = ROOT / "analysis/internvl35_joint_v1" / (
        "progress_audit.json" if incomplete else "full_audit.json"
    )
    write_json(output, report)
    print(f"{report['state']}: {output}; {len(incomplete)} incomplete dataset/method cells")
    if incomplete and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

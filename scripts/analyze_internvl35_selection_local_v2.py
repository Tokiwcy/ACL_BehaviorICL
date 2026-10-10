"""Audit how changing the retrieval prompt changes selected demos and ICL."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_query_retrieval_original_raw_v1 import raw_scores
from query_tam_local_plan_v1 import digest


def main() -> None:
    protocol = ROOT / "analysis/internvl35_retrieval_local_v2/protocol.json"
    plan = json.loads(protocol.read_text(encoding="utf-8"))
    output = {"protocol_SHA256": digest(protocol), "datasets": {}}
    all_rows = []
    for dataset, table in plan["datasets"].items():
        bank = table["bank"]
        query = table["query"]
        folder = ROOT / "results/internvl35_retrieval_local_v2" / dataset
        rankings = {}
        for arm in ("generic", "task_full"):
            state = np.load(folder / f"{arm}_states.npy", mmap_mode="r")
            rankings[arm] = np.argsort(-raw_scores(state, len(bank)), axis=1, kind="stable")[:, :4]
        icl_path = ROOT / "results/internvl35_icl_local_v2" / dataset / "predictions.jsonl"
        outcomes = {(r["arm"], r["query_id"]): r for line in icl_path.read_text(encoding="utf-8").splitlines()
                    for r in [json.loads(line)]}
        rows = []
        for index, item in enumerate(query):
            generic = set(rankings["generic"][index].tolist())
            full = set(rankings["task_full"][index].tolist())
            n_generic = sum(bank[j]["label"] == item["label"] for j in generic)
            n_full = sum(bank[j]["label"] == item["label"] for j in full)
            old = outcomes[("generic", item["sample_id"])]["correct"]
            new = outcomes[("task_full", item["sample_id"])]["correct"]
            rows.append({"query_id": item["sample_id"], "target": item["label"],
                         "shared_demo_count": len(generic & full),
                         "generic_correct_demo_count": n_generic, "task_full_correct_demo_count": n_full,
                         "generic_prediction_correct": old, "task_full_prediction_correct": new})
        output["datasets"][dataset] = {"n": len(rows),
            "mean_shared_demo_count": float(np.mean([r["shared_demo_count"] for r in rows])),
            "generic_any_correct_demo": sum(r["generic_correct_demo_count"] > 0 for r in rows),
            "task_full_any_correct_demo": sum(r["task_full_correct_demo_count"] > 0 for r in rows),
            "more_correct_demos": sum(r["task_full_correct_demo_count"] > r["generic_correct_demo_count"] for r in rows),
            "fewer_correct_demos": sum(r["task_full_correct_demo_count"] < r["generic_correct_demo_count"] for r in rows),
            "ICL_gain": sum(r["task_full_prediction_correct"] and not r["generic_prediction_correct"] for r in rows),
            "ICL_harm": sum(r["generic_prediction_correct"] and not r["task_full_prediction_correct"] for r in rows),
        }
        all_rows.extend(rows)
    output["pooled"] = {"n": len(all_rows),
        "mean_shared_demo_count": float(np.mean([r["shared_demo_count"] for r in all_rows])),
        "more_correct_demos": sum(r["task_full_correct_demo_count"] > r["generic_correct_demo_count"] for r in all_rows),
        "fewer_correct_demos": sum(r["task_full_correct_demo_count"] < r["generic_correct_demo_count"] for r in all_rows),
        "ICL_gain": sum(r["task_full_prediction_correct"] and not r["generic_prediction_correct"] for r in all_rows),
        "ICL_harm": sum(r["generic_prediction_correct"] and not r["task_full_prediction_correct"] for r in all_rows),
    }
    path = protocol.with_name("selection_icl_audit.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

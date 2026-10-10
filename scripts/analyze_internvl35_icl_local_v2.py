"""Paired five-dataset audit of small-bank InternVL four-shot outcomes."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_query_retrieval_local_v1 import ci
from query_tam_local_plan_v1 import digest


def main() -> None:
    plan_path = ROOT / "analysis/internvl35_retrieval_local_v2/protocol.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    output = {"retrieval_protocol_SHA256": digest(plan_path),
              "question": "Does task-conditioned versus generic retrieval change four-shot closed-set ICL with the ICL prompt fixed?",
              "scope": "32 official test queries and two bank images per class; no full-bank or full-test claim",
              "datasets": {}}
    all_generic, all_full = [], []
    for dataset, table in plan["datasets"].items():
        folder = ROOT / "results/internvl35_icl_local_v2" / dataset
        identity = json.loads((folder / "identity.json").read_text(encoding="utf-8"))
        if identity["retrieval_protocol_SHA256"] != digest(plan_path) or identity["image_max_patches"] != 1:
            raise ValueError(f"Wrong ICL identity: {dataset}")
        rows = [json.loads(line) for line in (folder / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        by_key = {(row["arm"], row["query_id"]): row for row in rows}
        if len(rows) != 64 or len(by_key) != 64:
            raise ValueError(f"Incomplete or duplicate ICL rows: {dataset}")
        generic, full = [], []
        for query in table["query"]:
            for arm, dest in (("generic", generic), ("task_full", full)):
                row = by_key[(arm, query["sample_id"])]
                if row["target"] != query["label"] or row["correct"] != (row["prediction"] == row["target"]):
                    raise ValueError(f"Wrong target/result for {dataset}/{query['sample_id']}")
                dest.append(int(row["correct"]))
        generic, full = np.asarray(generic), np.asarray(full)
        all_generic.extend(generic.tolist())
        all_full.extend(full.tolist())
        output["datasets"][dataset] = {
            "n": 32, "generic_correct": int(generic.sum()), "task_full_correct": int(full.sum()),
            "generic_accuracy": float(generic.mean()), "task_full_accuracy": float(full.mean()),
            "paired_delta": ci(full - generic),
        }
    generic, full = np.asarray(all_generic), np.asarray(all_full)
    output["pooled_160_queries"] = {"generic_correct": int(generic.sum()),
                                    "task_full_correct": int(full.sum()),
                                    "generic_accuracy": float(generic.mean()),
                                    "task_full_accuracy": float(full.mean()),
                                    "paired_delta": ci(full - generic)}
    out = ROOT / "analysis/internvl35_retrieval_local_v2/icl_analysis.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

"""Paired ICL comparison of generic, pre-answer Raw, and answer-state retrieval."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_query_retrieval_local_v1 import ci
from query_tam_local_plan_v1 import digest


def summarize(generic: np.ndarray, pre: np.ndarray, answer: np.ndarray) -> dict:
    return {"n": len(generic),
            "generic_correct": int(generic.sum()),
            "pre_answer_Raw_correct": int(pre.sum()),
            "answer_state_style_correct": int(answer.sum()),
            "pre_minus_generic": ci(pre - generic),
            "pre_minus_answer": ci(pre - answer),
            "answer_minus_generic": ci(answer - generic)}


def main() -> None:
    pre_path = ROOT / "analysis/internvl35_retrieval_local_v2/protocol.json"
    answer_path = ROOT / "analysis/internvl35_answer_state_local_v1/protocol.json"
    plan = json.loads(pre_path.read_text(encoding="utf-8"))
    if json.loads(answer_path.read_text(encoding="utf-8"))["base_protocol_SHA256"] != digest(pre_path):
        raise ValueError("Answer-state identity mismatch")
    output = {"pre_protocol_SHA256": digest(pre_path),
              "answer_protocol_SHA256": digest(answer_path),
              "scope": "same 32 test queries per dataset, same four-shot ICL prompt, two training examples per class",
              "datasets": {}}
    all_generic, all_pre, all_answer = [], [], []
    for dataset, table in plan["datasets"].items():
        pre_rows = [json.loads(line) for line in
                    (ROOT / "results/internvl35_icl_local_v2" / dataset / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        answer_rows = [json.loads(line) for line in
                       (ROOT / "results/internvl35_answer_state_icl_local_v1" / dataset / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        pre_by_key = {(r["arm"], r["query_id"]): r for r in pre_rows}
        answer_by_id = {r["query_id"]: r for r in answer_rows}
        if len(pre_rows) != 64 or len(pre_by_key) != 64 or len(answer_rows) != 32 or len(answer_by_id) != 32:
            raise ValueError(f"Incomplete or duplicate ICL rows: {dataset}")
        generic, pre, answer = [], [], []
        for query in table["query"]:
            key = query["sample_id"]
            rows = [pre_by_key[("generic", key)], pre_by_key[("task_full", key)], answer_by_id[key]]
            if any(r["target"] != query["label"] for r in rows):
                raise ValueError(f"Target mismatch: {dataset}/{key}")
            for row, dest in zip(rows, (generic, pre, answer)):
                dest.append(int(row["correct"]))
        arrays = [np.asarray(x) for x in (generic, pre, answer)]
        output["datasets"][dataset] = summarize(*arrays)
        for source, target in zip(arrays, (all_generic, all_pre, all_answer)):
            target.extend(source.tolist())
    output["pooled_160_queries"] = summarize(*[np.asarray(x) for x in (all_generic, all_pre, all_answer)])
    out = ROOT / "analysis/internvl35_answer_state_local_v1/icl_comparison.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

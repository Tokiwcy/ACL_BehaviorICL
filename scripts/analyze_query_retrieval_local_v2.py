"""CPU audit of full-task and label-list-only retrieval controls on the v1 bank."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_query_retrieval_local_v1 import ci, features, metrics
from query_tam_local_plan_v1 import digest


def main():
    p1 = ROOT / "analysis/query_retrieval_local_v1/protocol.json"
    p2 = ROOT / "analysis/query_retrieval_local_v2/protocol.json"
    v1 = json.loads(p1.read_text(encoding="utf-8"))
    v2 = json.loads(p2.read_text(encoding="utf-8"))
    if v2["base_protocol_SHA256"] != digest(p1):
        raise ValueError("Base protocol changed")
    out = {"protocol_SHA256": digest(p2), "base_protocol_SHA256": digest(p1), "datasets": {},
           "limits": ["This is two-per-class retrieval, not full-bank downstream ICL accuracy.",
                      "The label-list-only arm still implicitly defines a classification task.",
                      "The method and prompt contrasts are exploratory; 32 queries per dataset."]}
    methods = v1["prespecified_methods"]
    arms = ["generic", "task_short"] + v2["new_arms"]
    for dataset in v1["datasets"]:
        first = v1["datasets"][dataset]
        second = v2["datasets"][dataset]
        if first["bank"] != second["bank"] or first["query"] != second["query"]:
            raise ValueError("Bank/query mismatch")
        bank_n = len(first["bank"])
        bank_labels = [x["label"] for x in first["bank"]]
        query_labels = [x["label"] for x in first["query"]]
        source = ROOT / "results/query_retrieval_local_v1" / dataset
        rows = {}
        for arm in arms:
            expected = digest(p1 if arm in v1["arms"] else p2)
            progress = json.loads((source / f"{arm}_progress.json").read_text(encoding="utf-8"))
            if progress["protocol_SHA256"] != expected or progress["completed"] != bank_n + len(query_labels):
                raise ValueError(f"Incomplete/mismatched arm {dataset}/{arm}")
            states = np.load(source / f"{arm}_states.npy", mmap_mode="r")
            rows[arm] = {}
            for method in methods:
                embeddings = features(states, method)
                rows[arm][method] = metrics(embeddings[bank_n:] @ embeddings[:bank_n].T, bank_labels, query_labels)
        block = {"bank_n": bank_n, "query_n": len(query_labels), "methods": {}}
        for method in methods:
            measures = {}
            for metric in ("top1", "top4", "AP"):
                values = {arm: np.array([row[metric] for row in rows[arm][method]], dtype=float) for arm in arms}
                measures[metric] = {"rates": {arm: float(values[arm].mean()) for arm in arms},
                    "paired_deltas": {f"{right}_minus_{left}": ci(values[right] - values[left])
                        for left, right in (("generic", "task_short"), ("generic", "label_list_only"),
                                            ("generic", "task_full"), ("task_short", "task_full"),
                                            ("label_list_only", "task_full"))}}
            block["methods"][method] = measures
        out["datasets"][dataset] = block
    path = ROOT / "analysis/query_retrieval_local_v2/analysis.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({dataset: {method: value["top4"]["rates"] for method, value in block["methods"].items()}
        for dataset, block in out["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

"""Posthoc CPU diagnostic: exact equal-layer cosine Raw score on the local bank."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_query_retrieval_local_v1 import ci, metrics, normalize
from query_tam_local_plan_v1 import digest


def raw_scores(states, bank_n):
    bank = np.asarray(states[:bank_n], dtype=np.float32)
    query = np.asarray(states[bank_n:], dtype=np.float32)
    result = np.zeros((len(query), len(bank)), dtype=np.float32)
    for layer in range(36):
        result += normalize(query[:, layer]) @ normalize(bank[:, layer]).T / 36
    return result


def main():
    p1 = ROOT / "analysis/query_retrieval_local_v1/protocol.json"
    p2 = ROOT / "analysis/query_retrieval_local_v2/protocol.json"
    v1 = json.loads(p1.read_text(encoding="utf-8"))
    v2 = json.loads(p2.read_text(encoding="utf-8"))
    result = {"analysis_type": "posthoc diagnostic using the repository's exact Raw equal-layer cosine score",
              "v1_protocol_SHA256": digest(p1), "v2_protocol_SHA256": digest(p2), "datasets": {}}
    for dataset, table in v1["datasets"].items():
        bank_n = len(table["bank"])
        labels = [x["label"] for x in table["bank"]]
        query_labels = [x["label"] for x in table["query"]]
        source = ROOT / "results/query_retrieval_local_v1" / dataset
        arms = ["generic", "task_short"] + v2["new_arms"]
        values = {}
        for arm in arms:
            states = np.load(source / f"{arm}_states.npy", mmap_mode="r")
            measures = metrics(raw_scores(states, bank_n), labels, query_labels)
            values[arm] = {measure: np.array([x[measure] for x in measures], dtype=float)
                           for measure in ("top1", "top4", "AP")}
        block = {"bank_n": bank_n, "query_n": len(query_labels)}
        for measure in ("top1", "top4", "AP"):
            block[measure] = {"rates": {arm: float(values[arm][measure].mean()) for arm in arms},
                "paired_delta_task_short_minus_generic": ci(values["task_short"][measure] - values["generic"][measure]),
                "paired_delta_full_minus_generic": ci(values["task_full"][measure] - values["generic"][measure]),
                "paired_delta_full_minus_labels_only": ci(values["task_full"][measure] - values["label_list_only"][measure])}
        result["datasets"][dataset] = block
    out = ROOT / "analysis/query_retrieval_local_v2/original_raw_diagnostic.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["top4"]["rates"] for k, v in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

"""Posthoc exploratory retrieval from task-induced state changes, using saved GPU features."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_query_retrieval_local_v1 import features, metrics, normalize


def layerwise_score(states, bank_n):
    bank = np.asarray(states[:bank_n], dtype=np.float32)
    query = np.asarray(states[bank_n:], dtype=np.float32)
    score = np.zeros((len(query), len(bank)), dtype=np.float32)
    for layer in range(36):
        score += normalize(query[:, layer]) @ normalize(bank[:, layer]).T / 36
    return score


def main():
    plan = json.loads((ROOT / "analysis/query_retrieval_local_v1/protocol.json").read_text(encoding="utf-8"))
    result = {"status": "posthoc exploratory; not part of frozen v1/v2 methods", "datasets": {}}
    for dataset, table in plan["datasets"].items():
        source = ROOT / "results/query_retrieval_local_v1" / dataset
        generic = np.load(source / "generic_states.npy", mmap_mode="r").astype(np.float32)
        bank_n = len(table["bank"])
        labels = [x["label"] for x in table["bank"]]
        query_labels = [x["label"] for x in table["query"]]
        block = {}
        for arm in ("task_short", "task_full", "label_list_only"):
            task = np.load(source / f"{arm}_states.npy", mmap_mode="r").astype(np.float32)
            delta = task - generic
            block[arm] = {}
            for method in ("layer36", "mean_normalized_36_layers", "mean_normalized_layer_velocity", "equal_layer_cosine"):
                if method == "equal_layer_cosine":
                    scores = layerwise_score(delta, bank_n)
                else:
                    emb = features(delta, method)
                    scores = emb[bank_n:] @ emb[:bank_n].T
                values = metrics(scores, labels, query_labels)
                block[arm][method] = {metric: float(np.mean([x[metric] for x in values]))
                                      for metric in ("top1", "top4", "AP")}
        result["datasets"][dataset] = block
    out = ROOT / "analysis/query_retrieval_local_v2/query_delta_exploratory.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({d: {arm: methods["equal_layer_cosine"]["top4"] for arm, methods in block.items()}
                      for d, block in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

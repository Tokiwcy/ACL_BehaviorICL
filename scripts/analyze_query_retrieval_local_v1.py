"""CPU-only paired retrieval metrics for the frozen local bank and query panel."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from query_tam_local_plan_v1 import digest


def normalize(value):
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-12)


def features(states, method):
    x = np.asarray(states, dtype=np.float32)
    if method == "layer18":
        value = x[:, 17]
    elif method == "layer36":
        value = x[:, 35]
    elif method == "mean_normalized_36_layers":
        value = normalize(x).mean(axis=1)
    elif method == "mean_normalized_layer_velocity":
        value = normalize(x[:, 1:] - x[:, :-1]).mean(axis=1)
    else:
        raise ValueError(method)
    return normalize(value)


def metrics(scores, bank_labels, query_labels):
    ranked = np.argsort(-scores, axis=1, kind="stable")
    rows = []
    for order, label in zip(ranked, query_labels):
        hits = np.asarray([bank_labels[i] == label for i in order], dtype=bool)
        positions = np.flatnonzero(hits) + 1
        if len(positions) != 2:
            raise ValueError("Exactly two positive bank examples per query class required")
        rows.append({"top1": int(hits[0]), "top4": int(hits[:4].any()),
                     "AP": float(np.mean([rank / position for rank, position in enumerate(positions, 1)]))})
    return rows


def ci(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(73)
    bootstrap = values[rng.integers(0, len(values), (10000, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "95_percent_bootstrap_mean": [float(x) for x in np.quantile(bootstrap, [.025, .975])],
            "positive": int((values > 0).sum()), "negative": int((values < 0).sum()), "zero": int((values == 0).sum())}


def main():
    path = ROOT / "analysis/query_retrieval_local_v1/protocol.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    result = {"protocol_SHA256": digest(path), "datasets": {}, "scope": "two-per-class bank retrieval on 32 held-out test images per dataset; no model training or downstream ICL generation"}
    for dataset, table in plan["datasets"].items():
        bank_n = len(table["bank"])
        labels = [x["label"] for x in table["bank"]]
        query_labels = [x["label"] for x in table["query"]]
        source = ROOT / "results/query_retrieval_local_v1" / dataset
        results = {}
        for arm in plan["arms"]:
            progress = json.loads((source / f"{arm}_progress.json").read_text(encoding="utf-8"))
            if progress["completed"] != bank_n + len(query_labels) or progress["protocol_SHA256"] != result["protocol_SHA256"]:
                raise ValueError(f"Incomplete or inconsistent {dataset}/{arm}")
            states = np.load(source / f"{arm}_states.npy", mmap_mode="r")
            if states.shape != (bank_n + len(query_labels), 36, 2560) or not np.isfinite(states).all():
                raise ValueError("Invalid states")
            results[arm] = {}
            for method in plan["prespecified_methods"]:
                vectors = features(states, method)
                scores = vectors[bank_n:] @ vectors[:bank_n].T
                results[arm][method] = metrics(scores, labels, query_labels)
        summary = {"bank_n": bank_n, "query_n": len(query_labels), "methods": {}, "queries": []}
        for method in plan["prespecified_methods"]:
            entry = {}
            for measure in ("top1", "top4", "AP"):
                generic = np.asarray([x[measure] for x in results["generic"][method]], dtype=float)
                task = np.asarray([x[measure] for x in results["task_short"][method]], dtype=float)
                entry[measure] = {"generic": float(generic.mean()), "task_short": float(task.mean()),
                                  "paired_delta": ci(task - generic)}
            summary["methods"][method] = entry
        for i, sample in enumerate(table["query"]):
            summary["queries"].append({"sample_id": sample["sample_id"], "label": sample["label"],
                "scores": {arm: {method: results[arm][method][i] for method in plan["prespecified_methods"]} for arm in plan["arms"]}})
        result["datasets"][dataset] = summary
    out = ROOT / "analysis/query_retrieval_local_v1/analysis.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({dataset: {method: value["top4"] for method, value in table["methods"].items()}
        for dataset, table in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

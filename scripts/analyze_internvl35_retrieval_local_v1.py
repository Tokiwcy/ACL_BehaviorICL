"""Audited five-dataset InternVL retrieval and matched Qwen comparison."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_query_retrieval_local_v1 import ci, features, metrics
from analyze_query_retrieval_original_raw_v1 import raw_scores
from query_tam_local_plan_v1 import digest

ARMS = ("generic", "task_short", "task_full", "label_list_only")


def model_metrics(dataset: str, table: dict, source: Path, expected_sha: str) -> dict:
    bank_n = len(table["bank"])
    query_n = len(table["query"])
    bank_labels = [row["label"] for row in table["bank"]]
    query_labels = [row["label"] for row in table["query"]]
    values = {}
    controls = {}
    for arm in ARMS:
        progress = json.loads((source / f"{arm}_progress.json").read_text(encoding="utf-8"))
        if progress["protocol_SHA256"] != expected_sha or progress["completed"] != bank_n + query_n:
            raise ValueError(f"Incomplete or mismatched {source}/{dataset}/{arm}")
        states = np.load(source / f"{arm}_states.npy", mmap_mode="r")
        if states.shape != (bank_n + query_n, 36, 2560):
            raise ValueError(f"Wrong state shape for {dataset}/{arm}: {states.shape}")
        per_query = metrics(raw_scores(states, bank_n), bank_labels, query_labels)
        values[arm] = {measure: np.asarray([row[measure] for row in per_query], dtype=float)
                       for measure in ("top1", "top4", "AP")}
        controls[arm] = {}
        for method in ("layer18", "layer36", "mean_normalized_36_layers", "mean_normalized_layer_velocity"):
            vectors = features(states, method)
            rows = metrics(vectors[bank_n:] @ vectors[:bank_n].T, bank_labels, query_labels)
            controls[arm][method] = {measure: float(np.mean([row[measure] for row in rows]))
                                     for measure in ("top1", "top4", "AP")}
    result = {"bank_n": bank_n, "query_n": query_n}
    for measure in ("top1", "top4", "AP"):
        result[measure] = {
            "rates": {arm: float(values[arm][measure].mean()) for arm in ARMS},
            "paired_full_minus_generic": ci(values["task_full"][measure] - values["generic"][measure]),
            "paired_short_minus_generic": ci(values["task_short"][measure] - values["generic"][measure]),
            "paired_full_minus_labels_only": ci(values["task_full"][measure] - values["label_list_only"][measure]),
        }
    result["layer_controls"] = controls
    return result


def main() -> None:
    path = ROOT / "analysis/internvl35_retrieval_local_v1/protocol.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "protocol_SHA256": digest(path),
        "model": plan["model"], "revision": plan["revision"],
        "metric": "exact repository Raw: equal mean of 36 layerwise cosine similarities",
        "scope": plan["scope"], "datasets": {},
    }
    for name, table in plan["datasets"].items():
        result["datasets"][name] = model_metrics(
            name, table, ROOT / "results/internvl35_retrieval_local_v1" / name, digest(path)
        )
    out = path.with_name("analysis.json")
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: block["top4"]["rates"] for name, block in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()


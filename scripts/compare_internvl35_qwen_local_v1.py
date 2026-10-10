"""Paired cross-model effect contrast on identical 32-query panels."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_query_retrieval_local_v1 import ci, metrics
from analyze_query_retrieval_original_raw_v1 import raw_scores
from query_tam_local_plan_v1 import digest


def query_values(states_path: Path, bank_n: int, bank_labels: list[str], query_labels: list[str]) -> dict:
    states = np.load(states_path, mmap_mode="r")
    rows = metrics(raw_scores(states, bank_n), bank_labels, query_labels)
    return {measure: np.asarray([row[measure] for row in rows], dtype=float)
            for measure in ("top1", "top4", "AP")}


def main() -> None:
    intern_path = ROOT / "analysis/internvl35_retrieval_local_v2/protocol.json"
    qwen1_path = ROOT / "analysis/query_retrieval_local_v1/protocol.json"
    qwen2_path = ROOT / "analysis/query_retrieval_local_v2/protocol.json"
    plan = json.loads(intern_path.read_text(encoding="utf-8"))
    if plan["base_protocol_SHA256"] != digest(qwen1_path) or plan["follow_protocol_SHA256"] != digest(qwen2_path):
        raise ValueError("Matched Qwen protocol identity differs")
    result = {"internvl_protocol_SHA256": digest(intern_path),
              "qwen_base_protocol_SHA256": digest(qwen1_path),
              "qwen_follow_protocol_SHA256": digest(qwen2_path),
              "measure": "Raw equal-layer cosine retrieval; same images and prompt strings, model-specific image processing",
              "datasets": {}}
    for dataset, table in plan["datasets"].items():
        bank_n = len(table["bank"])
        bank_labels = [row["label"] for row in table["bank"]]
        query_labels = [row["label"] for row in table["query"]]
        model_values = {}
        for model_name, root in (("internvl35", ROOT / "results/internvl35_retrieval_local_v2"),
                                 ("qwen3vl4b", ROOT / "results/query_retrieval_local_v1")):
            model_values[model_name] = {}
            for arm in ("generic", "task_full"):
                progress = json.loads((root / dataset / f"{arm}_progress.json").read_text(encoding="utf-8"))
                expected = digest(intern_path) if model_name == "internvl35" else digest(qwen1_path if arm == "generic" else qwen2_path)
                if progress["protocol_SHA256"] != expected or progress["completed"] != bank_n + len(query_labels):
                    raise ValueError(f"Incomplete {model_name}/{dataset}/{arm}")
                model_values[model_name][arm] = query_values(root / dataset / f"{arm}_states.npy",
                                                             bank_n, bank_labels, query_labels)
        block = {"bank_n": bank_n, "query_n": len(query_labels)}
        for measure in ("top1", "top4", "AP"):
            i = model_values["internvl35"]
            q = model_values["qwen3vl4b"]
            intern_effect = i["task_full"][measure] - i["generic"][measure]
            qwen_effect = q["task_full"][measure] - q["generic"][measure]
            block[measure] = {
                "internvl_generic": float(i["generic"][measure].mean()),
                "internvl_full": float(i["task_full"][measure].mean()),
                "qwen_generic": float(q["generic"][measure].mean()),
                "qwen_full": float(q["task_full"][measure].mean()),
                "internvl_effect": ci(intern_effect),
                "qwen_effect": ci(qwen_effect),
                "paired_internvl_minus_qwen_effect": ci(intern_effect - qwen_effect),
            }
        result["datasets"][dataset] = block
    out = ROOT / "analysis/internvl35_retrieval_local_v2/qwen_comparison.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({dataset: block["top4"] for dataset, block in result["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

"""Compare same-model pre-answer Raw retrieval with GPT-MM-style answer states."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_query_retrieval_local_v1 import ci, metrics, normalize
from analyze_query_retrieval_original_raw_v1 import raw_scores
from query_tam_local_plan_v1 import digest


def main() -> None:
    path = ROOT / "analysis/internvl35_answer_state_local_v1/protocol.json"
    plan = json.loads(path.read_text(encoding="utf-8"))
    prepath = ROOT / "analysis/internvl35_retrieval_local_v2/protocol.json"
    if plan["base_protocol_SHA256"] != digest(prepath):
        raise ValueError("Pre-answer study identity changed")
    output = {"protocol_SHA256": digest(path), "base_protocol_SHA256": digest(prepath),
              "comparison": "same model, image budget, task-full prompt and fixed rows; pre-answer 36-layer Raw versus local GPT-MM-style generated-answer final state",
              "datasets": {}}
    for dataset, table in plan["datasets"].items():
        bank_n = len(table["bank"])
        query_n = len(table["query"])
        bank_labels = [x["label"] for x in table["bank"]]
        query_labels = [x["label"] for x in table["query"]]
        folder = ROOT / "results/internvl35_answer_state_local_v1" / dataset
        progress = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
        if progress["protocol_SHA256"] != digest(path) or progress["completed"] != bank_n + query_n:
            raise ValueError(f"Incomplete answer states for {dataset}")
        answer = np.asarray(np.load(folder / "answer_states.npy", mmap_mode="r"), dtype=np.float32)
        if answer.shape != (bank_n + query_n, 2560) or not np.isfinite(answer).all():
            raise ValueError(f"Wrong answer state shape/content: {dataset}")
        answer = normalize(answer)
        answer_rows = metrics(answer[bank_n:] @ answer[:bank_n].T, bank_labels, query_labels)
        pre = np.load(ROOT / "results/internvl35_retrieval_local_v2" / dataset / "task_full_states.npy", mmap_mode="r")
        pre_rows = metrics(raw_scores(pre, bank_n), bank_labels, query_labels)
        block = {"bank_n": bank_n, "query_n": query_n}
        for measure in ("top1", "top4", "AP"):
            a = np.asarray([row[measure] for row in answer_rows], dtype=float)
            p = np.asarray([row[measure] for row in pre_rows], dtype=float)
            block[measure] = {"pre_answer_Raw": float(p.mean()),
                              "answer_state_style": float(a.mean()),
                              "paired_pre_minus_answer": ci(p - a)}
        output["datasets"][dataset] = block
    out = path.with_name("analysis.json")
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({dataset: block["top4"] for dataset, block in output["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

"""Predeclared full-task and label-vocabulary followup on the same fixed bank/query rows."""
import json
from query_tam_local_plan_v1 import ROOT
from multimodal_model_adapter import TASK_NOUNS, task_instruction


def build():
    old_path = ROOT / "analysis/query_retrieval_local_v1/protocol.json"
    old = json.loads(old_path.read_text(encoding="utf-8"))
    plan = {"version": 2, "base_protocol_SHA256": __import__("hashlib").sha256(old_path.read_bytes()).hexdigest(),
            "model": old["model"], "revision": old["revision"],
            "bank_and_query": "exact same rows and order as retrieval v1; cache reuse verified by base protocol SHA",
            "new_arms": ["task_full", "label_list_only"],
            "question": "Is query-conditioned retrieval driven by task instruction, supplied class vocabulary, or both?",
            "datasets": {}}
    for dataset, table in old["datasets"].items():
        labels = table["labels"]
        plan["datasets"][dataset] = {"bank": table["bank"], "query": table["query"],
            "labels": labels,
            "prompts": {"task_full": task_instruction(dataset, labels),
                        "label_list_only": f"The only valid labels are: {', '.join(labels)}. Return exactly one valid label and no explanation. Label:"}}
    return plan


if __name__ == "__main__":
    out = ROOT / "analysis/query_retrieval_local_v2/protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    value = build()
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != value:
        raise ValueError("Frozen v2 retrieval plan differs")
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frozen {len(value['datasets'])} datasets x {len(value['new_arms'])} new arms")

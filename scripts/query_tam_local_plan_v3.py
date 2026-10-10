"""Deterministic 128-per-dataset spatial TAM extension independent of heatmaps."""
import hashlib
import json
from pathlib import Path
from query_tam_local_plan_v1 import ROOT, digest
from multidataset_protocol import DATASETS, load_dataset
from multimodal_model_adapter import TASK_NOUNS, task_instruction


def expanded():
    old = json.loads((ROOT / "analysis/query_tam_local_v2/protocol.json").read_text(encoding="utf-8"))
    old["version"] = 3
    old["sample_count_per_dataset"] = 128
    old["primary"] = "generic versus task_short, fixed first gold-label token, 128 independent test images per dataset"
    old["secondary"] = "task_full and natural first answer token on first 8 per dataset"
    old["limits"][-1] = "128 images per dataset are a spatial attribution study, not full-test ICL accuracy."
    all_examples = []
    for dataset in DATASETS:
        first = [x for x in old["examples"] if x["dataset"] == dataset]
        bank, query = load_dataset(dataset, ROOT / "datasets")
        labels = sorted({s.label for s in bank})
        used = {x["sample_id"] for x in first}
        seen_labels = {x["gold_label"] for x in first}
        ranked = sorted((s for s in query if s.sample_id not in used),
            key=lambda s: hashlib.sha256(("TAM-local-v3|" + s.sample_id).encode()).digest())
        distinct = []
        for sample in ranked:
            if sample.label not in seen_labels:
                distinct.append(sample)
                seen_labels.add(sample.label)
        distinct_ids = {sample.sample_id for sample in distinct}
        prioritized = distinct + [sample for sample in ranked if sample.sample_id not in distinct_ids]
        for sample in prioritized:
            if sample.sample_id in used:
                continue
            used.add(sample.sample_id)
            first.append({"dataset": dataset, "sample_id": sample.sample_id,
                "image_path": str(Path(sample.path).relative_to(ROOT)).replace("\\", "/"),
                "image_SHA256": digest(sample.path), "gold_label": sample.label,
                "selection": "SHA256_rank_v3_independent_of_TAM",
                "queries": {"generic": "Describe the image briefly.",
                    "task_short": f"Classify this image by its {TASK_NOUNS[dataset]}. Give the most specific label.",
                    "task_full": task_instruction(dataset, labels)}})
            if len(first) == 128:
                break
        if len(first) != 128:
            raise ValueError(dataset)
        all_examples.extend(first)
    old["examples"] = all_examples
    return old


if __name__ == "__main__":
    out = ROOT / "analysis/query_tam_local_v3/protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    value = expanded()
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != value:
        raise ValueError("Frozen TAM v3 protocol differs")
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frozen {len(value['examples'])} TAM images")


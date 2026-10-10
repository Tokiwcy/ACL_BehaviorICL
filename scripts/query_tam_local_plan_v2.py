"""Predeclared deterministic expansion of the eight-image TAM pilot to 32 per dataset."""
import hashlib
import json
from pathlib import Path
from query_tam_local_plan_v1 import ROOT, build, digest
from multidataset_protocol import DATASETS, load_dataset
from multimodal_model_adapter import TASK_NOUNS, task_instruction


def expanded():
    plan = build()
    plan["version"] = 2
    plan["sample_count_per_dataset"] = 32
    plan["primary"] = "generic versus task_short, first gold-label token fixed across queries; 32 test images per dataset"
    plan["secondary"] = "task_full and natural greedy first answer token on first 8 per dataset; pre-answer layer cosine"
    plan["limits"][-1] = "32 images per dataset support a paired exploratory analysis, not full-test accuracy."
    examples = []
    for dataset in DATASETS:
        first = [x for x in plan["examples"] if x["dataset"] == dataset]
        bank, query = load_dataset(dataset, ROOT / "datasets")
        labels = sorted({s.label for s in bank})
        used = {x["sample_id"] for x in first}
        seen = {x["gold_label"] for x in first}
        ranked = sorted((s for s in query if s.sample_id not in used),
                        key=lambda s: hashlib.sha256(("TAM-local-v2|" + s.sample_id).encode()).digest())
        for sample in ranked:
            if sample.label in seen:
                continue
            seen.add(sample.label)
            first.append({"dataset": dataset, "sample_id": sample.sample_id,
                "image_path": str(Path(sample.path).relative_to(ROOT)).replace("\\", "/"),
                "image_SHA256": digest(sample.path), "gold_label": sample.label,
                "selection": "SHA256_rank_distinct_label_v2",
                "queries": {"generic": "Describe the image briefly.",
                    "task_short": f"Classify this image by its {TASK_NOUNS[dataset]}. Give the most specific label.",
                    "task_full": task_instruction(dataset, labels)}})
            if len(first) == 32:
                break
        if len(first) != 32:
            raise ValueError(dataset)
        examples.extend(first)
    plan["examples"] = examples
    return plan


if __name__ == "__main__":
    out = ROOT / "analysis/query_tam_local_v2/protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    value = expanded()
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != value:
        raise ValueError("Frozen v2 protocol differs")
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frozen {len(value['examples'])} samples at {out}")

"""Freeze an outcome-independent five-dataset local TAM sample and prompt plan."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from multidataset_protocol import DATASETS, load_dataset
from multimodal_model_adapter import TASK_NOUNS, task_instruction


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def build():
    old = json.loads((ROOT / "analysis/query_tam_pilot_v1/protocol.json").read_text(encoding="utf-8"))
    examples = []
    for dataset in DATASETS:
        bank, query = load_dataset(dataset, ROOT / "datasets")
        labels = sorted({sample.label for sample in bank})
        original = [x for x in old["examples"] if x["dataset"] == dataset]
        chosen = [next(s for s in query if s.sample_id == x["sample_id"]) for x in original]
        used = {s.sample_id for s in chosen}
        ranked = sorted((s for s in query if s.sample_id not in used),
                        key=lambda s: hashlib.sha256(("TAM-local-v1|" + s.sample_id).encode()).digest())
        seen_labels = {s.label for s in chosen}
        for sample in ranked:
            if sample.label not in seen_labels:
                chosen.append(sample)
                seen_labels.add(sample.label)
            if len(chosen) == 8:
                break
        if len(chosen) != 8:
            raise ValueError(dataset)
        for sample in chosen:
            examples.append({
                "dataset": dataset, "sample_id": sample.sample_id,
                "image_path": str(Path(sample.path).relative_to(ROOT)).replace("\\", "/"),
                "image_SHA256": digest(sample.path), "gold_label": sample.label,
                "selection": "frozen_historical_outcome_stratum" if sample.sample_id in used else "SHA256_rank_distinct_label",
                "queries": {
                    "generic": "Describe the image briefly.",
                    "task_short": f"Classify this image by its {TASK_NOUNS[dataset]}. Give the most specific label.",
                    "task_full": task_instruction(dataset, labels),
                },
            })
    plan = {
        "version": 1, "model": old["model"], "revision": old["revision"],
        "TAM_commit": old["TAM"]["commit"], "TAM_SHA256": old["TAM"]["SHA"]["tam.py"],
        "seed": 73, "sample_count_per_dataset": 8,
        "primary": "generic versus task_short, first gold-label token fixed across queries; 8 test images per dataset",
        "secondary": "task_full on first 2 per dataset; natural greedy answer on first 2 per dataset; pre-answer layer cosine",
        "spatial_endpoint": "foreground-mass enrichment relative to region area for CUB/Aircraft/Dogs bounding boxes and Pets trimap; DTD map contrast only",
        "limits": ["Fixed gold token is teacher-forced, not a prediction.", "TAM is token attribution, not literal attention weights or causal proof.",
                   "Original image-before-text ordering makes raw vision-token CAM potentially query-invariant.",
                   "Eight images per dataset are exploratory, not a full-test benchmark."],
        "examples": examples,
    }
    return plan


if __name__ == "__main__":
    out = ROOT / "analysis/query_tam_local_v1/protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if json.loads(out.read_text(encoding="utf-8")) != build():
            raise ValueError("Existing frozen protocol differs")
    else:
        out.write_text(json.dumps(build(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"frozen {len(build()['examples'])} samples at {out}")


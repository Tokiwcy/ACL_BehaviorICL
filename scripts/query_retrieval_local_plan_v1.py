"""Freeze a test-independent two-per-class bank and TAM-selected query panel."""
import hashlib
import json
from pathlib import Path
from query_tam_local_plan_v1 import ROOT, digest
from multidataset_protocol import DATASETS, load_dataset


def build():
    tam = json.loads((ROOT / "analysis/query_tam_local_v2/protocol.json").read_text(encoding="utf-8"))
    result = {"version": 1, "model": tam["model"], "revision": tam["revision"],
        "bank_selection": "SHA256 rank by sample ID, two images per class; all dataset classes included; no test-label-dependent bank selection",
        "queries": "same 32 deterministic test images per dataset as TAM v2",
        "arms": ["generic", "task_short"], "features": "36 pre-answer final input decoder states, no generation or training",
        "retrieval_measures": ["top1_same_label", "top4_any_same_label", "same_label_average_precision"],
        "prespecified_methods": ["layer18", "layer36", "mean_normalized_36_layers", "mean_normalized_layer_velocity"],
        "datasets": {}}
    for dataset in DATASETS:
        bank, query = load_dataset(dataset, ROOT / "datasets")
        labels = sorted({x.label for x in bank})
        selected = []
        for label in labels:
            candidates = sorted((x for x in bank if x.label == label),
                key=lambda x: hashlib.sha256(("retrieval-local-v1|" + x.sample_id).encode()).digest())
            if len(candidates) < 2:
                raise ValueError(f"Insufficient bank: {dataset}/{label}")
            selected.extend(candidates[:2])
        qids = {x["sample_id"] for x in tam["examples"] if x["dataset"] == dataset}
        qselected = [x for x in query if x.sample_id in qids]
        if len(qselected) != 32:
            raise ValueError(dataset)
        def row(sample):
            return {"sample_id": sample.sample_id, "path": str(Path(sample.path).relative_to(ROOT)).replace("\\", "/"),
                "image_SHA256": digest(sample.path), "label": sample.label}
        result["datasets"][dataset] = {"bank": [row(x) for x in selected],
                                         "query": [row(x) for x in qselected],
                                         "labels": labels}
    return result


if __name__ == "__main__":
    out = ROOT / "analysis/query_retrieval_local_v1/protocol.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    value = build()
    if out.exists() and json.loads(out.read_text(encoding="utf-8")) != value:
        raise ValueError("Frozen retrieval plan differs")
    out.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print("frozen bank/query counts", {k: (len(v["bank"]), len(v["query"])) for k, v in value["datasets"].items()})

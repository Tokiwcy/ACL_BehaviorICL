"""Replay a saved context after planned OS update; never tune scientific settings."""
import argparse
import json
from pathlib import Path
import torch
from internvl35_joint_config_v2 import ANALYSIS, require_gate, load_model, write_json, digest
from run_internvl35_joint_methods_v2 import load_context
from multimodal_model_adapter import icl_messages, generate_legal_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    require_gate()
    folder, bank, queries, identity = load_context("cub")
    records = [json.loads(line) for line in (folder / "predictions/raw.jsonl").read_bytes().splitlines()]
    if len(records) != 2688:
        raise ValueError("Restart replay is bound to the saved2688-row prefix")
    row = records[-1]
    sample = queries[len(records)-1]
    selections = json.loads((folder / "selections/raw.json").read_text(encoding="utf-8"))
    demos = [bank[i] for i in selections[sample.sample_id]]
    if row["query_id"] != sample.sample_id or row["demo_ids"] != [d.sample_id for d in demos]:
        raise ValueError("Saved context identity changed")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    images = {r["sample_id"]:r for split in ("bank", "query") for r in manifest[split]}
    for image in demos+[sample]:
        if digest(image.path) != images[image.sample_id]["image_sha256"]:
            raise ValueError("Replay image changed")
    torch.manual_seed(73)
    model, processor = load_model(args.report.with_suffix(".inputs.jsonl"))
    torch.cuda.reset_peak_memory_stats()
    labels = sorted({b.label for b in bank})
    prediction, raw = generate_legal_labels(model, processor,
        [icl_messages(demos, sample, "cub", labels, None)], labels)[0]
    passed = prediction == row["prediction"] and raw == row["raw_output"]
    write_json(args.report, {"state":"complete" if passed else "failed",
        "query_id":sample.sample_id, "demo_ids":row["demo_ids"],
        "expected_prediction":row["prediction"], "prediction":prediction,
        "expected_raw":row["raw_output"], "raw":raw,
        "legal":prediction in labels, "exact_saved_output":passed,
        "peak_reserved_gb":torch.cuda.max_memory_reserved()/2**30,
        "accuracy_used_for_tuning":False})
    if not passed:
        raise RuntimeError("Updated-system replay differs from saved output; do not resume")

if __name__=="__main__":
    main()

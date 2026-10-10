"""Analyze full-label prompt and natural first-token controls on frozen 8-image panel."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_query_tam_local_v1 import cosine, mass_enrichment, region, summarize
from query_tam_local_plan_v1 import digest


def main():
    plan_path = ROOT / "analysis/query_tam_local_v1/protocol.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source = ROOT / "results/query_tam_local_v1"
    rows = []
    for example in plan["examples"]:
        runs = {}
        for arm, mode in (("generic", "fixed"), ("task_short", "fixed"),
                          ("task_full", "fixed"), ("generic", "natural"), ("task_short", "natural")):
            folder = source / f"{example['dataset']}__{example['sample_id']}__{arm}__{mode}"
            record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
            if record["maps_SHA256"] != digest(folder / "maps.npz") or record["query"] != example["queries"][arm]:
                raise ValueError("Run identity or map SHA changed")
            runs[(arm, mode)] = (record, dict(np.load(folder / "maps.npz", allow_pickle=False)))
        image = Image.open(ROOT / example["image_path"])
        roi = region(example, runs[("generic", "fixed")][1]["maps"].shape[1:], image.size)
        row = {"dataset": example["dataset"], "sample_id": example["sample_id"],
               "natural_first_token_same": runs[("generic", "natural")][0]["target_token_id"] == runs[("task_short", "natural")][0]["target_token_id"],
               "natural_generic_token": runs[("generic", "natural")][0]["target_token"],
               "natural_task_token": runs[("task_short", "natural")][0]["target_token"]}
        for label, left, right in (("short_vs_generic", ("generic", "fixed"), ("task_short", "fixed")),
                                   ("full_vs_generic", ("generic", "fixed"), ("task_full", "fixed")),
                                   ("full_vs_short", ("task_short", "fixed"), ("task_full", "fixed")),
                                   ("natural_task_vs_generic", ("generic", "natural"), ("task_short", "natural"))):
            a, b = runs[left], runs[right]
            if a[0]["processed_pixels_SHA256"] != b[0]["processed_pixels_SHA256"] or a[0]["grid"] != b[0]["grid"]:
                raise ValueError("Image/grid changed")
            if left[1] == "fixed" and a[0]["target_token_id"] != b[0]["target_token_id"]:
                raise ValueError("Fixed target changed")
            row[label] = {"tam_cosine": cosine(a[1]["maps"], b[1]["maps"]),
                "raw_CAM_max_difference": float(np.max(np.abs(a[1]["raw_CAM"].astype(float) - b[1]["raw_CAM"].astype(float)))),
                "last_layer_anchor_cosine": cosine(a[1]["anchor_states"][-1], b[1]["anchor_states"][-1])}
            if roi is not None:
                am = mass_enrichment(a[1]["maps"][0], roi)
                bm = mass_enrichment(b[1]["maps"][0], roi)
                row[label]["delta_excess"] = bm["excess_mass"] - am["excess_mass"] if am and bm else None
        rows.append(row)
    summary = {"protocol_SHA256": digest(plan_path), "n": len(rows), "datasets": {}, "rows": rows}
    for dataset in ("dtd", "aircraft", "cub", "dogs", "pets"):
        subset = [r for r in rows if r["dataset"] == dataset]
        block = {"n": len(subset), "natural_first_token_same": sum(x["natural_first_token_same"] for x in subset)}
        for label in ("short_vs_generic", "full_vs_generic", "full_vs_short", "natural_task_vs_generic"):
            block[label] = {measure: summarize([r[label][measure] for r in subset if r[label].get(measure) is not None])
                for measure in ("tam_cosine", "raw_CAM_max_difference", "last_layer_anchor_cosine", "delta_excess")}
        summary["datasets"][dataset] = block
    out = ROOT / "analysis/query_tam_local_v1/controls_analysis.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {"full_delta": v["full_vs_generic"]["delta_excess"],
                           "natural_same_token": v["natural_first_token_same"]} for k, v in summary["datasets"].items()}, indent=2))


if __name__ == "__main__":
    main()

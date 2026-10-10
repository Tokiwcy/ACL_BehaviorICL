"""CPU-only, paired spatial analysis of the frozen local five-dataset TAM study."""
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from query_tam_local_plan_v1 import digest


def region(example, shape, size):
    dataset = example["dataset"]
    if dataset == "dtd":
        return None
    width, height = size
    if dataset == "pets":
        stem = Path(example["image_path"]).stem
        path = ROOT / "datasets/oxford_iiit_pet/annotations/trimaps" / (stem + ".png")
        if not path.is_file():
            raise FileNotFoundError(path)
        source = Image.open(path)
        mask = Image.fromarray((np.asarray(source) == 1).astype(np.uint8) * 255)
        return np.asarray(mask.resize((shape[1], shape[0]), Image.Resampling.BOX), dtype=float) / 255
    if dataset == "cub":
        index = int(example["sample_id"].split("_")[1])
        lines = (ROOT / "datasets/CUB_200_2011/bounding_boxes.txt").read_text().splitlines()
        values = next(line.split() for line in lines if int(line.split()[0]) == index)
        x, y, w, h = map(float, values[1:])
        box = (x, y, x + w, y + h)
    elif dataset == "aircraft":
        ident = example["sample_id"]
        lines = (ROOT / "datasets/fgvc-aircraft-2013b/data/images_box.txt").read_text().splitlines()
        values = next(line.split() for line in lines if line.split()[0] == ident)
        box = tuple(map(float, values[1:]))
    elif dataset == "dogs":
        filename = Path(example["image_path"]).name
        with (ROOT / "datasets/stanford_dogs/metadata.csv").open(newline="", encoding="utf-8") as stream:
            values = next(json.loads(row["annotations"]) for row in csv.DictReader(stream) if row["name"] == filename)
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        for box in values:
            draw.rectangle(tuple(box), fill=255)
        return np.asarray(mask.resize((shape[1], shape[0]), Image.Resampling.BOX), dtype=float) / 255
    else:
        raise ValueError(dataset)
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return np.asarray(mask.resize((shape[1], shape[0]), Image.Resampling.BOX), dtype=float) / 255


def mass_enrichment(values, roi):
    values = np.maximum(np.asarray(values, dtype=float), 0)
    total = values.sum()
    area = float(roi.mean())
    if total <= 0 or area <= 0 or area >= 1:
        return None
    mass = float((values * roi).sum() / total)
    return {"foreground_mass": mass, "area_fraction": area, "enrichment": mass / area,
            "excess_mass": mass - area}


def cosine(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator else None


def summarize(values):
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"n": 0}
    rng = np.random.default_rng(73)
    means = np.mean(array[rng.integers(0, len(array), size=(10000, len(array)))], axis=1)
    return {"n": len(array), "mean": float(array.mean()), "median": float(np.median(array)),
            "bootstrap_mean_95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
            "positive": int((array > 0).sum()), "negative": int((array < 0).sum()), "zero": int((array == 0).sum())}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("analysis/query_tam_local_v2/protocol.json"))
    args = parser.parse_args()
    plan_path = ROOT / args.protocol
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source = ROOT / "results/query_tam_local_v1"
    output = plan_path.parent
    rows = []
    for example in plan["examples"]:
        runs = {}
        for arm in ("generic", "task_short"):
            key = f"{example['dataset']}__{example['sample_id']}__{arm}__fixed"
            folder = source / key
            record = json.loads((folder / "record.json").read_text(encoding="utf-8"))
            if record["query"] != example["queries"][arm] or record["image_SHA256"] != example["image_SHA256"] or record["maps_SHA256"] != digest(folder / "maps.npz"):
                raise ValueError(f"Run identity/hash mismatch: {key}")
            arrays = dict(np.load(folder / "maps.npz", allow_pickle=False))
            if arrays["anchor_states"].shape != (36, 2560) or any(not np.isfinite(x).all() for x in arrays.values()):
                raise ValueError(f"Invalid arrays: {key}")
            runs[arm] = (record, arrays)
        generic, task = runs["generic"], runs["task_short"]
        if generic[0]["target_token_id"] != task[0]["target_token_id"] or generic[0]["processed_pixels_SHA256"] != task[0]["processed_pixels_SHA256"] or generic[0]["grid"] != task[0]["grid"]:
            raise ValueError("Fixed token/image/grid control failed")
        image = Image.open(ROOT / example["image_path"])
        roi = region(example, generic[1]["maps"].shape[1:], image.size)
        row = {"dataset": example["dataset"], "sample_id": example["sample_id"],
               "gold_label": example["gold_label"], "selection": example["selection"],
               "target_token": generic[0]["target_token"], "roi_available": roi is not None,
               "last_layer_anchor_cosine": cosine(generic[1]["anchor_states"][-1], task[1]["anchor_states"][-1]),
               "layer_anchor_cosines": [cosine(a, b) for a, b in zip(generic[1]["anchor_states"], task[1]["anchor_states"])],
               "seconds_total": generic[0]["seconds"] + task[0]["seconds"]}
        for name in ("maps", "raw_CAM", "post_ECI"):
            a, b = generic[1][name][0], task[1][name][0]
            row[name + "_cosine"] = cosine(a, b)
            row[name + "_max_difference"] = float(np.max(np.abs(a.astype(float) - b.astype(float))))
            if roi is not None:
                row[name + "_generic"] = mass_enrichment(a, roi)
                row[name + "_task"] = mass_enrichment(b, roi)
                row[name + "_delta_excess"] = (row[name + "_task"]["excess_mass"] - row[name + "_generic"]["excess_mass"]
                    if row[name + "_task"] and row[name + "_generic"] else None)
        rows.append(row)
    summary = {"protocol_SHA256": digest(plan_path), "images": len(rows), "paired_runs": 2 * len(rows),
               "datasets": {}, "limits": plan["limits"]}
    for dataset in ("dtd", "aircraft", "cub", "dogs", "pets"):
        subset = [x for x in rows if x["dataset"] == dataset]
        summary["datasets"][dataset] = {"n": len(subset),
            "tam_cosine": summarize([x["maps_cosine"] for x in subset if x["maps_cosine"] is not None]),
            "tam_max_difference": summarize([x["maps_max_difference"] for x in subset]),
            "raw_CAM_max_difference": summarize([x["raw_CAM_max_difference"] for x in subset]),
            "last_layer_anchor_cosine": summarize([x["last_layer_anchor_cosine"] for x in subset if x["last_layer_anchor_cosine"] is not None]),
            "tam_delta_excess": summarize([x["maps_delta_excess"] for x in subset if x.get("maps_delta_excess") is not None]),
            "raw_CAM_delta_excess": summarize([x["raw_CAM_delta_excess"] for x in subset if x.get("raw_CAM_delta_excess") is not None]),
            "post_ECI_delta_excess": summarize([x["post_ECI_delta_excess"] for x in subset if x.get("post_ECI_delta_excess") is not None])}
    (output / "analysis.json").write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


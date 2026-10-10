"""Deterministic five-dataset TAM contact sheet on the first frozen sample each."""
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_query_tam_local_v1 import region


def tile(image, values=None, roi=None):
    source = image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    if values is None and roi is None:
        return source
    if values is not None:
        heat = Image.fromarray(np.uint8(np.clip(values, 0, 255))).resize((256, 256), Image.Resampling.BILINEAR)
        v = np.asarray(heat, dtype=np.float32) / 255
        overlay = Image.fromarray(np.uint8(np.stack([255 * v, np.zeros_like(v), 255 * (1 - v)], axis=-1)))
        return Image.blend(source, overlay, 0.45)
    mask = Image.fromarray(np.uint8(np.clip(roi * 255, 0, 255))).resize((256, 256), Image.Resampling.BILINEAR)
    original = np.asarray(source, dtype=np.float32)
    alpha = np.asarray(mask, dtype=np.float32)[..., None] / 255 * 0.45
    green = np.zeros_like(original)
    green[..., 1] = 255
    return Image.fromarray(np.uint8(original * (1 - alpha) + green * alpha))


def main():
    plan = json.loads((ROOT / "analysis/query_tam_local_v2/protocol.json").read_text(encoding="utf-8"))
    examples = [next(x for x in plan["examples"] if x["dataset"] == dataset)
                for dataset in ("dtd", "aircraft", "cub", "dogs", "pets")]
    sheet = Image.new("RGB", (1120, 5 * 300 + 50), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 19)
    for j, title in enumerate(("Original", "Generic TAM", "Task TAM", "Independent ROI")):
        draw.text((j * 280 + 12, 12), title, fill="black", font=font)
    for i, example in enumerate(examples):
        image = Image.open(ROOT / example["image_path"]).convert("RGB")
        maps = []
        for arm in ("generic", "task_short"):
            folder = ROOT / "results/query_tam_local_v1" / f"{example['dataset']}__{example['sample_id']}__{arm}__fixed"
            maps.append(np.load(folder / "maps.npz", allow_pickle=False)["maps"][0])
        roi = region(example, maps[0].shape, image.size)
        panels = [tile(image), tile(image, values=maps[0]), tile(image, values=maps[1]),
                  tile(image, roi=roi) if roi is not None else Image.new("RGB", (256, 256), "white")]
        y = 50 + i * 300
        for j, panel in enumerate(panels):
            sheet.paste(panel, (j * 280 + 12, y))
        caption = f"{example['dataset'].upper()} {example['sample_id']} | {example['gold_label']}"
        draw.text((12, y + 260), caption[:45], fill="black", font=font)
        if roi is None:
            draw.text((3 * 280 + 12, y + 120), "No object mask", fill="black", font=font)
    path = ROOT / "analysis/query_tam_local_v2/contact_sheet.png"
    sheet.save(path)
    print(path)


if __name__ == "__main__":
    main()

"""Resumable full official-split InternVL3.5-4B source extraction.

This local protocol is separate from the historical InternVL2.5-8B cloud lock.
It reuses the repository's canonical dataset loader and source kernels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = "OpenGVLab/InternVL3_5-4B-HF"
REVISION = "6bd4487402110ef9889ba50eb7aefeb302526fed"
DATASETS = ("dtd", "aircraft", "cub", "dogs", "pets")
OUTPUT = ROOT / "results/internvl35_full_v1"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def freeze_manifest(dataset: str, bank: list, query: list, folder: Path) -> dict:
    path = folder / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if ([r["sample_id"] for r in manifest["bank"]] != [r.sample_id for r in bank]
                or [r["sample_id"] for r in manifest["query"]] != [r.sample_id for r in query]
                or [r["label"] for r in manifest["bank"]] != [r.label for r in bank]
                or [r["label"] for r in manifest["query"]] != [r.label for r in query]):
            raise ValueError("Full official manifest order/labels changed")
    else:
        manifest = {"dataset": dataset, "bank": [], "query": []}
        for split, samples in (("bank", bank), ("query", query)):
            for sample in samples:
                manifest[split].append({**asdict(sample), "image_sha256": digest(Path(sample.path))})
        write_json(path, manifest)
    # Check the actual local image bytes before every resumed scientific stage.
    for split in ("bank", "query"):
        for row in manifest[split]:
            if digest(Path(row["path"])) != row["image_sha256"]:
                raise ValueError(f"Image changed: {row['path']}")
    return manifest


def load_model():
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    snapshot = ROOT / ".hf_cache/models--OpenGVLab--InternVL3_5-4B-HF/snapshots" / REVISION
    if not snapshot.is_dir():
        raise FileNotFoundError(snapshot)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)
    processor.image_processor.max_patches = 1
    model = AutoModelForImageTextToText.from_pretrained(
        snapshot, dtype=torch.bfloat16, local_files_only=True, low_cpu_mem_usage=True,
    ).eval().to("cuda")
    if (model.config.text_config.num_hidden_layers, model.config.text_config.hidden_size) != (36, 2560):
        raise ValueError("Unexpected model dimensions")
    return model, processor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--stage", choices=("manifest", "anchor", "gpt_mm", "clip"), required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    from multidataset_protocol import load_dataset
    import run_multidataset_main as main_run

    bank, query = load_dataset(args.dataset, ROOT / "datasets")
    labels = sorted({sample.label for sample in bank})
    if {sample.label for sample in query} != set(labels):
        raise ValueError("Official test label space differs from bank")
    folder = OUTPUT / args.dataset
    folder.mkdir(parents=True, exist_ok=True)
    manifest = freeze_manifest(args.dataset, bank, query, folder)
    identity = {"version": 1, "model": MODEL, "revision": REVISION,
                "dataset": args.dataset, "seed": 73, "shots": 4,
                "image_max_patches": 1, "image_patch_size": 448,
                "source_prompt": "multimodal_model_adapter.task_instruction",
                "manifest_sha256": digest(folder / "manifest.json"),
                "bank_n": len(bank), "query_n": len(query), "label_n": len(labels)}
    identity_path = folder / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text(encoding="utf-8")) != identity:
        raise ValueError("Full study identity changed")
    write_json(identity_path, identity)
    if args.stage == "manifest":
        print(json.dumps(identity, indent=2), flush=True)
        return
    samples = bank + query
    if args.stage == "clip":
        main_run.extract_clip_cache(samples, ROOT / ".hf_cache", max(1, args.batch_size),
                                    folder / "clip_features.npy", True)
        return
    model, processor = load_model()
    if args.stage == "anchor":
        main_run.extract_anchor_cache(
            model, processor, samples, args.dataset, labels, None,
            args.batch_size, folder / "anchor_states.npy", folder / "anchor_states_progress.json", True,
        )
    elif args.stage == "gpt_mm":
        main_run.extract_gpt_mm_cache(
            model, processor, samples, args.dataset, labels, None, args.batch_size, 12,
            folder / "gpt_mm_embeddings.npy", folder / "gpt_mm_progress.json",
            folder / "gpt_mm_zero_shot_outputs.jsonl", True,
        )
    print(f"{args.dataset}/{args.stage}: {len(samples)} complete", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build only frozen Qwen input states needed for a new DeTriever tuple.

Unlike the five-method main runner, this intentionally skips CLIP, GPT-MM and
all other retrievers. Query labels are never used for input-state extraction.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from multidataset_protocol import DATASETS, load_dataset, protocol_metadata, run_directory
from multimodal_model_adapter import MODEL_SPECS, load_model
from run_multidataset_main import extract_anchor_cache, validate_or_write_identity, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("results/cdr_main"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    spec = MODEL_SPECS["qwen3vl4b"]
    bank, query = load_dataset(args.dataset, args.datasets_root)
    labels = sorted({row.label for row in bank})
    run_dir = run_directory(args.output_root, args.dataset, spec.slug, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"dataset": args.dataset, "model_slug": spec.slug,
                "model_id": spec.model_id, "seed": args.seed}
    validate_or_write_identity(run_dir, identity, args.resume)
    manifest_path = run_dir / "manifest.json"
    manifest = {"bank": [asdict(row) for row in bank], "query": [asdict(row) for row in query]}
    if manifest_path.exists() and args.resume:
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Existing frozen-state manifest differs from this tuple")
    write_json(manifest_path, manifest)
    write_json(run_dir / "protocol.json", protocol_metadata(
        args.dataset, spec.model_id, args.seed, bank, query
    ))
    state_path = run_dir / "anchor_states.npy"
    progress_path = run_dir / "anchor_states_progress.json"
    if args.resume and state_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("completed") == len(bank) + len(query):
            states = np.load(state_path, mmap_mode="r")
            if states.shape[0] != len(bank) + len(query):
                raise RuntimeError("Frozen-state cache shape differs from manifest")
            print(f"Already complete: {len(states)} input-state rows", flush=True)
            return
    model, processor = load_model(spec, args.cache_dir)
    extract_anchor_cache(
        model, processor, bank + query, args.dataset, labels, spec.image_size,
        args.batch_size, state_path, progress_path, args.resume,
    )
    print(f"Input-state cache complete: {len(bank)} bank + {len(query)} query", flush=True)


if __name__ == "__main__":
    main()

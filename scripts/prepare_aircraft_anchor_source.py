#!/usr/bin/env python
"""Build only the frozen Aircraft anchor cache required by Decision ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

import run_multidataset_main as main_run
from multidataset_protocol import load_dataset, protocol_metadata, run_directory
from multimodal_model_adapter import MODEL_SPECS, load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--source-root", type=Path, default=Path("results/cdr_main"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    spec = MODEL_SPECS["qwen3vl4b"]
    bank, query = load_dataset("aircraft", args.datasets_root)
    samples = bank + query
    labels = sorted({item.label for item in bank})
    run_dir = run_directory(args.source_root, "aircraft", spec.slug, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"dataset": "aircraft", "model_slug": spec.slug,
                "model_id": spec.model_id, "seed": args.seed}
    main_run.validate_or_write_identity(run_dir, identity, args.resume)
    manifest = {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]}
    path = run_dir / "manifest.json"
    if path.exists() and args.resume:
        if json.loads(path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Aircraft source manifest mismatch")
    main_run.write_json(path, manifest)
    main_run.write_json(run_dir / "protocol.json",
                        protocol_metadata("aircraft", spec.model_id, args.seed, bank, query))
    state_path = run_dir / "anchor_states.npy"
    progress_path = run_dir / "anchor_states_progress.json"
    if args.resume and state_path.exists() and progress_path.exists():
        saved = json.loads(progress_path.read_text(encoding="utf-8"))
        if saved.get("completed") == len(samples) and saved.get("shape") == [len(samples), 36, 2560]:
            print("Aircraft anchor cache already complete", flush=True)
            return
    model, processor = load_model(spec, args.cache_dir)
    main_run.extract_anchor_cache(model, processor, samples, "aircraft", labels,
                                  spec.image_size, 2, state_path, progress_path, args.resume)
    print("Aircraft anchor cache complete", flush=True)


if __name__ == "__main__":
    main()

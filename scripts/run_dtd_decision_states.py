#!/usr/bin/env python
"""Behavior-Zero/Learn retrieval from 36 raw decoder-layer states.

Persisted method identifiers remain decision_zero/decision_learn so existing runs
and resumable cloud artifacts retain their original identity.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import run_dtd_learned_velocity as learned
import run_multidataset_main as main_run
from multidataset_protocol import DATASETS, load_dataset, protocol_metadata, run_directory
from multimodal_model_adapter import MODEL_SPECS


METHODS = ("decision_zero", "decision_learn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, default="dtd")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--source-root", type=Path, default=Path("results/cdr_main"))
    parser.add_argument("--output-root", type=Path, default=Path("results/decision_state_dtd"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def verify_source(source: Path, bank: list, query: list, spec, seed: int,
                  dataset: str) -> np.ndarray:
    identity = json.loads((source / "run_identity.json").read_text(encoding="utf-8"))
    expected = {"dataset": dataset, "model_slug": spec.slug, "model_id": spec.model_id, "seed": seed}
    if identity != expected:
        raise RuntimeError("Frozen-state source identity mismatch")
    expected_manifest = {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]}
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest != expected_manifest:
        raise RuntimeError("Frozen-state source bank/query manifest mismatch")
    progress = json.loads((source / "anchor_states_progress.json").read_text(encoding="utf-8"))
    expected_shape = [len(bank) + len(query), 36, 2560]
    if progress.get("completed") != expected_shape[0] or progress.get("shape") != expected_shape:
        raise RuntimeError("Frozen-state cache is incomplete or has the wrong shape")
    states = np.load(source / "anchor_states.npy", mmap_mode="r")
    if list(states.shape) != expected_shape:
        raise RuntimeError("Frozen-state array has the wrong shape")
    return states


@torch.inference_mode()
def zero_selections(states: np.ndarray, bank: list, query: list, shots: int) -> dict[str, list[int]]:
    # A chunked copy avoids simultaneous float16/float32 full-bank allocations
    # on 16 GB cards and the read-only-memmap PyTorch warning.
    encoded = torch.empty(states.shape, dtype=torch.float32, device="cuda")
    for offset in range(0, len(states), 64):
        chunk = torch.from_numpy(np.array(states[offset:offset + 64], copy=True)).to("cuda").float()
        encoded[offset:offset + len(chunk)] = F.normalize(chunk, dim=-1)
        del chunk
    bank_n = len(bank)
    result = {}
    for start in range(0, len(query), 24):
        batch = encoded[bank_n + start : bank_n + start + 24]
        scores = torch.einsum("qld,bld->qb", batch, encoded[:bank_n]) / encoded.shape[1]
        top = scores.topk(shots, dim=1).indices.flip(1).cpu().numpy()
        for offset, indices in enumerate(top):
            result[query[start + offset].sample_id] = indices.tolist()
        if (start + 24) % 240 == 0 or start + 24 >= len(query):
            print(f"Decision-Zero selections {min(start + 24, len(query))}/{len(query)}", flush=True)
    del encoded
    torch.cuda.empty_cache()
    return result


def learn_selections(states: np.ndarray, bank: list, query: list,
                     args: argparse.Namespace, run_dir: Path) -> dict[str, list[int]]:
    labels = [sample.label for sample in bank]
    train_indices, val_indices = learned.stratified_development_split(labels, 10, args.seed)
    train_args = argparse.Namespace(
        seed=args.seed, shots=args.shots, projection_dim=256,
        weights_lr=1e-2, projection_lr=3e-4, temperature=0.07,
        classes_per_batch=16, samples_per_class=4,
        train_steps=args.train_steps, eval_every=100,
    )
    model, _ = learned.train_variant(
        "projected_state", states[: len(bank)], labels,
        train_indices, val_indices, train_args, run_dir,
    )
    bank_encoded = learned.encode_indices(model, states, np.arange(len(bank)))
    query_encoded = learned.encode_indices(
        model, states, np.arange(len(bank), len(bank) + len(query))
    )
    result = {}
    with torch.inference_mode():
        for start in range(0, len(query), 64):
            scores = model.score(query_encoded[start : start + 64], bank_encoded)
            top = scores.topk(args.shots, dim=1).indices.flip(1).cpu().numpy()
            for offset, indices in enumerate(top):
                result[query[start + offset].sample_id] = indices.tolist()
    del model, bank_encoded, query_encoded
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.generation_batch_size < 1 or args.shots < 1 or args.train_steps < 1:
        raise ValueError("Batch size, shots, and training steps must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = MODEL_SPECS["qwen3vl4b"]
    bank, query = load_dataset(args.dataset, args.datasets_root)
    labels = sorted({sample.label for sample in bank})
    source = run_directory(args.source_root, args.dataset, spec.slug, args.seed)
    states = verify_source(source, bank, query, spec, args.seed, args.dataset)
    run_dir = run_directory(args.output_root, args.dataset, spec.slug, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"dataset": args.dataset, "model_slug": spec.slug, "model_id": spec.model_id,
                "seed": args.seed, "ablation": "36_raw_layer_states_v1"}
    main_run.validate_or_write_identity(run_dir, identity, args.resume)
    manifest = {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]}
    manifest_path = run_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Decision run manifest differs from the saved tuple")
    main_run.write_json(manifest_path, manifest)
    protocol = protocol_metadata(args.dataset, spec.model_id, args.seed, bank, query)
    protocol["ablation"] = "36 normalized raw layer states; no adjacent-layer subtraction"
    protocol["decision_learn"] = "bank-only supervised contrastive training; ten validation samples per class"
    main_run.write_json(run_dir / "protocol.json", protocol)
    main_run.write_json(run_dir / "config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "model": spec.slug, "methods": list(METHODS),
        "source_run": str(source), "state_shape": list(states.shape),
    })
    selections_path = run_dir / "selections.json"
    if args.resume and selections_path.exists():
        main_run.validate_generation_resume(run_dir, bank, query, list(METHODS), args.shots)
        selections = main_run.load_selections(selections_path, bank)
        if set(selections) != set(METHODS):
            raise RuntimeError("Decision selection method set mismatch")
    else:
        selections = {"decision_zero": zero_selections(states, bank, query, args.shots)}
        selections["decision_learn"] = learn_selections(states, bank, query, args, run_dir)
        main_run.save_selections(selections_path, selections, bank)
    main_run.write_json(run_dir / "retrieval_diagnostics.json",
                        main_run.retrieval_diagnostics(selections, bank, query))
    if args.selection_only:
        return
    args.methods = list(METHODS)
    args.vision_pixels = None
    main_run.run_generation(args, spec, bank, query, labels, run_dir, selections)


if __name__ == "__main__":
    main()

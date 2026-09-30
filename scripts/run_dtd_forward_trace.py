#!/usr/bin/env python
"""Fine-grained ablation: retrieve by a compressed zero-shot VLM prefill trace."""

from __future__ import annotations

import argparse
import gc
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

import run_dtd_learned_velocity as learned
import run_multidataset_main as main_run
from multidataset_protocol import load_dataset, protocol_metadata, run_directory
from multimodal_model_adapter import (
    MODEL_SPECS, load_model, model_dimensions, prepare_batch, zero_shot_messages,
)


METHODS = ("forward_zero", "forward_learn")
STREAMS = ("answer_anchor", "visual_token_mean", "answer_layer_update")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dtd", "aircraft"), default="dtd")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("results/full_forward_dtd"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--stage", choices=("probe", "features", "retrieval", "generation", "all"),
                        default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def forward_trace(model, inputs: dict) -> tuple[np.ndarray, int]:
    """Layer-major [anchor, visual mean, anchor update], never using a label answer."""
    layers, width = model_dimensions(model)
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("Trace extraction requires one unpadded sample at a time")
    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(model.config.text_config, "image_token_id", None)
    if image_token_id is None:
        raise RuntimeError("Qwen config lacks image_token_id; cannot locate visual positions")
    visual_mask = inputs["input_ids"][0].eq(image_token_id)
    visual_count = int(visual_mask.sum())
    if visual_count == 0:
        raise RuntimeError("No image placeholder tokens in the processed prompt")
    output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    states = output.hidden_states
    if states is None or len(states) != layers + 1:
        raise RuntimeError("Unexpected number of decoder hidden-state layers")
    if any(state.shape[1] != visual_mask.numel() or state.shape[-1] != width for state in states):
        raise RuntimeError("Hidden-state positions do not align with processor input IDs")
    trace = np.empty((layers, len(STREAMS), width), dtype=np.float16)
    previous_anchor = states[0][0, -1].float()
    for layer, state in enumerate(states[1:]):
        anchor = state[0, -1].float()
        visual_mean = state[0, visual_mask].float().mean(dim=0)
        update = anchor - previous_anchor
        trace[layer, 0] = anchor.cpu().numpy().astype(np.float16)
        trace[layer, 1] = visual_mean.cpu().numpy().astype(np.float16)
        trace[layer, 2] = update.cpu().numpy().astype(np.float16)
        previous_anchor = anchor
    if not np.isfinite(trace).all():
        raise RuntimeError("Non-finite trace values")
    return trace, visual_count


def feature_cache(model, processor, samples, labels, dataset: str, spec, run_dir: Path,
                  resume: bool) -> np.ndarray:
    layers, width = model_dimensions(model)
    shape = (len(samples), layers, len(STREAMS), width)
    feature_path = run_dir / "forward_trace.npy"
    progress_path = run_dir / "forward_trace_progress.json"
    start = 0
    if resume and feature_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("shape") != list(shape):
            raise RuntimeError("Saved trace shape differs from current model/protocol")
        start = int(progress["completed"])
        if not 0 <= start <= len(samples):
            raise RuntimeError("Invalid saved trace progress")
        cache = open_memmap(feature_path, mode="r+", dtype=np.float16, shape=shape)
    else:
        cache = open_memmap(feature_path, mode="w+", dtype=np.float16, shape=shape)
        main_run.write_json(progress_path, {"shape": list(shape), "completed": 0,
                                            "streams": list(STREAMS)})
    if start == len(samples):
        return np.load(feature_path, mmap_mode="r")
    for index in range(start, len(samples)):
        inputs = prepare_batch(processor, [zero_shot_messages(
            samples[index], dataset, labels, spec.image_size)])
        trace, visual_count = forward_trace(model, inputs)
        cache[index] = trace
        if (index + 1) % 10 == 0 or index + 1 == len(samples):
            cache.flush()
            main_run.write_json(progress_path, {"shape": list(shape), "completed": index + 1,
                                                "streams": list(STREAMS),
                                                "last_visual_tokens": visual_count})
        del inputs, trace
        if (index + 1) % 20 == 0:
            torch.cuda.empty_cache()
        if (index + 1) % 100 == 0 or index + 1 == len(samples):
            print(f"forward trace {index + 1}/{len(samples)} visual_tokens={visual_count}",
                  flush=True)
    return np.load(feature_path, mmap_mode="r")


@torch.inference_mode()
def zero_selections(states: np.ndarray, bank: list, query: list,
                    shots: int) -> dict[str, list[int]]:
    """Uniform mean cosine across corresponding layer/stream vectors."""
    bank_n = len(bank)
    device = "cuda"
    # Normalize in chunks so the 6,667-image Aircraft bank never creates
    # simultaneous float16 and float32 copies of the entire bank on GPU.
    candidates = torch.empty((bank_n, states.shape[1] * states.shape[2] * states.shape[3]),
                             dtype=torch.float32, device=device)
    for start in range(0, bank_n, 32):
        # The feature array also contains queries after bank_n; never let the
        # final short bank chunk spill into those query rows.
        chunk = torch.from_numpy(np.array(states[start:min(start + 32, bank_n)],
                                          copy=True)).to(device).float()
        candidates[start:start + len(chunk)] = F.normalize(chunk, dim=-1).flatten(1)
        del chunk
    result = {}
    for start in range(0, len(query), 16):
        batch = torch.from_numpy(np.array(states[bank_n + start : bank_n + start + 16],
                                          copy=True)).to(device).float()
        batch = F.normalize(batch, dim=-1).flatten(1)
        scores = batch @ candidates.T / (states.shape[1] * states.shape[2])
        top = scores.topk(shots, dim=1).indices.flip(1).cpu().numpy()
        for offset, indices in enumerate(top):
            result[query[start + offset].sample_id] = indices.tolist()
        if (start + 16) % 160 == 0 or start + 16 >= len(query):
            print(f"Forward-Zero selections {min(start + 16, len(query))}/{len(query)}",
                  flush=True)
    del candidates
    torch.cuda.empty_cache()
    return result


def learn_selections(states: np.ndarray, bank: list, query: list,
                     args: argparse.Namespace, run_dir: Path) -> dict[str, list[int]]:
    flattened = states.reshape(len(states), states.shape[1] * states.shape[2], states.shape[3])
    labels = [sample.label for sample in bank]
    train_indices, val_indices = learned.stratified_development_split(labels, 10, args.seed)
    train_args = argparse.Namespace(seed=args.seed, shots=args.shots, projection_dim=256,
        weights_lr=1e-2, projection_lr=3e-4, temperature=0.07, classes_per_batch=16,
        samples_per_class=4, train_steps=args.train_steps, eval_every=100)
    model, metadata = learned.train_variant("projected_state", flattened[:len(bank)],
        labels, train_indices, val_indices, train_args, run_dir)
    main_run.write_json(run_dir / "forward_learn_metadata.json", {
        "source_variant": "projected_state", "streams": list(STREAMS),
        "layer_stream_order": "layer-major", "bank_train_count": len(train_indices),
        "bank_validation_count": len(val_indices), "best_step": metadata["best_step"],
        "parameter_count": metadata["parameter_count"],
    })
    bank_encoded = learned.encode_indices(model, flattened, np.arange(len(bank)))
    query_encoded = learned.encode_indices(model, flattened,
        np.arange(len(bank), len(bank) + len(query)))
    result = {}
    with torch.inference_mode():
        for start in range(0, len(query), 64):
            scores = model.score(query_encoded[start:start + 64], bank_encoded)
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
    if args.train_steps < 1 or args.shots < 1 or args.generation_batch_size < 1:
        raise ValueError("Training steps, shots, and batch size must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = MODEL_SPECS["qwen3vl4b"]
    bank, query = load_dataset(args.dataset, args.datasets_root)
    samples = bank + query
    labels = sorted({sample.label for sample in bank})
    if args.stage == "probe":
        model, processor = load_model(spec, args.cache_dir)
        inputs = prepare_batch(processor, [zero_shot_messages(bank[0], args.dataset, labels,
                                                            spec.image_size)])
        trace, visual_count = forward_trace(model, inputs)
        print(json.dumps({"shape": list(trace.shape), "visual_tokens": visual_count,
                          "finite": bool(np.isfinite(trace).all()),
                          "input_tokens": int(inputs["input_ids"].shape[1])}), flush=True)
        return
    run_dir = run_directory(args.output_root, args.dataset, spec.slug, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {"dataset": args.dataset, "model_slug": spec.slug, "model_id": spec.model_id,
                "seed": args.seed, "ablation": "forward_trace_anchor_visual_update_v1"}
    main_run.validate_or_write_identity(run_dir, identity, args.resume)
    manifest = {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists() and args.resume:
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Saved DTD manifest differs from current data")
    main_run.write_json(manifest_path, manifest)
    protocol = protocol_metadata(args.dataset, spec.model_id, args.seed, bank, query)
    protocol.update({"ablation": identity["ablation"], "retrieval_feature_prompt":
                     "zero-shot image plus task; one prefill; no generated answer",
                     "streams": list(STREAMS), "stream_order": "layer-major",
                     "visual_pool": "mean of image placeholder positions after each decoder layer",
                     "feature_selection": "no query labels; bank-only supervised contrastive Learn"})
    main_run.write_json(run_dir / "protocol.json", protocol)
    main_run.write_json(run_dir / "config.json", {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": spec.slug, "methods": list(METHODS),
        "trace_shape": [len(samples), 36, 3, 2560], "vision_pixels": None})
    feature_path = run_dir / "forward_trace.npy"
    progress_path = run_dir / "forward_trace_progress.json"
    if args.stage in ("features", "all"):
        complete = args.resume and feature_path.exists() and progress_path.exists() and (
            json.loads(progress_path.read_text(encoding="utf-8")).get("completed") == len(samples))
        if complete:
            states = np.load(feature_path, mmap_mode="r")
        else:
            model, processor = load_model(spec, args.cache_dir)
            states = feature_cache(model, processor, samples, labels, args.dataset, spec, run_dir,
                                   args.resume)
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
    else:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("completed") != len(samples):
            raise RuntimeError("Forward-trace feature cache is incomplete")
        states = np.load(feature_path, mmap_mode="r")
    if list(states.shape) != [len(samples), 36, 3, 2560]:
        raise RuntimeError("Unexpected feature cache shape")
    if args.stage == "features":
        return
    selection_path = run_dir / "selections.json"
    if args.stage in ("retrieval", "all") and not (args.resume and selection_path.exists()):
        selections = {"forward_zero": zero_selections(states, bank, query, args.shots)}
        selections["forward_learn"] = learn_selections(states, bank, query, args, run_dir)
        main_run.save_selections(selection_path, selections, bank)
    else:
        main_run.validate_generation_resume(run_dir, bank, query, list(METHODS), args.shots)
        selections = main_run.load_selections(selection_path, bank)
    main_run.write_json(run_dir / "retrieval_diagnostics.json",
                        main_run.retrieval_diagnostics(selections, bank, query))
    if args.stage == "retrieval":
        return
    args.methods = list(METHODS)
    args.vision_pixels = None
    main_run.run_generation(args, spec, bank, query, labels, run_dir, selections)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Resumable five-dataset CDR main experiment runner.

Every output directory is scoped by dataset, model and seed.  Learned retrievers are
created inside that directory and are never loaded from a different experiment tuple.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

import run_dtd_learned_velocity as learned
import run_dtd_paper_baselines as paper
from multidataset_protocol import DATASETS, Sample, load_dataset, protocol_metadata, run_directory
from multimodal_model_adapter import (
    MODEL_SPECS,
    anchor_hidden_states,
    generate_legal_labels,
    icl_messages,
    load_model,
    model_dimensions,
    prepare_batch,
    task_instruction,
    zero_shot_messages,
)


METHODS = ("rices", "gpt_mm", "detriever", "cdr_zero", "cdr_learn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="qwen3vl4b")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("results/cdr_main"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--gpt-mm-new-tokens", type=int, default=12)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--detriever-steps", type=int, default=10_000)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument(
        "--stage", choices=("features", "retrieval", "generation", "all"), default="all"
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def validate_or_write_identity(run_dir: Path, identity: dict, resume: bool) -> None:
    path = run_dir / "run_identity.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != identity:
            raise RuntimeError(
                "Refusing to reuse a run directory with a different dataset/model/seed identity"
            )
        if not resume:
            raise RuntimeError(
                f"Run already exists at {run_dir}; use --resume only for this exact tuple"
            )
    else:
        write_json(path, identity)


def extract_anchor_cache(
    model,
    processor,
    samples: list[Sample],
    dataset: str,
    labels: list[str],
    image_size: int | None,
    batch_size: int,
    state_path: Path,
    progress_path: Path,
    resume: bool,
) -> np.ndarray:
    layers, width = model_dimensions(model)
    shape = (len(samples), layers, width)
    start = 0
    if resume and state_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("shape") != list(shape):
            raise ValueError("Existing anchor cache has an incompatible shape")
        start = int(progress.get("completed", 0))
        states = open_memmap(state_path, mode="r+", dtype=np.float16, shape=shape)
    else:
        states = open_memmap(state_path, mode="w+", dtype=np.float16, shape=shape)
        write_json(progress_path, {"shape": list(shape), "completed": 0})
    for offset in range(start, len(samples), batch_size):
        batch = samples[offset : offset + batch_size]
        messages = [zero_shot_messages(x, dataset, labels, image_size) for x in batch]
        inputs = prepare_batch(processor, messages)
        anchors = anchor_hidden_states(model, inputs)
        states[offset : offset + len(batch)] = anchors.float().cpu().numpy().astype(np.float16)
        states.flush()
        completed = offset + len(batch)
        write_json(progress_path, {"shape": list(shape), "completed": completed})
        del inputs, anchors
        # Long closed-set prompts (Aircraft/CUB) approach the 16 GB limit and
        # allocator cache growth can sharply reduce throughput before the next
        # 100-sample log point. Releasing cached blocks more often changes no
        # tensors or outputs; it only keeps the single-GPU run out of that regime.
        if completed % 20 == 0 or completed == len(samples):
            torch.cuda.empty_cache()
        if completed % 100 == 0 or completed == len(samples):
            print(f"anchor states {completed:05d}/{len(samples):05d}", flush=True)
    return np.load(state_path, mmap_mode="r")


def extract_clip_cache(
    samples: list[Sample], cache_dir: Path, batch_size: int, output_path: Path, resume: bool
) -> np.ndarray:
    progress_path = output_path.with_suffix(".progress.json")
    shape = (len(samples), 512)
    start = 0
    if resume and output_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("shape") != list(shape):
            raise ValueError("Existing CLIP cache has an incompatible shape")
        start = int(progress.get("completed", 0))
        result = open_memmap(output_path, mode="r+", dtype=np.float32, shape=shape)
    else:
        result = open_memmap(output_path, mode="w+", dtype=np.float32, shape=shape)
        write_json(progress_path, {"shape": list(shape), "completed": 0})
    if start == len(samples):
        return np.load(output_path, mmap_mode="r")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32", cache_dir=cache_dir, dtype=torch.float32
    ).eval().to("cuda")
    with torch.inference_mode():
        for offset in range(start, len(samples), batch_size):
            batch = samples[offset : offset + batch_size]
            images = []
            for sample in batch:
                with Image.open(sample.path) as image:
                    images.append(image.convert("RGB"))
            inputs = processor(images=images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to("cuda")
            value = model.get_image_features(pixel_values=pixel_values)
            if not torch.is_tensor(value):
                value = value.pooler_output
            value = F.normalize(value.float(), dim=-1).cpu().numpy()
            result[offset : offset + len(batch)] = value
            result.flush()
            completed = offset + len(batch)
            write_json(progress_path, {"shape": list(shape), "completed": completed})
            if completed % 500 == 0 or completed == len(samples):
                print(f"CLIP {completed:05d}/{len(samples):05d}", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return np.load(output_path, mmap_mode="r")


def extract_gpt_mm_cache(
    model,
    processor,
    samples: list[Sample],
    dataset: str,
    labels: list[str],
    image_size: int | None,
    batch_size: int,
    max_new_tokens: int,
    embedding_path: Path,
    progress_path: Path,
    outputs_path: Path,
    resume: bool,
) -> np.ndarray:
    _, width = model_dimensions(model)
    shape = (len(samples), width)
    start = 0
    if resume and embedding_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("shape") != list(shape):
            raise ValueError("Existing GPT-MM cache has an incompatible shape")
        start = int(progress.get("completed", 0))
        embeddings = open_memmap(embedding_path, mode="r+", dtype=np.float16, shape=shape)
    else:
        embeddings = open_memmap(embedding_path, mode="w+", dtype=np.float16, shape=shape)
        write_json(progress_path, {"shape": list(shape), "completed": 0})
    mode = "a" if start else "w"
    with outputs_path.open(mode, encoding="utf-8") as output_handle:
        for offset in range(start, len(samples), batch_size):
            batch = samples[offset : offset + batch_size]
            inputs = prepare_batch(
                processor,
                [zero_shot_messages(x, dataset, labels, image_size) for x in batch],
            )
            prompt_length = inputs["input_ids"].shape[1]
            with torch.inference_mode():
                generation = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                )
            tail = generation.sequences[:, prompt_length:]
            eos = processor.tokenizer.eos_token_id
            lengths = []
            for row in tail:
                positions = (row == eos).nonzero(as_tuple=False).flatten()
                lengths.append(max(1, int(positions[0]) if len(positions) else len(row)))
            hidden_steps = generation.hidden_states
            if hidden_steps is None:
                raise RuntimeError("GPT-MM generation returned no hidden states")
            vectors = torch.stack(
                [
                    hidden_steps[min(length, len(hidden_steps) - 1)][-1][row_index, -1]
                    for row_index, length in enumerate(lengths)
                ]
            )
            embeddings[offset : offset + len(batch)] = vectors.float().cpu().numpy().astype(np.float16)
            embeddings.flush()
            texts = [x.strip() for x in processor.batch_decode(tail, skip_special_tokens=True)]
            for sample, text, length in zip(batch, texts, lengths):
                output_handle.write(
                    json.dumps(
                        {"sample_id": sample.sample_id, "raw_zero_shot_answer": text, "answer_token_count": length},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            output_handle.flush()
            completed = offset + len(batch)
            write_json(progress_path, {"shape": list(shape), "completed": completed})
            del inputs, generation, hidden_steps, vectors, tail
            if completed % 20 == 0 or completed == len(samples):
                torch.cuda.empty_cache()
            if completed % 100 == 0 or completed == len(samples):
                print(f"GPT-MM {completed:05d}/{len(samples):05d}", flush=True)
    return np.load(embedding_path, mmap_mode="r")


def nearest_from_embeddings(
    embeddings: np.ndarray, bank_n: int, query: list[Sample], shots: int
) -> dict[str, list[int]]:
    value = embeddings.astype(np.float32)
    value /= np.linalg.norm(value, axis=-1, keepdims=True).clip(1e-12)
    bank = value[:bank_n]
    selections = {}
    for start in range(0, len(query), 256):
        scores = value[bank_n + start : bank_n + start + 256] @ bank.T
        for offset, row in enumerate(scores):
            # Ascending within the selected set: most similar demonstration appears last.
            selections[query[start + offset].sample_id] = np.argsort(row)[-shots:].tolist()
    return selections


def cdr_zero_selections(
    states: np.ndarray, bank_n: int, query: list[Sample], shots: int
) -> dict[str, list[int]]:
    device = torch.device("cuda")
    hidden = torch.as_tensor(np.asarray(states), dtype=torch.bfloat16, device=device)
    velocity = F.normalize((hidden[:, 1:] - hidden[:, :-1]).float(), dim=-1)
    del hidden
    bank = velocity[:bank_n]
    result = {}
    for start in range(0, len(query), 24):
        q = velocity[bank_n + start : bank_n + start + 24]
        scores = torch.einsum("qld,bld->qb", q, bank) / q.shape[1]
        top = scores.topk(shots, dim=1).indices.flip(1).cpu().numpy()
        for offset, indices in enumerate(top):
            result[query[start + offset].sample_id] = indices.tolist()
        print(f"CDR-Zero selections {min(start + 24, len(query)):05d}/{len(query):05d}", flush=True)
    del velocity, bank
    torch.cuda.empty_cache()
    return result


def cdr_learn_selections(
    states: np.ndarray,
    bank: list[Sample],
    query: list[Sample],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[dict[str, list[int]], dict]:
    labels = [sample.label for sample in bank]
    # Ten held-out development examples per class matches the established DTD protocol
    # and remains feasible for CUB's roughly 30 training examples per class.
    train_indices, val_indices = learned.stratified_development_split(labels, 10, args.seed)
    train_args = argparse.Namespace(
        seed=args.seed,
        shots=args.shots,
        projection_dim=256,
        weights_lr=1e-2,
        projection_lr=3e-4,
        temperature=0.07,
        classes_per_batch=16,
        samples_per_class=4,
        train_steps=args.train_steps,
        eval_every=100,
    )
    model, metadata = learned.train_variant(
        "projected_velocity", states[: len(bank)], labels, train_indices, val_indices, train_args, run_dir
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
    return result, metadata


def detriever_selections(
    states: np.ndarray,
    bank: list[Sample],
    query: list[Sample],
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[dict[str, list[int]], dict]:
    det_args = argparse.Namespace(
        seed=args.seed,
        train_steps=args.detriever_steps,
        train_batch_size=64,
        positive_count=40,
        negative_count=100,
        memory_refresh=100,
        learning_rate=1e-4,
        temperature=0.07,
        resume=args.resume,
    )
    model, chosen, metadata = paper.train_detriever(
        states[: len(bank)],
        [sample.label for sample in bank],
        det_args,
        run_dir / "detriever_checkpoint.pt",
        run_dir / "detriever_metadata.json",
    )
    embeddings = paper.detriever_embeddings(model, states, chosen)
    result = nearest_from_embeddings(embeddings, len(bank), query, args.shots)
    np.save(run_dir / "detriever_embeddings.npy", embeddings)
    del model, embeddings
    torch.cuda.empty_cache()
    return result, metadata


def retrieval_diagnostics(
    selections: dict[str, dict[str, list[int]]], bank: list[Sample], query: list[Sample]
) -> dict:
    by_id = {sample.sample_id: sample for sample in query}
    output = {}
    for method, rows in selections.items():
        same, total, distinct = 0, 0, []
        for query_id, indices in rows.items():
            labels = [bank[index].label for index in indices]
            same += sum(label == by_id[query_id].label for label in labels)
            total += len(labels)
            distinct.append(len(set(labels)))
        output[method] = {
            "same_label_demo_fraction": same / total,
            "mean_distinct_demo_labels": float(np.mean(distinct)),
        }
    return output


def save_selections(path: Path, selections: dict[str, dict[str, list[int]]], bank: list[Sample]) -> None:
    write_json(
        path,
        {
            method: {
                query_id: [asdict(bank[index]) for index in indices]
                for query_id, indices in rows.items()
            }
            for method, rows in selections.items()
        },
    )


def load_selections(path: Path, bank: list[Sample]) -> dict[str, dict[str, list[int]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    index = {sample.sample_id: position for position, sample in enumerate(bank)}
    return {
        method: {
            query_id: [index[item["sample_id"]] for item in demos]
            for query_id, demos in rows.items()
        }
        for method, rows in raw.items()
    }


def validate_generation_resume(run_dir: Path, bank: list[Sample], query: list[Sample], methods: list[str], shots: int) -> None:
    """Require the original split and demo choices before a cache-free generation resume."""
    manifest_path = run_dir / "manifest.json"
    selections_path = run_dir / "selections.json"
    if not manifest_path.exists() or not selections_path.exists():
        raise RuntimeError("Generation-only resume requires manifest.json and selections.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for split_name, samples in (("bank", bank), ("query", query)):
        saved = manifest.get(split_name, [])
        expected = [(sample.sample_id, sample.label, sample.split) for sample in samples]
        actual = [(row["sample_id"], row["label"], row["split"]) for row in saved]
        if actual != expected:
            raise RuntimeError(f"Generation-only resume has a different {split_name} split or order")
    raw = json.loads(selections_path.read_text(encoding="utf-8"))
    bank_by_id = {sample.sample_id: sample for sample in bank}
    query_ids = {sample.sample_id for sample in query}
    for method in methods:
        rows = raw.get(method)
        if rows is None or set(rows) != query_ids:
            raise RuntimeError(f"Missing or incomplete {method} selections")
        for demos in rows.values():
            if len(demos) != shots:
                raise RuntimeError(f"Expected {shots} demonstrations for {method}")
            for demo in demos:
                sample = bank_by_id.get(demo["sample_id"])
                if sample is None or sample.label != demo["label"]:
                    raise RuntimeError(f"{method} selection does not match the bank")


def run_generation(args: argparse.Namespace, spec, bank: list[Sample], query: list[Sample],
                   labels: list[str], run_dir: Path, selections: dict[str, dict[str, list[int]]]) -> None:
    model, processor = load_model(spec, args.cache_dir)
    predictions_path = run_dir / "predictions.jsonl"
    completed: dict[tuple[str, str], dict] = {}
    if args.resume and predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[(row["method"], row["query_id"])] = row
    started = time.time()
    with predictions_path.open("a", encoding="utf-8") as handle:
        for method in args.methods:
            pending = [sample for sample in query if (method, sample.sample_id) not in completed]
            for start in range(0, len(pending), args.generation_batch_size):
                batch = pending[start : start + args.generation_batch_size]
                demo_batches = [
                    [bank[index] for index in selections[method][sample.sample_id]] for sample in batch
                ]
                outputs = generate_legal_labels(
                    model,
                    processor,
                    [
                        icl_messages(demos, sample, args.dataset, labels, spec.image_size)
                        for demos, sample in zip(demo_batches, batch)
                    ],
                    labels,
                )
                for sample, demos, (prediction, raw) in zip(batch, demo_batches, outputs):
                    row = {
                        "method": method,
                        "query_id": sample.sample_id,
                        "target": sample.label,
                        "prediction": prediction,
                        "correct": prediction == sample.label,
                        "raw_output": raw,
                        "demo_ids": [demo.sample_id for demo in demos],
                        "demo_labels": [demo.label for demo in demos],
                    }
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed[(method, sample.sample_id)] = row
                done = len(query) - len(pending) + min(start + len(batch), len(pending))
                if done % 20 == 0 or done == len(query):
                    torch.cuda.empty_cache()
                if done % 40 == 0 or done == len(query):
                    print(f"{method} predictions {done:05d}/{len(query):05d}", flush=True)
    summary = {}
    for method in args.methods:
        rows = [completed[(method, sample.sample_id)] for sample in query]
        correct = sum(row["correct"] for row in rows)
        summary[method] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
    write_json(run_dir / "summary.json", {"summary": summary, "elapsed_seconds": time.time() - started})
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = MODEL_SPECS[args.model]
    bank, query = load_dataset(args.dataset, args.datasets_root)
    all_samples = bank + query
    labels = sorted({sample.label for sample in bank})
    run_dir = run_directory(args.output_root, args.dataset, spec.slug, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "dataset": args.dataset,
        "model_slug": spec.slug,
        "model_id": spec.model_id,
        "seed": args.seed,
    }
    validate_or_write_identity(run_dir, identity, args.resume)
    if args.stage == "generation":
        if not args.resume:
            raise RuntimeError("Generation-only stage requires --resume")
        validate_generation_resume(run_dir, bank, query, args.methods, args.shots)
    write_json(run_dir / "protocol.json", protocol_metadata(args.dataset, spec.model_id, args.seed, bank, query))
    write_json(run_dir / "manifest.json", {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]})
    write_json(run_dir / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
    if args.stage == "generation":
        selections = load_selections(run_dir / "selections.json", bank)
        run_generation(args, spec, bank, query, labels, run_dir, selections)
        return

    state_path = run_dir / "anchor_states.npy"
    state_progress = run_dir / "anchor_states_progress.json"
    clip_path = run_dir / "clip_features.npy"
    gpt_path = run_dir / "gpt_mm_embeddings.npy"
    gpt_progress = run_dir / "gpt_mm_progress.json"
    gpt_outputs = run_dir / "gpt_mm_zero_shot_outputs.jsonl"
    model = processor = None

    # Extract CLIP first so it can be released before the VLM is loaded. Keeping
    # both models resident needlessly pushes 16 GB GPUs close to the OOM edge.
    clip = extract_clip_cache(all_samples, args.cache_dir, args.clip_batch_size, clip_path, args.resume)
    if not (args.resume and state_path.exists() and state_progress.exists() and json.loads(state_progress.read_text())["completed"] == len(all_samples)):
        model, processor = load_model(spec, args.cache_dir)
        states = extract_anchor_cache(
            model, processor, all_samples, args.dataset, labels, spec.image_size,
            args.extract_batch_size, state_path, state_progress, args.resume,
        )
    else:
        states = np.load(state_path, mmap_mode="r")
    if model is None and not (
        args.resume and gpt_path.exists() and gpt_progress.exists()
        and json.loads(gpt_progress.read_text())["completed"] == len(all_samples)
    ):
        model, processor = load_model(spec, args.cache_dir)
    if model is not None:
        gpt_embedding = extract_gpt_mm_cache(
            model, processor, all_samples, args.dataset, labels, spec.image_size,
            args.extract_batch_size, args.gpt_mm_new_tokens, gpt_path, gpt_progress,
            gpt_outputs, args.resume,
        )
    else:
        gpt_embedding = np.load(gpt_path, mmap_mode="r")
    if args.stage == "features":
        return

    if model is not None:
        del model, processor
        model = processor = None
        gc.collect()
        torch.cuda.empty_cache()

    selections_path = run_dir / "selections.json"
    selections = load_selections(selections_path, bank) if args.resume and selections_path.exists() else {}
    if "rices" not in selections:
        selections["rices"] = nearest_from_embeddings(clip, len(bank), query, args.shots)
    if "gpt_mm" not in selections:
        selections["gpt_mm"] = nearest_from_embeddings(gpt_embedding, len(bank), query, args.shots)
    if "cdr_zero" not in selections:
        selections["cdr_zero"] = cdr_zero_selections(states, len(bank), query, args.shots)
    if "cdr_learn" not in selections:
        selections["cdr_learn"], _ = cdr_learn_selections(states, bank, query, args, run_dir)
    if "detriever" not in selections:
        selections["detriever"], _ = detriever_selections(states, bank, query, args, run_dir)
    save_selections(selections_path, selections, bank)
    diagnostics = retrieval_diagnostics(selections, bank, query)
    write_json(run_dir / "retrieval_diagnostics.json", diagnostics)
    print(json.dumps(diagnostics, indent=2), flush=True)
    if args.stage == "retrieval":
        return

    run_generation(args, spec, bank, query, labels, run_dir, selections)


if __name__ == "__main__":
    main()

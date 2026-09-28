#!/usr/bin/env python
"""Small, reproducible DTD pilot for hidden-state-assisted ICL retrieval.

The script deliberately separates representation extraction from ICL generation:
CLIP supplies the RICES visual score; Qwen3-VL supplies layer-wise decision-anchor
states and performs the final five-shot classification.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, CLIPModel, CLIPProcessor


DEFAULT_CLASSES = ("braided", "bubbly", "cracked", "paisley", "waffled")
METHODS = (
    "class_balanced_random",
    "rices",
    "rices_final_hidden",
    "rices_selected_matrix",
    "rices_delta_matrix",
    "direct_selected_matrix",
    "direct_delta_matrix",
    "rices_visual_final_hidden",
    "rices_visual_selected_matrix",
    "rices_visual_delta_matrix",
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    path: str
    label: str
    split: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("datasets/dtd"))
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/dtd_qwen3vl4b_pilot"))
    parser.add_argument(
        "--feature-source",
        type=Path,
        default=None,
        help="Optional seed run directory containing a compatible features.npz cache.",
    )
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--all-classes", action="store_true")
    parser.add_argument("--bank-splits", nargs="+", default=["train1.txt"])
    parser.add_argument("--bank-per-class", type=int, default=5)
    parser.add_argument("--query-per-class", type=int, default=2)
    parser.add_argument("--shots", type=int, default=5)
    parser.add_argument("--rices-pool", type=int, default=15)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_split(data_root: Path, split_name: str) -> list[str]:
    path = data_root / "labels" / split_name
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_samples(args: argparse.Namespace) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(args.seed)
    wanted = set(args.classes)
    train_by_class = {name: [] for name in args.classes}
    test_by_class = {name: [] for name in args.classes}
    for split_name in args.bank_splits:
        for rel in read_split(args.data_root, split_name):
            label = rel.split("/", 1)[0]
            if label in wanted:
                train_by_class[label].append(rel)
    for rel in read_split(args.data_root, "test1.txt"):
        label = rel.split("/", 1)[0]
        if label in wanted:
            test_by_class[label].append(rel)

    def choose(grouped: dict[str, list[str]], n: int, split: str) -> list[Sample]:
        result: list[Sample] = []
        for label in args.classes:
            candidates = sorted(grouped[label])
            if len(candidates) < n:
                raise ValueError(f"Only {len(candidates)} {split} images for {label}; need {n}")
            for rel in rng.sample(candidates, n):
                full_path = (args.data_root / "images" / Path(rel)).resolve()
                if not full_path.is_file():
                    raise FileNotFoundError(full_path)
                result.append(Sample(Path(rel).stem, str(full_path), label, split))
        return result

    return choose(train_by_class, args.bank_per_class, "bank"), choose(
        test_by_class, args.query_per_class, "query"
    )


def l2_normalize(x: np.ndarray, axis: int = -1) -> np.ndarray:
    norm = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(norm, 1e-12)


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between one vector/matrix and a bank along the last axis."""
    return np.sum(l2_normalize(a) * l2_normalize(b), axis=-1)


def matrix_similarity(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """Mean row-aligned cosine; query [L,D], bank [N,L,D] -> [N]."""
    q = l2_normalize(query, axis=-1)
    b = l2_normalize(bank, axis=-1)
    return np.mean(np.sum(b * q[None, :, :], axis=-1), axis=-1)


def minmax(x: np.ndarray) -> np.ndarray:
    span = float(x.max() - x.min())
    return np.zeros_like(x) if span < 1e-12 else (x - x.min()) / span


def extract_clip_features(
    samples: list[Sample], model_name: str, cache_dir: Path, image_size: int
) -> np.ndarray:
    print(f"Loading CLIP: {model_name}", flush=True)
    processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir).eval().to("cuda")
    features: list[np.ndarray] = []
    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            with Image.open(sample.path) as image:
                image = image.convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to("cuda")
            value = model.get_image_features(pixel_values=pixel_values)
            # Transformers 5 returns BaseModelOutputWithPooling; 4.x returned a tensor.
            if not torch.is_tensor(value):
                value = value.pooler_output
            features.append(l2_normalize(value.float().cpu().numpy()[0]))
            print(f"CLIP {index:02d}/{len(samples):02d}: {sample.sample_id}", flush=True)
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return np.stack(features).astype(np.float32)


def classification_instruction(classes: Iterable[str]) -> str:
    labels = ", ".join(classes)
    return (
        "Classify the texture in the image. "
        f"The only valid labels are: {labels}. "
        "Return exactly one label and no explanation. Answer:"
    )


def image_part(path: str, image_size: int) -> dict:
    return {
        "type": "image",
        "image": path,
        "resized_height": image_size,
        "resized_width": image_size,
    }


def load_qwen(model_name: str, cache_dir: Path):
    print(f"Loading Qwen3-VL: {model_name}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).eval().to("cuda")
    return model, processor


def move_inputs(inputs: dict, device: str = "cuda") -> dict:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def qwen_inputs(processor, messages: list[dict]) -> dict:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def extract_hidden_states(
    model,
    processor,
    samples: list[Sample],
    classes: list[str],
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    final_vectors: list[np.ndarray] = []
    matrices: list[np.ndarray] = []
    delta_matrices: list[np.ndarray] = []
    visual_final_vectors: list[np.ndarray] = []
    visual_matrices: list[np.ndarray] = []
    visual_delta_matrices: list[np.ndarray] = []
    selected_layers: list[int] | None = None
    instruction = classification_instruction(classes)

    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            messages = [{"role": "user", "content": [image_part(sample.path, image_size), {"type": "text", "text": instruction}]}]
            inputs = move_inputs(qwen_inputs(processor, messages))
            outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            states = outputs.hidden_states
            if states is None:
                raise RuntimeError("Qwen did not return language hidden_states")
            layer_count = len(states) - 1  # hidden_states[0] is the embedding output.
            if selected_layers is None:
                selected_layers = sorted(
                    {max(1, min(layer_count, round(layer_count * p))) for p in (0.25, 0.50, 0.75, 1.0)}
                )
                print(f"Selected language layers: {selected_layers} / {layer_count}", flush=True)
            anchor = int(inputs["attention_mask"][0].sum().item() - 1)
            image_token_id = int(model.config.image_token_id)
            image_mask = inputs["input_ids"][0] == image_token_id
            if not bool(image_mask.any()):
                raise RuntimeError(f"No image tokens found for {sample.sample_id}")
            selected = torch.stack([states[layer][0, anchor].float().cpu() for layer in selected_layers])
            deltas = torch.stack(
                [(states[layer][0, anchor] - states[layer - 1][0, anchor]).float().cpu() for layer in selected_layers]
            )
            visual_selected = torch.stack(
                [states[layer][0, image_mask].float().mean(dim=0).cpu() for layer in selected_layers]
            )
            visual_deltas = torch.stack(
                [
                    (states[layer][0, image_mask] - states[layer - 1][0, image_mask])
                    .float()
                    .mean(dim=0)
                    .cpu()
                    for layer in selected_layers
                ]
            )
            final_vectors.append(selected[-1].numpy())
            matrices.append(selected.numpy())
            delta_matrices.append(deltas.numpy())
            visual_final_vectors.append(visual_selected[-1].numpy())
            visual_matrices.append(visual_selected.numpy())
            visual_delta_matrices.append(visual_deltas.numpy())
            del outputs, states, inputs, selected, deltas, visual_selected, visual_deltas
            torch.cuda.empty_cache()
            print(f"Qwen hidden {index:02d}/{len(samples):02d}: {sample.sample_id}", flush=True)
    assert selected_layers is not None
    return (
        np.stack(final_vectors).astype(np.float32),
        np.stack(matrices).astype(np.float32),
        np.stack(delta_matrices).astype(np.float32),
        np.stack(visual_final_vectors).astype(np.float32),
        np.stack(visual_matrices).astype(np.float32),
        np.stack(visual_delta_matrices).astype(np.float32),
        selected_layers,
    )


def rank_methods(
    bank: list[Sample],
    query: list[Sample],
    clip: np.ndarray,
    final_hidden: np.ndarray,
    matrix: np.ndarray,
    delta: np.ndarray,
    visual_final_hidden: np.ndarray,
    visual_matrix: np.ndarray,
    visual_delta: np.ndarray,
    shots: int,
    pool_size: int,
    seed: int,
    class_order: list[str],
) -> dict[str, dict[str, list[int]]]:
    bank_n = len(bank)
    bank_clip, query_clip = clip[:bank_n], clip[bank_n:]
    bank_final, query_final = final_hidden[:bank_n], final_hidden[bank_n:]
    bank_matrix, query_matrix = matrix[:bank_n], matrix[bank_n:]
    bank_delta, query_delta = delta[:bank_n], delta[bank_n:]
    bank_visual_final, query_visual_final = visual_final_hidden[:bank_n], visual_final_hidden[bank_n:]
    bank_visual_matrix, query_visual_matrix = visual_matrix[:bank_n], visual_matrix[bank_n:]
    bank_visual_delta, query_visual_delta = visual_delta[:bank_n], visual_delta[bank_n:]
    # Normalize the full bank once. Re-normalizing [N,L,D] for every query is
    # negligible in a pilot but dominates runtime for full benchmark splits.
    bank_final_norm = l2_normalize(bank_final)
    query_final_norm = l2_normalize(query_final)
    bank_matrix_norm = l2_normalize(bank_matrix, axis=-1)
    query_matrix_norm = l2_normalize(query_matrix, axis=-1)
    bank_delta_norm = l2_normalize(bank_delta, axis=-1)
    query_delta_norm = l2_normalize(query_delta, axis=-1)
    bank_visual_final_norm = l2_normalize(bank_visual_final)
    query_visual_final_norm = l2_normalize(query_visual_final)
    bank_visual_matrix_norm = l2_normalize(bank_visual_matrix, axis=-1)
    query_visual_matrix_norm = l2_normalize(query_visual_matrix, axis=-1)
    bank_visual_delta_norm = l2_normalize(bank_visual_delta, axis=-1)
    query_visual_delta_norm = l2_normalize(query_visual_delta, axis=-1)
    clip_scores_all = query_clip @ bank_clip.T
    hidden_scores_all = query_final_norm @ bank_final_norm.T
    matrix_scores_all = (
        query_matrix_norm.reshape(len(query), -1)
        @ bank_matrix_norm.reshape(bank_n, -1).T
    ) / query_matrix_norm.shape[1]
    delta_scores_all = (
        query_delta_norm.reshape(len(query), -1)
        @ bank_delta_norm.reshape(bank_n, -1).T
    ) / query_delta_norm.shape[1]
    visual_final_scores_all = query_visual_final_norm @ bank_visual_final_norm.T
    visual_matrix_scores_all = (
        query_visual_matrix_norm.reshape(len(query), -1)
        @ bank_visual_matrix_norm.reshape(bank_n, -1).T
    ) / query_visual_matrix_norm.shape[1]
    visual_delta_scores_all = (
        query_visual_delta_norm.reshape(len(query), -1)
        @ bank_visual_delta_norm.reshape(bank_n, -1).T
    ) / query_visual_delta_norm.shape[1]
    selections: dict[str, dict[str, list[int]]] = {method: {} for method in METHODS}

    label_indices = {label: [i for i, sample in enumerate(bank) if sample.label == label] for label in class_order}
    for q_idx, sample in enumerate(query):
        rng_seed = int(hashlib.sha256(f"{seed}:{sample.sample_id}".encode()).hexdigest()[:8], 16)
        rng = random.Random(rng_seed)
        if shots <= len(class_order):
            sampled_labels = rng.sample(class_order, shots)
        else:
            repeats, remainder = divmod(shots, len(class_order))
            sampled_labels = class_order * repeats + rng.sample(class_order, remainder)
            rng.shuffle(sampled_labels)
        balanced = [rng.choice(label_indices[label]) for label in sampled_labels]
        rng.shuffle(balanced)
        selections["class_balanced_random"][sample.sample_id] = balanced

        clip_scores = clip_scores_all[q_idx]
        pool = np.argsort(clip_scores)[-min(pool_size, bank_n) :]
        rices = pool[np.argsort(clip_scores[pool])[-shots:]]
        selections["rices"][sample.sample_id] = rices.tolist()

        hidden_scores = hidden_scores_all[q_idx]
        matrix_scores = matrix_scores_all[q_idx]
        delta_scores = delta_scores_all[q_idx]
        visual_final_scores = visual_final_scores_all[q_idx]
        visual_matrix_scores = visual_matrix_scores_all[q_idx]
        visual_delta_scores = visual_delta_scores_all[q_idx]
        selections["direct_selected_matrix"][sample.sample_id] = np.argsort(matrix_scores)[
            -shots:
        ].tolist()
        selections["direct_delta_matrix"][sample.sample_id] = np.argsort(delta_scores)[
            -shots:
        ].tolist()
        for method, extra in (
            ("rices_final_hidden", hidden_scores),
            ("rices_selected_matrix", matrix_scores),
            ("rices_delta_matrix", delta_scores),
            ("rices_visual_final_hidden", visual_final_scores),
            ("rices_visual_selected_matrix", visual_matrix_scores),
            ("rices_visual_delta_matrix", visual_delta_scores),
        ):
            combined = 0.5 * minmax(clip_scores[pool]) + 0.5 * minmax(extra[pool])
            chosen = pool[np.argsort(combined)[-shots:]]
            selections[method][sample.sample_id] = chosen.tolist()
    return selections


def build_icl_messages(
    demos: list[Sample], query: Sample, classes: list[str], image_size: int
) -> list[dict]:
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Learn the image-to-label task from the demonstrations. "
                f"Valid labels are: {', '.join(classes)}.\n"
            ),
        }
    ]
    for number, demo in enumerate(demos, start=1):
        content.extend(
            [
                {"type": "text", "text": f"Demonstration {number}:"},
                image_part(demo.path, image_size),
                {"type": "text", "text": f"Label: {demo.label}\n"},
            ]
        )
    content.extend(
        [
            {"type": "text", "text": "Query image:"},
            image_part(query.path, image_size),
            {
                "type": "text",
                "text": "Return exactly one valid label and no explanation. Label:",
            },
        ]
    )
    return [{"role": "user", "content": content}]


def parse_prediction(text: str, classes: list[str]) -> str | None:
    clean = text.strip().lower()
    for label in sorted(classes, key=len, reverse=True):
        if re.search(rf"(?<![a-z-]){re.escape(label.lower())}(?![a-z-])", clean):
            return label
    return None


def generate_prediction(
    model,
    processor,
    messages: list[dict],
    classes: list[str],
    max_new_tokens: int,
) -> tuple[str | None, str]:
    inputs = move_inputs(qwen_inputs(processor, messages))
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    text = processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)[0].strip()
    del generated, inputs
    torch.cuda.empty_cache()
    return parse_prediction(text, classes), text


def generate_predictions_batch(
    model,
    processor,
    messages_batch: list[list[dict]],
    classes: list[str],
    max_new_tokens: int,
) -> list[tuple[str | None, str]]:
    if len(messages_batch) == 1:
        return [generate_prediction(model, processor, messages_batch[0], classes, max_new_tokens)]
    previous_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        inputs = processor.apply_chat_template(
            messages_batch,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_dict=True,
            return_tensors="pt",
        )
    finally:
        processor.tokenizer.padding_side = previous_padding_side
    inputs = move_inputs(inputs)
    input_length = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    texts = [text.strip() for text in processor.batch_decode(generated[:, input_length:], skip_special_tokens=True)]
    del generated, inputs
    torch.cuda.empty_cache()
    return [(parse_prediction(value, classes), value) for value in texts]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires a CUDA GPU")
    if args.all_classes:
        args.classes = sorted({rel.split("/", 1)[0] for rel in read_split(args.data_root, args.bank_splits[0])})
    seed_everything(args.seed)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    bank, query = make_samples(args)
    all_samples = bank + query
    write_json(run_dir / "manifest.json", {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]})
    write_json(run_dir / "config.json", {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})

    feature_dir = args.feature_source if args.feature_source is not None else run_dir
    feature_path = feature_dir / "features.npz"
    clip_path = feature_dir / "clip_features.npy"
    if args.feature_source is not None:
        source_manifest_path = feature_dir / "manifest.json"
        if not source_manifest_path.exists():
            raise FileNotFoundError(f"Missing feature-source manifest: {source_manifest_path}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        expected_ids = [sample.sample_id for sample in all_samples]
        source_ids = [sample["sample_id"] for sample in source_manifest["bank"] + source_manifest["query"]]
        if source_ids != expected_ids:
            raise ValueError("Feature-source manifest does not match the current bank/query ordering")
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature-source cache: {feature_path}")
    if args.resume and feature_path.exists():
        cached = np.load(feature_path)
        clip = cached["clip"]
        final_hidden = cached["final_hidden"]
        matrix = cached["matrix"]
        delta = cached["delta"]
        visual_final_hidden = cached["visual_final_hidden"]
        visual_matrix = cached["visual_matrix"]
        visual_delta = cached["visual_delta"]
        selected_layers = cached["selected_layers"].tolist()
        model, processor = load_qwen(args.qwen_model, args.cache_dir)
    else:
        if args.resume and clip_path.exists():
            clip = np.load(clip_path)
        else:
            clip = extract_clip_features(all_samples, args.clip_model, args.cache_dir, args.image_size)
            np.save(clip_path, clip)
        model, processor = load_qwen(args.qwen_model, args.cache_dir)
        (
            final_hidden,
            matrix,
            delta,
            visual_final_hidden,
            visual_matrix,
            visual_delta,
            selected_layers,
        ) = extract_hidden_states(
            model, processor, all_samples, args.classes, args.image_size
        )
        np.savez_compressed(
            feature_path,
            clip=clip,
            final_hidden=final_hidden,
            matrix=matrix,
            delta=delta,
            visual_final_hidden=visual_final_hidden,
            visual_matrix=visual_matrix,
            visual_delta=visual_delta,
            selected_layers=np.asarray(selected_layers),
        )

    selections = rank_methods(
        bank,
        query,
        clip,
        final_hidden,
        matrix,
        delta,
        visual_final_hidden,
        visual_matrix,
        visual_delta,
        args.shots,
        args.rices_pool,
        args.seed,
        args.classes,
    )
    selections = {method: selections[method] for method in args.methods}
    detailed_selections = {
        method: {
            query_id: [asdict(bank[index]) for index in indices]
            for query_id, indices in by_query.items()
        }
        for method, by_query in selections.items()
    }
    write_json(run_dir / "selections.json", detailed_selections)

    predictions_path = run_dir / "predictions.jsonl"
    completed: dict[tuple[str, str], dict] = {}
    if args.resume and predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            completed[(record["method"], record["query_id"])] = record

    started = time.time()
    with predictions_path.open("a", encoding="utf-8") as handle:
        for method in args.methods:
            pending = [sample for sample in query if (method, sample.sample_id) not in completed]
            for start in range(0, len(pending), args.generation_batch_size):
                batch_samples = pending[start : start + args.generation_batch_size]
                batch_demos = [
                    [bank[index] for index in selections[method][sample.sample_id]]
                    for sample in batch_samples
                ]
                outputs = generate_predictions_batch(
                    model,
                    processor,
                    [
                        build_icl_messages(demos, sample, args.classes, args.image_size)
                        for demos, sample in zip(batch_demos, batch_samples)
                    ],
                    args.classes,
                    args.max_new_tokens,
                )
                for offset, (query_sample, demos, (prediction, raw)) in enumerate(
                    zip(batch_samples, batch_demos, outputs), start=start + 1
                ):
                    record = {
                        "method": method,
                        "query_id": query_sample.sample_id,
                        "target": query_sample.label,
                        "prediction": prediction,
                        "correct": prediction == query_sample.label,
                        "raw_output": raw,
                        "demo_ids": [demo.sample_id for demo in demos],
                        "demo_labels": [demo.label for demo in demos],
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed[(method, query_sample.sample_id)] = record
                    print(
                        f"{method} {offset:04d}/{len(query):04d}: "
                        f"target={query_sample.label} pred={prediction} raw={raw!r}",
                        flush=True,
                    )

    records = list(completed.values())
    summary = {}
    for method in args.methods:
        rows = [row for row in records if row["method"] == method]
        summary[method] = {
            "correct": sum(bool(row["correct"]) for row in rows),
            "total": len(rows),
            "accuracy": (sum(bool(row["correct"]) for row in rows) / len(rows)) if rows else None,
            "parse_failures": sum(row["prediction"] is None for row in rows),
        }
    report = {
        "summary": summary,
        "selected_layers": selected_layers,
        "matrix_shape_per_image": list(matrix.shape[1:]),
        "visual_matrix_shape_per_image": list(visual_matrix.shape[1:]),
        "delta_definition": "For selected layer l, h_l(answer-anchor) - h_(l-1)(answer-anchor)",
        "visual_definition": "Mean hidden state over image-pad token positions at each selected language layer",
        "combined_score": "0.5 * minmax(CLIP cosine) + 0.5 * minmax(hidden similarity), within RICES top pool",
        "elapsed_seconds_generation_stage": round(time.time() - started, 1),
    }
    write_json(run_dir / "summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

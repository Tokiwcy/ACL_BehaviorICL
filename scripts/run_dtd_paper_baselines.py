#!/usr/bin/env python
"""Paper-faithful DeTriever and GPT-MM adaptations for DTD multimodal ICL.

Neither paper provides a public implementation linked from its publication page.  This
script follows the published equations and hyperparameters while keeping the existing
DTD/Qwen3-VL evaluation protocol fixed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.lib.format import open_memmap

import run_dtd_hidden_rices_pilot as base


METHODS = ("detriever", "gpt_mm")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("results/dtd_qwen3vl4b_trajectory_full47/seed_73"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dtd_qwen3vl4b_paper_baselines_full47"),
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--gpt-mm-new-tokens", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=10_000)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--positive-count", type=int, default=40)
    parser.add_argument("--negative-count", type=int, default=100)
    parser.add_argument("--memory-refresh", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_samples(path: Path) -> tuple[list[base.Sample], list[base.Sample]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return (
        [base.Sample(**item) for item in manifest["bank"]],
        [base.Sample(**item) for item in manifest["query"]],
    )


def detriever_layers(layer_count: int) -> list[int]:
    """Paper samples layers 0,5,...,40; retain that cadence and the final layer."""
    layers = [0] + list(range(4, layer_count, 5))
    if layer_count - 1 not in layers:
        layers.append(layer_count - 1)
    return sorted(set(layers))


class DeTriever(nn.Module):
    """Equation 4: weighted sum of layer-specific three-layer MLPs."""

    def __init__(self, input_size: int, layer_count: int) -> None:
        super().__init__()
        self.mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_size, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 512),
                )
                for _ in range(layer_count)
            ]
        )
        self.layer_logits = nn.Parameter(torch.zeros(layer_count))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        transformed = torch.stack(
            [mlp(states[:, index]) for index, mlp in enumerate(self.mlps)], dim=1
        )
        weights = self.layer_logits.softmax(dim=0)
        return F.normalize((transformed * weights[None, :, None]).sum(dim=1), dim=-1)


def proxy_candidate_indices(
    labels: list[str], positive_count: int, negative_count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build the output-query proxy pools used by Equation 5.

    For a classification output, identical target labels have identical query-only
    representations, so the paper's nearest-output proxy reduces to same-label positives.
    """
    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels)
    positives = np.empty((len(labels), positive_count), dtype=np.int64)
    negatives = np.empty((len(labels), negative_count), dtype=np.int64)
    all_indices = np.arange(len(labels))
    for index, label in enumerate(labels):
        pos_pool = all_indices[(labels_np == label) & (all_indices != index)]
        neg_pool = all_indices[labels_np != label]
        positives[index] = rng.choice(
            pos_pool, size=positive_count, replace=len(pos_pool) < positive_count
        )
        negatives[index] = rng.choice(
            neg_pool, size=negative_count, replace=len(neg_pool) < negative_count
        )
    return positives, negatives


@torch.inference_mode()
def encode_in_chunks(
    retriever: DeTriever, states: torch.Tensor, chunk_size: int = 128
) -> torch.Tensor:
    result = []
    for start in range(0, len(states), chunk_size):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            result.append(retriever(states[start : start + chunk_size]).float())
    return torch.cat(result)


def train_detriever(
    states: np.ndarray,
    labels: list[str],
    args: argparse.Namespace,
    checkpoint_path: Path,
    metadata_path: Path,
) -> tuple[DeTriever, list[int], dict]:
    chosen_layers = detriever_layers(states.shape[1])
    device = torch.device("cuda")
    # CPU tensor remains backed by the mmap until selected minibatches are copied to CUDA.
    train_states = torch.from_numpy(np.asarray(states[:, chosen_layers], dtype=np.float32))
    retriever = DeTriever(states.shape[2], len(chosen_layers)).to(device)
    optimizer = torch.optim.AdamW(
        retriever.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    start_step = 0
    losses: list[float] = []
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        retriever.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
        losses = list(saved.get("losses", []))

    positives, negatives = proxy_candidate_indices(
        labels, args.positive_count, args.negative_count, args.seed
    )
    generator = np.random.default_rng(args.seed + start_step)
    memory: torch.Tensor | None = None
    retriever.train()
    for step in range(start_step, args.train_steps):
        if memory is None or step % args.memory_refresh == 0:
            retriever.eval()
            gpu_states = train_states.to(device, non_blocking=True)
            memory = encode_in_chunks(retriever, gpu_states).detach()
            del gpu_states
            retriever.train()

        anchors = generator.integers(0, len(labels), size=args.train_batch_size)
        anchor_states = train_states[anchors].to(device, non_blocking=True)
        candidate_np = np.concatenate([positives[anchors], negatives[anchors]], axis=1)
        candidate_indices = torch.as_tensor(candidate_np, device=device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            anchor_embeddings = retriever(anchor_states)
            logits = torch.einsum("bd,bkd->bk", anchor_embeddings, memory[candidate_indices])
            logits = logits / args.temperature
            log_denominator = torch.logsumexp(logits, dim=1)
            loss = -(logits[:, : args.positive_count].mean(dim=1) - log_denominator).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

        completed = step + 1
        if completed % 100 == 0 or completed == args.train_steps:
            recent = float(np.mean(losses[-100:]))
            print(f"DeTriever train {completed:05d}/{args.train_steps:05d} loss={recent:.5f}", flush=True)
        if completed % 1000 == 0 or completed == args.train_steps:
            torch.save(
                {
                    "model": retriever.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": completed,
                    "losses": losses,
                    "layers_zero_based": chosen_layers,
                },
                checkpoint_path,
            )

    metadata = {
        "paper": "DeTriever (Li et al., COLING 2025)",
        "implementation": "paper-faithful task adaptation; no official code was published",
        "layers_one_based": [value + 1 for value in chosen_layers],
        "mlp": "layer-specific 2560->1024->1024->512 with learned softmax layer weights",
        "steps": args.train_steps,
        "batch_size": args.train_batch_size,
        "positive_count": args.positive_count,
        "negative_count": args.negative_count,
        "temperature": args.temperature,
        "optimizer": "AdamW(lr=1e-4, weight_decay=0.01, betas=(0.9,0.98))",
        "proxy": "query-only label identity; same-label positives for classification",
        "memory_refresh_steps": args.memory_refresh,
        "final_loss_mean_100": float(np.mean(losses[-100:])),
        "learned_layer_weights": retriever.layer_logits.softmax(0).detach().cpu().tolist(),
    }
    write_json(metadata_path, metadata)
    return retriever, chosen_layers, metadata


@torch.inference_mode()
def detriever_embeddings(
    retriever: DeTriever, all_states: np.ndarray, chosen_layers: list[int]
) -> np.ndarray:
    retriever.eval()
    states = torch.from_numpy(
        np.asarray(all_states[:, chosen_layers], dtype=np.float32)
    ).to("cuda")
    result = encode_in_chunks(retriever, states).cpu().numpy().astype(np.float32)
    del states
    return result


def zero_shot_messages(samples: list[base.Sample], classes: list[str], image_size: int):
    instruction = base.classification_instruction(classes)
    return [
        [
            {
                "role": "user",
                "content": [
                    base.image_part(sample.path, image_size),
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        for sample in samples
    ]


def extract_gpt_mm_embeddings(
    model,
    processor,
    samples: list[base.Sample],
    classes: list[str],
    image_size: int,
    batch_size: int,
    max_new_tokens: int,
    embedding_path: Path,
    progress_path: Path,
    outputs_path: Path,
    resume: bool,
) -> np.ndarray:
    width = int(model.config.text_config.hidden_size)
    start = 0
    output_rows: list[dict] = []
    if resume and embedding_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("shape") != [len(samples), width]:
            raise ValueError("Existing GPT-MM cache has an incompatible shape")
        start = int(progress["completed"])
        embeddings = open_memmap(
            embedding_path, mode="r+", dtype=np.float16, shape=(len(samples), width)
        )
        if outputs_path.exists():
            output_rows = [
                json.loads(line)
                for line in outputs_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
    else:
        embeddings = open_memmap(
            embedding_path, mode="w+", dtype=np.float16, shape=(len(samples), width)
        )
        write_json(progress_path, {"shape": [len(samples), width], "completed": 0})

    previous_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        with outputs_path.open("a" if start else "w", encoding="utf-8") as output_handle:
            for offset in range(start, len(samples), batch_size):
                batch = samples[offset : offset + batch_size]
                inputs = processor.apply_chat_template(
                    zero_shot_messages(batch, classes, image_size),
                    tokenize=True,
                    add_generation_prompt=True,
                    padding=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = base.move_inputs(inputs)
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

                generated = generation.sequences
                generated_tail = generated[:, prompt_length:]
                eos_id = processor.tokenizer.eos_token_id
                content_lengths = []
                for row in generated_tail:
                    eos_positions = (row == eos_id).nonzero(as_tuple=False).flatten()
                    length = int(eos_positions[0]) if len(eos_positions) else len(row)
                    content_lengths.append(max(1, length))
                # At generation step t>0, the cached decoder processes the token
                # generated at t-1.  Therefore step `content_length` contains the
                # state of the final answer token (the next token is normally EOS).
                hidden_steps = generation.hidden_states
                if hidden_steps is None:
                    raise RuntimeError("Generation did not return hidden states")
                answer_vectors = torch.stack(
                    [
                        hidden_steps[min(length, len(hidden_steps) - 1)][-1][
                            row_index, -1
                        ]
                        for row_index, length in enumerate(content_lengths)
                    ]
                )
                embeddings[offset : offset + len(batch)] = (
                    answer_vectors.float().cpu().numpy().astype(np.float16)
                )
                embeddings.flush()

                texts = [
                    text.strip()
                    for text in processor.batch_decode(generated_tail, skip_special_tokens=True)
                ]
                for sample, text, length in zip(batch, texts, content_lengths):
                    record = {
                        "sample_id": sample.sample_id,
                        "split": sample.split,
                        "raw_zero_shot_answer": text,
                        "parsed_zero_shot_answer": base.parse_prediction(text, classes),
                        "answer_token_count": length,
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    output_rows.append(record)
                output_handle.flush()
                completed = offset + len(batch)
                write_json(
                    progress_path,
                    {"shape": [len(samples), width], "completed": completed},
                )
                del generation, hidden_steps, answer_vectors, generated, inputs
                if completed % 40 == 0 or completed == len(samples):
                    torch.cuda.empty_cache()
                    print(f"GPT-MM answer states {completed:04d}/{len(samples):04d}", flush=True)
    finally:
        processor.tokenizer.padding_side = previous_padding_side
    return np.load(embedding_path, mmap_mode="r")


def nearest_selections(
    embeddings: np.ndarray,
    bank: list[base.Sample],
    query: list[base.Sample],
    shots: int,
) -> dict[str, list[int]]:
    bank_n = len(bank)
    normalized = embeddings.astype(np.float32)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True).clip(1e-12)
    scores = normalized[bank_n:] @ normalized[:bank_n].T
    return {
        sample.sample_id: np.argsort(scores[index])[-shots:].tolist()
        for index, sample in enumerate(query)
    }


def retrieval_diagnostics(
    selections: dict[str, dict[str, list[int]]],
    bank: list[base.Sample],
    query: list[base.Sample],
) -> dict:
    query_by_id = {sample.sample_id: sample for sample in query}
    result = {}
    for method, by_query in selections.items():
        same = 0
        total = 0
        distinct = []
        for query_id, indices in by_query.items():
            demo_labels = [bank[index].label for index in indices]
            same += sum(label == query_by_id[query_id].label for label in demo_labels)
            total += len(indices)
            distinct.append(len(set(demo_labels)))
        result[method] = {
            "same_label_demo_fraction": same / total,
            "mean_distinct_demo_labels": float(np.mean(distinct)),
        }
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    base.seed_everything(args.seed)
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    bank, query = load_samples(args.source_run / "manifest.json")
    all_samples = bank + query
    classes = sorted({sample.label for sample in bank})
    write_json(
        run_dir / "manifest.json",
        {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]},
    )
    write_json(
        run_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )

    selections: dict[str, dict[str, list[int]]] = {}
    detriever_embedding_path = run_dir / "detriever_embeddings.npy"
    if args.resume and detriever_embedding_path.exists():
        detriever_embedding = np.load(detriever_embedding_path)
    else:
        all_layer_states = np.load(
            args.source_run / "all_layer_anchor_states.npy", mmap_mode="r"
        )
        retriever, chosen_layers, _ = train_detriever(
            all_layer_states[: len(bank)],
            [sample.label for sample in bank],
            args,
            run_dir / "detriever_checkpoint.pt",
            run_dir / "detriever_metadata.json",
        )
        detriever_embedding = detriever_embeddings(
            retriever, all_layer_states, chosen_layers
        )
        np.save(detriever_embedding_path, detriever_embedding)
        del retriever
        torch.cuda.empty_cache()
    selections["detriever"] = nearest_selections(
        detriever_embedding, bank, query, args.shots
    )

    model = processor = None
    gpt_mm_path = run_dir / "gpt_mm_answer_embeddings.npy"
    if args.resume and gpt_mm_path.exists() and (run_dir / "gpt_mm_progress.json").exists():
        progress = json.loads((run_dir / "gpt_mm_progress.json").read_text(encoding="utf-8"))
        gpt_mm_embedding = np.load(gpt_mm_path, mmap_mode="r")
        if progress.get("completed") != len(all_samples):
            model, processor = base.load_qwen(args.qwen_model, args.cache_dir)
            gpt_mm_embedding = extract_gpt_mm_embeddings(
                model, processor, all_samples, classes, args.image_size, args.batch_size,
                args.gpt_mm_new_tokens, gpt_mm_path, run_dir / "gpt_mm_progress.json",
                run_dir / "gpt_mm_zero_shot_outputs.jsonl", True,
            )
    else:
        model, processor = base.load_qwen(args.qwen_model, args.cache_dir)
        gpt_mm_embedding = extract_gpt_mm_embeddings(
            model, processor, all_samples, classes, args.image_size, args.batch_size,
            args.gpt_mm_new_tokens, gpt_mm_path, run_dir / "gpt_mm_progress.json",
            run_dir / "gpt_mm_zero_shot_outputs.jsonl", args.resume,
        )
    selections["gpt_mm"] = nearest_selections(gpt_mm_embedding, bank, query, args.shots)

    write_json(
        run_dir / "selections.json",
        {
            method: {
                query_id: [asdict(bank[index]) for index in indices]
                for query_id, indices in by_query.items()
            }
            for method, by_query in selections.items()
        },
    )
    write_json(
        run_dir / "retrieval_diagnostics.json",
        retrieval_diagnostics(selections, bank, query),
    )
    print(
        json.dumps(retrieval_diagnostics(selections, bank, query), indent=2),
        flush=True,
    )
    if args.selection_only:
        return

    if model is None:
        model, processor = base.load_qwen(args.qwen_model, args.cache_dir)
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
            for start in range(0, len(pending), args.batch_size):
                batch_samples = pending[start : start + args.batch_size]
                batch_demos = [
                    [bank[index] for index in selections[method][sample.sample_id]]
                    for sample in batch_samples
                ]
                outputs = base.generate_predictions_batch(
                    model,
                    processor,
                    [
                        base.build_icl_messages(demos, sample, classes, args.image_size)
                        for demos, sample in zip(batch_demos, batch_samples)
                    ],
                    classes,
                    args.max_new_tokens,
                )
                for offset, (sample, demos, (prediction, raw)) in enumerate(
                    zip(batch_samples, batch_demos, outputs), start=start + 1
                ):
                    record = {
                        "method": method,
                        "query_id": sample.sample_id,
                        "target": sample.label,
                        "prediction": prediction,
                        "correct": prediction == sample.label,
                        "raw_output": raw,
                        "demo_ids": [demo.sample_id for demo in demos],
                        "demo_labels": [demo.label for demo in demos],
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed[(method, sample.sample_id)] = record
                    print(
                        f"{method} {offset:04d}/{len(query):04d}: "
                        f"target={sample.label} pred={prediction} raw={raw!r}",
                        flush=True,
                    )

    summary = {}
    for method in args.methods:
        rows = [row for (name, _), row in completed.items() if name == method]
        correct = sum(bool(row["correct"]) for row in rows)
        summary[method] = {
            "correct": correct,
            "total": len(rows),
            "accuracy": correct / len(rows) if rows else None,
            "parse_failures": sum(row["prediction"] is None for row in rows),
        }
    write_json(
        run_dir / "summary.json",
        {"summary": summary, "elapsed_seconds_generation_stage": time.time() - started},
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

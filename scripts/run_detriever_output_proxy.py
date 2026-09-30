#!/usr/bin/env python
"""Re-train/evaluate a separate DeTriever with the paper's gold-output proxy.

The existing five-method results use a label-identity adaptation and are never
modified here. Frozen *input* states may be reused from the same tuple; gold
output representations and retriever weights are newly created per tuple.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from numpy.lib.format import open_memmap

import run_dtd_paper_baselines as paper
import run_multidataset_main as main_run
from multidataset_protocol import DATASETS, Sample, run_directory
from multimodal_model_adapter import (
    MODEL_SPECS, load_model, model_dimensions, move_inputs, zero_shot_messages,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--source-root", type=Path, default=Path("results/cdr_main"))
    parser.add_argument("--output-root", type=Path, default=Path("results/detriever_output_proxy"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--extract-batch-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--detriever-steps", type=int, default=10_000)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--stage", choices=("proxy", "retrieval", "generation", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def verify_source(source: Path, dataset: str, seed: int) -> tuple[list[Sample], list[Sample], dict, str]:
    identity = json.loads((source / "run_identity.json").read_text(encoding="utf-8"))
    if identity.get("dataset") != dataset or identity.get("model_slug") != "qwen3vl4b" or identity.get("seed") != seed:
        raise RuntimeError("Source tuple identity does not match requested dataset/model/seed")
    if identity.get("model_id") != MODEL_SPECS["qwen3vl4b"].model_id:
        raise RuntimeError("Source model ID differs from the expected Qwen baseline")
    manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    bank = [Sample(**row) for row in manifest["bank"]]
    query = [Sample(**row) for row in manifest["query"]]
    if not bank or not query or {row.sample_id for row in bank} & {row.sample_id for row in query}:
        raise RuntimeError("Source manifest has empty or overlapping bank/query")
    progress = json.loads((source / "anchor_states_progress.json").read_text(encoding="utf-8"))
    if progress.get("completed") != len(bank) + len(query):
        raise RuntimeError("Source frozen input states are incomplete")
    return bank, query, identity, hashlib.sha256(manifest_bytes).hexdigest()


@torch.inference_mode()
def gold_answer_eos_state(model, processor, sample: Sample, dataset: str,
                          labels: list[str], image_size: int | None) -> torch.Tensor:
    """Teacher-force the bank gold label after the same zero-shot input prompt.

    The final chat EOS state is an input+gold-answer proxy; no query gold label
    is ever passed to this function. A complete chat template keeps multimodal
    position and auxiliary tensors aligned with the assistant answer.
    """
    messages = zero_shot_messages(sample, dataset, labels, image_size) + [
        {"role": "assistant", "content": sample.label}
    ]
    inputs = move_inputs(
        processor.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True},
        )
    )
    output = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
    result = output.hidden_states[-1][0, -1].detach().float().cpu()
    del inputs, output
    return result


def extract_proxy(model, processor, bank: list[Sample], dataset: str, labels: list[str],
                  image_size: int | None, run_dir: Path, resume: bool,
                  manifest_sha256: str) -> np.ndarray:
    _, width = model_dimensions(model)
    shape = (len(bank), width)
    path = run_dir / "gold_input_answer_eos.npy"
    progress_path = run_dir / "gold_input_answer_eos_progress.json"
    expected = {"shape": list(shape), "manifest_sha256": manifest_sha256}
    if resume and path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if any(progress.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Gold proxy cache belongs to a different bank or shape")
        start = int(progress["completed"])
        if not 0 <= start <= len(bank):
            raise RuntimeError("Invalid gold proxy progress")
        output = open_memmap(path, mode="r+", dtype=np.float16, shape=shape)
    else:
        start = 0
        output = open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        main_run.write_json(progress_path, {**expected, "completed": 0})
    for index in range(start, len(bank)):
        output[index] = gold_answer_eos_state(
            model, processor, bank[index], dataset, labels, image_size
        ).numpy().astype(np.float16)
        output.flush()
        main_run.write_json(progress_path, {**expected, "completed": index + 1})
        if (index + 1) % 20 == 0:
            torch.cuda.empty_cache()
        if (index + 1) % 100 == 0 or index + 1 == len(bank):
            print(f"DeTriever gold proxy {index + 1:05d}/{len(bank):05d}", flush=True)
    return np.load(path, mmap_mode="r")


def main() -> None:
    args = parse_args()
    if args.extract_batch_size != 1:
        raise ValueError("Gold-output extraction currently uses batch size 1 to avoid padding ambiguity")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = MODEL_SPECS["qwen3vl4b"]
    source = run_directory(args.source_root, args.dataset, spec.slug, args.seed)
    bank, query, source_identity, manifest_sha256 = verify_source(source, args.dataset, args.seed)
    labels = sorted({row.label for row in bank})
    run_dir = run_directory(args.output_root, args.dataset, spec.slug, args.seed)
    if source.resolve() == run_dir.resolve():
        raise RuntimeError("Output directory must be separate from the legacy run")
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {**source_identity, "baseline": "detriever_gold_input_answer_eos_v1",
                "manifest_sha256": manifest_sha256}
    main_run.validate_or_write_identity(run_dir, identity, args.resume)
    main_run.write_json(run_dir / "manifest.json", {
        "bank": [asdict(row) for row in bank], "query": [asdict(row) for row in query]
    })
    protocol = json.loads((source / "protocol.json").read_text(encoding="utf-8"))
    protocol["detriever_proxy"] = "bank_only_gold_input_answer_eos_dot_product_v1"
    main_run.write_json(run_dir / "protocol.json", protocol)
    main_run.write_json(run_dir / "config.json", {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    })
    states = np.load(source / "anchor_states.npy", mmap_mode="r")
    if states.ndim != 3 or len(states) != len(bank) + len(query):
        raise RuntimeError("Source input-state cache shape differs from manifest")
    proxy_path = run_dir / "gold_input_answer_eos.npy"
    progress_path = run_dir / "gold_input_answer_eos_progress.json"
    if args.stage != "generation":
        complete = (args.resume and proxy_path.exists() and progress_path.exists()
                    and json.loads(progress_path.read_text(encoding="utf-8")).get("completed") == len(bank))
        if not complete:
            vision_pixels = source_identity.get("vision_pixels")
            model, processor = load_model(spec, args.cache_dir, vision_pixels)
            proxy = extract_proxy(model, processor, bank, args.dataset, labels,
                                  spec.image_size if vision_pixels is None else None,
                                  run_dir, args.resume, manifest_sha256)
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
        else:
            proxy = np.load(proxy_path, mmap_mode="r")
        if proxy.shape != (len(bank), states.shape[2]):
            raise RuntimeError("Gold proxy cache has an incompatible shape")
        if args.stage == "proxy":
            return
        det_args = argparse.Namespace(
            seed=args.seed, train_steps=args.detriever_steps, train_batch_size=64,
            positive_count=40, negative_count=100, memory_refresh=100,
            learning_rate=1e-4, temperature=0.07, resume=args.resume,
        )
        retriever, layers, _ = paper.train_detriever(
            states[:len(bank)], [row.label for row in bank], det_args,
            run_dir / "detriever_checkpoint.pt", run_dir / "detriever_metadata.json",
            output_proxy=proxy,
        )
        embeddings = paper.detriever_embeddings(retriever, states, layers)
        selections = {"detriever": main_run.nearest_from_embeddings(
            embeddings, len(bank), query, args.shots
        )}
        main_run.save_selections(run_dir / "selections.json", selections, bank)
        del retriever, embeddings
        gc.collect()
        torch.cuda.empty_cache()
        if args.stage == "retrieval":
            return
    else:
        if not args.resume:
            raise RuntimeError("Generation-only stage requires --resume")
        main_run.validate_generation_resume(run_dir, bank, query, ["detriever"], args.shots)
        selections = main_run.load_selections(run_dir / "selections.json", bank)
    args.methods = ["detriever"]
    args.model = "qwen3vl4b"
    args.vision_pixels = source_identity.get("vision_pixels")
    main_run.run_generation(args, spec, bank, query, labels, run_dir, selections)


if __name__ == "__main__":
    main()

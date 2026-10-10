"""Full-bank training and retrieval for the local dense InternVL3.5-4B study.

Uses the existing bank-only BehaviorICL and DeTriever adaptation kernels without
changing their historical source. All stages are scoped by one pinned manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from multidataset_protocol import Sample
from internvl35_qwen_config_v4 import (DATASETS, METHODS, MODEL, OUTPUT, REVISION, PROTOCOL_PATH, digest, load_model, write_json, ensure_protocol)


def load_context(dataset: str) -> tuple[Path, list[Sample], list[Sample], dict]:
    folder = OUTPUT / dataset
    identity = json.loads((folder / "identity.json").read_text(encoding="utf-8"))
    if identity["model"] != MODEL or identity["revision"] != REVISION or identity["shots"] != 4:
        raise ValueError("Wrong full source identity")
    if identity.get("full_protocol_sha256") != digest(PROTOCOL_PATH):
        raise ValueError("Source belongs to a different full protocol")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if digest(folder / "manifest.json") != identity["manifest_sha256"]:
        raise ValueError("Full source manifest changed")
    sample = lambda row: Sample(row["sample_id"], row["path"], row["label"], row["split"])
    bank = [sample(row) for row in manifest["bank"]]
    query = [sample(row) for row in manifest["query"]]
    if (len(bank), len(query)) != (identity["bank_n"], identity["query_n"]):
        raise ValueError("Full source count changed")
    return folder, bank, query, identity


def require_anchor(folder: Path, total: int) -> np.ndarray:
    progress = json.loads((folder / "anchor_states_progress.json").read_text(encoding="utf-8"))
    if progress["completed"] != total or progress["shape"] != [total, 36, 2560]:
        raise ValueError("Full pre-answer source incomplete")
    states = np.load(folder / "anchor_states.npy", mmap_mode="r")
    if states.shape != (total, 36, 2560) or states.dtype != np.float16:
        raise ValueError("Full pre-answer source shape/dtype changed")
    return states


def train_proj(folder: Path, bank: list[Sample], states: np.ndarray) -> None:
    import argparse as ap
    from run_dtd_learned_velocity import stratified_development_split
    from run_behavior_ablation import train_control

    out = folder / "trained/proj"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "projection_only_best.pt"
    if checkpoint.exists() and (out / "projection_only_metadata.json").exists():
        print(f"{folder.name}/proj checkpoint already complete", flush=True)
        return
    labels = [x.label for x in bank]
    train, dev = stratified_development_split(labels, 10, 73)
    args = ap.Namespace(seed=73, shots=4, train_steps=1500, projection_dim=256)
    model = train_control("projection_only", states, labels, train, dev, args, out)
    del model


def train_learn(folder: Path, bank: list[Sample], states: np.ndarray) -> None:
    import argparse as ap
    from run_dtd_learned_velocity import stratified_development_split, train_variant

    out = folder / "trained/learn"
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "projected_state_best.pt"
    if checkpoint.exists() and (out / "projected_state_metadata.json").exists():
        print(f"{folder.name}/learn checkpoint already complete", flush=True)
        return
    labels = [x.label for x in bank]
    train, dev = stratified_development_split(labels, 10, 73)
    args = ap.Namespace(seed=73, shots=4, projection_dim=256, weights_lr=1e-2,
                        projection_lr=3e-4, temperature=.07, classes_per_batch=16,
                        samples_per_class=4, train_steps=1500, eval_every=100)
    model, _ = train_variant("projected_state", states, labels, train, dev, args, out)
    del model


def extract_detriever_proxy(folder: Path, dataset: str, bank: list[Sample], identity: dict) -> None:
    import torch
    from run_detriever_output_proxy import extract_proxy

    out = folder / "trained/detriever"
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "gold_input_answer_eos_progress.json"
    if progress_path.exists() and json.loads(progress_path.read_text(encoding="utf-8")).get("completed") == len(bank):
        print(f"{dataset}/detriever gold proxy already complete", flush=True)
        return
    model, processor = load_model(folder / "telemetry/proxy.jsonl")
    labels = sorted({x.label for x in bank})
    extract_proxy(model, processor, bank, dataset, labels, None, out, True, identity["manifest_sha256"])
    del model, processor
    torch.cuda.empty_cache()


def train_detriever(folder: Path, bank: list[Sample], states: np.ndarray) -> None:
    import argparse as ap
    import run_dtd_paper_baselines as paper

    out = folder / "trained/detriever"
    checkpoint = out / "detriever_checkpoint.pt"
    metadata = out / "detriever_metadata.json"
    if checkpoint.exists() and metadata.exists() and json.loads(metadata.read_text(encoding="utf-8")).get("steps") == 10000:
        print(f"{folder.name}/detriever checkpoint already complete", flush=True)
        return
    proxy = np.load(out / "gold_input_answer_eos.npy", mmap_mode="r")
    if proxy.shape != (len(bank), 2560):
        raise ValueError("Gold output proxy incomplete")
    args = ap.Namespace(seed=73, train_steps=10000, train_batch_size=64,
                        positive_count=40, negative_count=100, memory_refresh=100,
                        learning_rate=1e-4, temperature=.07, resume=False)
    model, _, _ = paper.train_detriever(states[:len(bank)], [x.label for x in bank],
                                        args, checkpoint, metadata, output_proxy=proxy)
    del model


def encode_layered(model, states: np.ndarray, indices: range, chunk: int = 32):
    import torch

    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(indices), chunk):
            chosen = list(indices[start:start + chunk])
            batch = torch.from_numpy(np.asarray(states[chosen], dtype=np.float32)).to("cuda")
            parts.append(model(batch).float())
            del batch
    return torch.cat(parts)


def select_layered(model, states: np.ndarray, bank_n: int, query_n: int) -> list[list[int]]:
    import torch
    from run_dtd_learned_velocity import encode_indices
    bank = encode_indices(model, states, np.arange(bank_n))
    query = encode_indices(model, states, np.arange(bank_n, bank_n + query_n))
    selected = []
    with torch.inference_mode():
        for start in range(0, query_n, 64):
            scores = model.score(query[start:start + 64], bank)
            selected.extend(scores.topk(4, dim=1).indices.flip(1).cpu().tolist())
    return selected


def select_dense(embeddings: np.ndarray, bank_n: int, query: list[Sample]) -> list[list[int]]:
    from run_multidataset_main import nearest_from_embeddings
    selections = nearest_from_embeddings(embeddings, bank_n, query, 4)
    return [selections[q.sample_id] for q in query]


def retrieve(method: str, folder: Path, bank: list[Sample], query: list[Sample], states: np.ndarray) -> None:
    import torch
    from run_behavior_ablation import BehaviorControl
    from run_dtd_learned_velocity import StateRetriever
    import run_dtd_paper_baselines as paper

    out = folder / "selections"
    out.mkdir(exist_ok=True)
    path = out / f"{method}.json"
    if path.exists():
        value = json.loads(path.read_text(encoding="utf-8"))
        if len(value) != len(query) or list(value) != [q.sample_id for q in query]:
            raise ValueError("Existing selection scope differs")
        print(f"{folder.name}/{method} selections already complete", flush=True)
        return
    bank_n, query_n = len(bank), len(query)
    if method == "raw":
        from run_dtd_decision_states import zero_selections
        selections = zero_selections(states, bank, query, 4)
        indices = [selections[q.sample_id] for q in query]
    elif method == "proj":
        model = BehaviorControl("projection_only", 2560, 36, 256).to("cuda")
        model.load_state_dict(torch.load(folder / "trained/proj/projection_only_best.pt", map_location="cuda", weights_only=True)["model"])
        indices = select_layered(model, states, bank_n, query_n)
    elif method == "learn":
        model = StateRetriever(2560, 36, 256).to("cuda")
        model.load_state_dict(torch.load(folder / "trained/learn/projected_state_best.pt", map_location="cuda", weights_only=True)["model"])
        indices = select_layered(model, states, bank_n, query_n)
    elif method == "detriever":
        metadata = json.loads((folder / "trained/detriever/detriever_metadata.json").read_text(encoding="utf-8"))
        layers = [x - 1 for x in metadata["layers_one_based"]]
        model = paper.DeTriever(2560, len(layers)).to("cuda")
        model.load_state_dict(torch.load(folder / "trained/detriever/detriever_checkpoint.pt", map_location="cuda", weights_only=False)["model"])
        encoded = paper.detriever_embeddings(model, states, layers)
        indices = select_dense(encoded, bank_n, query)
    elif method == "gpt_mm":
        progress = json.loads((folder / "gpt_mm_progress.json").read_text(encoding="utf-8"))
        if progress["completed"] != bank_n + query_n:
            raise ValueError("GPT-MM answer-state source incomplete")
        encoded = np.load(folder / "gpt_mm_embeddings.npy", mmap_mode="r")
        indices = select_dense(encoded, bank_n, query)
    elif method == "rices":
        encoded = np.load(folder / "clip_features.npy", mmap_mode="r")
        if encoded.shape != (bank_n + query_n, 512):
            raise ValueError("CLIP source incomplete")
        indices = select_dense(encoded, bank_n, query)
    else:
        raise ValueError(method)
    if len(indices) != query_n or any(len(set(row)) != 4 or min(row) < 0 or max(row) >= bank_n for row in indices):
        raise ValueError("Invalid top-four selections")
    result = {q.sample_id: row for q, row in zip(query, indices)}
    write_json(path, result)
    print(f"{folder.name}/{method}: {len(result)} full-test selections", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--stage", choices=("protocol", "train_proj", "train_learn", "proxy", "train_detriever", "retrieve"), required=True)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    ensure_protocol()
    import random
    import torch
    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    torch.cuda.manual_seed_all(73)
    torch.set_num_threads(4)
    import run_multidataset_main
    run_multidataset_main.write_json = write_json
    if args.stage == "protocol":
        print(PROTOCOL_PATH)
        return
    folder, bank, query, identity = load_context(args.dataset)
    states = require_anchor(folder, len(bank) + len(query))
    if args.stage == "train_proj":
        train_proj(folder, bank, states)
    elif args.stage == "train_learn":
        train_learn(folder, bank, states)
    elif args.stage == "proxy":
        extract_detriever_proxy(folder, args.dataset, bank, identity)
    elif args.stage == "train_detriever":
        train_detriever(folder, bank, states)
    elif args.stage == "retrieve":
        if not args.method:
            raise ValueError("--method required for retrieve")
        retrieve(args.method, folder, bank, query, states)


if __name__ == "__main__":
    main()

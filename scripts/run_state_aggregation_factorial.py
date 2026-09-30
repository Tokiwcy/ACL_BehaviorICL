#!/usr/bin/env python
"""DTD 2x2: raw/delta states x pooled/layerwise retrieval scoring.

All four cells use the same frozen Qwen state cache, bank-only class supervision,
projection dimension, optimization schedule, and four-shot generation protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import run_dtd_decision_states as decision
import run_dtd_learned_velocity as learned
import run_multidataset_main as main_run
from multidataset_protocol import load_dataset, protocol_metadata, run_directory
from multimodal_model_adapter import MODEL_SPECS


METHODS = ("raw_pooled", "raw_layerwise", "delta_pooled", "delta_layerwise")


class FactorialRetriever(nn.Module):
    def __init__(self, width: int, layers: int, projection_dim: int,
                 state_kind: str, aggregation: str) -> None:
        super().__init__()
        if state_kind not in {"raw", "delta"} or aggregation not in {"pooled", "layerwise"}:
            raise ValueError("Unknown factorial cell")
        self.state_kind = state_kind
        self.aggregation = aggregation
        self.layer_logits = nn.Parameter(torch.zeros(layers))
        self.projection = nn.Linear(width, projection_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(dim=0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if self.state_kind == "delta":
            states = states[:, 1:] - states[:, :-1]
        encoded = F.normalize(states.float(), dim=-1)
        encoded = F.normalize(self.projection(encoded), dim=-1)
        if self.aggregation == "pooled":
            return F.normalize(torch.einsum("l,bld->bd", self.weights(), encoded), dim=-1)
        return encoded

    def score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.aggregation == "pooled":
            return left @ right.T
        return torch.einsum("ald,bld,l->ab", left, right, self.weights())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dtd", "aircraft"), default="dtd")
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--source-root", type=Path, default=Path("results/cdr_main"))
    parser.add_argument("--output-root", type=Path, default=Path("results/state_aggregation_factorial"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--stage", choices=("retrieval", "generation", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def train_cell(method: str, states: np.ndarray, labels: list[str],
               train_indices: np.ndarray, val_indices: np.ndarray,
               args: argparse.Namespace, run_dir: Path) -> FactorialRetriever:
    state_kind, aggregation = method.split("_", 1)
    layers = states.shape[1] - int(state_kind == "delta")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = FactorialRetriever(states.shape[2], layers, args.projection_dim,
                               state_kind, aggregation).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    label_names = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    train_label_ids = np.asarray([label_to_id[labels[index]] for index in train_indices])
    by_label = {
        label_id: train_indices[train_label_ids == label_id]
        for label_id in range(len(label_names))
    }
    rng = np.random.default_rng(args.seed + 10_000)
    checkpoint = run_dir / f"{method}_best.pt"
    metadata_path = run_dir / f"{method}_metadata.json"
    best_step = 0
    initial = learned.retrieval_metrics(model, states, train_indices, val_indices, labels, args.shots)
    best_metric = initial["top4_same_label_fraction"]
    history = [{"step": 0, "loss": None, "validation": initial}]
    torch.save({"model": model.state_dict(), "step": 0}, checkpoint)
    print(f"{method} step 0 validation={initial}", flush=True)

    for step in range(1, args.train_steps + 1):
        batch_indices, batch_labels = learned.balanced_batch(by_label, 16, 4, rng)
        batch = torch.from_numpy(np.asarray(states[batch_indices], dtype=np.float32)).to("cuda")
        target = torch.as_tensor(batch_labels, device="cuda")
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = model(batch)
            scores = model.score(encoded, encoded)
            loss = learned.supervised_contrastive_loss(scores, target, 0.07)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        del batch, target, encoded, scores, loss
        if step % 100 == 0 or step == args.train_steps:
            validation = learned.retrieval_metrics(
                model, states, train_indices, val_indices, labels, args.shots
            )
            history.append({"step": step, "loss": loss_value, "validation": validation})
            metric = validation["top4_same_label_fraction"]
            print(f"{method} step {step} loss={loss_value:.5f} validation={validation}", flush=True)
            if metric > best_metric:
                best_metric = metric
                best_step = step
                torch.save({"model": model.state_dict(), "step": step}, checkpoint)

    saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(saved["model"])
    metadata = {
        "method": method,
        "best_step": best_step,
        "train_steps": args.train_steps,
        "projection_dim": args.projection_dim,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "initial_validation": initial,
        "best_validation": learned.retrieval_metrics(
            model, states, train_indices, val_indices, labels, args.shots
        ),
        "selection_metric": "bank development top4_same_label_fraction",
        "learned_layer_weights": model.weights().detach().cpu().tolist(),
        "history": history,
    }
    main_run.write_json(metadata_path, metadata)
    return model


@torch.inference_mode()
def select_demos(model: FactorialRetriever, states: np.ndarray, bank: list, query: list,
                 shots: int) -> dict[str, list[int]]:
    bank_n = len(bank)
    bank_encoded = learned.encode_indices(model, states, np.arange(bank_n))
    query_encoded = learned.encode_indices(model, states, np.arange(bank_n, len(states)))
    result = {}
    for start in range(0, len(query), 64):
        scores = model.score(query_encoded[start:start + 64], bank_encoded)
        top = scores.topk(shots, dim=1).indices.flip(1).cpu().numpy()
        for offset, indices in enumerate(top):
            result[query[start + offset].sample_id] = indices.tolist()
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed Qwen protocol")
    if args.shots != 4 or args.projection_dim != 256 or args.train_steps != 1500:
        raise ValueError("This factorial fixes shots=4, projection_dim=256, train_steps=1500")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = MODEL_SPECS["qwen3vl4b"]
    bank, query = load_dataset(args.dataset, args.datasets_root)
    labels = sorted({sample.label for sample in bank})
    source = run_directory(args.source_root, args.dataset, spec.slug, args.seed)
    states = decision.verify_source(source, bank, query, spec, args.seed, args.dataset)
    cache_sha256 = file_sha256(source / "anchor_states.npy")
    run_dir = run_directory(args.output_root, args.dataset, spec.slug, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "dataset": args.dataset, "model_id": spec.model_id, "seed": args.seed,
        "ablation": "raw_delta_x_pooled_layerwise_v1", "shots": args.shots,
        "projection_dim": args.projection_dim, "train_steps": args.train_steps,
        "source_cache_sha256": cache_sha256,
    }
    main_run.validate_or_write_identity(run_dir, identity, args.resume)
    manifest = {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]}
    manifest_path = run_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Manifest differs from the saved factorial run")
    main_run.write_json(manifest_path, manifest)
    protocol = protocol_metadata(args.dataset, spec.model_id, args.seed, bank, query)
    protocol["factorial"] = {
        "state": ["36 raw decoder-layer states", "35 adjacent-layer differences"],
        "aggregation": ["weighted pooled vector then cosine", "weighted per-layer cosine"],
        "training": "bank-only supervised contrastive; ten validation examples per class",
        "source_cache_sha256": cache_sha256,
    }
    main_run.write_json(run_dir / "protocol.json", protocol)
    main_run.write_json(run_dir / "config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "methods": list(METHODS), "source_run": str(source), "state_shape": list(states.shape),
    })
    selections_path = run_dir / "selections.json"
    if args.stage != "generation" and not (args.resume and selections_path.exists()):
        train_indices, val_indices = learned.stratified_development_split(
            [sample.label for sample in bank], 10, args.seed
        )
        main_run.write_json(run_dir / "development_split.json", {
            "train_ids": [bank[index].sample_id for index in train_indices],
            "validation_ids": [bank[index].sample_id for index in val_indices],
        })
        selections = {}
        for method in METHODS:
            metadata_path = run_dir / f"{method}_metadata.json"
            checkpoint = run_dir / f"{method}_best.pt"
            if args.resume and metadata_path.exists() and checkpoint.exists():
                state_kind, aggregation = method.split("_", 1)
                layers = states.shape[1] - int(state_kind == "delta")
                model = FactorialRetriever(states.shape[2], layers, args.projection_dim,
                                           state_kind, aggregation).to("cuda")
                saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
                model.load_state_dict(saved["model"])
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("method") != method or metadata.get("train_steps") != args.train_steps:
                    raise RuntimeError(f"Saved {method} training metadata is incompatible")
                print(f"Reusing completed same-tuple {method} checkpoint", flush=True)
            else:
                model = train_cell(method, states, [sample.label for sample in bank],
                                   train_indices, val_indices, args, run_dir)
            selections[method] = select_demos(model, states, bank, query, args.shots)
            del model
            torch.cuda.empty_cache()
            print(f"{method} selections {len(selections[method])}/{len(query)}", flush=True)
        main_run.save_selections(selections_path, selections, bank)
    else:
        main_run.validate_generation_resume(run_dir, bank, query, list(METHODS), args.shots)
        selections = main_run.load_selections(selections_path, bank)
    main_run.write_json(run_dir / "retrieval_diagnostics.json",
                        main_run.retrieval_diagnostics(selections, bank, query))
    if args.stage == "retrieval":
        return
    args.methods = list(METHODS)
    args.vision_pixels = None
    main_run.run_generation(args, spec, bank, query, labels, run_dir, selections)


if __name__ == "__main__":
    main()

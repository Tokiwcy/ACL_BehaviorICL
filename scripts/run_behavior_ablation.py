#!/usr/bin/env python
"""Behavior representation and component ablations on one full test split.

The frozen zero-shot 36-layer state cache, bank-only supervised contrastive
objective, four-shot generation protocol, and test queries are shared with the
established Behavior-Zero/Learn runs. Existing Behavior results are references;
this script trains and generates only the four missing control variants.
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

import run_dtd_decision_states as behavior
import run_dtd_learned_velocity as learned
import run_multidataset_main as main_run
from multidataset_protocol import load_dataset, protocol_metadata, run_directory
from multimodal_model_adapter import MODEL_SPECS


METHODS = ("final_projected", "layer_weights_only", "projection_only")
LEGACY_METHODS = ("final_projected", "pooled_projected", "layer_weights_only", "projection_only")


class BehaviorControl(nn.Module):
    def __init__(self, mode: str, width: int = 2560, layers: int = 36, projection_dim: int = 256):
        super().__init__()
        if mode not in LEGACY_METHODS:
            raise ValueError(mode)
        self.mode = mode
        self.projection = None if mode == "layer_weights_only" else nn.Linear(width, projection_dim, bias=False)
        if self.projection is not None:
            nn.init.orthogonal_(self.projection.weight)
        self.layer_logits = nn.Parameter(torch.zeros(layers)) if mode in {
            "pooled_projected", "layer_weights_only"
        } else None

    def weights(self, layers: int, device: torch.device) -> torch.Tensor:
        if self.layer_logits is None:
            return torch.full((layers,), 1.0 / layers, device=device)
        return self.layer_logits.softmax(dim=0)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        states = F.normalize(states.float(), dim=-1)
        if self.mode == "final_projected":
            states = states[:, -1:, :]
        if self.projection is not None:
            states = self.projection(states)
        states = F.normalize(states, dim=-1)
        if self.mode == "pooled_projected":
            weights = self.weights(states.shape[1], states.device)
            return F.normalize(torch.einsum("l,bld->bd", weights, states), dim=-1)
        return states

    def score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if self.mode == "pooled_projected":
            return left @ right.T
        return torch.einsum(
            "ald,bld,l->ab", left, right, self.weights(left.shape[1], left.device)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("dtd", "aircraft", "cub", "pets", "dogs"), required=True)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--source-root", type=Path, default=Path("results/cdr_main"))
    parser.add_argument("--output-root", type=Path, default=Path("results/behavior_ablation"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache/hub"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--generation-batch-size", type=int, default=1)
    parser.add_argument("--stage", choices=("retrieval", "generation", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def train_control(mode: str, states: np.ndarray, labels: list[str],
                  train_indices: np.ndarray, val_indices: np.ndarray,
                  args: argparse.Namespace, run_dir: Path) -> BehaviorControl:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = BehaviorControl(mode, states.shape[2], states.shape[1], args.projection_dim).to("cuda")
    learning_rate = 1e-2 if mode == "layer_weights_only" else 3e-4
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    label_names = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    train_label_ids = np.asarray([label_to_id[labels[index]] for index in train_indices])
    by_label = {
        label_id: train_indices[train_label_ids == label_id]
        for label_id in range(len(label_names))
    }
    rng = np.random.default_rng(args.seed + 10_000)
    checkpoint = run_dir / f"{mode}_best.pt"
    history = []
    initial = learned.retrieval_metrics(model, states, train_indices, val_indices, labels, args.shots)
    best_metric, best_step = initial["top4_same_label_fraction"], 0
    history.append({"step": 0, "loss": None, "validation": initial})
    torch.save({"model": model.state_dict(), "step": 0}, checkpoint)
    print(f"{mode} step 0 validation={initial}", flush=True)
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
            validation = learned.retrieval_metrics(model, states, train_indices, val_indices, labels, args.shots)
            history.append({"step": step, "loss": loss_value, "validation": validation})
            metric = validation["top4_same_label_fraction"]
            print(f"{mode} step {step} loss={loss_value:.5f} validation={validation}", flush=True)
            if metric > best_metric:
                best_metric, best_step = metric, step
                torch.save({"model": model.state_dict(), "step": step}, checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True)["model"])
    metadata = {
        "method": mode, "best_step": best_step, "train_steps": args.train_steps,
        "projection_dim": None if model.projection is None else args.projection_dim,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "learning_rate": learning_rate,
        "selection_metric": "bank development top4_same_label_fraction",
        "initial_validation": initial,
        "best_validation": learned.retrieval_metrics(model, states, train_indices, val_indices, labels, args.shots),
        "layer_weights": model.weights(states.shape[1] if mode != "final_projected" else 1,
                                       torch.device("cuda")).detach().cpu().tolist(),
        "history": history,
    }
    main_run.write_json(run_dir / f"{mode}_metadata.json", metadata)
    return model


@torch.inference_mode()
def select_demos(model: BehaviorControl, states: np.ndarray, bank: list, query: list,
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
        raise RuntimeError("CUDA is required")
    if (args.seed, args.shots, args.train_steps, args.projection_dim) != (73, 4, 1500, 256):
        raise ValueError("Fixed main protocol requires seed73, 4-shot, 1500 steps, 256 dimensions")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = MODEL_SPECS["qwen3vl4b"]
    bank, query = load_dataset(args.dataset, args.datasets_root)
    labels = sorted({sample.label for sample in bank})
    source = run_directory(args.source_root, args.dataset, spec.slug, args.seed)
    states = behavior.verify_source(source, bank, query, spec, args.seed, args.dataset)
    cache_sha256 = sha256_file(source / "anchor_states.npy")
    run_dir = run_directory(args.output_root, args.dataset, spec.slug, args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "dataset": args.dataset, "model_id": spec.model_id, "seed": args.seed,
        "ablation": "behavior_representation_components_v2", "shots": args.shots,
        "projection_dim": args.projection_dim, "train_steps": args.train_steps,
        "source_cache_sha256": cache_sha256,
    }
    main_run.validate_or_write_identity(run_dir, identity, args.resume)
    manifest = {"bank": [asdict(row) for row in bank], "query": [asdict(row) for row in query]}
    manifest_path = run_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("Manifest differs from saved ablation tuple")
    main_run.write_json(manifest_path, manifest)
    protocol = protocol_metadata(args.dataset, spec.model_id, args.seed, bank, query)
    protocol["behavior_ablation"] = {
        "reference_methods": ["decision_zero", "decision_learn"],
        "new_methods": list(METHODS),
        "feature": "36 zero-shot answer-position decoder states; fixed source cache",
        "training": "bank-only same-label supervised contrastive; ten validation samples per class",
        "source_cache_sha256": cache_sha256,
    }
    main_run.write_json(run_dir / "protocol.json", protocol)
    main_run.write_json(run_dir / "config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "methods": list(METHODS), "source_run": str(source), "state_shape": list(states.shape),
    })
    selections_path = run_dir / "selections.json"
    if args.stage == "generation" or (args.resume and selections_path.exists()):
        main_run.validate_generation_resume(run_dir, bank, query, list(METHODS), args.shots)
        selections = main_run.load_selections(selections_path, bank)
    else:
        train_indices, val_indices = learned.stratified_development_split(
            [sample.label for sample in bank], 10, args.seed
        )
        main_run.write_json(run_dir / "development_split.json", {
            "train_ids": [bank[index].sample_id for index in train_indices],
            "validation_ids": [bank[index].sample_id for index in val_indices],
        })
        selections = {}
        for mode in METHODS:
            checkpoint = run_dir / f"{mode}_best.pt"
            metadata_path = run_dir / f"{mode}_metadata.json"
            if args.resume and checkpoint.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("method") != mode or metadata.get("train_steps") != args.train_steps:
                    raise RuntimeError(f"Incompatible saved {mode} training")
                model = BehaviorControl(mode, states.shape[2], states.shape[1], args.projection_dim).to("cuda")
                model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True)["model"])
                print(f"Reusing completed same-tuple {mode} checkpoint", flush=True)
            else:
                model = train_control(mode, states[:len(bank)], [row.label for row in bank],
                                      train_indices, val_indices, args, run_dir)
            selections[mode] = select_demos(model, states, bank, query, args.shots)
            del model
            torch.cuda.empty_cache()
            print(f"{mode} selections {len(selections[mode])}/{len(query)}", flush=True)
        main_run.save_selections(selections_path, selections, bank)
    main_run.write_json(run_dir / "retrieval_diagnostics.json",
                        main_run.retrieval_diagnostics(selections, bank, query))
    if args.stage == "retrieval":
        return
    args.methods = list(METHODS)
    args.vision_pixels = None
    main_run.run_generation(args, spec, bank, query, labels, run_dir, selections)


if __name__ == "__main__":
    main()

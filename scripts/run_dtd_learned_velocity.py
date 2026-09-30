#!/usr/bin/env python
"""Train a small retrieval head on frozen 35-step Qwen velocity trajectories.

The Qwen hidden-state cache is never updated.  Retriever training uses a deterministic
40/10-per-class split of the DTD development bank; the 940 test labels are touched only
after the variant and checkpoint have been selected on the development validation split.
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

import run_dtd_hidden_rices_pilot as base


VARIANTS = ("layer_weights", "projected_velocity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("results/dtd_qwen3vl4b_trajectory_full47/seed_73"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/dtd_qwen3vl4b_learned_velocity_full47")
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--val-per-class", type=int, default=10)
    parser.add_argument("--classes-per-batch", type=int, default=16)
    parser.add_argument("--samples-per-class", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--projection-lr", type=float, default=3e-4)
    parser.add_argument("--weights-lr", type=float, default=1e-2)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
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


def stratified_development_split(
    labels: list[str], val_per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels_np = np.asarray(labels)
    train, val = [], []
    for label in sorted(set(labels)):
        indices = np.flatnonzero(labels_np == label)
        shuffled = rng.permutation(indices)
        if len(indices) <= val_per_class + 1:
            raise ValueError(f"Not enough examples of {label} for the requested split")
        val.extend(shuffled[:val_per_class].tolist())
        train.extend(shuffled[val_per_class:].tolist())
    return np.asarray(sorted(train)), np.asarray(sorted(val))


class VelocityRetriever(nn.Module):
    """Layer-weighted cosine kernel, optionally with one shared low-rank projection."""

    def __init__(self, width: int, layer_count: int, projection_dim: int | None) -> None:
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(layer_count))
        self.projection = (
            nn.Linear(width, projection_dim, bias=False) if projection_dim is not None else None
        )
        if self.projection is not None:
            nn.init.orthogonal_(self.projection.weight)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        velocity = states[:, 1:] - states[:, :-1]
        velocity = F.normalize(velocity.float(), dim=-1)
        if self.projection is not None:
            velocity = self.projection(velocity)
        return F.normalize(velocity, dim=-1)

    def weights(self) -> torch.Tensor:
        return self.layer_logits.softmax(dim=0)

    def score(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.einsum("ald,bld,l->ab", left, right, self.weights())


class StateRetriever(VelocityRetriever):
    """Parameter-matched control that scores normalized raw layer states."""

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        encoded = F.normalize(states.float(), dim=-1)
        if self.projection is not None:
            encoded = self.projection(encoded)
        return F.normalize(encoded, dim=-1)


def supervised_contrastive_loss(
    scores: torch.Tensor, labels: torch.Tensor, temperature: float
) -> torch.Tensor:
    if scores.ndim != 2 or scores.shape[0] != scores.shape[1]:
        raise ValueError("Training scores must be a square pairwise matrix")
    same = labels[:, None].eq(labels[None, :])
    diagonal = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positives = same & ~diagonal
    valid = ~diagonal
    if not positives.any(dim=1).all():
        raise ValueError("Every anchor needs at least one positive")
    logits = scores / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    negative_inf = torch.finfo(logits.dtype).min
    numerator = torch.logsumexp(logits.masked_fill(~positives, negative_inf), dim=1)
    denominator = torch.logsumexp(logits.masked_fill(~valid, negative_inf), dim=1)
    return -(numerator - denominator).mean()


def balanced_batch(
    by_label: dict[int, np.ndarray], classes_per_batch: int, samples_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    keys = np.asarray(sorted(by_label))
    chosen = rng.choice(keys, size=classes_per_batch, replace=classes_per_batch > len(keys))
    indices, labels = [], []
    for label in chosen:
        pool = by_label[int(label)]
        selected = rng.choice(pool, size=samples_per_class, replace=samples_per_class > len(pool))
        indices.extend(selected.tolist())
        labels.extend([int(label)] * samples_per_class)
    return np.asarray(indices), np.asarray(labels)


@torch.inference_mode()
def encode_indices(
    model: VelocityRetriever, states: np.ndarray, indices: np.ndarray, chunk_size: int = 64
) -> torch.Tensor:
    model.eval()
    encoded = []
    for start in range(0, len(indices), chunk_size):
        batch = torch.from_numpy(
            np.asarray(states[indices[start : start + chunk_size]], dtype=np.float32)
        ).to("cuda")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded.append(model(batch).float())
        del batch
    return torch.cat(encoded)


@torch.inference_mode()
def retrieval_metrics(
    model: VelocityRetriever,
    states: np.ndarray,
    candidate_indices: np.ndarray,
    query_indices: np.ndarray,
    labels: list[str],
    shots: int,
) -> dict:
    candidate = encode_indices(model, states, candidate_indices)
    query = encode_indices(model, states, query_indices)
    scores = model.score(query, candidate)
    top = scores.topk(shots, dim=1).indices.cpu().numpy()
    candidate_labels = np.asarray(labels)[candidate_indices]
    query_labels = np.asarray(labels)[query_indices]
    retrieved = candidate_labels[top]
    matches = retrieved == query_labels[:, None]
    reciprocal = []
    full_order = scores.argsort(dim=1, descending=True).cpu().numpy()
    for row, label in zip(full_order, query_labels):
        ranks = np.flatnonzero(candidate_labels[row] == label)
        reciprocal.append(1.0 / (int(ranks[0]) + 1))
    return {
        "top4_same_label_fraction": float(matches.mean()),
        "top4_any_same_label": float(matches.any(axis=1).mean()),
        "mean_reciprocal_rank": float(np.mean(reciprocal)),
    }


def train_variant(
    variant: str,
    states: np.ndarray,
    labels: list[str],
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    args: argparse.Namespace,
    run_dir: Path,
) -> tuple[VelocityRetriever, dict]:
    projection_dim = None if variant == "layer_weights" else args.projection_dim
    if variant == "projected_state":
        model = StateRetriever(states.shape[2], states.shape[1], projection_dim).to("cuda")
    else:
        model = VelocityRetriever(states.shape[2], states.shape[1] - 1, projection_dim).to("cuda")
    lr = args.weights_lr if projection_dim is None else args.projection_lr
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    checkpoint = run_dir / f"{variant}_best.pt"
    label_names = sorted(set(labels))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    train_label_ids = np.asarray([label_to_id[labels[index]] for index in train_indices])
    by_label = {
        label_id: train_indices[train_label_ids == label_id] for label_id in range(len(label_names))
    }
    rng = np.random.default_rng(args.seed + (0 if variant == "layer_weights" else 10_000))
    history: list[dict] = []
    best_metric = -1.0
    best_step = 0

    uniform = retrieval_metrics(model, states, train_indices, val_indices, labels, args.shots)
    history.append({"step": 0, "loss": None, "validation": uniform})
    best_metric = uniform["top4_same_label_fraction"]
    torch.save({"model": model.state_dict(), "step": 0}, checkpoint)
    print(f"{variant} step 0000 validation={uniform}", flush=True)

    for step in range(1, args.train_steps + 1):
        batch_indices, batch_labels = balanced_batch(
            by_label, args.classes_per_batch, args.samples_per_class, rng
        )
        batch = torch.from_numpy(np.asarray(states[batch_indices], dtype=np.float32)).to("cuda")
        target = torch.as_tensor(batch_labels, device="cuda")
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            encoded = model(batch)
            scores = model.score(encoded, encoded)
            loss = supervised_contrastive_loss(scores, target, args.temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        del batch, target, encoded, scores, loss

        if step % args.eval_every == 0 or step == args.train_steps:
            validation = retrieval_metrics(
                model, states, train_indices, val_indices, labels, args.shots
            )
            history.append({"step": step, "loss": loss_value, "validation": validation})
            metric = validation["top4_same_label_fraction"]
            print(
                f"{variant} step {step:04d} loss={loss_value:.5f} validation={validation}",
                flush=True,
            )
            if metric > best_metric:
                best_metric = metric
                best_step = step
                torch.save({"model": model.state_dict(), "step": step}, checkpoint)

    saved = torch.load(checkpoint, map_location="cuda", weights_only=True)
    model.load_state_dict(saved["model"])
    metadata = {
        "variant": variant,
        "best_step": best_step,
        "selection_metric": "development validation top4_same_label_fraction",
        "best_validation": retrieval_metrics(
            model, states, train_indices, val_indices, labels, args.shots
        ),
        "initial_validation": uniform,
        "history": history,
        "learned_layer_weights": model.weights().detach().cpu().tolist(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    write_json(run_dir / f"{variant}_metadata.json", metadata)
    return model, metadata


@torch.inference_mode()
def final_selections(
    model: VelocityRetriever,
    states: np.ndarray,
    bank: list[base.Sample],
    query: list[base.Sample],
    shots: int,
) -> tuple[dict[str, list[int]], dict]:
    bank_n = len(bank)
    bank_encoded = encode_indices(model, states, np.arange(bank_n))
    query_encoded = encode_indices(model, states, np.arange(bank_n, len(states)))
    selections: dict[str, list[int]] = {}
    same = 0
    distinct = []
    for start in range(0, len(query), 64):
        score = model.score(query_encoded[start : start + 64], bank_encoded)
        top = score.topk(shots, dim=1).indices.cpu().numpy()
        for offset, indices in enumerate(top):
            sample = query[start + offset]
            selections[sample.sample_id] = indices.tolist()
            retrieved_labels = [bank[index].label for index in indices]
            same += sum(label == sample.label for label in retrieved_labels)
            distinct.append(len(set(retrieved_labels)))
    diagnostics = {
        "same_label_demo_fraction": same / (len(query) * shots),
        "mean_distinct_demo_labels": float(np.mean(distinct)),
    }
    return selections, diagnostics


def prediction_summary(rows: list[dict]) -> dict:
    return {
        "correct": sum(bool(row["correct"]) for row in rows),
        "total": len(rows),
        "accuracy": sum(bool(row["correct"]) for row in rows) / len(rows),
        "parse_failures": sum(row["prediction"] is None for row in rows),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    base.seed_everything(args.seed)
    random.seed(args.seed)
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    bank, query = load_samples(args.source_run / "manifest.json")
    labels = [sample.label for sample in bank]
    classes = sorted(set(labels))
    states = np.load(args.source_run / "all_layer_anchor_states.npy", mmap_mode="r")
    if len(states) != len(bank) + len(query):
        raise ValueError("Hidden-state cache and manifest disagree")
    train_indices, val_indices = stratified_development_split(
        labels, args.val_per_class, args.seed
    )
    write_json(run_dir / "config.json", {
        **{key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "development_train_count": len(train_indices),
        "development_validation_count": len(val_indices),
        "test_count": len(query),
        "selection_rule": "highest validation top4 same-label fraction; tie -> fewer parameters",
    })
    write_json(run_dir / "manifest.json", {
        "bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query],
        "development_train_ids": [bank[index].sample_id for index in train_indices],
        "development_validation_ids": [bank[index].sample_id for index in val_indices],
    })

    models: dict[str, VelocityRetriever] = {}
    metadata: dict[str, dict] = {}
    for variant in args.variants:
        checkpoint = run_dir / f"{variant}_best.pt"
        meta_path = run_dir / f"{variant}_metadata.json"
        if args.resume and checkpoint.exists() and meta_path.exists():
            projection_dim = None if variant == "layer_weights" else args.projection_dim
            model = VelocityRetriever(states.shape[2], states.shape[1] - 1, projection_dim).to("cuda")
            model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True)["model"])
            models[variant] = model
            metadata[variant] = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            models[variant], metadata[variant] = train_variant(
                variant, states, labels, train_indices, val_indices, args, run_dir
            )

    winner = max(
        args.variants,
        key=lambda name: (
            metadata[name]["best_validation"]["top4_same_label_fraction"],
            -metadata[name]["parameter_count"],
        ),
    )
    selections, retrieval = final_selections(
        models[winner], states, bank, query, args.shots
    )
    write_json(run_dir / "selection_result.json", {
        "selected_variant": winner,
        "variant_validation": {
            name: value["best_validation"] for name, value in metadata.items()
        },
        "test_retrieval_diagnostics_after_selection": retrieval,
    })
    write_json(run_dir / "selections.json", {
        query_id: [asdict(bank[index]) for index in indices]
        for query_id, indices in selections.items()
    })
    print(json.dumps(json.loads((run_dir / "selection_result.json").read_text()), indent=2), flush=True)
    if args.selection_only:
        return

    del models
    torch.cuda.empty_cache()
    model, processor = base.load_qwen(args.qwen_model, args.cache_dir)
    predictions_path = run_dir / "predictions.jsonl"
    completed: dict[str, dict] = {}
    if args.resume and predictions_path.exists():
        for line in predictions_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            completed[row["query_id"]] = row
    pending = [sample for sample in query if sample.sample_id not in completed]
    started = time.time()
    with predictions_path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(pending), args.generation_batch_size):
            batch_samples = pending[start : start + args.generation_batch_size]
            batch_demos = [
                [bank[index] for index in selections[sample.sample_id]] for sample in batch_samples
            ]
            outputs = base.generate_predictions_batch(
                model, processor,
                [
                    base.build_icl_messages(demos, sample, classes, args.image_size)
                    for demos, sample in zip(batch_demos, batch_samples)
                ],
                classes, args.max_new_tokens,
            )
            for offset, (sample, demos, (prediction, raw)) in enumerate(
                zip(batch_samples, batch_demos, outputs), start=start + 1
            ):
                row = {
                    "method": f"learned_35_step_velocity_{winner}",
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
                completed[sample.sample_id] = row
                print(
                    f"learned velocity {offset:04d}/{len(query):04d}: "
                    f"target={sample.label} pred={prediction} raw={raw!r}", flush=True,
                )
    rows = [completed[sample.sample_id] for sample in query]
    write_json(run_dir / "summary.json", {
        "selected_variant": winner,
        "summary": prediction_summary(rows),
        "elapsed_seconds_generation_stage": time.time() - started,
    })
    print(json.dumps(json.loads((run_dir / "summary.json").read_text()), indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Training-free computation-trajectory retrieval for multimodal ICL on DTD."""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.format import open_memmap

import run_dtd_hidden_rices_pilot as base


METHODS = ("all_layer_velocity", "adaptive_dynamics", "trajectory_dtw", "velocity_dtw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-run",
        type=Path,
        default=Path("results/dtd_qwen3vl4b_standard_4shot_full47/seed_73"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/dtd_qwen3vl4b_trajectory_full47")
    )
    parser.add_argument("--qwen-model", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--shots", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--extract-batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtw-pool", type=int, default=64)
    parser.add_argument("--dtw-band", type=int, default=6)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--recompute-selections", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_samples(path: Path) -> tuple[list[base.Sample], list[base.Sample]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    bank = [base.Sample(**item) for item in manifest["bank"]]
    query = [base.Sample(**item) for item in manifest["query"]]
    return bank, query


def _trajectory_messages(samples: list[base.Sample], classes: list[str], image_size: int):
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


def extract_all_layer_states(
    model,
    processor,
    samples: list[base.Sample],
    classes: list[str],
    image_size: int,
    batch_size: int,
    state_path: Path,
    progress_path: Path,
    resume: bool,
) -> np.ndarray:
    layers = int(model.config.text_config.num_hidden_layers)
    width = int(model.config.text_config.hidden_size)
    start = 0
    if resume and state_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = [len(samples), layers, width]
        if progress.get("shape") != expected:
            raise ValueError("Existing trajectory cache has an incompatible shape")
        start = int(progress.get("completed", 0))
        states = open_memmap(state_path, mode="r+", dtype=np.float16, shape=tuple(expected))
    else:
        states = open_memmap(
            state_path, mode="w+", dtype=np.float16, shape=(len(samples), layers, width)
        )
        write_json(progress_path, {"shape": [len(samples), layers, width], "completed": 0})

    previous_padding_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "left"
    try:
        for offset in range(start, len(samples), batch_size):
            batch = samples[offset : offset + batch_size]
            inputs = processor.apply_chat_template(
                _trajectory_messages(batch, classes, image_size),
                tokenize=True,
                add_generation_prompt=True,
                padding=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = base.move_inputs(inputs)
            with torch.inference_mode():
                outputs = model(**inputs, output_hidden_states=True, use_cache=False, return_dict=True)
            if outputs.hidden_states is None or len(outputs.hidden_states) != layers + 1:
                raise RuntimeError("Unexpected hidden-state structure from Qwen")
            anchor_states = torch.stack(
                [layer_state[:, -1].detach() for layer_state in outputs.hidden_states[1:]], dim=1
            )
            states[offset : offset + len(batch)] = anchor_states.float().cpu().numpy().astype(np.float16)
            states.flush()
            completed = offset + len(batch)
            write_json(
                progress_path,
                {"shape": [len(samples), layers, width], "completed": completed},
            )
            del outputs, anchor_states, inputs
            if completed % 40 == 0 or completed == len(samples):
                torch.cuda.empty_cache()
                print(f"trajectory hidden {completed:04d}/{len(samples):04d}", flush=True)
    finally:
        processor.tokenizer.padding_side = previous_padding_side
    return np.load(state_path, mmap_mode="r")


def activity_weights(speed: torch.Tensor, curvature: torch.Tensor) -> torch.Tensor:
    """Parameter-free query weights from relative speed and turning activity."""
    if speed.ndim != 2 or curvature.shape != speed.shape:
        raise ValueError("speed and curvature must both be [N,L]")
    relative_speed = speed / speed.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    raw = relative_speed * (1.0 + curvature.clamp_min(0.0))
    return raw / raw.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def dtw_scores(local_similarity: np.ndarray, band: int) -> np.ndarray:
    """Banded monotonic alignment score for [C,L,L] local cosine matrices."""
    candidates, left_len, right_len = local_similarity.shape
    if left_len != right_len:
        raise ValueError("Current DTW implementation expects equal trajectory lengths")
    length = left_len
    inf = np.float32(1e9)
    dp = np.full((candidates, length + 1, length + 1), inf, dtype=np.float32)
    dp[:, 0, 0] = 0.0
    temporal = np.abs(np.arange(length)[:, None] - np.arange(length)[None, :]) / max(1, length - 1)
    cost = 1.0 - local_similarity.astype(np.float32) + 0.05 * temporal[None]
    for i in range(1, length + 1):
        for j in range(max(1, i - band), min(length, i + band) + 1):
            previous = np.minimum(
                np.minimum(dp[:, i - 1, j], dp[:, i, j - 1]), dp[:, i - 1, j - 1]
            )
            dp[:, i, j] = cost[:, i - 1, j - 1] + previous
    return -dp[:, length, length] / length


def compute_selections(
    states: np.ndarray,
    bank: list[base.Sample],
    query: list[base.Sample],
    shots: int,
    dtw_pool: int,
    dtw_band: int,
) -> tuple[dict[str, dict[str, list[int]]], dict]:
    bank_n = len(bank)
    device = torch.device("cuda")
    hidden = torch.as_tensor(np.asarray(states), dtype=torch.bfloat16, device=device)
    velocity = hidden[:, 1:] - hidden[:, :-1]
    del hidden
    acceleration = velocity[:, 1:] - velocity[:, :-1]
    speed = torch.linalg.vector_norm(velocity.float(), dim=-1)
    velocity_norm = F.normalize(velocity, dim=-1)
    del velocity
    acceleration_norm = F.normalize(acceleration, dim=-1)
    del acceleration
    curvature_inner = 1.0 - (velocity_norm[:, 1:].float() * velocity_norm[:, :-1].float()).sum(-1)
    curvature = torch.cat(
        [torch.zeros((len(states), 1), device=device), curvature_inner], dim=1
    ).clamp(0.0, 2.0)
    weights = activity_weights(speed, curvature)

    bank_v, query_v = velocity_norm[:bank_n], velocity_norm[bank_n:]
    bank_a, query_a = acceleration_norm[:bank_n], acceleration_norm[bank_n:]
    bank_speed, query_speed = speed[:bank_n], speed[bank_n:]
    bank_curv, query_curv = curvature[:bank_n], curvature[bank_n:]
    query_weights = weights[bank_n:]
    del velocity_norm, acceleration_norm, speed, curvature, weights

    velocity_scores = torch.empty((len(query), bank_n), dtype=torch.float32, device="cpu")
    dynamics_scores = torch.empty_like(velocity_scores)
    batch_size = 20
    log_bank_speed = torch.log(bank_speed.clamp_min(1e-12))
    for start in range(0, len(query), batch_size):
        end = min(len(query), start + batch_size)
        qv = query_v[start:end]
        qa = query_a[start:end]
        qw = query_weights[start:end]
        velocity_equal = torch.einsum("qld,bld->qb", qv.float(), bank_v.float()) / qv.shape[1]
        velocity_adaptive = torch.einsum("qld,bld,ql->qb", qv.float(), bank_v.float(), qw)
        acceleration_weights = qw[:, 1:]
        acceleration_weights = acceleration_weights / acceleration_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-12)
        acceleration_adaptive = torch.einsum(
            "qld,bld,ql->qb", qa.float(), bank_a.float(), acceleration_weights
        )
        speed_similarity = torch.exp(
            -torch.abs(
                torch.log(query_speed[start:end].clamp_min(1e-12))[:, None, :]
                - log_bank_speed[None, :, :]
            )
        )
        speed_similarity = torch.einsum("qbl,ql->qb", speed_similarity, qw)
        curvature_similarity = torch.exp(
            -torch.abs(query_curv[start:end, None, :] - bank_curv[None, :, :])
        )
        curvature_similarity = torch.einsum("qbl,ql->qb", curvature_similarity, qw)
        combined = (
            velocity_adaptive
            + acceleration_adaptive
            + speed_similarity
            + curvature_similarity
        ) / 4.0
        velocity_scores[start:end] = velocity_equal.cpu()
        dynamics_scores[start:end] = combined.cpu()
        print(f"trajectory kernel {end:04d}/{len(query):04d}", flush=True)

    selections: dict[str, dict[str, list[int]]] = {method: {} for method in METHODS}
    velocity_np = velocity_scores.numpy()
    dynamics_np = dynamics_scores.numpy()
    dtw_candidates = np.argsort(dynamics_np, axis=1)[:, -min(dtw_pool, bank_n) :]
    velocity_dtw_candidates = np.argsort(velocity_np, axis=1)[:, -min(dtw_pool, bank_n) :]

    velocity_dtw_all = np.empty(
        (len(query), velocity_dtw_candidates.shape[1]), dtype=np.float32
    )
    for q_idx, candidates in enumerate(velocity_dtw_candidates):
        local = torch.einsum(
            "id,cjd->cij", query_v[q_idx].float(), bank_v[candidates].float()
        )
        velocity_dtw_all[q_idx] = dtw_scores(local.cpu().numpy(), dtw_band)
        if (q_idx + 1) % 40 == 0 or q_idx + 1 == len(query):
            print(f"velocity DTW {q_idx + 1:04d}/{len(query):04d}", flush=True)

    # DTW aligns the actual layer states. The dynamics kernel above is the full-bank prefilter,
    # so no external visual/text retriever participates in candidate generation.
    del bank_v, query_v, bank_a, query_a, bank_speed, query_speed, bank_curv, query_curv
    torch.cuda.empty_cache()
    hidden = torch.as_tensor(np.asarray(states), dtype=torch.bfloat16, device=device)
    hidden_norm = F.normalize(hidden, dim=-1)
    del hidden
    bank_h, query_h = hidden_norm[:bank_n], hidden_norm[bank_n:]
    dtw_all = np.empty((len(query), dtw_candidates.shape[1]), dtype=np.float32)
    for q_idx, candidates in enumerate(dtw_candidates):
        local = torch.einsum(
            "id,cjd->cij", query_h[q_idx].float(), bank_h[candidates].float()
        )
        dtw_all[q_idx] = dtw_scores(local.cpu().numpy(), dtw_band)
        if (q_idx + 1) % 40 == 0 or q_idx + 1 == len(query):
            print(f"trajectory DTW {q_idx + 1:04d}/{len(query):04d}", flush=True)

    for q_idx, sample in enumerate(query):
        velocity_top = np.argsort(velocity_np[q_idx])[-shots:]
        dynamics_top = np.argsort(dynamics_np[q_idx])[-shots:]
        candidates = dtw_candidates[q_idx]
        dtw_top = candidates[np.argsort(dtw_all[q_idx])[-shots:]]
        velocity_candidates = velocity_dtw_candidates[q_idx]
        velocity_dtw_top = velocity_candidates[
            np.argsort(velocity_dtw_all[q_idx])[-shots:]
        ]
        selections["all_layer_velocity"][sample.sample_id] = velocity_top.tolist()
        selections["adaptive_dynamics"][sample.sample_id] = dynamics_top.tolist()
        selections["trajectory_dtw"][sample.sample_id] = dtw_top.tolist()
        selections["velocity_dtw"][sample.sample_id] = velocity_dtw_top.tolist()

    diagnostics = {
        "layers": int(states.shape[1]),
        "hidden_size": int(states.shape[2]),
        "dtw_pool": int(dtw_candidates.shape[1]),
        "dtw_band": dtw_band,
        "kernel": (
            "Equal mean of query-activity-weighted velocity direction, acceleration direction, "
            "log-speed similarity, and curvature similarity"
        ),
        "query_weights": "relative_speed * (1 + curvature), normalized across layers",
        "dtw_local_score": "cosine between layer states with monotonic banded alignment",
        "velocity_dtw_local_score": (
            "cosine between layer velocities with monotonic banded alignment; "
            "full-bank prefilter is all-layer velocity"
        ),
    }
    return selections, diagnostics


def selection_diagnostics(
    selections: dict[str, dict[str, list[int]]],
    bank: list[base.Sample],
    query: list[base.Sample],
) -> dict:
    query_by_id = {sample.sample_id: sample for sample in query}
    result = {}
    for method, by_query in selections.items():
        same = 0
        total = 0
        distinct_labels = []
        for query_id, indices in by_query.items():
            labels = [bank[index].label for index in indices]
            same += sum(label == query_by_id[query_id].label for label in labels)
            total += len(labels)
            distinct_labels.append(len(set(labels)))
        result[method] = {
            "same_label_demo_fraction": same / total,
            "mean_distinct_demo_labels": float(np.mean(distinct_labels)),
        }
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    base.seed_everything(args.seed)
    source_manifest = args.source_run / "manifest.json"
    bank, query = load_samples(source_manifest)
    classes = sorted({sample.label for sample in bank})
    all_samples = bank + query
    run_dir = args.output_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "manifest.json",
        {"bank": [asdict(x) for x in bank], "query": [asdict(x) for x in query]},
    )
    write_json(
        run_dir / "config.json",
        {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    )

    state_path = run_dir / "all_layer_anchor_states.npy"
    progress_path = run_dir / "trajectory_progress.json"
    model = processor = None
    cache_complete = False
    if args.resume and state_path.exists() and progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        cache_complete = int(progress.get("completed", 0)) == len(all_samples)
    if cache_complete:
        states = np.load(state_path, mmap_mode="r")
    else:
        model, processor = base.load_qwen(args.qwen_model, args.cache_dir)
        states = extract_all_layer_states(
            model,
            processor,
            all_samples,
            classes,
            args.image_size,
            args.extract_batch_size,
            state_path,
            progress_path,
            args.resume,
        )
    selections_path = run_dir / "selections.json"
    diagnostics_path = run_dir / "trajectory_diagnostics.json"
    if (
        args.resume
        and not args.recompute_selections
        and selections_path.exists()
        and diagnostics_path.exists()
    ):
        detailed = json.loads(selections_path.read_text(encoding="utf-8"))
        sample_index = {sample.sample_id: index for index, sample in enumerate(bank)}
        selections = {
            method: {
                query_id: [sample_index[item["sample_id"]] for item in demos]
                for query_id, demos in by_query.items()
            }
            for method, by_query in detailed.items()
        }
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))["trajectory"]
    else:
        selections, diagnostics = compute_selections(
            states, bank, query, args.shots, args.dtw_pool, args.dtw_band
        )
        detailed = {
            method: {
                query_id: [asdict(bank[index]) for index in indices]
                for query_id, indices in by_query.items()
            }
            for method, by_query in selections.items()
        }
        write_json(selections_path, detailed)
        write_json(
            diagnostics_path,
            {
                "trajectory": diagnostics,
                "retrieval": selection_diagnostics(selections, bank, query),
            },
        )

    print(json.dumps(json.loads(diagnostics_path.read_text(encoding="utf-8")), indent=2), flush=True)
    if args.selection_only:
        return

    if model is None:
        model, processor = base.load_qwen(args.qwen_model, args.cache_dir)

    selections = {method: selections[method] for method in args.methods}
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

    summary = {}
    summary_methods = [
        method
        for method in METHODS
        if any(name == method for name, _ in completed)
    ]
    for method in summary_methods:
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

#!/usr/bin/env python
"""Check label reproducibility when resuming a tuple on different GPU hardware."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from multidataset_protocol import Sample
from multimodal_model_adapter import MODEL_SPECS, generate_legal_labels, icl_messages, load_model
from run_multidataset_main import load_selections


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 1 or args.batch_size < 1 or args.offset < 0:
        raise ValueError("samples and batch size must be positive; offset must be nonnegative")

    manifest = json.loads((args.run_dir / "manifest.json").read_text(encoding="utf-8"))
    bank = [Sample(**row) for row in manifest["bank"]]
    query = {row["sample_id"]: Sample(**row) for row in manifest["query"]}
    labels = sorted({sample.label for sample in bank})
    selections = load_selections(args.run_dir / "selections.json", bank)
    previous = [json.loads(line) for line in
                (args.run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    previous = [row for row in previous if row["method"] == "rices"]
    previous = previous[args.offset:args.offset + args.samples]
    if len(previous) != args.samples:
        raise RuntimeError("Not enough prior RICES predictions for hardware check")

    model, processor = load_model(MODEL_SPECS["qwen3vl4b"], args.cache_dir)
    mismatched_labels: list[str] = []
    mismatched_raw: list[str] = []
    seconds = 0.0
    for start in range(0, len(previous), args.batch_size):
        batch = previous[start:start + args.batch_size]
        messages = []
        for row in batch:
            demos = [bank[index] for index in selections["rices"][row["query_id"]]]
            if [sample.sample_id for sample in demos] != row["demo_ids"]:
                raise RuntimeError("Saved demonstrations do not match retrieval selections")
            messages.append(icl_messages(demos, query[row["query_id"]], "aircraft", labels, 224))
        torch.cuda.synchronize()
        begun = time.perf_counter()
        outputs = generate_legal_labels(model, processor, messages, labels)
        torch.cuda.synchronize()
        seconds += time.perf_counter() - begun
        for row, (label, raw) in zip(batch, outputs):
            if label != row["prediction"]:
                mismatched_labels.append(row["query_id"])
            if raw != row["raw_output"]:
                mismatched_raw.append(row["query_id"])
        print(f"checked {min(start + args.batch_size, len(previous))}/{len(previous)}", flush=True)
    print(json.dumps({
        "offset": args.offset,
        "checked": len(previous),
        "label_mismatch_ids": mismatched_labels,
        "raw_mismatch_ids": mismatched_raw,
        "seconds": seconds,
        "queries_per_minute": len(previous) * 60 / seconds,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

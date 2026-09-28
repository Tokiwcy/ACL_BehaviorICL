#!/usr/bin/env python
"""One-batch GPU smoke test for hidden trajectories and legal-label decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from multidataset_protocol import load_dataset
from multimodal_model_adapter import (
    MODEL_SPECS,
    anchor_hidden_states,
    generate_legal_labels,
    icl_messages,
    load_model,
    model_dimensions,
    prepare_batch,
    zero_shot_messages,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="qwen3vl4b")
    parser.add_argument("--dataset", choices=["dtd", "aircraft", "cub", "dogs", "pets"], default="dtd")
    parser.add_argument("--cache-dir", type=Path, default=Path(".hf_cache"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    spec = MODEL_SPECS[args.model]
    bank, query = load_dataset(args.dataset)
    labels = sorted({sample.label for sample in bank})
    model, processor = load_model(spec, args.cache_dir)
    zero_messages = [
        zero_shot_messages(sample, args.dataset, labels, spec.image_size) for sample in query[:2]
    ]
    inputs = prepare_batch(processor, zero_messages)
    states = anchor_hidden_states(model, inputs)
    expected_layers, expected_width = model_dimensions(model)
    if tuple(states.shape) != (2, expected_layers, expected_width):
        raise RuntimeError(f"Unexpected anchor shape: {tuple(states.shape)}")
    del inputs, states
    torch.cuda.empty_cache()

    messages = [
        icl_messages(bank[index * 4 : index * 4 + 4], sample, args.dataset, labels, spec.image_size)
        for index, sample in enumerate(query[:2])
    ]
    outputs = generate_legal_labels(model, processor, messages, labels)
    if any(prediction not in labels for prediction, _ in outputs):
        raise RuntimeError(f"Constrained generation escaped label set: {outputs}")
    result = {
        "model": spec.model_id,
        "dataset": args.dataset,
        "layers": expected_layers,
        "hidden_size": expected_width,
        "legal_label_count": len(labels),
        "outputs": [
            {"query_id": sample.sample_id, "target": sample.label, "prediction": prediction, "raw": raw}
            for sample, (prediction, raw) in zip(query[:2], outputs)
        ],
        "status": "passed",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
